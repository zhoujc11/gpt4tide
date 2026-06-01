from typing import Optional
import torch
from torch import nn, Tensor
import torch.nn.functional as F
import numpy as np
import math
# 假设这些从您的 utils 导入
from utils.TS_Pos_Enc import get_activation_fn, Transpose


# ======================
# Cross-modal block
# ======================
class CrossModal(nn.Module):
    def __init__(self, d_model, n_heads, d_k=None, d_v=None, d_ff=None,
                 norm='LayerNorm', attn_dropout=0., dropout=0., activation='relu',
                 res_attention=False, n_layers=1, pre_norm=False, store_attn=False,
                 rank=216, temperature_init=None, gate_init=1,
                 # *** PE 参数 ***
                 use_pe=False, pe_type='sincos', max_q_len=1024): 
        super().__init__()

        self.use_pe = use_pe
        if self.use_pe:
            from utils.TS_Pos_Enc import positional_encoding 
            pe_param = positional_encoding(pe=pe_type, learn_pe=False, 
                                           q_len=max_q_len, d_model=d_model)
            self.register_buffer('pe_weight', pe_param.data)
        
        # ============================================================
        # [核心修改 1] 语义适配器 (Semantic Adapter)
        # 解决泛化问题：让静态的词向量通过一个可训练的网络，适应当前任务
        # ============================================================
        self.semantic_adapter = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model)
        )
        # ============================================================

        self.layers = nn.ModuleList([
            TSTEncoderLayer(
                d_model, n_heads=n_heads, d_k=d_k, d_v=d_v, d_ff=d_ff, norm=norm,
                attn_dropout=attn_dropout, dropout=dropout, activation=activation,
                res_attention=res_attention, pre_norm=pre_norm, store_attn=store_attn,
                rank=rank, temperature_init=temperature_init, gate_init=gate_init
            ) for _ in range(n_layers)
        ])
        self.res_attention = res_attention
        self.last_attn_weights = None  # [B,H,Q,K]

    def forward(self, q: Tensor, k: Tensor, v: Tensor,
                key_padding_mask: Optional[Tensor] = None,  # [B, K]
                attn_mask: Optional[Tensor] = None):        # [1, Q, K]

        # ============================================================
        # [核心修改 2] 在 Attention 之前，先对 K 和 V 做适配变换
        # ============================================================
        # k, v 是原始的 GPT-2 向量，我们让它们过一遍 Adapter
        # k = self.semantic_adapter(k)
        # v = self.semantic_adapter(v)
        # ============================================================

        scores = None
        
        output = q
        for mod in self.layers:
            if self.res_attention:
                output, scores, attn_weights = mod(
                    output, k, v, prev=scores,
                    key_padding_mask=key_padding_mask, attn_mask=attn_mask
                )
            else:
                output, attn_weights = mod(
                    output, k, v,
                    key_padding_mask=key_padding_mask, attn_mask=attn_mask
                )
            self.last_attn_weights = attn_weights  # [B,H,Q,K]
        return output


# ======================
# Encoder layer
# ======================
class TSTEncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_k=None, d_v=None, d_ff=256, store_attn=False,
                 norm='LayerNorm', attn_dropout=0., dropout=0., bias=True, activation="gelu",
                 res_attention=False, pre_norm=False, rank=16, temperature_init=None, gate_init=0.1):
        super().__init__()
        assert d_model % n_heads == 0, f"d_model ({d_model}) must be divisible by n_heads ({n_heads})"
        d_k = d_model // n_heads if d_k is None else d_k
        d_v = d_model // n_heads if d_v is None else d_v

        self.res_attention = res_attention
        self.self_attn = _MultiheadAttention(
            d_model, n_heads, d_k, d_v,
            attn_dropout=attn_dropout, proj_dropout=dropout,
            res_attention=res_attention, rank=rank, temperature_init=temperature_init
        )

        # Add & Norm
        self.dropout_attn = nn.Dropout(dropout)
        if "batch" in norm.lower():
            self.norm_attn = nn.Sequential(Transpose(1, 2), nn.BatchNorm1d(d_model), Transpose(1, 2))
        else:
            self.norm_attn = nn.LayerNorm(d_model)

        # FFN
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff, bias=bias),
            get_activation_fn(activation),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model, bias=bias)
        )
        self.dropout_ffn = nn.Dropout(dropout)
        if "batch" in norm.lower():
            self.norm_ffn = nn.Sequential(Transpose(1, 2), nn.BatchNorm1d(d_model), Transpose(1, 2))
        else:
            self.norm_ffn = nn.LayerNorm(d_model)

        self.pre_norm = pre_norm
        self.store_attn = store_attn

        # 跨模态残差门控
        self.cross_gate = nn.Parameter(torch.tensor(float(gate_init)))

    def forward(self, q: Tensor, k: Tensor, v: Tensor, prev: Optional[Tensor] = None,
                key_padding_mask: Optional[Tensor] = None, attn_mask: Optional[Tensor] = None):
        # Multi-Head attention sublayer
        if self.pre_norm:
            q = self.norm_attn(q)

        if self.res_attention:
            q2, attn, scores = self.self_attn(q, k, v, prev,
                                              key_padding_mask=key_padding_mask, attn_mask=attn_mask)
        else:
            q2, attn = self.self_attn(q, k, v,
                                      key_padding_mask=key_padding_mask, attn_mask=attn_mask)

        if self.store_attn:
            self.attn = attn  # [B,H,Q,K]

        # 残差 + 门控
        q = q + self.cross_gate * self.dropout_attn(q2)
        if not self.pre_norm:
            q = self.norm_attn(q)

        # FFN
        if self.pre_norm:
            q = self.norm_ffn(q)
        q2 = self.ff(q)
        q = q + self.dropout_ffn(q2)
        if not self.pre_norm:
            q = self.norm_ffn(q)

        if self.res_attention:
            return q, scores, attn
        else:
            return q, attn


# ======================
# Multi-head Attention
# ======================
class _MultiheadAttention(nn.Module):
    def __init__(self, d_model, n_heads, d_k=None, d_v=None, res_attention=False,
                 attn_dropout=0., proj_dropout=0., qkv_bias=True, lsa=False,
                 rank=256, temperature_init=None):
        super().__init__()
        d_k = d_model // n_heads if d_k is None else d_k
        d_v = d_model // n_heads if d_v is None else d_v

        self.n_heads, self.d_k, self.d_v = n_heads, d_k, d_v

        self.W_Q = nn.Linear(d_model, d_k * n_heads, bias=qkv_bias)
        self.W_K = nn.Linear(d_model, d_k * n_heads, bias=qkv_bias)
        self.W_V = nn.Linear(d_model, d_v * n_heads, bias=qkv_bias)

        self.res_attention = res_attention

        self.sdp_attn = _ScaledDotProductAttention(
            d_model, n_heads, attn_dropout=attn_dropout, res_attention=self.res_attention,
            lsa=lsa, rank=rank, temperature_init=temperature_init
        )

        self.to_out = nn.Sequential(nn.Linear(n_heads * d_v, d_model), nn.Dropout(proj_dropout))

    def forward(self, Q: Tensor, K: Optional[Tensor] = None, V: Optional[Tensor] = None,
                prev: Optional[Tensor] = None,
                key_padding_mask: Optional[Tensor] = None, attn_mask: Optional[Tensor] = None):

        bs = Q.size(0)
        if K is None: K = Q
        if V is None: V = Q

        q_s = self.W_Q(Q).view(bs, -1, self.n_heads, self.d_k).transpose(1, 2)  # [B,H,Q,D]
        k_s = self.W_K(K).view(bs, -1, self.n_heads, self.d_k).transpose(1, 2)  # [B,H,K,D]
        v_s = self.W_V(V).view(bs, -1, self.n_heads, self.d_v).transpose(1, 2)  # [B,H,K,Dv]

        if self.res_attention:
            output, attn_weights, attn_scores = self.sdp_attn(
                q_s, k_s, v_s, prev=prev, key_padding_mask=key_padding_mask, attn_mask=attn_mask
            )
        else:
            output, attn_weights = self.sdp_attn(
                q_s, k_s, v_s, key_padding_mask=key_padding_mask, attn_mask=attn_mask
            )

        output = output.transpose(1, 2).contiguous().view(bs, -1, self.n_heads * self.d_v)
        output = self.to_out(output)

        if self.res_attention:
            return output, attn_weights, attn_scores
        else:
            return output, attn_weights


# ======================
# Low-rank Bilinear SDPA
# ======================
class _ScaledDotProductAttention(nn.Module):
    def __init__(self, d_model, n_heads, attn_dropout=0., res_attention=False, lsa=False,
                 rank=16, temperature_init=None):
        super().__init__()
        self.attn_dropout = nn.Dropout(attn_dropout)
        self.res_attention = res_attention

        D = d_model // n_heads
        r = rank

        # 低秩分解参数
        self.U = nn.Parameter(torch.randn(n_heads, D, r))  # 使用标准正态分布初始化
        self.V = nn.Parameter(torch.randn(n_heads, D, r))  # 使用标准正态分布初始化
        self.w = nn.Parameter(torch.randn(n_heads, r)) 
        # nn.init.normal_(self.w, mean=0.0, std=1.0)
        # nn.init.orthogonal_(self.U, gain=5.0)
        # nn.init.orthogonal_(self.V, gain=5.0)
        self.norm_q = nn.LayerNorm(D)
        self.norm_k = nn.LayerNorm(D)

        # 温度系数初始化
        if temperature_init is None:
            temperature_init = 0.1 # 默认尖锐
        
        self.log_temperature = nn.Parameter(torch.tensor(math.log(temperature_init)))

    def forward(self, q: Tensor, k: Tensor, v: Tensor, prev: Optional[Tensor] = None,
                key_padding_mask: Optional[Tensor] = None, attn_mask: Optional[Tensor] = None):
        
        B, H, Q, D = q.shape

        # 1. Norm
        q_r = self.norm_q(q)
        k_r = self.norm_k(k)

        # # 2. Low-rank projection
        q_r = torch.einsum('bhqd,hdr->bhqr', q, self.U)
        k_r = torch.einsum('bhkd,hdr->bhkr', k, self.V)

        # 3. L2 Normalize
        q_r = F.normalize(q_r, p=2, dim=-1)
        k_r = F.normalize(k_r, p=2, dim=-1)

        scores = torch.einsum('bhqr,hr,bhkr->bhqk', q_r, self.w, k_r) 

        temperature = self.log_temperature.exp()
        temperature = torch.clamp(temperature, min=0.001, max=100) 
        
        scores = scores / (temperature + 1e-6)
        #print(scores)
        
        
        # 6. 残差与掩码
        if prev is not None:
            scores = scores + prev

        if attn_mask is not None:
            if attn_mask.dtype == torch.bool:
                scores = scores.masked_fill(attn_mask, torch.finfo(scores.dtype).min)
            else:
                scores = scores + attn_mask

        if key_padding_mask is not None:
            scores = scores.masked_fill(
                key_padding_mask[:, None, None, :],
                torch.finfo(scores.dtype).min
            )

        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)

        out = torch.einsum('bhqk,bhkd->bhqd', attn_weights, v)

        if self.res_attention:
            return out, attn_weights, scores
        else:
            return out, attn_weights