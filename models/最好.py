from typing import Callable, Optional
import torch
from torch import nn
from torch import Tensor
import torch.nn.functional as F
import numpy as np
# 假设 get_activation_fn 和 Transpose 在这个包中
from utils.TS_Pos_Enc import *
    
# Cell
class CrossModal(nn.Module):
    def __init__(self, d_model, n_heads, d_k=None, d_v=None, d_ff=None, 
                 norm='LayerNorm', attn_dropout=0., dropout=0., activation='gelu',
                 res_attention=False, n_layers=1, pre_norm=False, store_attn=False):
        super().__init__()
    
        self.layers = nn.ModuleList([TSTEncoderLayer( d_model, n_heads=n_heads, d_k=d_k, d_v=d_v, d_ff=d_ff, norm=norm,
                                                      attn_dropout=attn_dropout, dropout=dropout,
                                                      activation=activation, res_attention=res_attention,
                                                      pre_norm=pre_norm, store_attn=store_attn) for i in range(n_layers)])
        self.res_attention = res_attention
        self.last_attn_weights = None
    def forward(self, q:Tensor,k:Tensor,v:Tensor, key_padding_mask:Optional[Tensor]=None, attn_mask:Optional[Tensor]=None):
        '''
        q [bs * nvars x (q_len) x d_model]
        k [bs * nvars x (k_len) x d_model]
        v [bs * nvars x (v_len) x d_model]
        '''
        scores = None
        output = q # 初始化输出为 q
        
        for mod in self.layers:
            # [修复 2]: 确保捕获 attn_weights
            if self.res_attention: 
                # TSTEncoderLayer 必须返回 (output, scores, attn_weights)
                output, scores, attn_weights = mod(output, k, v, prev=scores, key_padding_mask=key_padding_mask, attn_mask=attn_mask)
            else:
                # TSTEncoderLayer 必须返回 (output, attn_weights)
                output, attn_weights = mod(output, k, v, key_padding_mask=key_padding_mask, attn_mask=attn_mask)
            
            # [修复 3]: 存储权重

            self.last_attn_weights = attn_weights

        return output


class TSTEncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_k=None, d_v=None, d_ff=256, store_attn=False,
                 norm='LayerNorm', attn_dropout=0, dropout=0., bias=True, activation="gelu", res_attention=False, pre_norm=False):
        super().__init__()
        assert not d_model%n_heads, f"d_model ({d_model}) must be divisible by n_heads ({n_heads})"
        d_k = d_model // n_heads if d_k is None else d_k
        d_v = d_model // n_heads if d_v is None else d_v

        # Multi-Head attention
        self.res_attention = res_attention
        self.self_attn = _MultiheadAttention(d_model, n_heads, d_k, d_v, attn_dropout=attn_dropout, proj_dropout=dropout, res_attention=res_attention)

        # Add & Norm
        self.dropout_attn = nn.Dropout(dropout)
        if "batch" in norm.lower():
            self.norm_attn = nn.Sequential(Transpose(1,2), nn.LayerNorm1d(d_model), Transpose(1,2))
        else:
            self.norm_attn = nn.LayerNorm(d_model)

        # Position-wise Feed-Forward
        self.ff = nn.Sequential(nn.Linear(d_model, d_ff, bias=bias),
                                 get_activation_fn(activation),
                                 nn.Dropout(dropout),
                                 nn.Linear(d_ff, d_model, bias=bias))

        # Add & Norm
        self.dropout_ffn = nn.Dropout(dropout)
        if "batch" in norm.lower():
            self.norm_ffn = nn.Sequential(Transpose(1,2), nn.LayerNorm1d(d_model), Transpose(1,2))
        else:
            self.norm_ffn = nn.LayerNorm(d_model)

        self.pre_norm = pre_norm
        self.store_attn = store_attn


    def forward(self, q:Tensor,k:Tensor,v:Tensor, prev:Optional[Tensor]=None, key_padding_mask:Optional[Tensor]=None, attn_mask:Optional[Tensor]=None) -> Tensor:

        # Multi-Head attention sublayer
        if self.pre_norm:
            q = self.norm_attn(q)
            k = self.norm_attn(k)
            v = self.norm_attn(v)
        ## Multi-Head attention
        if self.res_attention:
            #print(q.shape,k.shape,v.shape)
            q2, attn, scores = self.self_attn(q, k, v, prev, key_padding_mask=key_padding_mask, attn_mask=attn_mask)
        else:
            q2, attn = self.self_attn(q, k, v, key_padding_mask=key_padding_mask, attn_mask=attn_mask)
        if self.store_attn:
            self.attn = attn
        ## Add & Norm
        q = q + self.dropout_attn(q2) # Add: residual connection with residual dropout
        if not self.pre_norm:
            q = self.norm_attn(q)

        # Feed-forward sublayer
        if self.pre_norm:
            q = self.norm_ffn(q)
        ## Position-wise Feed-Forward
        q2 = self.ff(q)
        ## Add & Norm
        q = q + self.dropout_ffn(q2) # Add: residual connection with residual dropout
        if not self.pre_norm:
            q = self.norm_ffn(q)

        if self.res_attention:
            return q, scores, attn # 返回 (output, scores, attn_weights)
        else:
            return q, attn # 返回 (output, attn_weights)


class _MultiheadAttention(nn.Module):
    def __init__(self, d_model, n_heads, d_k=None, d_v=None, res_attention=False, attn_dropout=0., proj_dropout=0., qkv_bias=True, lsa=False):
        super().__init__()
        d_k = d_model // n_heads if d_k is None else d_k
        d_v = d_model // n_heads if d_v is None else d_v

        self.n_heads, self.d_k, self.d_v = n_heads, d_k, d_v

        self.W_Q = nn.Linear(d_model, d_k * n_heads, bias=qkv_bias)
        self.W_K = nn.Linear(d_model, d_k * n_heads, bias=qkv_bias)
        self.W_V = nn.Linear(d_model, d_v * n_heads, bias=qkv_bias)

        # Scaled Dot-Product Attention (multiple heads)
        self.res_attention = res_attention
        # [!!!] 这里的 _ScaledDotProductAttention 已经被替换为下面的 Bilinear 版本
        self.sdp_attn = _ScaledDotProductAttention(d_model, n_heads, attn_dropout=attn_dropout, res_attention=self.res_attention, lsa=lsa)

        # Poject output
        self.to_out = nn.Sequential(nn.Linear(n_heads * d_v, d_model), nn.Dropout(proj_dropout))


    def forward(self, Q:Tensor, K:Optional[Tensor]=None, V:Optional[Tensor]=None, prev:Optional[Tensor]=None,
                key_padding_mask:Optional[Tensor]=None, attn_mask:Optional[Tensor]=None):

        bs = Q.size(0)
        if K is None: K = Q
        if V is None: V = Q

        # Linear (+ split in multiple heads)
        q_s = self.W_Q(Q).view(bs, -1, self.n_heads, self.d_k).transpose(1,2)      # q_s    : [bs x n_heads x max_q_len x d_k]
        k_s = self.W_K(K).view(bs, -1, self.n_heads, self.d_k).permute(0,2,3,1)    # k_s    : [bs x n_heads x d_k x q_len] - transpose(1,2) + transpose(2,3)
        v_s = self.W_V(V).view(bs, -1, self.n_heads, self.d_v).transpose(1,2)      # v_s    : [bs x n_heads x q_len x d_v]

        # Apply Scaled Dot-Product Attention (multiple heads)
        if self.res_attention:
            output, attn_weights, attn_scores = self.sdp_attn(q_s, k_s, v_s, prev=prev, key_padding_mask=key_padding_mask, attn_mask=attn_mask)
        else:
            output, attn_weights = self.sdp_attn(q_s, k_s, v_s, key_padding_mask=key_padding_mask, attn_mask=attn_mask)
        # output: [bs x n_heads x q_len x d_v], attn: [bs x n_heads x q_len x q_len], scores: [bs x n_heads x max_q_len x q_len]

        # back to the original inputs dimensions
        output = output.transpose(1, 2).contiguous().view(bs, -1, self.n_heads * self.d_v) # output: [bs x q_len x n_heads * d_v]
        output = self.to_out(output)

        if self.res_attention: return output, attn_weights, attn_scores
        else: return output, attn_weights

# =============================================================================
# [!!! 创新点 2：修改后的注意力得分计算 !!!]
# =============================================================================

class _ScaledDotProductAttention(nn.Module):
    r""" 
    [MODIFIED]
    Scaled Dot-Product Attention (SDPA)
    
    This implementation replaces the standard dot-product similarity (q @ k)
    with a Low-Rank Bilinear Attention mechanism (v.T @ tanh(q * k + b)).
    This is designed to capture multi-modal feature *interactions* (element-wise product)
    rather than just *similarity* (dot-product).
    """

    def __init__(self, d_model, n_heads, attn_dropout=0., res_attention=False, lsa=False):
        super().__init__()
        self.attn_dropout = nn.Dropout(attn_dropout)
        self.res_attention = res_attention
        
        head_dim = d_model // n_heads
        
        # [INNOVATION 2] 移除 SDPA 的 scale
        # self.scale = nn.Parameter(torch.tensor(head_dim ** -0.5), requires_grad=lsa)
        self.lsa = lsa # 保持 lsa 参数，即使 scale 被移除，以防 lsa 有其他用途

        # --- [INNOVATION 2: 双线性注意力参数] ---
        d_k = head_dim
        # v_a 向量，用于将 d_k 维的交互向量压缩为 1 维的得分
        self.bilinear_v = nn.Parameter(torch.randn(d_k, 1))
        # 偏置项
        self.bilinear_b = nn.Parameter(torch.zeros(1))
        # --- [INNOVATION 2 END] ---


    def forward(self, q:Tensor, k:Tensor, v:Tensor, prev:Optional[Tensor]=None, key_padding_mask:Optional[Tensor]=None, attn_mask:Optional[Tensor]=None):
        '''
        Input shape:
            q                 : [bs x n_heads x max_q_len x d_k]
            k                 : [bs x n_heads x d_k x seq_len]  <-- 原始 k 形状
            v                 : [bs x n_heads x seq_len x d_v]
            prev              : [bs x n_heads x q_len x seq_len]
            key_padding_mask: [bs x seq_len]
            attn_mask         : [1 x seq_len x seq_len]
        Output shape:
            output:  [bs x n_heads x q_len x d_v]
            attn   : [bs x n_heads x q_len x seq_len]
            scores : [bs x n_heads x q_len x seq_len]
        '''

        # --- [INNOVATION 2: 双线性注意力计算] ---
        
        # 1. 转置 K 的形状以进行广播
        # k: [bs x n_heads x d_k x seq_len] -> [bs x n_heads x seq_len x d_k]
        k_s = k.transpose(-1, -2) 
        
        # 2. 广播元素乘法 (q * k) (Hadamard product)
        # q: [bs x n_heads x max_q_len x 1       x d_k]
        # k: [bs x n_heads x 1         x seq_len x d_k]
        # -> [bs x n_heads x max_q_len x seq_len x d_k]
        q_k_interaction = q.unsqueeze(-2) * k_s.unsqueeze(-3)
        
        # 3. 应用非线性激活
        activated = torch.tanh(q_k_interaction)
        
        # 4. 计算得分: v.T @ tanh(...) + b
        # (..., d_k) @ (d_k, 1) -> (..., 1)
        attn_scores = torch.matmul(activated, self.bilinear_v) + self.bilinear_b
        
        # 5. 移除最后一个维度
        attn_scores = attn_scores.squeeze(-1) # [bs x n_heads x max_q_len x seq_len]
        # --- [INNOVATION 2 END] ---


        # --- [以下为原始代码，保持不变] ---

        # Add pre-softmax attention scores from the previous layer (optional)
        if prev is not None: attn_scores = attn_scores + prev

        # Attention mask (optional)
        if attn_mask is not None:
            if attn_mask.dtype == torch.bool:
                attn_scores.masked_fill_(attn_mask, -np.inf)
            else:
                attn_scores += attn_mask

        # Key padding mask (optional)
        if key_padding_mask is not None:
            attn_scores.masked_fill_(key_padding_mask.unsqueeze(1).unsqueeze(2), -np.inf)

        # normalize the attention weights
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)

        # compute the new values given the attention weights
        output = torch.matmul(attn_weights, v)

        if self.res_attention: return output, attn_weights, attn_scores
        else: return output, attn_weights