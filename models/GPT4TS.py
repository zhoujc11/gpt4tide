import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# --- 可视化所需的库 ---
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

from torch import optim
from transformers import AutoTokenizer, AutoConfig
from einops import rearrange
from transformers.models.gpt2.modeling_gpt2 import GPT2Model
from transformers.models.gpt2.configuration_gpt2 import GPT2Config
from transformers import BertTokenizer, BertModel
from peft import LoraConfig, get_peft_model

# (假设这些 utils 存在于你的项目中)
from utils.Cross_Modal_Align import CrossModal
from utils.Retrieval import RetrievalTool
from utils.StandardNorm import Normalize


# =============================================================================
# Prompt Attention 可视化辅助函数 (增强版：过滤纯空格)
# =============================================================================
def visualize_prompt_importance_heatmap(attn_weights_tensor: torch.Tensor,
                                        prompt_tokens: list,
                                        n_heads: int,
                                        filename: str,
                                        sample_interval: int = 1,
                                        show_words_on_x: bool = True): # [新增参数] 控制横坐标显示模式
    """
    计算并可视化所有 Patch Token 对 Prompt Token 的注意力热力图。
    增强功能：
    1. 自动过滤掉仅包含空格的无意义 Token。
    2. 支持切换横坐标显示单词或下标。
    """
    print(f"Generating Patch-to-Prompt Heatmap...")

    # 1. 数据准备 [L_Patch, L_Prompt]
    avg_attn_per_patch = attn_weights_tensor.mean(dim=(0, 1)).cpu().numpy()
    plot_data = avg_attn_per_patch  
    
    # 2. 强制对齐长度 (防止 IndexError)
    L_Patch, tensor_width = plot_data.shape
    list_len = len(prompt_tokens)
    valid_len = min(tensor_width, list_len)
    
    plot_data = plot_data[:, :valid_len]
    prompt_tokens = prompt_tokens[:valid_len]
    
    # 3. 过滤掉纯空格 Token (如 'Ġ', ' ', '') 
    valid_indices = []
    cleaned_labels = []
    
    for idx, token in enumerate(prompt_tokens):
        text = str(token).replace('Ġ', '').strip()
        if len(text) > 0:
            valid_indices.append(idx)
            cleaned_labels.append(text)
            
    if not valid_indices:
        print("Warning: No valid tokens found for visualization!")
        return

    # 只保留有效的列
    plot_data = plot_data[:, valid_indices]
    L_Prompt = plot_data.shape[1]

    # 4. 绘制热力图
    figsize_height = max(6, L_Patch * 0.08)
    # 根据显示模式调整宽度：显示单词需要更宽，显示下标可以窄一点
    width_factor = 0.2 if show_words_on_x else 0.1
    plt.figure(figsize=(L_Prompt * width_factor + 2, figsize_height)) 
    
    plt.imshow(plot_data, aspect='auto', cmap='viridis')
    
    # --- [核心修改] 横坐标控制逻辑 ---
    x_indices_all = np.arange(L_Prompt)
    
    if show_words_on_x:
        # 模式 A: 显示单词 (使用传入的 sample_interval)
        sampled_indices = x_indices_all[::sample_interval] 
        x_labels = [cleaned_labels[i] for i in sampled_indices]
        rotation = 90
        fontsize = 8
    else:
        # 模式 B: 显示下标 (固定间隔 20)
        index_interval = 20
        sampled_indices = x_indices_all[::index_interval]
        x_labels = [str(i) for i in sampled_indices] # 显示数字下标
        rotation = 0 # 数字横着放更好看
        fontsize = 10

    plt.xticks(sampled_indices, x_labels, rotation=rotation, ha='center', fontsize=fontsize)
    # --------------------------------
    
    y_ticks = np.linspace(0, L_Patch - 1, min(10, L_Patch), dtype=int)
    plt.yticks(y_ticks, [f'Patch {i}' for i in y_ticks], fontsize=8) 
    
    plt.tight_layout()
    plt.savefig(filename, dpi=600)
    print(f">>> Patch-to-Prompt Heatmap saved to: {filename}")
    plt.close()

    # 5. 打印 Top 5 (始终打印单词，方便分析)
    overall_scores = plot_data.mean(axis=0)
    sorted_indices = np.argsort(overall_scores)[::-1]
    
    print("\n--- Top 5 Most Important Prompt Words (Filtered) ---")
    for i in sorted_indices[:5]:
        if i < len(cleaned_labels):
            print(f"- Index {i} ('{cleaned_labels[i]}'): {overall_scores[i]:.4f}") 
    print("----------------------------------------------------------")


# =============================================================================
# GPT4TS 模型类 (集成 Tokenizer 注册机制 & 空格优化)
# =============================================================================
class GPT4TS(nn.Module):
    
    def __init__(self, configs, device, initial_periods_hours=None, initial_powers=None):
        super(GPT4TS, self).__init__()
        
        # --- 配置与状态 ---
        self.configs = configs 
        self.do_visualize = getattr(configs, "do_visualize", True)
        self.do_tfce = getattr(configs, "do_tfce", False)
        self.do_attn_viz = getattr(configs, "do_attn_viz", True)
        
        # [修改] 使用集合来记录已绘图的状态 (mode, epoch)，防止在同一个 Epoch 内重复画图
        self.viz_done_states = set()
        
        self.device = device
        self.is_gpt = configs.is_gpt
        self.is_cross = configs.is_cross
        self.is_tpe = configs.is_tpe
        self.patch_size = configs.patch_size
        self.pretrain = configs.pretrain
        self.stride = configs.stride
        self.pred_len = configs.pred_len
        self.seq_len = configs.seq_len
        self.patch_num = (configs.seq_len - self.patch_size) // self.stride + 1
        self.dropout_n = getattr(configs, "dropout", 0.1) 
        self.top_k = getattr(configs, "top_k_lags", 5) 
        
        self.gpt2_local_dir = getattr(configs, "gpt2_local_dir", None)
        self.local_files_only = bool(getattr(configs, "local_files_only", True))
        
        self.normalize_layers = Normalize(configs.enc_in, affine=False)
        self.padding_patch_layer = nn.ReplicationPad1d((0, self.stride))
        self.patch_num += 1
        
        # CrossModal: L-RBA 模块
        self.cross = CrossModal(
            d_model=configs.d_model, n_heads=1, d_ff=128, norm='LayerNorm',
            attn_dropout=self.dropout_n, dropout=self.dropout_n,
            pre_norm=True, activation="gelu", res_attention=True,
            n_layers=1, store_attn=self.do_attn_viz or self.do_visualize
        ).to(device)
        
        # ---------------- GPT-2 加载与 Token 注册 ----------------
        if configs.is_gpt:
            model_path = '/home/gpt4tide/llm/openai-community/gpt2'
            
            # 1. 先加载 Tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_path,
                trust_remote_code=True,
                local_files_only=self.local_files_only
            )
            if self.tokenizer.eos_token:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            else:
                self.tokenizer.pad_token = '[PAD]'

            # 注册物理词汇 (含空格版本)
            base_physics_words = [
                'Abnormal', 'Abrupt', 'Accelerate', 'Active', 'Angle', 'Arc', 'Ascend', 'Asymptote', 
                'Average', 'Basin', 'Boost', 'Bounce', 'Brake', 'Brief', 'Bumpy', 'Calm', 'Canyon', 
                'Chaotic', 'Clean', 'Clear', 'Climb', 'Collapse', 'Common', 'Concave', 'Continuous', 
                'Convex', 'Correction', 'Cosine', 'Crash', 'Creep', 'Crest', 'Critical', 'Curve', 
                'Cyclic', 'Danger', 'Decay', 'Decelerate', 'Delayed', 'Dense', 'Depleted', 'Descend', 
                'Deterministic', 'Dip', 'Discrete', 'Distorted', 'Dive', 'Dormant', 'Double-Peak', 
                'Downward', 'Drift', 'Drop', 'Dynamic', 'Early', 'Empty', 'Erratic', 'Extreme', 
                'Fall', 'Fast', 'Flat', 'Flatline', 'Frequent', 'Full', 'Fuzzy', 'Gain', 'Gap', 
                'Gentle', 'Glide', 'Gradual', 'Growth', 'Harmonic', 'Harsh', 'Head-and-Shoulders', 
                'Heavy', 'High', 'Impulse', 'Inert', 'Inflection', 'Instant', 'Intense', 'Intermittent', 
                'Irregular', 'Jagged', 'Jittery', 'Jump', 'Kinetic', 'Kink', 'Lagged', 'Late', 
                'Leading', 'Light', 'Line', 'Long', 'Loop', 'Loss', 'Low', 'Maximum', 'Median', 
                'Messy', 'Mid', 'Mild', 'Minimum', 'Negative', 'Noisy', 'Nominal', 'Normal', 'Null', 
                'Passive', 'Peak', 'Periodic', 'Permanent', 'Persistent', 'Plateau', 'Plunge', 
                'Positive', 'Potential', 'Prolonged', 'Pulse', 'Quiet', 'Rally', 'Ramp', 'Random', 
                'Rapid', 'Rare', 'Rebound', 'Recover', 'Regular', 'Retrace', 'Reverse', 'Rhythmic', 
                'Ridge', 'Ripple', 'Rise', 'Rocket', 'Rough', 'Safe', 'Saturated', 'Sawtooth', 
                'Severe', 'Shaky', 'Short', 'Sine', 'Slide', 'Slope', 'Slow', 'Slump', 'Smooth', 
                'Soar', 'Soft', 'Sparse', 'Spike', 'Square', 'Stable', 'Static', 'Steady', 'Step', 
                'Stochastic', 'Strong', 'Sudden', 'Surge', 'Tangent', 'Temporary', 'Trailing', 
                'Transient', 'Triangle', 'Trough', 'Tumble', 'Turbulent', 'Unstable', 'Upward', 
                'Valley', 'Violent', 'Void', 'Volatile', 'Warning', 'Wave', 'Weak', 'Wild'
            ]
            
            tokens_to_add = []
            for w in base_physics_words:
                tokens_to_add.append(w)          
                tokens_to_add.append("Ġ" + w)    
                
            num_added = self.tokenizer.add_tokens(tokens_to_add)
            print(f">>> Registered {num_added} special physics tokens (including space-prefixed).")

            self.gpt2 = GPT2Model.from_pretrained(
                model_path,
                output_attentions=True,
                output_hidden_states=True,
                local_files_only=self.local_files_only,
            )
            self.gpt2.h = self.gpt2.h[:configs.gpt_layers]
            
            self.gpt2.resize_token_embeddings(len(self.tokenizer))
            print(f">>> Resized model embeddings to {len(self.tokenizer)}")

            if hasattr(self.gpt2.config, "n_embd"):
                assert configs.d_model == self.gpt2.config.n_embd

            strategy = configs.strategy
            print(f"Applying Finetune Strategy: {strategy}")

            if strategy == 'lora':
                lora_r = getattr(configs, "lora_r", 16)
                lora_alpha = getattr(configs, "lora_alpha", 32)
                peft_config = LoraConfig(
                    r=lora_r,
                    lora_alpha=lora_alpha,
                    target_modules=["wpe", "c_attn"], 
                    lora_dropout=self.dropout_n, 
                    bias="none",
                    modules_to_save=["wte"] 
                )
                self.gpt2 = get_peft_model(self.gpt2, peft_config)
                print("GPT-2 (PEFT) trainable parameters:")
                self.gpt2.print_trainable_parameters()
            elif strategy == 'frozen':
                for name, param in self.gpt2.named_parameters():
                    param.requires_grad = False
                self.gpt2.get_input_embeddings().weight.requires_grad = True
            elif strategy == 'partial_wpe_ln':
                for name, param in self.gpt2.named_parameters():
                    if ('ln' in name) or ('wpe' in name) or ('wte' in name):
                        param.requires_grad = True
                    else:
                        param.requires_grad = False
            elif strategy == 'full':
                pass
                
        self.llm_model = self.gpt2.to(device)
        
        self.in_layer = nn.Linear(configs.patch_size, configs.d_model).to(device)
        self.out_layer = nn.Linear(configs.d_model * self.patch_num, configs.pred_len).to(device)

        # ==========================================================
        #         TPE 初始化 
        # ==========================================================
        d_model = configs.d_model
        self.k_periods = getattr(configs, "k_periods", 7) 
        
        if initial_periods_hours is None:
             initial_periods_hours = torch.tensor([12.42, 12.0, 12.66, 11.97, 23.93, 25.82, 24.07], dtype=torch.float32)
             initial_powers = torch.ones(self.k_periods, dtype=torch.float32)

        initial_periods_hours = initial_periods_hours.to(torch.float32)
        initial_powers = initial_powers.to(torch.float32)

        initial_w = 2.0 * math.pi / (initial_periods_hours + 1e-6)
        self.tpe_w = nn.Parameter(initial_w)
        self.tpe_phase = nn.Parameter(torch.zeros(self.k_periods))

        initial_amplitudes = torch.sqrt(initial_powers + 1e-8)
        init_logits = torch.log(initial_amplitudes + 1e-6)
        self.tpe_amp_logits = nn.Parameter(init_logits)

        self.tpe_temp = nn.Parameter(torch.tensor(0.4)) 
        self.tpe_sharpness = nn.Parameter(torch.tensor(2.5))
        self.tpe_proj = nn.Linear(self.k_periods * 2, d_model).to(device) 
        self.tpe_scale = nn.Parameter(torch.tensor(1.0))
        self.tpe_gate = nn.Parameter(torch.tensor(0.0))
        
        self.cached_prompt_embeddings = None
        self.cached_prompt_ids = None

    # ====================== t-SNE 画图 ======================
    @staticmethod
    def _plot_tsne(ax, data: torch.Tensor, title: str):
        data_np = data.detach().cpu().numpy()
        n_tokens = data_np.shape[0]
        perplexity = min(30.0, float(max(n_tokens - 1, 2)))
        tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42, init='pca', learning_rate='auto')
        tsne_results = tsne.fit_transform(data_np)
        ax.scatter(tsne_results[:, 0], tsne_results[:, 1], alpha=0.7, s=15)
        ax.set_title(title)
        ax.set_xticks([]); ax.set_yticks([])

    def _build_tpe(self, tpe_x_mark: torch.Tensor, device, dtype):
        # ... (TPE 逻辑保持不变) ...
        proj_dtype = self.tpe_proj.weight.dtype
        day_of_month = tpe_x_mark[..., 1]
        day_of_week  = tpe_x_mark[..., 2]
        hour_of_day  = tpe_x_mark[..., 3]
        minute_of_hour = tpe_x_mark[..., 4]
        t = day_of_week * 24 + hour_of_day + (minute_of_hour / 60.0)
        t = t.to(dtype=proj_dtype)
        w   = self.tpe_w.to(device=device, dtype=proj_dtype)
        phi = self.tpe_phase.to(device=device, dtype=proj_dtype)
        amp_logits = self.tpe_amp_logits.to(device=device, dtype=proj_dtype)
        tau = torch.clamp(self.tpe_temp.to(device=device, dtype=proj_dtype), 0.1, 2.0)
        weights = F.softmax(amp_logits / tau, dim=-1)
        gamma = torch.clamp(self.tpe_sharpness.to(device=device, dtype=proj_dtype), 1.0, 5.0)
        weights = weights ** gamma
        weights = weights / (weights.sum() + 1e-8)
        wt_phi = t.unsqueeze(-1) * w + phi
        cos_terms = torch.cos(wt_phi)
        sin_terms = torch.sin(wt_phi)
        feats = torch.cat([sin_terms, cos_terms], dim=-1)
        amp_expanded = weights.repeat(2).unsqueeze(0).unsqueeze(0)
        feats = feats * amp_expanded
        tpe = self.tpe_proj(feats)
        return tpe

    def calcute_lags(self, x_enc):
        x_enc_permuted = x_enc.permute(0, 2, 1).contiguous()
        x_enc_fft = torch.fft.rfft(x_enc_permuted, dim=-1)
        res = x_enc_fft * torch.conj(x_enc_fft)
        corr = torch.fft.irfft(res, dim=-1)
        mean_value = torch.mean(corr, dim=1)
        _, lags = torch.topk(mean_value, self.top_k, dim=-1)
        return lags
    
    # ====================== 前向传播 ======================
    def forward(self, x, batch_x_mark, itr, index, mode, epoch=None):
        x = self.normalize_layers(x, 'norm')
        
        x_enc = x
        B0, T, N = x_enc.size()
        x_enc_prompt = x_enc.permute(0, 2, 1).contiguous().reshape(B0 * N, T, 1)
        current_device = x_enc.device
        
        # [优化核心] 静态 Prompt 处理：只计算一次
        if self.cached_prompt_embeddings is None:
            static_text = (
                f"Abnormal Abrupt Accelerate Active Angle Arc Ascend Asymptote Average Basin "
                f"Boost Bounce Brake Brief Bumpy Calm Canyon Chaotic Clean Clear "
                f"Climb Collapse Common Concave Continuous Convex Correction Cosine Crash Creep "
                f"Crest Critical Curve Cyclic Danger Decay Decelerate Delayed Dense Depleted "
                f"Descend Deterministic Dip Discrete Distorted Dive Dormant Double-Peak Downward Drift "
                f"Drop Dynamic Early Empty Erratic Extreme Fall Fast Flat Flatline "
                f"Frequent Full Fuzzy Gain Gap Gentle Glide Gradual Growth Harmonic "
                f"Harsh Head-and-Shoulders Heavy High Impulse Inert Inflection Instant Intense Intermittent "
                f"Irregular Jagged Jittery Jump Kinetic Kink Lagged Late Leading Light "
                f"Line Long Loop Loss Low Maximum Median Messy Mid Mild "
                f"Minimum Negative Noisy Nominal Normal Null Passive Peak Periodic Permanent "
                f"Persistent Plateau Plunge Positive Potential Prolonged Pulse Quiet Rally Ramp "
                f"Random Rapid Rare Rebound Recover Regular Retrace Reverse Rhythmic Ridge "
                f"Ripple Rise Rocket Rough Safe Saturated Sawtooth Severe Shaky Short "
                f"Sine Slide Slope Slow Slump Smooth Soar Soft Sparse Spike "
                f"Square Stable Static Steady Step Stochastic Strong Sudden Surge Tangent "
                f"Temporary Trailing Transient Triangle Trough Tumble Turbulent Unstable Upward Valley "
                f"Violent Void Volatile Warning Wave Weak Wild Zero "
            )
            
            single_prompt = f"{static_text}"
            encoding = self.tokenizer([single_prompt], return_tensors="pt", padding=True, truncation=True, max_length=512)
            input_ids_1 = encoding.input_ids.to(current_device)
            self.cached_prompt_ids = input_ids_1 # 缓存 ID
            
        # 每次 forward 重新计算 embedding 以支持梯度回传 (如果 wte 可训练)
        prompt_embeddings_1 = self.llm_model.get_input_embeddings()(self.cached_prompt_ids)
        batch_size = x_enc_prompt.shape[0]
        prompt_embeddings = prompt_embeddings_1.expand(batch_size, -1, -1)
        viz_input_ids = self.cached_prompt_ids[0]
        
        # --- 数据 Patching 处理 ---
        x_tokens_in_raw = rearrange(x, 'b l m -> b m l')
        x_tokens_in_raw = self.padding_patch_layer(x_tokens_in_raw)
        x_tokens_in_raw = x_tokens_in_raw.unfold(dimension=-1, size=self.patch_size, step=self.stride)
        x_tokens_in = rearrange(x_tokens_in_raw, 'b m n p -> (b m) n p')
        tokens = self.in_layer(x_tokens_in)
        
        # ============================================================
        # [绘图判断逻辑] 
        # ============================================================
        current_epoch = epoch if epoch is not None else -1
        target_epochs = [0,2,8, 5,10,12,15,18, 20,22,25,28,30,32,35,38,40,42,45]
        
        should_plot = False
        plot_suffix = ""
        
        if mode == 'test':
            should_plot = True
            plot_suffix = "test"
        elif mode == 'train' and current_epoch in target_epochs:
            should_plot = True
            plot_suffix = f"train_epoch{current_epoch}"
            
        # 检查是否重复画图 (每个 epoch 只画一次)
        state_key = (mode, current_epoch)
        if state_key in self.viz_done_states:
            should_plot = False
        
        run_visualization = self.do_visualize and should_plot
        # ============================================================

        base_tokens_viz = None
        prompt_tokens_viz = None
        tpe_viz = None
        
        if run_visualization:
            base_tokens_viz = tokens.detach().clone()
            
        # --- CrossModal (L-RBA) 融合 ---
        if self.is_cross:
            tokens = self.cross(tokens, prompt_embeddings, prompt_embeddings)
            
            if run_visualization:
                prompt_tokens_viz = tokens.detach().clone()
            
            if self.do_attn_viz and run_visualization:
                attn_weights = self.cross.last_attn_weights
                
                # 捕获 Prompt Tokens 用于可视化
                prompt_tokens_list = self.tokenizer.convert_ids_to_tokens(viz_input_ids.tolist())
                
                # 截断 Pad Token
                try:
                    pad_token_id = self.tokenizer.pad_token_id
                    if (viz_input_ids == pad_token_id).any():
                         first_pad_index = (viz_input_ids == pad_token_id).nonzero(as_tuple=True)[0][0].item()
                    else:
                         first_pad_index = len(prompt_tokens_list)
                    
                    prompt_tokens_list_cleaned = prompt_tokens_list[:first_pad_index]
                    attn_weights = attn_weights[:, :, :, :first_pad_index]
                except (IndexError, AttributeError):
                    prompt_tokens_list_cleaned = prompt_tokens_list

                # 生成文件名
                base_path = getattr(self.configs, "viz_prompt_path", "prompt_attention_heatmap.jpg")
                if "." in base_path:
                    name, ext = base_path.rsplit(".", 1)
                    heatmap_filename = f"{name}_{plot_suffix}.{ext}"
                else:
                    heatmap_filename = f"{base_path}_{plot_suffix}"

                n_heads_val = self.cross.layers[0].self_attn.n_heads 
                
                visualize_prompt_importance_heatmap(
                    attn_weights.detach(), 
                    prompt_tokens_list_cleaned, 
                    n_heads=n_heads_val, 
                    filename=heatmap_filename,
                    sample_interval=1,
                    show_words_on_x=False
                )
                
        # --- TPE 注入 ---
        if self.is_tpe:
            patch_start_indices = torch.arange(0, T - self.patch_size + 1, self.stride, device=current_device)
            tpe_x_mark = batch_x_mark[:, patch_start_indices, :] 
            last_time_mark = tpe_x_mark[:, -1:, :] 
            tpe_x_mark = torch.cat([tpe_x_mark, last_time_mark], dim=1) 
            _, n_patch, n_feat = tpe_x_mark.shape 
            tpe_x_mark = tpe_x_mark.unsqueeze(2).expand(-1, -1, N, -1) 
            tpe_x_mark = tpe_x_mark.permute(0, 2, 1, 3).reshape(B0 * N, n_patch, n_feat) 
            
            tpe_raw = self._build_tpe(tpe_x_mark, device=tokens.device, dtype=tokens.dtype)
            tpe_scaled = tpe_raw.to(tokens.dtype)
            
            if run_visualization:
                tpe_viz = tpe_scaled.clone()
            
            gate = torch.sigmoid(self.tpe_gate)
            tokens = tokens + gate * self.tpe_scale * tpe_scaled

        # --- GPT-2 Backbone ---
        if self.is_gpt:
            tokens = self.gpt2(inputs_embeds=tokens).last_hidden_state
            
            if run_visualization and base_tokens_viz is not None:
                try:
                    last_tokens_viz = tokens.detach().clone()
                    sample_idx = 0
                    data_1 = base_tokens_viz[sample_idx]
                    data_2 = prompt_tokens_viz[sample_idx] if prompt_tokens_viz is not None else base_tokens_viz[sample_idx]
                    if tpe_viz is not None:
                        tpe_sum = (prompt_tokens_viz if prompt_tokens_viz is not None else base_tokens_viz) + tpe_viz
                        data_3 = tpe_sum[sample_idx]
                    else:
                        data_3 = data_2
                    data_4 = last_tokens_viz[sample_idx]
                    
                    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
                    self._plot_tsne(axes[0, 0], data_1, '1. Original Patch')
                    self._plot_tsne(axes[0, 1], data_2, '2. + Prompt (L-RBA)')
                    self._plot_tsne(axes[1, 0], data_3, '3. + TPE')
                    self._plot_tsne(axes[1, 1], data_4, '4. Final GPT Output')
                    
                    base_tsne_path = getattr(self.configs, "viz_plot_path", "embedding_tsne_combined.png")
                    if "." in base_tsne_path:
                        name, ext = base_tsne_path.rsplit(".", 1)
                        tsne_filename = f"{name}_{plot_suffix}.{ext}"
                    else:
                        tsne_filename = f"{base_tsne_path}_{plot_suffix}"
                    
                    plt.savefig(tsne_filename)
                    plt.close(fig)
                    
                    # 标记已完成
                    self.viz_done_states.add(state_key)
                    
                except Exception as e:
                    print(f"t-SNE Viz Error: {e}")
        
        outputs = self.out_layer(tokens.reshape(B0 * N, -1))
        outputs = rearrange(outputs, '(b m) l -> b l m', b=B0)
        outputs = self.normalize_layers(outputs, 'denorm')
        return outputs