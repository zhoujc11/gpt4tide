import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# --- [新导入] 可视化所需的库 ---
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
# --- [导入结束] ---

from torch import optim
from transformers import AutoTokenizer, AutoConfig
from einops import rearrange
from transformers.models.gpt2.modeling_gpt2 import GPT2Model
from transformers.models.gpt2.configuration_gpt2 import GPT2Config
from transformers import BertTokenizer, BertModel

# (假设这些 utils 存在于你的项目中)
from utils.Cross_Modal_Align import CrossModal
from utils.Retrieval import RetrievalTool
from utils.StandardNorm import Normalize

# =============================================================================
# (FFT 辅助函数 - 保持原样)
# =============================================================================

def find_top_k_periods_fft(time_series_data, dt_hours, k=4):
    """
    使用 FFT 分析整个时间序列数据集（或一个大样本），找到最强的 K 个周期。
    ... (函数内容保持不变) ...
    """
    print(f"Running FFT to find top-{k} periods...")
    
    if time_series_data.ndim == 3:
        ts = time_series_data[:, :, 0].flatten()
    elif time_series_data.ndim == 2:
        ts = time_series_data[:, 0].flatten()
    else:
        ts = time_series_data.flatten()
        
    ts = ts - np.mean(ts)
    n = len(ts)
    if n < 100: 
        print(f"Warning: FFT data length ({n}) is very short.")
        
    fft_result = np.fft.rfft(ts)
    fft_power = np.abs(fft_result)**2
    frequencies_hz = np.fft.rfftfreq(n, d=dt_hours) 
    
    if len(frequencies_hz) > 0:
        fft_power[0] = 0
    
    top_k_indices = np.argsort(fft_power[1:])[-k:] + 1
    top_k_frequencies = frequencies_hz[top_k_indices]
    
    valid_freq_mask = np.abs(top_k_frequencies) > 1e-8
    top_k_periods_hours = np.zeros_like(top_k_frequencies)
    top_k_periods_hours[valid_freq_mask] = 1.0 / top_k_frequencies[valid_freq_mask]
    
    top_k_periods_hours = np.sort(top_k_periods_hours)
    print(f"Found periods (hours): {top_k_periods_hours}")
    return torch.tensor(top_k_periods_hours, dtype=torch.float32)

# =============================================================================
# (优化后) GPT4TS 模型类
# =============================================================================

class GPT4TS(nn.Module):
    # <-- [MODIFIED] 移除了 __init__ 中的 device 参数
    def __init__(self, configs, device,initial_periods_hours=None):
        super(GPT4TS, self).__init__()
        
        # --- [新代码] 添加配置和状态位 ---
        self.configs = configs # 保存 configs 以便后续使用
        self.do_visualize = getattr(configs, "do_visualize", False)
        self._visualization_done = False # 确保只画一次
        # --- [新代码结束] ---
        
        self.device = device
        self.is_gpt = configs.is_gpt
        self.is_cross = configs.is_cross
        self.is_tpe = configs.is_tpe
        self.is_retrieval = configs.is_retrieval
        self.topm = configs.topm
        self.patch_size = configs.patch_size
        self.pretrain = configs.pretrain
        self.stride = configs.stride
        self.pred_len = configs.pred_len
        self.seq_len = configs.seq_len
        self.patch_num = (configs.seq_len - self.patch_size) // self.stride + 1
        self.dropout_n = getattr(configs, "dropout", 0.1)
        self.top_k = getattr(configs, "top_k_lags", 5) 
        self.step_minutes = getattr(configs, "step_minutes", 15)
        
        # <-- [NEW] 从 configs 读取时间特征的维度
        # 假设 data_loader 输出 [M, D, W, H, Min] 5个特征
        self.time_feat_dim = getattr(configs, "time_feat_dim", 5) 

        self.gpt2_local_dir = getattr(configs, "gpt2_local_dir", None)
        self.local_files_only = bool(getattr(configs, "local_files_only", True))
        
        self.normalize_layers = Normalize(configs.enc_in, affine=False)
        self.padding_patch_layer = nn.ReplicationPad1d((0, self.stride))
        self.patch_num += 1
        self.retrieval_dict = {'train': None, 'valid': None, 'test': None}

        self.cross = CrossModal(
            d_model=configs.d_model, n_heads=1, d_ff=128, norm='LayerNorm',
            attn_dropout=self.dropout_n, dropout=self.dropout_n,
            pre_norm=True, activation="gelu", res_attention=True,
            n_layers=1, store_attn=False
        ).to(device)
        
        if configs.is_gpt:
            model_source = self.gpt2_local_dir if self.gpt2_local_dir else "gpt2"
            self.gpt2 = GPT2Model.from_pretrained(
                '/home/gpt4tide/llm/openai-community/gpt2',
                output_attentions=True,
                output_hidden_states=True,
                local_files_only=self.local_files_only,
            )
            self.gpt2.h = self.gpt2.h[:configs.gpt_layers]
            if hasattr(self.gpt2.config, "n_embd"):
                assert configs.d_model == self.gpt2.config.n_embd, \
                    f"d_model({configs.d_model}) must equal GPT-2 n_embd({self.gpt2.config.n_embd})"
        
        self.llm_model = self.gpt2.to(device)
        
        tokenizer_source = self.gpt2_local_dir if self.gpt2_local_dir else "gpt2"
        self.tokenizer = AutoTokenizer.from_pretrained(
            '/home/gpt4tide/llm/openai-community/gpt2',
            trust_remote_code=True,
            local_files_only=self.local_files_only
        )
        if self.tokenizer.eos_token:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        else:
            pad_token = '[PAD]'
            self.tokenizer.add_special_tokens({'pad_token': pad_token})
            self.tokenizer.pad_token = pad_token
        
        self.in_layer = nn.Linear(configs.patch_size, configs.d_model).to(device)
        self.out_layer = nn.Linear(configs.d_model * self.patch_num, configs.pred_len).to(device)

        if configs.freeze and configs.pretrain and self.is_gpt:
            for name, param in self.gpt2.named_parameters():
                if ('ln' in name) or ('wpe' in name):
                    param.requires_grad = True
                else:
                    param.requires_grad = False

        # ==========================================================
        #         TPE (已修改: 动态幅度和绝对时间)
        # ==========================================================
        d_model = configs.d_model
        
        # 1. 周期数量 (从 main.py 的 k_periods 参数读取)
        self.k_periods = getattr(configs, "k_periods", 4) 
        
        # 2. 初始化可学习的角频率 (w)
        if initial_periods_hours is None:
            print("Warning: No initial_periods_hours provided for TPE. Using defaults.")
            initial_periods_hours = torch.tensor([12.42, 12.00, 23.93, 25.82]) # 默认 5 个
        else:
            if len(initial_periods_hours) != self.k_periods:
                print(f"Warning: k_periods in config ({self.k_periods}) mismatch "
                      f"FFT results ({len(initial_periods_hours)}). Using {len(initial_periods_hours)} periods.")
                self.k_periods = len(initial_periods_hours)
            print(f"Initializing TPE with FFT-derived periods: {initial_periods_hours}")

        initial_w = 2.0 * math.pi / initial_periods_hours.to(dtype=torch.float32)
        self.tpe_w = nn.Parameter(initial_w) # [k_periods]
        
        # 3. 可学习的相位 (phi)
        self.tpe_phase = nn.Parameter(torch.zeros(self.k_periods)) # [k_periods]
        self.tpe_amp = nn.Parameter(torch.ones(self.k_periods)) 
        self.tpe_amp = nn.Parameter(self.tpe_amp.to(device))
        

        # 5. 投影和门控
        self.tpe_proj = nn.Linear(self.k_periods * 2, d_model).to(device) # (k*sin + k*cos)
        self.tpe_scale = nn.Parameter(torch.tensor(1.0))
        self.tpe_gate = nn.Parameter(torch.tensor(1.0))

    # ==========================================================
    # 
    #           <-- [新代码] t-SNE 绘图辅助函数
    # 
    # ==========================================================
    @staticmethod
    def _plot_tsne(ax, data: torch.Tensor, title: str, n_patch: int):
        """
        在给定的 matplotlib a 上运行 t-SNE 并绘制散点图。
        """
        print(f"Running t-SNE for: {title} (Data shape: {data.shape})")
        
        # 1. 确保数据在 CPU 上并转为 numpy
        data_np = data.detach().cpu().numpy()
        
        # 2. t-SNE 对 perplexity 有要求，必须小于样本数
        #    我们的样本数是 n_patch (例如 47)
        perplexity = min(30.0, float(n_patch - 1))
        
        # 3. 初始化 t-SNE
        tsne = TSNE(
            n_components=2, 
            perplexity=perplexity, 
            random_state=42, 
            init='pca', 
            learning_rate='auto'
        )
        
        # 4. 拟合和转换
        tsne_results = tsne.fit_transform(data_np)
        
        # 5. 绘制散点图
        #    我们使用 patch 的索引 (0 到 n_patch-1) 作为颜色
        #    这样可以观察到是否有序列顺序结构
        colors = np.arange(n_patch)
        scatter = ax.scatter(
            tsne_results[:, 0], 
            tsne_results[:, 1], 
            c=colors, 
            cmap='viridis', 
            alpha=0.7
        )
        
        # 6. 添加颜色条和标题
        plt.colorbar(scatter, ax=ax, label='Patch Index')
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])

    # =============== TPE 生成 & 注入 (已修改) ===============
    
    # <-- [MODIFIED] 签名已更改，现在接收 tpe_x_mark
    def _build_tpe(self, tpe_x_mark: torch.Tensor, device, dtype):
        """
        生成 [B*, L, d_model] 的绝对时间位置编码
        tpe_x_mark: [B*, L_patch, N_time_features]
        """
        proj_dtype = self.tpe_proj.weight.dtype

        # --- [CRITICAL] 1. 计算绝对时间 t ---
        # 假设 tpe_x_mark 的时间特征索引为:
        # 2: day_of_week (0-6)
        # 3: hour_of_day (0-23)
        # 4: minute_of_hour (0-59) or (0, 15, 30, 45)
        # 你必须根据你的 data_loader 确认这些索引！
        try:
            day_of_week = tpe_x_mark[..., 2]
            hour_of_day = tpe_x_mark[..., 3]
            minute_of_hour = tpe_x_mark[..., 4]
        except IndexError:
            print("="*80)
            print(f"错误: TPE 时间戳索引失败。期望 time_feat_dim >= 5。")
            print(f"tpe_x_mark shape: {tpe_x_mark.shape}")
            print(f"请检查你的 data_loader 和 main.py 中的 'time_feat_dim' 参数。")
            print("="*80)
            raise

        # 计算 "周小时数" (e.g., 周一 8:15 -> 1*24 + 8 + 15/60 = 32.25)
        # [修改]：为了简化，这里我们只使用 "天小时数" (0-23.9x)，因为潮汐主要周期是日和半日
        #         如果需要周周期，可以使用 t = day_of_week * 24 + hour_of_day + ...
        t = hour_of_day*60 + minute_of_hour # 形状 [B*, L_patch]
        t = t.to(dtype=proj_dtype)
        # --- [修改结束] ---

        # 2. 获取可学习的角频率 (w) 和相位 (phi)
        w = F.softplus(self.tpe_w.to(device=device, dtype=proj_dtype)) # [k_periods]
        phi = self.tpe_phase.to(device=device, dtype=proj_dtype) # [k_periods]
        
        # 3. [NEW] 计算动态幅度 (amp)
        # amp 形状 [B*, L_patch, k_periods]
        amp = torch.clamp(self.tpe_amp.to(device=device, dtype=proj_dtype), min=0.0)
        
        # 4. 计算 sin/cos 项
        # t.unsqueeze(-1) -> [B*, L_patch, 1]
        # wt_phi          -> [B*, L_patch, k_periods]
        wt_phi = t.unsqueeze(-1) * w + phi
        
        cos_terms = amp * torch.cos(wt_phi) # [B*, L_patch, k_periods]
        sin_terms = amp * torch.sin(wt_phi) # [B*, L_patch, k_periods]
        
        # 5. 拼接
        feats = torch.cat([sin_terms, cos_terms], dim=-1) # [B*, L_patch, k_periods * 2]

        # 6. 投影
        tpe = self.tpe_proj(feats) # [B*, L_patch, d_model]

        # 返回形状 [B*, L, d_model]
        return tpe

    @staticmethod
    def _rms(x, eps=1e-8):
        return torch.sqrt(torch.mean(x.pow(2), dim=(-1, -2), keepdim=True) + eps)

    # [注意] _inject_tpe 函数已被移除，其逻辑合并到 forward 函数中

    # ====================== 前向 (已修改) ======================
    def calcute_lags(self, x_enc):
        x_enc_permuted = x_enc.permute(0, 2, 1).contiguous()
        x_enc_fft = torch.fft.rfft(x_enc_permuted, dim=-1)
        res = x_enc_fft * torch.conj(x_enc_fft)
        corr = torch.fft.irfft(res, dim=-1)
        mean_value = torch.mean(corr, dim=1)
        _, lags = torch.topk(mean_value, self.top_k, dim=-1)
        return lags
    
    # <-- [MODIFIED] 签名已更改，现在接收 batch_x_mark
    def forward(self, x, batch_x_mark, itr, index, mode):
        """
        x: [B, L_seq, M]
        batch_x_mark: [B, L_seq, N_time_features] (e.g., [B, 512, 5])
        """

        x = self.normalize_layers(x, 'norm')
        
        x_enc = x
        B0, T, N = x_enc.size() # T = L_seq
        x_enc_prompt = x_enc.permute(0, 2, 1).contiguous().reshape(B0 * N, T, 1)

        min_values = torch.min(x_enc_prompt, dim=1)[0]
        max_values = torch.max(x_enc_prompt, dim=1)[0]
        medians = torch.median(x_enc_prompt, dim=1).values
        lags = self.calcute_lags(x_enc_prompt)
        trends = x_enc_prompt.diff(dim=1).sum(dim=1)

        prompt = []
        for b in range(x_enc_prompt.shape[0]):
            min_values_str = str(min_values[b].tolist()[0])
            max_values_str = str(max_values[b].tolist()[0])
            median_values_str = str(medians[b].tolist()[0])
            lags_values_str = str(lags[b].tolist())
            prompt_ = (
                f"Task description: forecast the next {str(self.pred_len)} steps given the previous {str(self.seq_len)} steps information; "
                "Input statistics: "
                f"min value {min_values_str}, "
                f"max value {max_values_str}, "
                f"median value {median_values_str}, "
                f"the trend of input is {'upward' if trends[b] > 0 else 'downward'}, "
                f"top 5 lags are : {lags_values_str}<|<end_prompt>|>"
            )
            prompt.append(prompt_)

        current_device = x_enc.device
        prompt_ids = self.tokenizer(
            prompt, return_tensors="pt", padding=True, truncation=True, max_length=512
        ).input_ids
        prompt_embeddings = self.llm_model.get_input_embeddings()(prompt_ids.to(current_device))

        x_tokens_in = rearrange(x, 'b l m -> b m l')
        x_tokens_in = self.padding_patch_layer(x_tokens_in)
        x_tokens_in = x_tokens_in.unfold(dimension=-1, size=self.patch_size, step=self.stride)
        x_tokens_in = rearrange(x_tokens_in, 'b m n p -> (b m) n p')

        tokens = self.in_layer(x_tokens_in)  # [(B*M), Npatch, d_model]

        # --- [新代码：可视化触发器] ---
        # 检查是否需要运行可视化
        run_visualization = (
            mode == 'test' and 
            self.do_visualize and 
            not self._visualization_done
        )
        
        # 准备用于可视化的独立向量
        base_tokens_viz = None
        tpe_viz = None
        wpe_viz = None
        
        if run_visualization:
            # 1. 克隆 base tokens
            base_tokens_viz = tokens.clone()
        # --- [新代码结束] ---

        if self.is_tpe:
            # --- [FIXED] TPE 注入 ---
            # 1. 选取每个 patch 起始点的时间戳
            patch_start_indices = torch.arange(0, T - self.patch_size + 1, self.stride, device=current_device)
            
            # 2. 提取时间戳
            tpe_x_mark = batch_x_mark[:, patch_start_indices, :] # [B, 46, N_feat]
            
            # --- [关键修复 开始] ---
            # 复制最后一个时间戳，以匹配由 ReplicationPad1d 创建的第 47 个数据补丁
            # last_time_mark 形状为 [B, 1, N_feat]
            last_time_mark = tpe_x_mark[:, -1:, :] 
            # 将其拼接到 tpe_x_mark 后面，使其形状变为 [B, 47, N_feat]
            tpe_x_mark = torch.cat([tpe_x_mark, last_time_mark], dim=1) 
            # --- [关键修复 结束] ---

            # 3. 扩展时间戳以匹配 'tokens' 的 (B*M) 维度
            _, n_patch, n_feat = tpe_x_mark.shape # n_patch 现在是 47
            tpe_x_mark = tpe_x_mark.unsqueeze(2).expand(-1, -1, N, -1) # [B, 47, M, N_feat]
            tpe_x_mark = tpe_x_mark.permute(0, 2, 1, 3).reshape(B0 * N, n_patch, n_feat) # [(B*M), 47, N_feat]
            
            # --- [修改：从 _inject_tpe 拆分] ---
            # 4. 先 *构建* TPE
            tpe_raw = self._build_tpe(
                tpe_x_mark,
                device=tokens.device,
                dtype=tokens.dtype
            ) # [(B*M), N_patch, d_model]
            
            tpe_scaled = tpe_raw.to(tokens.dtype) * 0.1
            
            if run_visualization:
                # 5. 克隆 TPE
                tpe_viz = tpe_scaled.clone()

            # 6. *注入* TPE
            tokens = tokens + tpe_scaled 
            # --- [修改结束] ---
            
        if self.is_gpt:
            
            # --- [新代码：在 GPT-2 之前执行绘图] ---
            if run_visualization:
                # 标记为已完成，防止重复执行
                self._visualization_done = True 
                
                print(">>> [GPT4TS.forward] 触发 t-SNE 绘图 (仅在测试模式下运行一次)...")
                
                B_M, N_patch, D_model = base_tokens_viz.shape
                
                # 3. 手动获取 WPE
                position_ids = torch.arange(0, N_patch, dtype=torch.long, device=tokens.device)
                position_ids = position_ids.unsqueeze(0) # [1, N_patch]
                wpe_viz = self.gpt2.wpe(position_ids).clone() # [1, N_patch, d_model]

                # 确保 TPE (如果 is_tpe=False) 是一个零向量
                if tpe_viz is None:
                    tpe_viz = torch.zeros_like(base_tokens_viz)
                    
                # 4. 准备四组数据 (仅使用第一个样本)
                #    我们选择 [0] 来代表 (b=0, m=0) 的样本
                data_1 = base_tokens_viz[0]
                data_2 = (base_tokens_viz + tpe_viz)[0]
                data_3 = (base_tokens_viz + wpe_viz)[0]
                data_4 = (base_tokens_viz + tpe_viz + wpe_viz)[0]

                # 5. 绘图并保存
                try:
                    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
                    fig.suptitle('t-SNE Visualization of Patch Token Embeddings (Sample 0)', fontsize=16)
                    
                    self._plot_tsne(axes[0, 0], data_1, '1. Patch Tokens Only', N_patch)
                    self._plot_tsne(axes[0, 1], data_2, '2. Patch + TPE (Tidal)', N_patch)
                    self._plot_tsne(axes[1, 0], data_3, '3. Patch + WPE (GPT-2)', N_patch)
                    self._plot_tsne(axes[1, 1], data_4, '4. Patch + TPE + WPE', N_patch)
                    
                    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
                    
                    # 从 configs 读取保存路径，默认为 'embedding_tsne.png'
                    plot_filename = getattr(self.configs, "viz_plot_path", "embedding_tsne.png")
                    plt.savefig(plot_filename) # 保存文件
                    
                    print(f">>> [GPT4TS.forward] t-SNE 图像已保存到: {plot_filename}")
                    plt.close(fig) # 关闭图像，释放内存
                    
                except Exception as e:
                    print(f"!!! [GPT4TS.forward] t-SNE 绘图失败: {e}")
                    # 即使失败也不要中断测试
            
            # --- [新代码结束] ---
            
            # 正常的 GPT-2 前向传播 (它会在内部添加 WPE)
            tokens = self.gpt2(inputs_embeds=tokens).last_hidden_state 
            
        if self.is_cross:
            tokens = self.cross(tokens, prompt_embeddings, prompt_embeddings)

        outputs = self.out_layer(tokens.reshape(B0 * N, -1)) 
        outputs = rearrange(outputs, '(b m) l -> b l m', b=B0) 

        outputs = self.normalize_layers(outputs, 'denorm')
        return outputs