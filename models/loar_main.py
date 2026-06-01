from data_provider.data_factory import data_provider
from utils.tools import EarlyStopping, adjust_learning_rate, visual, vali, test
from tqdm import tqdm
from models.PatchTST import PatchTST
from models.GPT4TS import GPT4TS
# --- [新导入] 导入 FFT 辅助函数 ---
from models.GPT4TS import find_top_k_periods_fft
# --- [导入结束] ---
from models.DLinear import DLinear
import numpy as np
import torch
import torch.nn as nn
from torch import optim
import os
import time
import warnings
import matplotlib.pyplot as plt
import numpy as np
import argparse
import random
os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings('ignore')
fix_seed = 3047
random.seed(fix_seed)
torch.manual_seed(fix_seed)
np.random.seed(fix_seed)
parser = argparse.ArgumentParser(description='GPT4TS')
# --- 基础配置 ---
parser.add_argument('--model_id', type=str, required=True, default='test')
parser.add_argument('--checkpoints', type=str, default='/root/GPT4TS-main/checkpoints/')
parser.add_argument('--model', type=str, default='model', help='model name, options: [GPT4TS, PatchTST, DLinear]')
# --- 数据加载器 ---
parser.add_argument('--root_path', type=str, default='./dataset/traffic/')
parser.add_argument('--data_path', type=str, default='traffic.csv')
parser.add_argument('--data', type=str, default='custom')
parser.add_argument('--features', type=str, default='S')
parser.add_argument('--freq', type=str, default='15min')
parser.add_argument('--target', type=str, default='target')
parser.add_argument('--embed', type=str, default='timeF')
parser.add_argument('--percent', type=int, default=10)
# --- 序列长度 ---
parser.add_argument('--seq_len', type=int, default=512)
parser.add_argument('--pred_len', type=int, default=96)
parser.add_argument('--label_len', type=int, default=48)
# --- 训练设置 ---
parser.add_argument('--decay_fac', type=float, default=0.75)
parser.add_argument('--learning_rate', type=float, default=0.0001)
parser.add_argument('--batch_size', type=int, default=512)
parser.add_argument('--num_workers', type=int, default=10)
parser.add_argument('--train_epochs', type=int, default=10)
parser.add_argument('--lradj', type=str, default='type1')
parser.add_argument('--patience', type=int, default=3)
parser.add_argument('--loss_func', type=str, default='mse')
parser.add_argument('--itr', type=int, default=3)
parser.add_argument('--cos', type=int, default=0)
parser.add_argument('--tmax', type=int, default=10)
# --- 模型参数 ---
parser.add_argument('--gpt_layers', type=int, default=3)
parser.add_argument('--is_gpt', type=int, default=1)
parser.add_argument('--e_layers', type=int, default=3)
parser.add_argument('--d_model', type=int, default=768)
parser.add_argument('--n_heads', type=int, default=16)
parser.add_argument('--d_ff', type=int, default=512)
parser.add_argument('--dropout', type=float, default=0.2)
parser.add_argument('--enc_in', type=int, default=1)
parser.add_argument('--c_out', type=int, default=1)
parser.add_argument('--patch_size', type=int, default=16)
parser.add_argument('--kernel_size', type=int, default=25)
parser.add_argument('--pretrain', type=int, default=1)
parser.add_argument('--freeze', type=int, default=1) # 注意：这个 freeze 将被 PEFT 覆盖
parser.add_argument('--stride', type=int, default=8)
parser.add_argument('--max_len', type=int, default=-1)
parser.add_argument('--hid_dim', type=int, default=16)
# --- GPT4TS 特定参数 ---
parser.add_argument('--is_tpe', type=int, default=1)
parser.add_argument('--is_cross', type=int, default=1)
parser.add_argument('--is_retrieval', type=int, default=1)
parser.add_argument('--topm', type=int, default=5)
parser.add_argument('--do_visualize', action='store_true', help='flag to run embedding visualization')
# --- [新] TPE 参数 ---
parser.add_argument('--k_periods', type=int, default=4, help='Number of top periods to find for TPE')
parser.add_argument('--step_minutes', type=int, default=15, help='Time step in minutes (e.g., 15 for 15min data)')
# --- [新] LoRA (PEFT) 参数 ---
parser.add_argument('--lora_r', type=int, default=8, help='LoRA rank (r)')
parser.add_argument('--lora_alpha', type=int, default=16, help='LoRA alpha')
args = parser.parse_args()
# (SEASONALITY_MAP 保持不变)
mses = []
maes = []
for ii in range(args.itr):
    setting = '{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_gl{}_df{}_eb{}_itr{}'.format(args.model_id, 336, args.label_len, args.pred_len,
                                                                         args.d_model, args.n_heads, args.e_layers, args.gpt_layers, 
                                                                         args.d_ff, args.embed, ii)
    path = os.path.join(args.checkpoints, setting)
    if not os.path.exists(path):
        os.makedirs(path)
    train_data, train_loader = data_provider(args, 'train')
    vali_data, vali_loader = data_provider(args, 'val')
    test_data, test_loader = data_provider(args, 'test')
    device = torch.device('cuda:0')
    time_now = time.time()
    train_steps = len(train_loader)
    if args.model == 'PatchTST':
        model = PatchTST(args, device)
        model.to(device)
    elif args.model == 'DLinear':
        model = DLinear(args, device)
        model.to(device)
    else:
        # --- [修改] TPE FFT 计算 ---
        initial_periods = None
        if args.is_tpe:
            print("Running FFT on training data for TPE initialization...")
            # 从分钟计算小时
            dt_hours = args.step_minutes / 60.0
            
            # 假设 train_data.data_x 是 (N_samples, L_seq, M_features) 的 NumPy 数组
            ts_data_for_fft = train_data.data_x 
            if ts_data_for_fft is None:
                print("Warning: train_data.data_x is None. TPE will use default periods.")
            else:
                initial_periods = find_top_k_periods_fft(
                    ts_data_for_fft, 
                    dt_hours=dt_hours, 
                    k=args.k_periods
                )
        # --- [TPE 结束] ---
        # 将 initial_periods 传递给模型
        model = GPT4TS(args, device)
    
    # --- [修改] 优化器 (Optimizer) ---
    # 仅获取需要梯度更新的参数
    # 这将自动包括:
    # 1. LoRA 适配器参数 (来自 self.gpt2)
    # 2. 您的 TPE 参数 (self.tpe_w, self.tpe_proj 等)
    # 3. 您的 in_layer, out_layer, cross 等
    print("Filtering for trainable parameters (LoRA + TPE + other heads)...")
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    model_optim = torch.optim.AdamW(trainable_params, lr=args.learning_rate)
    # --- [修改结束] ---
    
    early_stopping = EarlyStopping(patience=args.patience, verbose=True)
    if args.loss_func == 'mse':
        criterion = nn.MSELoss()
    elif args.loss_func == 'smape':
        class SMAPE(nn.Module):
            def __init__(self):
                super(SMAPE, self).__init__()
            def forward(self, pred, true):
                return torch.mean(200 * torch.abs(pred - true) / (torch.abs(pred) + torch.abs(true) + 1e-8))
        criterion = SMAPE()
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(model_optim, T_max=args.tmax, eta_min=1e-8)
    for epoch in range(args.train_epochs):
        iter_count = 0
        train_loss = []
        epoch_time = time.time()
        for i, (index,batch_x, batch_y, batch_x_mark, batch_y_mark) in tqdm(enumerate(train_loader),disable=True):
            iter_count += 1
            model_optim.zero_grad()
            batch_x = batch_x.float().to(device)
            batch_y = batch_y.float().to(device)
            batch_x_mark = batch_x_mark.float().to(device)
            batch_y_mark = batch_y_mark.float().to(device)
            
            outputs = model(batch_x,batch_x_mark, ii, index, mode='train')
        
            outputs = outputs[:, -args.pred_len:, :].to(device)
            batch_y = batch_y[:, -args.pred_len:, :].to(device)
            loss = criterion(outputs, batch_y)
            train_loss.append(loss.item())
            if (i + 1) % 1000 == 0:
                speed = (time.time() - time_now) / iter_count
                left_time = speed * ((args.train_epochs - epoch) * train_steps - i)
                iter_count = 0
                time_now = time.time()
            
            loss.backward()
            model_optim.step()
        train_loss = np.average(train_loss)
        vali_loss = vali(model, vali_data, vali_loader, criterion, args, device, ii)
        
        print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f}".format(
            epoch + 1, train_steps, train_loss, vali_loss))
        # --- [修改] 模型保存 ---
        # 1. 正常调用 EarlyStopping。它会保存完整的 checkpoint.pth
        #    (这对于 LoRA 来说是低效的，但能确保与 utils.tools 兼容)
        early_stopping(vali_loss, model, path)
        # 2. [新] 如果是最佳模型，我们*额外*保存 LoRA 适配器
        #    我们通过检查计数器是否为 0 来判断是否刚更新了最佳损失
        if not early_stopping.early_stop and early_stopping.counter == 0:
            print(f"Validation loss improved. Saving LoRA adapters and model head...")
            adapter_save_path = os.path.join(path, "lora_adapters")
            head_save_path = os.path.join(path, "model_head.pth")
            # 仅在模型是 GPT4TS (或任何 PEFT 模型) 时执行
            if hasattr(model, 'gpt2') and hasattr(model.gpt2, 'save_pretrained'):
                model.gpt2.save_pretrained(adapter_save_path)
            
            # 保存模型的其余部分 (TPE, in/out layers, cross-attn, etc.)
            non_llm_state_dict = {
                k: v for k, v in model.state_dict().items() 
                if "gpt2" not in k
            }
            torch.save(non_llm_state_dict, head_save_path)
        # --- [修改结束] ---
        
        if args.cos:
            scheduler.step()
            print("lr = {:.10f}".format(model_optim.param_groups[0]['lr']))
        else:
            adjust_learning_rate(model_optim, epoch + 1, args)
        
        if early_stopping.early_stop:
            print("Early stopping")
            break
    # --- [修改] 模型加载 ---
    # 不再加载 'checkpoint.pth'，而是加载适配器和 head
    print("Loading best model (LoRA adapters + head) for final test...")
    adapter_load_path = os.path.join(path, "lora_adapters")
    head_load_path = os.path.join(path, "model_head.pth")
    try:
        # 1. 加载 LoRA 适配器
        if hasattr(model, 'gpt2') and hasattr(model.gpt2, 'load_adapter'):
            # is_trainable=False 适用于推理/测试
            model.gpt2.load_adapter(adapter_load_path, is_trainable=False) 
            print(f"Successfully loaded LoRA adapters from {adapter_load_path}")
        
        # 2. 加载模型的其余部分 (head)
        #    strict=False 至关重要，因为它会忽略 'gpt2' 相关的键
        model.load_state_dict(
            torch.load(head_load_path), 
            strict=False 
        )
        print(f"Successfully loaded model head from {head_load_path}")
    except Exception as e:
        print(f"Error loading LoRA model: {e}")
        print("Fallback: Loading full 'checkpoint.pth'. (This is inefficient but ensures testing).")
        best_model_path = path + '/' + 'checkpoint.pth'
        model.load_state_dict(torch.load(best_model_path))
    # --- [修改结束] ---
    print("------------------------------------")
    mse, mae = test(model, test_data, test_loader, args, device, ii)
    mses.append(mse)
    maes.append(mae)
mses = np.array(mses)
maes = np.array(maes)
print("mse_mean = {:.4f}, mse_std = {:.4f}".format(np.mean(mses), np.std(mses)))
print("mae_mean = {:.4f}, mae_std = {:.4f}".format(np.mean(maes), np.std(maes)))