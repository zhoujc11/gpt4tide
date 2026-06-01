import torch
import torch.nn.functional as F
import numpy as np
import math
from tqdm import tqdm
from torch.utils.data import DataLoader


class RetrievalTool():
    def __init__(
        self,
        seq_len,
        pred_len,
        channels,
        n_period=1,
        temperature=0.1,
        topm=20,
        with_dec=False,
        return_key=False,
    ):
        period_num = [16, 8, 4, 2, 1]
        period_num = period_num[-1 * n_period:]

        self.seq_len = seq_len
        self.pred_len = pred_len
        self.channels = channels

        self.n_period = n_period
        self.period_num = sorted(period_num, reverse=True)

        self.temperature = temperature
        self.topm = topm

        self.with_dec = with_dec
        self.return_key = return_key

    def prepare_dataset(self, train_data):
        train_data_all = []
        y_data_all = []

        for i in range(len(train_data)):
            td = train_data[i]
            train_data_all.append(td[1])

            if self.with_dec:
                y_data_all.append(td[2][-(train_data.pred_len + train_data.label_len):])
            else:
                y_data_all.append(td[2][-train_data.pred_len:])

        self.train_data_all = torch.tensor(np.stack(train_data_all, axis=0)).float()

        self.y_data_all = torch.tensor(np.stack(y_data_all, axis=0)).float()

        self.n_train = self.train_data_all.shape[0]


    def periodic_batch_corr(self, data_all, key, in_bsz = 512):
        _, bsz, features = key.shape
        _, train_len, _ = data_all.shape

        bx = key - torch.mean(key, dim=2, keepdim=True)

        iters = math.ceil(train_len / in_bsz)

        sim = []
        for i in range(iters):
            start_idx = i * in_bsz
            end_idx = min((i + 1) * in_bsz, train_len)

            cur_data = data_all[:, start_idx:end_idx].to(key.device)
            ax = cur_data - torch.mean(cur_data, dim=2, keepdim=True)

            cur_sim = torch.bmm(F.normalize(bx, dim=2), F.normalize(ax, dim=2).transpose(-1, -2))
            sim.append(cur_sim)

        sim = torch.cat(sim, dim=2)

        return sim

    def retrieve(self, x, index, train=True):
        index = index.to(x.device)

        bsz, seq_len, channels = x.shape

        train_data_all=self.train_data_all.unsqueeze(0)   # 变为 [1, G, T, S, C]
        x = x.unsqueeze(0)  # 在第0维插入新的维度，变为 [1, b, seq, d]

        sim = self.periodic_batch_corr(
            train_data_all.flatten(start_dim=2), # G, T, S * C
            x.flatten(start_dim=2), # G, B, S * C
        ) # G, B, T
        #防止模型"作弊"检索到与当前序列重叠的历史片段,创建一个滑动窗口掩码，将与当前序列相邻的区域设为负无穷
        if train:
            sliding_index = torch.arange(2 * (self.seq_len + self.pred_len) - 1).to(x.device)
            sliding_index = sliding_index.unsqueeze(dim=0).repeat(len(index), 1)
            sliding_index = sliding_index + (index - self.seq_len - self.pred_len + 1).unsqueeze(dim=1)

            sliding_index = torch.where(sliding_index >= 0, sliding_index, 0)
            sliding_index = torch.where(sliding_index < self.n_train, sliding_index, self.n_train - 1)

            self_mask = torch.zeros((bsz, self.n_train)).to(x.device)
            self_mask = self_mask.scatter_(1, sliding_index, 1.)
            self_mask = self_mask.unsqueeze(dim=0).repeat(self.n_period, 1, 1)

            sim = sim.masked_fill_(self_mask.bool(), float('-inf')) # G, B, T

        sim = sim.reshape(self.n_period * bsz, self.n_train) # G X B, T

        topm_index = torch.topk(sim, self.topm, dim=1).indices

        ranking_sim = torch.ones_like(sim) * float('-inf')

        rows = torch.arange(sim.size(0)).unsqueeze(-1).to(sim.device)
        ranking_sim[rows, topm_index] = sim[rows, topm_index]

        sim = sim.reshape(self.n_period, bsz, self.n_train) # G, B, T
        ranking_sim = ranking_sim.reshape(self.n_period, bsz, self.n_train) # G, B, T

        data_len, seq_len, channels = self.train_data_all.shape

        ranking_prob = F.softmax(ranking_sim / self.temperature, dim=2)
        ranking_prob = ranking_prob.detach().cpu() # G, B, T
        self.train_data_all = self.train_data_all.to(topm_index.device)
        self.y_data_all = self.y_data_all.to(topm_index.device)
        x_data_all = self.train_data_all[topm_index]
        y_data_all = self.y_data_all[topm_index] # G, T, P * C
        return x_data_all,y_data_all,topm_index

    def retrieve_all(self, data, train=False, device=torch.device('cpu')):


        rt_loader = DataLoader(
            data,
            batch_size=1024,
            shuffle=False,
            num_workers=8,
            drop_last=False
        )

        x_data = []
        y_data = []
        topm = []
        with torch.no_grad():
            for index, batch_x, batch_y, batch_x_mark, batch_y_mark in tqdm(rt_loader):
                x_data_all,y_data_all,topm_index = self.retrieve(batch_x.float().to(device), index, train=train)
                x_data_all = x_data_all.cpu()
                y_data_all = y_data_all.cpu()
                topm_index = topm_index.cpu()
                
                x_data.append(x_data_all)
                y_data.append(y_data_all)
                topm.append(topm_index)
    # 最终将列表转换为 Tensor 或其他适当格式
        x_data = torch.cat(x_data, dim=0)  # 按照需要拼接
        y_data = torch.cat(y_data, dim=0)  # 按照需要拼接
        topm = torch.cat(topm, dim=0)  # 按照需要拼接
        return x_data,y_data,topm


