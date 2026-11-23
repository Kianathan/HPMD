import torch
import numpy as np
import random
import pynvml
import logging
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger('HPMD')

class GGDUnitDistillLoss(nn.Module):
    def __init__(self, d_feat=64, logits_dim=2, sinkhorn_lambda=1.0, sinkhorn_iter=5):
        
        super(GGDUnitDistillLoss, self).__init__()
        self.d_feat = d_feat                
        self.logits_dim = logits_dim         
        self.sinkhorn_lambda = sinkhorn_lambda  
        self.sinkhorn_iter = sinkhorn_iter      
        
        input_dim_g = 2 * (self.logits_dim + self.d_feat)
        self.g = nn.Sequential(
            nn.Linear(input_dim_g, input_dim_g // 2),
            nn.ReLU(),
            nn.Linear(input_dim_g // 2, 1),
            nn.Sigmoid()
        )

    def _calc_distill_strength(self, logits_c, logits_p_list, feat_c, feat_p_list):
        batch_size = logits_c.shape[0]
        num_modalities = len(logits_p_list)
        ω_c2p = torch.zeros(batch_size, num_modalities, device=logits_c.device)
        ω_p2c = torch.zeros(batch_size, num_modalities, device=logits_c.device)

        for i in range(num_modalities):
            logits_p = logits_p_list[i]
            feat_p = feat_p_list[i]

            concat_c = torch.cat([logits_c, feat_c], dim=1)
            concat_cp = torch.cat([logits_c, feat_p], dim=1)
            input_g_c2p = torch.cat([concat_c, concat_cp], dim=1)
            ω_c2p[:, i] = self.g(input_g_c2p).squeeze(1)

            concat_p = torch.cat([logits_p, feat_p], dim=1)
            concat_pc = torch.cat([logits_p, feat_c], dim=1)
            input_g_p2c = torch.cat([concat_p, concat_pc], dim=1)
            ω_p2c[:, i] = self.g(input_g_p2c).squeeze(1)

        return ω_c2p, ω_p2c

    def _sinkhorn_batch(self, t, s):
        batch_size = t.shape[0]
        t_expand = t.unsqueeze(1).expand(-1, batch_size, -1) 
        s_expand = s.unsqueeze(0).expand(batch_size, -1, -1)  
        D = torch.abs(t_expand - s_expand).sum(dim=2)        

        K = torch.exp(-D / self.sinkhorn_lambda)  
        u = torch.ones(batch_size, device=t.device)  
        v = torch.ones(batch_size, device=t.device)  

        for _ in range(self.sinkhorn_iter):
            u = (torch.ones(batch_size, device=t.device) / (K.T @ v))
            v = (torch.ones(batch_size, device=t.device) / (K @ u))

  
        P_lambda = torch.diag(v) @ K @ torch.diag(u)  
        sinkhorn_loss = (P_lambda * D).sum()  
        return sinkhorn_loss

    def _calc_distill_error(self, logits_c, logits_p_list):
        batch_size = logits_c.shape[0]
        num_modalities = len(logits_p_list)
        ε_c2p = torch.zeros(batch_size, num_modalities, device=logits_c.device)
        ε_p2c = torch.zeros(batch_size, num_modalities, device=logits_c.device)
        loss_cc = self._sinkhorn_batch(logits_c, logits_c)

        for i in range(num_modalities):
            logits_p = logits_p_list[i]
            loss_pp = self._sinkhorn_batch(logits_p, logits_p)
            loss_c2p = self._sinkhorn_batch(logits_c, logits_p)
            debiased_loss_c2p = loss_c2p - 0.5 * loss_cc - 0.5 * loss_pp
            ε_c2p[:, i] = torch.relu(debiased_loss_c2p) / batch_size 
            loss_p2c = self._sinkhorn_batch(logits_p, logits_c)
            debiased_loss_p2c = loss_p2c - 0.5 * loss_pp - 0.5 * loss_cc
            ε_p2c[:, i] = torch.relu(debiased_loss_p2c) / batch_size
        return ε_c2p, ε_p2c

    def forward(self, logits_c, feat_c, logits_p_list, feat_p_list):
        assert len(logits_p_list) == len(feat_p_list)
        assert logits_c.shape[1] == self.logits_dim == 2
        num_modalities = len(logits_p_list)
        ω_c2p, ω_p2c = self._calc_distill_strength(logits_c, logits_p_list, feat_c, feat_p_list)
        ε_c2p, ε_p2c = self._calc_distill_error(logits_c, logits_p_list)

        W = torch.cat([ω_c2p, ω_p2c], dim=1)
        E = torch.cat([ε_c2p, ε_p2c], dim=1)
        L_dtl = torch.norm(W * E, p=2) / logits_c.shape[0]

        return L_dtl

def dict_to_str(src_dict):
    dst_str = ""
    for key in src_dict.keys():
        dst_str += " %s: %.4f " %(key, src_dict[key]) 
    return dst_str

def setup_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def assign_gpu(gpu_ids, memory_limit=1e16):
    if len(gpu_ids) == 0 and torch.cuda.is_available():
        # find most free gpu
        pynvml.nvmlInit()
        n_gpus = pynvml.nvmlDeviceGetCount()
        dst_gpu_id, min_mem_used = 0, memory_limit
        for g_id in range(n_gpus):
            handle = pynvml.nvmlDeviceGetHandleByIndex(g_id)
            meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
            mem_used = meminfo.used
            if mem_used < min_mem_used:
                min_mem_used = mem_used
                dst_gpu_id = g_id
        logger.info(f'Found gpu {dst_gpu_id}, used memory {min_mem_used}.')
        gpu_ids.append(dst_gpu_id)
    # device
    using_cuda = len(gpu_ids) > 0 and torch.cuda.is_available()
    device = torch.device('cuda:%d' % int(gpu_ids[0]) if using_cuda else 'cpu')
    return device

def count_parameters(model):
    res = 0
    for p in model.parameters():
        if p.requires_grad:
            res += p.numel()
            # print(p)
    return res

def eva_imp(y_true, y_pred):
    res = (y_pred - y_true) ** 2
    return res

def entropy_balance(probs):
    probs = torch.clamp(probs, min=1e-9)
    N = probs.size(1)
    entropy = torch.sum(probs * torch.log(probs), dim=1) + math.log(N)  
    return torch.mean(entropy)