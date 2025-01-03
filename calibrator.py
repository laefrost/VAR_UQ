import time
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader

import dist
from models import VAR, VQVAE, VectorQuantizer2
from utils.amp_sc import AmpOptimizer
from utils.misc import MetricLogger, TensorboardLogger

Ten = torch.Tensor
FTen = torch.Tensor
ITen = torch.LongTensor
BTen = torch.BoolTensor

class VARCalibrator(object): 
    def  __init__(self, device, patch_nums: Tuple[int, ...], resos: Tuple[int, ...],
            vae_local: VQVAE, var_wo_ddp: VAR): 
        super(VARCalibrator, self).__init__()
        
        self.vae_local, self.quantize_local = vae_local, vae_local.quantize
        self.quantize_local: VectorQuantizer2
        self.var_wo_ddp: VAR = var_wo_ddp  # after torch.compile
        
        del self.var_wo_ddp.rng
        self.var_wo_ddp.rng = torch.Generator(device=device)
        
        self.val_loss = nn.CrossEntropyLoss(label_smoothing=0.0, reduction='mean')
        self.L = sum(pn * pn for pn in patch_nums)
        self.last_l = patch_nums[-1] * patch_nums[-1]
        self.loss_weight = torch.ones(1, self.L, device=device) / self.L
        
        self.patch_nums, self.resos = patch_nums, resos
        self.begin_ends = []
        cur = 0
        for i, pn in enumerate(patch_nums):
            self.begin_ends.append((cur, cur + pn * pn))
            cur += pn*pn
        
        self.prog_it = 0
        self.last_prog_si = -1
        self.first_prog = True
    

    @torch.no_grad()
    def teacher_enforced_cp(self, ld_val: DataLoader):
        tot = 0
        L_mean, L_tail, acc_mean, acc_tail = 0, 0, 0, 0
        stt = time.time()
        #training = self.var_wo_ddp.training
        self.var_wo_ddp.eval()
        for inp_B3HW, label_B in ld_val:
            B, V = label_B.shape[0], self.vae_local.vocab_size
            inp_B3HW = inp_B3HW.to(dist.get_device(), non_blocking=True)
            label_B = label_B.to(dist.get_device(), non_blocking=True)
            
            # true indeces for all token maps
            gt_idx_Bl: List[ITen] = self.vae_local.img_to_idxBl(inp_B3HW)
            # flattens indices into one sequence 
            gt_BL = torch.cat(gt_idx_Bl, dim=1)
            # teacherforcing input for var
            x_BLCv_wo_first_l: Ten = self.quantize_local.idxBl_to_var_input(gt_idx_Bl)
            
            self.var_wo_ddp.forward
            logits_BLV = self.var_wo_ddp(label_B, x_BLCv_wo_first_l)
            print(logits_BLV.shape)
            print(logits_BLV[0, 0,: ])
            #TODO: compute scores, append scores across batch dimension
            
            # L_mean += self.val_loss(logits_BLV.data.view(-1, V), gt_BL.view(-1)) * B
            # L_tail += self.val_loss(logits_BLV.data[:, -self.last_l:].reshape(-1, V), gt_BL[:, -self.last_l:].reshape(-1)) * B
            # acc_mean += (logits_BLV.data.argmax(dim=-1) == gt_BL).sum() * (100/gt_BL.shape[1])
            # acc_tail += (logits_BLV.data[:, -self.last_l:].argmax(dim=-1) == gt_BL[:, -self.last_l:]).sum() * (100 / self.last_l)
            # tot += B
        #self.var_wo_ddp.train(training)
        
        stats = L_mean.new_tensor([L_mean.item(), L_tail.item(), acc_mean.item(), acc_tail.item(), tot])
        dist.allreduce(stats)
        tot = round(stats[-1].item())
        stats /= tot
        L_mean, L_tail, acc_mean, acc_tail, _ = stats.tolist()
        return L_mean, L_tail, acc_mean, acc_tail, tot, time.time()-stt