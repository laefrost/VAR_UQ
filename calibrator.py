import time
from typing import List, Optional, Tuple, Union
import numpy as np
import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from models.basic_var import AdaLNBeforeHead, AdaLNSelfAttn
from models.helpers import gumbel_softmax_with_rng, sample_with_top_k_top_p_
import scipy



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
            vae_local: VQVAE, var_wo_ddp: VAR, num_classes = 1000): 
        super(VARCalibrator, self).__init__()
        
        self.vae_local, self.quantize_local = vae_local, vae_local.quantize
        self.quantize_local: VectorQuantizer2
        self.var_wo_ddp: VAR = var_wo_ddp  # after torch.compile
        self.num_classes = num_classes
        
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
        
        self.qhats = []
    
    @torch.no_grad()
    def autoregressive_cp(self, ld_val, cp_type = 'global_vanilla_cp', alpha = 0.1, 
        g_seed: Optional[int] = None, 
    ) -> torch.Tensor:   # returns reconstructed image (B, 3, H, W) in [0, 1]
        """
        only used for inference, on autoregressive mode
        :param cfg: classifier-free guidance ratio
        :param top_k: top-k sampling
        :param top_p: top-p sampling
        :return: if returns_vemb: list of embedding h_BChw := vae_embed(idx_Bl), else: list of idx_Bl
        """
        rng = self.var_wo_ddp.rng
        # TODO: Decide where to init
        self.var_wo_ddp.cp_type = cp_type
        self.var_wo_ddp.alpha = alpha
        for inp_B3HW, label_B in ld_val: 
           # TODO: Remove this for proper calib. on cluster
            print(label_B)
            for i, label in enumerate(label_B): 
                if label == 1: 
                    label_B[i] = 2
            print(label_B)
            
            if g_seed is None: rng = None
            else: self.var_wo_ddp.rng.manual_seed(g_seed); rng = self.var_wo_ddp.rng
            B, V = label_B.shape[0], self.vae_local.vocab_size
            
            #label_B = torch.full((B,), fill_value=self.num_classes if label_B < 0 else label_B, device=self.lvl_1L.device)
            
            inp_B3HW = inp_B3HW.to(dist.get_device(), non_blocking=True)
            label_B = label_B.to(dist.get_device(), non_blocking=True)
            
            sos = cond_BD = self.var_wo_ddp.class_emb(torch.cat((label_B, torch.full_like(label_B, fill_value=self.var_wo_ddp.num_classes)), dim=0))
              
            lvl_pos = self.var_wo_ddp.lvl_embed(self.var_wo_ddp.lvl_1L) + self.var_wo_ddp.pos_1LC
            next_token_map = sos.unsqueeze(1).expand(2 * B, self.var_wo_ddp.first_l, -1) + self.var_wo_ddp.pos_start.expand(2 * B, self.var_wo_ddp.first_l, -1) + lvl_pos[:, :self.var_wo_ddp.first_l]
                
            cur_L = 0
            f_hat = sos.new_zeros(B, self.var_wo_ddp.Cvae, self.var_wo_ddp.patch_nums[-1], self.var_wo_ddp.patch_nums[-1])
            
            # true indeces for all token maps
            gt_idx_Bl: List[ITen] = self.vae_local.img_to_idxBl(inp_B3HW)
            # flattens indices into one sequence 
            gt_BL = torch.cat(gt_idx_Bl, dim=1)
            # MAYBE_ len(self.begin_ends) == len(self.var_wo_ddp.patch_nums) 
            for b in self.var_wo_ddp.blocks: b.attn.kv_caching(True)
            for si, pn in enumerate(self.var_wo_ddp.patch_nums):   # si: i-th segment 
                ratio = si / self.var_wo_ddp.num_stages_minus_1
                # last_L = cur_L
                cur_L += pn*pn
                # assert self.attn_bias_for_masking[:, :, last_L:cur_L, :cur_L].sum() == 0, f'AR with {(self.attn_bias_for_masking[:, :, last_L:cur_L, :cur_L] != 0).sum()} / {self.attn_bias_for_masking[:, :, last_L:cur_L, :cur_L].numel()} mask item'
                cond_BD_or_gss = self.var_wo_ddp.shared_ada_lin(cond_BD)
                x = next_token_map
                AdaLNSelfAttn.forward
                for b in self.var_wo_ddp.blocks:
                    x = b(x=x, cond_BD=cond_BD_or_gss, attn_bias=None)
                logits_BlV = self.var_wo_ddp.get_logits(x, cond_BD)
                
                # TODO: Is that sensible? --> other approach: simply use cond drop rate as t? 
                # cfg = self.var_wo_ddp.cond_drop_rate
                t = self.var_wo_ddp.cfg * ratio
                # Combines scaled logits 
                logits_BlV = (1+t) * logits_BlV[:B] - t * logits_BlV[B:]
                
                begin = self.begin_ends[si][0]
                end = self.begin_ends[si][1]
                true_indcs_expanded = gt_BL[: , begin:end].unsqueeze(-1)
                
                if cp_type == 'global_vanilla_cp':
                    softmax_BlV = logits_BlV.softmax(dim=-1)
                    max_values, _ = softmax_BlV.max(dim=2)
                    print("Max values along the last dimension (V):", max_values)
                    # logits of the true classes
                    probs_true_cls = torch.gather(softmax_BlV, dim = 2, index = true_indcs_expanded).squeeze()  # Shape [B, begin:end]
                    # 1) get probs via sofmax
                    print(f"max prob true class {probs_true_cls.max(dim=-1)}")
                    cal_scores = 1 - probs_true_cls # Shape [B, begin:end]
                    
                    if probs_true_cls.shape == true_indcs_expanded.squeeze().shape: 
                       print(f"All fine shapes are matching: {probs_true_cls.shape}, {true_indcs_expanded.squeeze().shape}")
                    cal_scores_flat = torch.flatten(cal_scores)
                    # n = number of total pixels, since we are in global vanilla cp 
                    n = B * gt_BL[: , begin:end].shape[-1]
                    print(f"n:{n}")
                    # coverage guarantee for alpha with small finite sample correction
                    q_level = np.ceil((n+1)*(1-alpha))/n
                    # at least qlevel examples have true class scores above qhat 
                    qhat = np.quantile(cal_scores_flat, q_level, interpolation='higher')
                    print(f"qhat: {qhat},qlevel: {q_level}")
                    self.qhats.append(qhat.item())
                
                # --------------------------------------------------------------- assign index, adapt token map for next resolution
                # returns the indices of associated codebook vectors for respective resolution
                # TODO: Maybe sample from credal prediction set? 
                idx_Bl = sample_with_top_k_top_p_(logits_BlV, rng=rng, top_k=0, top_p=0, num_samples=1)[:, :, 0]
                h_BChw = self.var_wo_ddp.vae_quant_proxy[0].embedding(idx_Bl)   # B, l, Cvae
                h_BChw = h_BChw.transpose_(1, 2).reshape(B, self.var_wo_ddp.Cvae, pn, pn)
                f_hat, next_token_map = self.var_wo_ddp.vae_quant_proxy[0].get_next_autoregressive_input(si, len(self.var_wo_ddp.patch_nums), f_hat, h_BChw)
                print(f"shape of f_hat: {f_hat.shape}")
                print(f"shape of next_token_map before doubeling: {next_token_map.shape}")
                                
                # ---------------------------------------------------------------
                if si != self.var_wo_ddp.num_stages_minus_1:   # prepare for next stage
                    next_token_map = next_token_map.view(B, self.var_wo_ddp.Cvae, -1).transpose(1, 2)
                    next_token_map = self.var_wo_ddp.word_embed(next_token_map) + lvl_pos[:, cur_L:cur_L + self.patch_nums[si+1] ** 2]
                    # TODO: Don't think that this is necessary
                    next_token_map = next_token_map.repeat(2, 1, 1)   # double the batch sizes due to CFG
                    print(f"shape of next_token_map before doubeling: {next_token_map.shape}")
        print("QHAAAAAAAAAAAAATS")
        print(self.qhats)
        for b in self.var_wo_ddp.blocks: b.attn.kv_caching(False)
        return self.var_wo_ddp.vae_proxy[0].fhat_to_img(f_hat).add_(1).mul_(0.5)   # de-normalize, from [-1, 1] to [0, 1]
        

    @torch.no_grad()
    def teacher_enforced_cp(self, ld_val, cp_type = 'global_vanilla_cp', alpha = 0.1, 
        g_seed: Optional[int] = None,
    ) -> torch.Tensor:   # returns reconstructed image (B, 3, H, W) in [0, 1]
        """
        only used for inference, on autoregressive mode
        :param cfg: classifier-free guidance ratio
        :param top_k: top-k sampling
        :param top_p: top-p sampling
        :return: if returns_vemb: list of embedding h_BChw := vae_embed(idx_Bl), else: list of idx_Bl
        """
        rng = self.var_wo_ddp.rng
        # TODO: Decide where to init
        self.var_wo_ddp.cp_type = cp_type
        self.var_wo_ddp.alpha = alpha
        for inp_B3HW, label_B in ld_val: 
           # TODO: Remove this for proper calib. on cluster
            #print(label_B)
            for i, label in enumerate(label_B): 
                if label == 1: 
                    label_B[i] = 2
            #print(label_B)
            
            if g_seed is None: rng = None
            else: self.var_wo_ddp.rng.manual_seed(g_seed); rng = self.var_wo_ddp.rng
            B, V = label_B.shape[0], self.vae_local.vocab_size
            
            #label_B = torch.full((B,), fill_value=self.num_classes if label_B < 0 else label_B, device=self.lvl_1L.device)
            
            inp_B3HW = inp_B3HW.to(dist.get_device(), non_blocking=True)
            label_B = label_B.to(dist.get_device(), non_blocking=True)
            
            sos = cond_BD = self.var_wo_ddp.class_emb(torch.cat((label_B, torch.full_like(label_B, fill_value=self.var_wo_ddp.num_classes)), dim=0))
              
            lvl_pos = self.var_wo_ddp.lvl_embed(self.var_wo_ddp.lvl_1L) + self.var_wo_ddp.pos_1LC
            next_token_map = sos.unsqueeze(1).expand(2 * B, self.var_wo_ddp.first_l, -1) + self.var_wo_ddp.pos_start.expand(2 * B, self.var_wo_ddp.first_l, -1) + lvl_pos[:, :self.var_wo_ddp.first_l]
                
            cur_L = 0
            f_hat = sos.new_zeros(B, self.var_wo_ddp.Cvae, self.var_wo_ddp.patch_nums[-1], self.var_wo_ddp.patch_nums[-1])
            
            # true indeces for all token maps
            gt_idx_Bl: List[ITen] = self.vae_local.img_to_idxBl(inp_B3HW)
            
            ###################################################################################################
            # TODO: Possible extension: Get e.g. "ambigous" ground truth per resultion step k and patch via e.g. 5 nearest codebook vectors, 
            # --> create lambdas via rescaling the distances to sum up to 1 
            ###################################################################################################
            
            # flattens indices into one sequence 
            gt_BL = torch.cat(gt_idx_Bl, dim=1)
            # MAYBE_ len(self.begin_ends) == len(self.var_wo_ddp.patch_nums) 
            for b in self.var_wo_ddp.blocks: b.attn.kv_caching(True)
            for si, pn in enumerate(self.var_wo_ddp.patch_nums):   # si: i-th segment 
                ratio = si / self.var_wo_ddp.num_stages_minus_1
                # last_L = cur_L
                cur_L += pn*pn
                # assert self.attn_bias_for_masking[:, :, last_L:cur_L, :cur_L].sum() == 0, f'AR with {(self.attn_bias_for_masking[:, :, last_L:cur_L, :cur_L] != 0).sum()} / {self.attn_bias_for_masking[:, :, last_L:cur_L, :cur_L].numel()} mask item'
                cond_BD_or_gss = self.var_wo_ddp.shared_ada_lin(cond_BD)
                x = next_token_map
                AdaLNSelfAttn.forward
                for b in self.var_wo_ddp.blocks:
                    x = b(x=x, cond_BD=cond_BD_or_gss, attn_bias=None)
                logits_BlV = self.var_wo_ddp.get_logits(x, cond_BD)
                
                # TODO: Is that sensible? --> other approach: simply use cond drop rate as t? 
                # cfg = self.var_wo_ddp.cond_drop_rate
                t = self.var_wo_ddp.cfg * ratio
                # Combines scaled logits 
                logits_BlV = (1+t) * logits_BlV[:B] - t * logits_BlV[B:]
                
                # TODO: insert sth. similar to above
                begin = self.begin_ends[si][0]
                end = self.begin_ends[si][1]
                
                if cp_type == "conformal_cr": 
                    true_indcs_expanded = gt_BL[: , begin:end].unsqueeze(-1) # B, begin:end, 1
                    softmax_BlV = logits_BlV.softmax(dim=-1) # B, l, V --> B, begin:end, 40xx
                    B, L, V = softmax_BlV.shape
                    cat_true = torch.zeros_like(softmax_BlV)

                    # One hot tensor of true class
                    cat_true.scatter_(-1, true_indcs_expanded, 1)
                    print(f"cat true shape {cat_true.shape}")
                    
                                        
                    # Initialize wasserstein tensor
                    ws_tensor = torch.zeros((B, L), device=softmax_BlV.device)

                    # ws between pixel/patch and true cat. distribution
                    for b in range(B):
                        for l in range(L):
                            p = softmax_BlV[b, l].cpu().numpy() #
                            q = cat_true[b, l].cpu().numpy()

                            # Compute Wasserstein distance
                            ws_tensor[b, l] = scipy.stats.wasserstein_distance(range(V), range(V), p, q) 
                                        
                    print(f"ws_tensor shape: {ws_tensor.shape}")
                    print(ws_tensor)
                    
                    n = B
                    print(f"n:{n}")
                    # coverage guarantee for alpha with small finite sample correction
                    q_level = np.ceil((n+1)*(1-alpha))/n
                    qhat = torch.quantile(ws_tensor, q_level, dim=0, interpolation='higher') # shape: (1, start:end)
                    print(f"qhat: {qhat.shape},qlevel: {q_level}")
                    qhat_numpy = qhat.cpu().numpy()
                    if qhat_numpy.ndim == 0:  # If scalar
                        qhat_flat = [qhat_numpy.item()]  # Convert single scalar to list
                    else:
                        qhat_flat = qhat_numpy.flatten().tolist()  # Convert array to list
                    self.qhats.extend(qhat_flat)
                
                if cp_type == "local_vanilla_cp": 
                    true_indcs_expanded = gt_BL[: , begin:end].unsqueeze(-1)

                    softmax_BlV = logits_BlV.softmax(dim=-1)
                    max_values, _ = softmax_BlV.max(dim=2)
                    #print("Max values along the last dimension (V):", max_values)
                    # logits of the true classes
                    probs_true_cls = torch.gather(softmax_BlV, dim = 2, index = true_indcs_expanded)  # Shape [B, begin:end]
                    # 1) get probs via sofmax
                    #print(f"max prob true class {probs_true_cls.max(dim=-1)}")
                    cal_scores = 1 - probs_true_cls # Shape [B, begin:end]
                    
                    #if probs_true_cls.shape == true_indcs_expanded.squeeze().shape: 
                       #print(f"All fine shapes are matching: {probs_true_cls.shape}, {true_indcs_expanded.squeeze().shape}")
                    #cal_scores_flat = torch.flatten(cal_scores)
                    # n = B since we are in local vanilla cp 
                    print(f"Cal scores {cal_scores.shape}")
                    n = B
                    print(f"n:{n}")
                    # coverage guarantee for alpha with small finite sample correction
                    q_level = np.ceil((n+1)*(1-alpha))/n
                    qhat = torch.quantile(cal_scores, q_level, dim=0, interpolation='higher') # shape: (1, start:end)
                    print(f"qhat: {qhat.shape},qlevel: {q_level}")
                    qhat_numpy = qhat.cpu().numpy()
                    if qhat_numpy.ndim == 0:  # If scalar
                        qhat_flat = [qhat_numpy.item()]  # Convert single scalar to list
                    else:
                        qhat_flat = qhat_numpy.flatten().tolist()  # Convert array to list
                    self.qhats.extend(qhat_flat)
                    
                
                if cp_type == 'global_vanilla_cp':
                    true_indcs_expanded = gt_BL[: , begin:end].unsqueeze(-1)

                    softmax_BlV = logits_BlV.softmax(dim=-1)
                    max_values, _ = softmax_BlV.max(dim=2)
                    #print("Max values along the last dimension (V):", max_values)
                    # logits of the true classes
                    probs_true_cls = torch.gather(softmax_BlV, dim = 2, index = true_indcs_expanded).squeeze()  # Shape [B, begin:end]
                    # 1) get probs via sofmax
                    print(f"max prob true class {probs_true_cls.max(dim=-1)}")
                    cal_scores = 1 - probs_true_cls # Shape [B, begin:end]
                    
                    if probs_true_cls.shape == true_indcs_expanded.squeeze().shape: 
                       print(f"All fine shapes are matching: {probs_true_cls.shape}, {true_indcs_expanded.squeeze().shape}")
                    cal_scores_flat = torch.flatten(cal_scores)
                    # n = number of total pixels, since we are in global vanilla cp 
                    n = B * gt_BL[: , begin:end].shape[-1]
                    print(f"n:{n}")
                    # coverage guarantee for alpha with small finite sample correction
                    q_level = np.ceil((n+1)*(1-alpha))/n
                    # at least qlevel examples have true class scores above qhat 
                    qhat = np.quantile(cal_scores_flat, q_level, interpolation='higher')
                    print(f"qhat: {qhat},qlevel: {q_level}")
                    self.qhats.append(qhat.item())
                    
                elif cp_type == "global_adaptive_cp": 
                    # general procedure: 
                    # Score construction: Greedily include classes until we reach true class 
                    # --> uses softmax outputs of all classes with higher or equal pred. prob
                    softmax_BlV = logits_BlV.softmax(dim=-1) # shape: B, begin:end, V
                    softmax_BV = softmax_BlV.view(softmax_BlV.shape[0]*softmax_BlV.shape[1], softmax_BlV.shape[2]) # shape: B*l, V
                    print(f"sofmax shape after: {softmax_BV.shape}")
                    # Get scores. calib_X.shape[0] == calib_Y.shape[0] == n
                    # sort across dim V, get 
                    l = end-begin
                    print(f"numbers of patches:{l}")
                    # eg. 0.3, 0.2, 0.2, 0.1.............., 1, 3, 2, 0, 14, # B*l, V
                    sorted_softmax_BV, sorted_indices = torch.sort(softmax_BV, dim=-1, descending=True)
                    # TODO: regularize sorted 
                    # e.g. 15, 16, 17, 1 , 2, 300, usw. 
                    true_indcs = gt_BL[: , begin:end]
                    print(f"shape of true indcs : {true_indcs.shape}")
                    true_indcs_expanded = true_indcs.reshape(true_indcs.shape[0]*true_indcs.shape[1])
                    print(f"shape of true indcs expanded: {true_indcs_expanded.shape}")
                    
                    # Get position of true class out of sorted indices: 
                    position_tc = (sorted_indices == true_indcs_expanded[:, None]).nonzero(as_tuple=True)[1].unsqueeze(-1)# shape B*l, 1
                    print(f"Positions of true classes: {position_tc}")
                    # cumulative sum across sorted  softmax BV
                    cal_scores = torch.cumsum(sorted_softmax_BV, dim=1) 
                    # TODO: add regularization; Select index of true class 
                    cal_scores = torch.gather(cal_scores, dim=1, index=position_tc)
                    n = B * gt_BL[: , begin:end].shape[-1]
                    qlevel = np.ceil((n+1)*(1-alpha))/n
                    qhat = np.quantile(cal_scores, qlevel, interpolation='higher')
                    
                
                # --------------------------------------------------------------- assign index, adapt token map for next resolution
                # returns the indices of associated codebook vectors for respective resolution
                idx_Bl = gt_BL[: , begin:end]                
                
                h_BChw = self.var_wo_ddp.vae_quant_proxy[0].embedding(idx_Bl)   # B, l, Cvae
                h_BChw = h_BChw.transpose_(1, 2).reshape(B, self.var_wo_ddp.Cvae, pn, pn)
                f_hat, next_token_map = self.var_wo_ddp.vae_quant_proxy[0].get_next_autoregressive_input(si, len(self.var_wo_ddp.patch_nums), f_hat, h_BChw)                                
                # ---------------------------------------------------------------
                if si != self.var_wo_ddp.num_stages_minus_1:   # prepare for next stage
                    next_token_map = next_token_map.view(B, self.var_wo_ddp.Cvae, -1).transpose(1, 2)
                    next_token_map = self.var_wo_ddp.word_embed(next_token_map) + lvl_pos[:, cur_L:cur_L + self.patch_nums[si+1] ** 2]
                    next_token_map = next_token_map.repeat(2, 1, 1)   # double the batch sizes due to CFG

        self.var_wo_ddp.qhats = torch.FloatTensor(self.qhats)
        for b in self.var_wo_ddp.blocks: b.attn.kv_caching(False)
        