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
    def teacher_enforced_cp(self, ld_val: DataLoader, cp_type = 1, alpha = 0.1,
        g_seed: Optional[int] = None, autoregressive = False) -> torch.Tensor:   # returns reconstructed image (B, 3, H, W) in [0, 1]
        """
        only used for inference, on autoregressive mode
        :param cfg: classifier-free guidance ratio
        :param top_k: top-k sampling
        :param top_p: top-p sampling
        :return: if returns_vemb: list of embedding h_BChw := vae_embed(idx_Bl), else: list of idx_Bl
        """
        rng = self.var_wo_ddp.rng

        distances = self.var_wo_ddp.vae_proxy[0].create_embedding_distances() # 4096 * 4096

        assert torch.allclose(self.vae_local.quantize.embedding.weight, self.var_wo_ddp.vae_proxy[0].quantize.embedding.weight)
        assert id(self.vae_local) == id(self.var_wo_ddp.vae_proxy[0])

        cal_scores_list = [None] * len(self.var_wo_ddp.patch_nums)
        for i, (inp_B3HW, label_B) in enumerate(ld_val):
            if g_seed is None: rng = None
            else: self.var_wo_ddp.rng.manual_seed(g_seed); rng = self.var_wo_ddp.rng
            B, V = label_B.shape[0], self.vae_local.vocab_size

            inp_B3HW = inp_B3HW.to(dist.get_device(), non_blocking=True)
            label_B = label_B.to(dist.get_device(), non_blocking=True)

            sos = cond_BD = self.var_wo_ddp.class_emb(torch.cat((label_B, torch.full_like(label_B, fill_value=self.var_wo_ddp.num_classes)), dim=0))
            #sos = cond_BD = self.var_wo_ddp.class_emb(label_B)

            lvl_pos = self.var_wo_ddp.lvl_embed(self.var_wo_ddp.lvl_1L) + self.var_wo_ddp.pos_1LC
            next_token_map = sos.unsqueeze(1).expand(2 * B, self.var_wo_ddp.first_l, -1) + self.var_wo_ddp.pos_start.expand(2 * B, self.var_wo_ddp.first_l, -1) + lvl_pos[:, :self.var_wo_ddp.first_l]
            cur_L = 0
            f_hat = sos.new_zeros(B, self.var_wo_ddp.Cvae, self.var_wo_ddp.patch_nums[-1], self.var_wo_ddp.patch_nums[-1])

            # true indeces for all token maps
            gt_idx_Bl: List[ITen] = self.vae_local.img_to_idxBl(inp_B3HW)

            # To get the ambigous labels for the for the CCR 
            if cp_type == 4 or cp_type == 6:
                distances_BlV: List[ITen] = self.vae_local.image_to_distances(inp_B3HW)
                distances_BlV = torch.cat(distances_BlV, dim=1)

            # flattens indices into one sequence
            gt_BL = torch.cat(gt_idx_Bl, dim=1)
            for b in self.var_wo_ddp.blocks: b.attn.kv_caching(True)
            for si, pn in enumerate(self.var_wo_ddp.patch_nums):

                ratio = si / self.var_wo_ddp.num_stages_minus_1
                # last_L = cur_L
                cur_L += pn*pn
                # assert self.attn_bias_for_masking[:, :, last_L:cur_L, :cur_L].sum() == 0, f'AR with {(self.attn_bias_for_masking[:, :, last_L:cur_L, :cur_L] != 0).sum()} / {self.attn_bias_for_masking[:, :, last_L:cur_L, :cur_L].numel()} mask item'
                cond_BD_or_gss = self.var_wo_ddp.shared_ada_lin(cond_BD)
                x = next_token_map
                # attn_bias = self.var_wo_ddp.attn_bias_for_masking[:, :, :cur_L, :cur_L]
                AdaLNSelfAttn.forward
                for b in self.var_wo_ddp.blocks:
                    x = b(x=x, cond_BD=cond_BD_or_gss, attn_bias=None)
                logits_BlV = self.var_wo_ddp.get_logits(x, cond_BD)

                # cfg = self.var_wo_ddp.cond_drop_rate
                t = 4 * ratio
                # Combines scaled logits
                logits_BlV = (1+t) * logits_BlV[:B] - t * logits_BlV[B:]

                begin = self.begin_ends[si][0]
                end = self.begin_ends[si][1]

                # if True:
                #     idx_to_remove = logits_BlV < logits_BlV.topk(900, largest=True, sorted=False, dim=-1)[0].amin(dim=-1, keepdim=True)
                #     logits_BlV.masked_fill_(idx_to_remove, -torch.inf)
                # if True:
                #     sorted_logits, sorted_idx = logits_BlV.sort(dim=-1, descending=False)
                #     sorted_idx_to_remove = sorted_logits.softmax(dim=-1).cumsum_(dim=-1) <= (1 - 0-95)
                #     sorted_idx_to_remove[..., -1:] = False
                #     logits_BlV.masked_fill_(sorted_idx_to_remove.scatter(sorted_idx.ndim - 1, sorted_idx, sorted_idx_to_remove), -torch.inf)

                true_indcs_expanded = gt_BL[: , begin:end].unsqueeze(-1) # B, begin:end, 1
                softmax_BlV = logits_BlV.softmax(dim=-1) # B, l, V --> B, begin:end, 40xx
                B, L, V = softmax_BlV.shape

                # traditional CP with neg. prob. of true class as score function
                if cp_type == 1:
                    probs_true_cls = torch.gather(softmax_BlV, dim = 2, index = true_indcs_expanded)  # Shape [B, begin:end]
                    cal_scores = - probs_true_cls # Shape [B, L]
                    log10 = torch.log10(softmax_BlV)
                    torch.nan_to_num_(log10, nan=0)
                    emp_entropy = (- torch.sum(softmax_BlV * log10, dim = -1).unsqueeze(-1)) / np.log10(4096)

                    # define neighbourhood of true class --> max.
                    p_true = torch.gather(softmax_BlV, dim=2, index=true_indcs_expanded).squeeze(-1)  # [B, L]
                    dists, topk_idxs = distances.topk(50, dim=-1, largest=False) # V, k | B, L, 1
                    neighbours = topk_idxs[true_indcs_expanded.squeeze(-1)]
                    p_neighbourhood = torch.gather(softmax_BlV, dim=2, index=neighbours)

                # cumulative CP
                if cp_type == 2:
                    sorted_softmax_BlV, sorted_indices = torch.sort(softmax_BlV, dim=-1, descending=True)
                    position_tc = (sorted_indices == true_indcs_expanded)# shape B,l, V
                    position_tc = torch.argmax(position_tc.to(torch.int64), dim=-1)  # B, l, 1
                    cal_scores = torch.cumsum(sorted_softmax_BlV, dim=-1)
                    cal_scores = torch.gather(cal_scores, dim=-1, index=position_tc.unsqueeze(-1))

                # Conformalized Credal Region with neg. prob. as score function
                if cp_type == 4:
                    distances = distances_BlV[:, begin:end, :] # B, l, V
                    probs_distances_BlV = torch.nn.functional.softmax(-distances, dim=-1)
                    cal_scores = probs_distances_BlV * - softmax_BlV
                    cal_scores = torch.sum(cal_scores, dim = -1).squeeze()

                # CCR with crisp ground truth with scaled max_other_prob as score function
                if cp_type == 5:
                    probs_true_cls = torch.gather(softmax_BlV, dim = 2, index = true_indcs_expanded) # shape: B, L
                    softmax_BlV.scatter_(-1, true_indcs_expanded, torch.zeros_like(true_indcs_expanded, dtype=softmax_BlV.dtype))
                    max_other, idx = torch.max(softmax_BlV, dim=-1)
                    gamma = 0.00001
                    if si == 0:
                        a = (probs_true_cls.squeeze().unsqueeze(-1) + gamma)
                    else:
                        a = (probs_true_cls.squeeze() + gamma)
                    cal_scores = max_other / a
                    
                # CCR with ambigous ground truth with scaled max_other_prob as score function
                if cp_type == 6:
                    distances = distances_BlV[:, begin:end, :]
                    probs_distances_BlV = torch.nn.functional.softmax(-distances, dim=-1)
                    probs_true_cls = torch.gather(softmax_BlV, dim = 2, index = true_indcs_expanded) # shape: B, L
                    max_values, _ = softmax_BlV.max(dim=-1, keepdim=True)  # Shape: (B, L, 1)

                    # Create a mask for elements that are equal to the max
                    mask = softmax_BlV == max_values  # Shape: (B, L, V)

                    # Replace max values with -inf temporarily to get second max
                    X_masked = softmax_BlV.clone()
                    X_masked[mask] = float('-inf')
                    second_max_values, _ = X_masked.max(dim=-1, keepdim = True)  # Shape: (B, L)
                    max_excluding_self = torch.where(mask, second_max_values, max_values)

                    gamma = 0.001
                    a = max_excluding_self / (softmax_BlV + gamma)
                    scores = torch.mul(probs_distances_BlV, a)
                    cal_scores = torch.sum(scores, dim = 2)
                
                # CPS Experimental with scaled logits (shannon entropy as temperature) and cumulative score function
                if cp_type == 7:
                    # scale with shannon entropy
                    log10 = torch.log10(softmax_BlV)
                    torch.nan_to_num_(log10, nan=0)
                    emp_entropy = (- torch.sum(softmax_BlV * log10, dim = -1).unsqueeze(-1)) / np.log10(4096)
                    logits_BlV_emp = logits_BlV / (2 * emp_entropy + 0.0000001)

                    softmax_BlV_emp = torch.softmax(logits_BlV_emp, dim = -1)

                    sorted_softmax_BlV, sorted_indices = torch.sort(softmax_BlV_emp, dim=-1, descending=True)
                    position_tc = (sorted_indices == true_indcs_expanded)# shape B,l, V
                    position_tc = torch.argmax(position_tc.to(torch.int64), dim=-1)  # B, l, 1
                    cal_scores = torch.cumsum(sorted_softmax_BlV, dim=-1)
                    cal_scores = torch.gather(cal_scores, dim=-1, index=position_tc.unsqueeze(-1))

                # CPS Experimental with scaled logits (rao entropy as temperature) and cumulative score function
                if cp_type == 8:
                    p_flat = softmax_BlV.reshape(-1, V)  # shape (B*l, V)
                    dp = torch.matmul(p_flat, distances)
                    rao_entropy_flat = torch.sum(p_flat * dp, dim=-1)
                    rao_entropy = rao_entropy_flat.reshape(B, L).unsqueeze(-1)
                    logits_BlV_emp = logits_BlV / rao_entropy
                    softmax_BlV_emp = torch.softmax(logits_BlV_emp, dim = -1)

                    sorted_softmax_BlV, sorted_indices = torch.sort(softmax_BlV_emp, dim=-1, descending=True)
                    position_tc = (sorted_indices == true_indcs_expanded)# shape B,l, V
                    position_tc = torch.argmax(position_tc.to(torch.int64), dim=-1)  # B, l, 1
                    cal_scores = torch.cumsum(sorted_softmax_BlV, dim=-1)
                    cal_scores = torch.gather(cal_scores, dim=-1, index=position_tc.unsqueeze(-1))

                # CPS Experimental: Scale the Logits based on expected distance from all other classes
                if cp_type == 9:
                    p_flat = softmax_BlV.reshape(-1, V)  # shape (B*L, V)
                    # Matrix multiply to get expected distances per class
                    # Each row: expected distances from all other classes
                    distance_prob_flat = torch.matmul(p_flat, distances)  # shape (B*L, V)
                    distance_prob = distance_prob_flat.reshape(B, L, V)

                    logits_BlV_scaled = logits_BlV / distance_prob
                    softmax_BlV_scaled = torch.softmax(logits_BlV_scaled, dim = -1)

                    probs_true_cls = torch.gather(softmax_BlV_scaled, dim = 2, index = true_indcs_expanded) # shape: B, L
                    softmax_BlV_scaled.scatter_(-1, true_indcs_expanded, torch.zeros_like(true_indcs_expanded, dtype=softmax_BlV_scaled.dtype))
                    max_other, _ = torch.max(softmax_BlV_scaled, dim=-1)
                    gamma = 0.00001
                    if si == 0:
                        a = (probs_true_cls.squeeze().unsqueeze(-1) + gamma)
                    else:
                        a = (probs_true_cls.squeeze() + gamma)
                    cal_scores = max_other / a
                    print(cal_scores)

                # CPS Experimental: Scale the Logits based on max. distance from other classes
                if cp_type == 10:
                    p_flat = softmax_BlV.reshape(-1, V)
                    distance_prob_flat = torch.matmul(p_flat, distances)
                    distance_prob = distance_prob_flat.reshape(B, L, V)
                    #print("distance prob", distance_prob.shape)
                    max_dist, _ = distance_prob.max(dim = -1, keepdim = True)
                    #print("max_dist", max_dist.shape)
                    dist_entropy = torch.sum(softmax_BlV * (max_dist), dim = -1).unsqueeze(-1)
                    #print("Dist entr", dist_entropy.shape)
                    logits_BlV_emp = logits_BlV / dist_entropy

                    softmax_BlV_emp = torch.softmax(logits_BlV_emp, dim = -1)

                    # # eg. 0.3, 0.2, 0.2, 0.1.............., 1, 3, 2, 0, 14, # B, l, V
                    # sorted_softmax_BlV, sorted_indices = torch.sort(softmax_BlV_emp, dim=-1, descending=True)
                    # # Get position of true class out of sorted indices:
                    # position_tc = (sorted_indices == true_indcs_expanded)# shape B,l, V
                    # position_tc = torch.argmax(position_tc.to(torch.int64), dim=-1)  # B, l, 1
                    # # cumulative sum across sorted softmax BlV
                    # cal_scores = torch.cumsum(sorted_softmax_BlV, dim=-1)
                    # cal_scores = torch.gather(cal_scores, dim=-1, index=position_tc.unsqueeze(-1))
                    probs_true_cls = torch.gather(softmax_BlV_emp, dim = 2, index = true_indcs_expanded) # shape: B, L
                    softmax_BlV_emp.scatter_(-1, true_indcs_expanded, torch.zeros_like(true_indcs_expanded, dtype=softmax_BlV_emp.dtype))
                    max_other, _ = torch.max(softmax_BlV_emp, dim=-1)
                    gamma = 0.00001
                    if si == 0:
                        a = (probs_true_cls.squeeze().unsqueeze(-1) + gamma)
                    else:
                        a = (probs_true_cls.squeeze() + gamma)
                    cal_scores = max_other / a
                    
                    
                # CPS Experimental: Scale scores based on entropy of top-k logits
                # if cp_type == 3:
                #     _, topk_idxs = distances.topk(100, dim=-1, largest=False) # V, k | B, L, 1
                #     neighbours = topk_idxs[true_indcs_expanded.squeeze(-1)]
                #     mask = torch.zeros_like(softmax_BlV).bool()
                #     mask.scatter_(dim=-1, index=neighbours, value=True)
                #     # Zero out non-top-k entries
                #     softmax_BlV = softmax_BlV * mask

                #     # ------------- compute entropy of top-k classes
                #     topk_logits = torch.gather(logits_BlV, dim=-1, index=topk_idxs)
                #     topk_softmax = torch.softmax(topk_logits, dim = -1)
                #     log10 = torch.log10(topk_softmax)
                #     torch.nan_to_num_(log10, nan=0)
                #     emp_entropy = (- torch.sum(topk_softmax * log10, dim = -1).unsqueeze(-1)) / np.log10(100)
                #     # --------------

                #     sorted_softmax_BlV, sorted_indices = torch.sort(softmax_BlV, dim=-1, descending=True)
                #     position_tc = (sorted_indices == true_indcs_expanded)# shape B,l, V
                #     position_tc = torch.argmax(position_tc.to(torch.int64), dim=-1)  # B, l, 1
                #     cal_scores = torch.cumsum(sorted_softmax_BlV, dim=-1)
                #     cal_scores = torch.gather(cal_scores, dim=-1, index=position_tc.unsqueeze(-1)) * emp_entropy

                
                # concat cal_scores for given resolution along batch axis
                # if no element for resolution exists --> first iteration of ld_val
                if cal_scores_list[si] is None:
                    cal_scores_list[si] = cal_scores
                else:
                    cal_scores_list[si] = torch.cat((cal_scores_list[si], cal_scores), axis = 0)
                # --------------------------------------------------------------- assign index, adapt token map for next resolution
                # returns the indices of associated codebook vectors for respective resolution
                if not autoregressive:
                    idx_Bl = gt_BL[: , begin:end]
                else:
                    idx_Bl = sample_with_top_k_top_p_(logits_BlV, rng=rng, top_k=0, top_p=0, num_samples=1)[:, :, 0]

                h_BChw = self.var_wo_ddp.vae_quant_proxy[0].embedding(idx_Bl)   # B, l, Cvae
                h_BChw = h_BChw.transpose_(1, 2).reshape(B, self.var_wo_ddp.Cvae, pn, pn)
                f_hat, next_token_map = self.var_wo_ddp.vae_quant_proxy[0].get_next_autoregressive_input(si, len(self.var_wo_ddp.patch_nums), f_hat, h_BChw)
                # ---------------------------------------------------------------
                if si != self.var_wo_ddp.num_stages_minus_1:   # prepare for next stage
                    next_token_map = next_token_map.view(B, self.var_wo_ddp.Cvae, -1).transpose(1, 2)
                    next_token_map = self.var_wo_ddp.word_embed(next_token_map) + lvl_pos[:, cur_L:cur_L + self.patch_nums[si+1] ** 2]
                    next_token_map = next_token_map.repeat(2, 1, 1)   # double the batch sizes due to CFG

        n = len(ld_val.dataset)
        if np.ceil((n+1)*(1-alpha))/n > 1:
            q_level = 1-alpha
        else:
            q_level = np.ceil((n+1)*(1-alpha))/n
        for si, pn in enumerate(self.var_wo_ddp.patch_nums):
            cal_scores = cal_scores_list[si]
            qhat = torch.quantile(cal_scores, q_level, dim=0, interpolation='higher') # shape: (1, start:end)
            qhat_numpy = qhat.cpu().numpy()
            if qhat_numpy.ndim == 0:  # If scalar
                qhat_flat = [qhat_numpy.item()]  # Convert single scalar to list
            else:
                qhat_flat = qhat_numpy.flatten().tolist()  # Convert array to list
            self.qhats.extend(qhat_flat)

        num_samples = 20000
        random_indices = torch.randint(0, cal_scores.size(0), (num_samples,))
        random_indices = random_indices.to(cal_scores.device)
        sampled_random = torch.index_select(cal_scores, dim=0, index=random_indices)
        self.var_wo_ddp.qhats = torch.FloatTensor(self.qhats)
        self.var_wo_ddp.cp_type = torch.FloatTensor([cp_type])
        self.var_wo_ddp.alpha = torch.FloatTensor([alpha])
        self.var_wo_ddp.cal_scores = torch.FloatTensor(sampled_random.squeeze().cpu())

        for b in self.var_wo_ddp.blocks: b.attn.kv_caching(False)
