import math
from functools import partial
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
from huggingface_hub import PyTorchModelHubMixin

import dist
from models.basic_var import AdaLNBeforeHead, AdaLNSelfAttn
from models.helpers import gumbel_softmax_with_rng, sample_with_top_k_top_p_
from models.vqvae import VQVAE, VectorQuantizer2
from torch.autograd.functional import jacobian
from models.helpers_calib import calc_entropy, sample_from_high_uq, generate_high_uq_samples, sample_from_prediction_set, calc_quadratic_entropy
import scipy
import torch.nn.functional as F
import numpy as np


class SharedAdaLin(nn.Linear):
    def forward(self, cond_BD):
        C = self.weight.shape[0] // 6
        return super().forward(cond_BD).view(-1, 1, 6, C)   # B16C


class VAR(nn.Module):
    def __init__(
        self, vae_local: VQVAE,
        num_classes=1000, depth=16, embed_dim=1024, num_heads=16, mlp_ratio=4., drop_rate=0., attn_drop_rate=0., drop_path_rate=0.,
        norm_eps=1e-6, shared_aln=False, cond_drop_rate=0.1,
        attn_l2_norm=False,
        patch_nums=(1, 2, 3, 4, 5, 6, 8, 10, 13, 16),   # 10 steps by default
        flash_if_available=True, fused_if_available=True, alpha = None, qhats = None, cp_type = None, cal_scores = None
    ):
        super().__init__()
        # 0. hyperparameters
        assert embed_dim % num_heads == 0
        self.Cvae, self.V = vae_local.Cvae, vae_local.vocab_size
        self.depth, self.C, self.D, self.num_heads = depth, embed_dim, embed_dim, num_heads
        
        self.cond_drop_rate = cond_drop_rate
        self.prog_si = -1   # progressive training
        
        self.patch_nums: Tuple[int] = patch_nums
        # total number of tokens across all scales
        self.L = sum(pn ** 2 for pn in self.patch_nums)
        # first number of tokens
        self.first_l = self.patch_nums[0] ** 2
        self.begin_ends = []
        cur = 0
        for i, pn in enumerate(self.patch_nums):
            self.begin_ends.append((cur, cur+pn ** 2))
            cur += pn ** 2
        
        self.num_stages_minus_1 = len(self.patch_nums) - 1
        self.rng = torch.Generator(device=dist.get_device())
        
        # 1. input (word) embedding
        quant: VectorQuantizer2 = vae_local.quantize
        self.vae_proxy: Tuple[VQVAE] = (vae_local,)
        self.vae_quant_proxy: Tuple[VectorQuantizer2] = (quant,)
        self.word_embed = nn.Linear(self.Cvae, self.C)
        
        # 2. class embedding
        init_std = math.sqrt(1 / self.C / 3)
        self.num_classes = num_classes
        self.uniform_prob = torch.full((1, num_classes), fill_value=1.0 / num_classes, dtype=torch.float32, device=dist.get_device())
        self.class_emb = nn.Embedding(self.num_classes + 1, self.C)
        nn.init.trunc_normal_(self.class_emb.weight.data, mean=0, std=init_std)
        self.pos_start = nn.Parameter(torch.empty(1, self.first_l, self.C))
        nn.init.trunc_normal_(self.pos_start.data, mean=0, std=init_std)
        
        # 3. absolute position embedding
        pos_1LC = []
        for i, pn in enumerate(self.patch_nums):
            pe = torch.empty(1, pn*pn, self.C)
            nn.init.trunc_normal_(pe, mean=0, std=init_std)
            pos_1LC.append(pe)
        pos_1LC = torch.cat(pos_1LC, dim=1)     # 1, L, C
        assert tuple(pos_1LC.shape) == (1, self.L, self.C)
        self.pos_1LC = nn.Parameter(pos_1LC)
        # level embedding (similar to GPT's segment embedding, used to distinguish different levels of token pyramid)
        self.lvl_embed = nn.Embedding(len(self.patch_nums), self.C)
        nn.init.trunc_normal_(self.lvl_embed.weight.data, mean=0, std=init_std)
        
        # 4. backbone blocks
        self.shared_ada_lin = nn.Sequential(nn.SiLU(inplace=False), SharedAdaLin(self.D, 6*self.C)) if shared_aln else nn.Identity()
        
        norm_layer = partial(nn.LayerNorm, eps=norm_eps)
        self.drop_path_rate = drop_path_rate
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule (linearly increasing)
        self.blocks = nn.ModuleList([
            AdaLNSelfAttn(
                cond_dim=self.D, shared_aln=shared_aln,
                block_idx=block_idx, embed_dim=self.C, norm_layer=norm_layer, num_heads=num_heads, mlp_ratio=mlp_ratio,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[block_idx], last_drop_p=0 if block_idx == 0 else dpr[block_idx-1],
                attn_l2_norm=attn_l2_norm,
                flash_if_available=flash_if_available, fused_if_available=fused_if_available,
            )
            for block_idx in range(depth)
        ])
        
        fused_add_norm_fns = [b.fused_add_norm_fn is not None for b in self.blocks]
        self.using_fused_add_norm_fn = any(fused_add_norm_fns)
        print(
            f'\n[constructor]  ==== flash_if_available={flash_if_available} ({sum(b.attn.using_flash for b in self.blocks)}/{self.depth}), fused_if_available={fused_if_available} (fusing_add_ln={sum(fused_add_norm_fns)}/{self.depth}, fusing_mlp={sum(b.ffn.fused_mlp_func is not None for b in self.blocks)}/{self.depth}) ==== \n'
            f'    [VAR config ] embed_dim={embed_dim}, num_heads={num_heads}, depth={depth}, mlp_ratio={mlp_ratio}\n'
            f'    [drop ratios ] drop_rate={drop_rate}, attn_drop_rate={attn_drop_rate}, drop_path_rate={drop_path_rate:g} ({torch.linspace(0, drop_path_rate, depth)})',
            end='\n\n', flush=True
        )
        
        # 5. attention mask used in training (for masking out the future)
        #    it won't be used in inference, since kv cache is enabled
        d: torch.Tensor = torch.cat([torch.full((pn*pn,), i) for i, pn in enumerate(self.patch_nums)]).view(1, self.L, 1)
        dT = d.transpose(1, 2)    # dT: 11L
        lvl_1L = dT[:, 0].contiguous()
        self.register_buffer('lvl_1L', lvl_1L)
        attn_bias_for_masking = torch.where(d >= dT, 0., -torch.inf).reshape(1, 1, self.L, self.L)
        self.register_buffer('attn_bias_for_masking', attn_bias_for_masking.contiguous())
        
        # 6. classifier head
        self.head_nm = AdaLNBeforeHead(self.C, self.D, norm_layer=norm_layer)
        self.head = nn.Linear(self.C, self.V)
        
        # 7. conformal prediction stuff
        self.register_buffer('qhats', None if qhats is None else torch.FloatTensor(qhats), persistent=True)
        self.register_buffer('cp_type', None if cp_type is None else torch.FloatTensor(cp_type), persistent=True)
        self.register_buffer('alpha', None if alpha is None else torch.FloatTensor(alpha), persistent=True)
        self.cp_count_sets = []
        self.register_buffer('cal_scores', None if cal_scores is None else torch.FloatTensor(cal_scores), persistent=True)
        self.cfg = 1.5 
        
    def get_logits(self, h_or_h_and_residual: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]], cond_BD: Optional[torch.Tensor]):
        if not isinstance(h_or_h_and_residual, torch.Tensor):
            h, resi = h_or_h_and_residual   # fused_add_norm must be used
            h = resi + self.blocks[-1].drop_path(h)
        else:                               # fused_add_norm is not used
            h = h_or_h_and_residual
        return self.head(self.head_nm(h.float(), cond_BD).float()).float()
    
    def get_prediction_sets(self, si, begin, end, logits_BlV, rng, top_k, top_p, num_samples=1, distances = None): 
        qhats = self.qhats[begin:end] # shape begin:end
        qhats = qhats.view(1, qhats.shape[0], 1) # shape: 1, l, 1
        softmax_BlV = logits_BlV.softmax(dim=-1) #shape[B, begin:end, V]
        count_set_l = None
        B, L, V = softmax_BlV.shape 
        
        emp_entropy = - torch.sum(softmax_BlV * torch.log10(softmax_BlV), dim = -1).unsqueeze(-1)
        emp_entropy = emp_entropy / np.log10(4096)
        p_flat = softmax_BlV.reshape(-1, V)  # shape (B*l, V)

        # Matrix multiply: (B*l, V) x (V, V) → (B*l, V)
        dp = torch.matmul(p_flat, distances)  # each row is D @ p
        # Elementwise product + sum over last dim: (B*l,) = sum_i p_i * [D @ p]_i
        rao_entropy_flat = torch.sum(p_flat * dp, dim=-1)        
        # Reshape back to (B, l)
        rao_entropy = rao_entropy_flat.reshape(B, L).unsqueeze(-1)
        idx_Bl = sample_with_top_k_top_p_(logits_BlV, rng=rng, top_k=top_k, top_p=top_p, num_samples=1)[:, :, 0] # B, l , 1
        idx_Bl = idx_Bl.unsqueeze(-1)
        ratio_q_scores = None

        # traditional CP with neg. prob. of true class as score function
        if self.cp_type == 1:    
            cal_scores = - softmax_BlV
            mask = cal_scores <= qhats
            count_set_l = mask.sum(dim=-1)
            self.cp_count_sets.append(count_set_l)
            
        # cumulative CP
        if self.cp_type == 2: 
            sorted_softmax_BlV, sorted_indices = torch.sort(softmax_BlV, dim=-1, descending=True) # B, L, V
            probs = torch.cumsum(sorted_softmax_BlV, dim=-1)  # shape B, l
            mask = probs <= qhats
            count_set_l = mask.sum(dim=-1) 
            self.cp_count_sets.append(count_set_l)
        
        # Conformalized Credal Region with neg. prob. as score function         
        if self.cp_type == 4: 
            # mask = softmax_BlV >= qhats
            for b in range(softmax_BlV.shape[0]): 
                max_entropies = torch.zeros(size = (softmax_BlV.shape[1],)) #l entries
                min_entropies = torch.zeros(size = (softmax_BlV.shape[1],)) #l entries
                # qhats = qhats.squeeze()
                qhats = qhats[0, :, 0]
                for l in range(qhats.shape[0]): 
                    min_entropy = calc_entropy(-softmax_BlV[b, l, :].squeeze(), tau = qhats[l], maximize = False)
                    max_entropy = calc_entropy(-softmax_BlV[b, l, :].squeeze(), tau = qhats[l], maximize = True)
                    
                    max_entropies[l] = max_entropy
                    min_entropies[l] = min_entropy
                if b == 0: 
                    max_count_set_l = max_entropies.unsqueeze(0)
                    min_count_set_l = min_entropies.unsqueeze(0)
                else: 
                    max_count_set_l = torch.cat((max_count_set_l, max_entropies.unsqueeze(0)), dim = 0)
                    min_count_set_l = torch.cat((min_count_set_l, min_entropies.unsqueeze(0)), dim = 0)
            #count_set_l = torch.cat((max_count_set_l.unsqueeze(0), min_count_set_l.unsqueeze(0)), dim = 0)
            #self.cp_count_sets.append(count_set_l)
            count_set_l = max_count_set_l#.unsqueeze(0)
            count_set_l_min = min_count_set_l#.unsqueeze(0)
            self.cp_count_sets.append(count_set_l)
            self.cp_count_sets_min.append(count_set_l_min)

        
        # Conformalized Credal Region eihter crisp or ambig. ground truths with scaled max_other_prob as score function
        if self.cp_type == 6 or self.cp_type == 5: 
            # -------------- Experimental: Only necessary for rao entropy
            # D = self.vae_proxy[0].create_embedding_gram_matrix()
            # eigvals = torch.linalg.eigvalsh(D)        
            # if eigvals.min() < 0: 
            #     A = D.cpu().numpy()
            #     eigvals, eigvecs = np.linalg.eigh(A)
                
            #     # Clip negative eigenvalues
            #     eigvals[eigvals < 0] = 0.0
            
            #     # Reconstruct the matrix
            #     D = eigvecs @ np.diag(eigvals) @ eigvecs.T
            # --------------
            
            # cal_scores_b = []
            for b in range(softmax_BlV.shape[0]): 
                max_entropies = torch.zeros(size = (softmax_BlV.shape[1],)) #l entries
                min_entropies = torch.zeros(size = (softmax_BlV.shape[1],)) #l entries
                qhats = qhats[0, :, 0]
                cal_scores_l = []
                for l in range(qhats.shape[0]): 
                    if self.cp_type == 6: 
                        gamma = 0.001
                    else: 
                        gamma = 0.00001
                    probs = softmax_BlV[b, l, :].squeeze() # V
                    max_values = torch.full_like(probs, float('-inf'))  # Initialize with very small values
                    for i in range(4096):
                        max_values[i] = torch.max(torch.cat((probs[:i], probs[i+1:])))  # Max without i-th element

                    # Step 2: Compute the required ratio
                    cal_scores = max_values / (probs + gamma) # V
                    cal_scores_l.append(cal_scores)
                    min_entropy = calc_entropy(cal_scores, tau = qhats[l], maximize = False)
                    max_entropy = calc_entropy(cal_scores, tau = qhats[l], maximize = True)

                    # ------ Experimental -------
                    # Here we can also compute the shannon entropy as in type 4 ---> Rao was just experimental
                    # min_rao_entropy = 0
                    # max_rao_entropy = calc_quadratic_entropy(cal_scores, D, tau = qhats[l], maximize = True)
                    max_entropies[l] = max_entropy
                    min_entropies[l] = min_entropy
                if b == 0: 
                    max_count_set_l = max_entropies.unsqueeze(0)
                    min_count_set_l = min_entropies.unsqueeze(0)
                else: 
                    max_count_set_l = torch.cat((max_count_set_l, max_entropies.unsqueeze(0)), dim = 0)
                    min_count_set_l = torch.cat((min_count_set_l, min_entropies.unsqueeze(0)), dim = 0)
                
                # if si == len(self.patch_nums)-1: 
                #     cal_scores_l = torch.stack(cal_scores_l, dim=0)
                #     cal_scores_b.append(cal_scores_l)
            
            # count_set_l = torch.cat((max_count_set_l.unsqueeze(0), min_count_set_l.unsqueeze(0)), dim = 0)
            count_set_l = max_count_set_l#.unsqueeze(0)
            count_set_l_min = min_count_set_l#.unsqueeze(0)
            self.cp_count_sets.append(count_set_l)
            self.cp_count_sets_min.append(count_set_l_min)
        
        # CPS with scaled logits (shannon entropy as temperature) and cumulative score function
        if self.cp_type == 7: 
            log10 = torch.log10(softmax_BlV)
            torch.nan_to_num_(log10, nan=0)
            emp_entropy = (- torch.sum(softmax_BlV * log10, dim = -1).unsqueeze(-1)) / np.log10(4096)
            # scale the logits:
            logits_BlV_emp = logits_BlV / (2 * emp_entropy + 0.0000001)
            softmax_BlV_emp = torch.softmax(logits_BlV_emp, dim = -1)
            sorted_softmax_BlV, sorted_indices = torch.sort(softmax_BlV_emp, dim=-1, descending=True) # B, L, V
            probs = torch.cumsum(sorted_softmax_BlV, dim=-1)  # shape B, l
            cal_scores = probs
            mask = probs <= qhats
            count_set_l = mask.sum(dim=-1) 
            self.cp_count_sets.append(count_set_l)
        
        # CPS with scaled logits (rao entropy as temperature) and cumulative score function
        if self.cp_type == 8:
            emp_entropy = - torch.sum(softmax_BlV * torch.log10(softmax_BlV), dim = -1).unsqueeze(-1)
            emp_entropy = emp_entropy / np.log10(4096)
            p_flat = softmax_BlV.reshape(-1, V)  # shape (B*l, V)

            # Matrix multiply: (B*l, V) x (V, V) → (B*l, V)
            dp = torch.matmul(p_flat, distances)  # each row is D @ p
            # Elementwise product + sum over last dim: (B*l,) = sum_i p_i * [D @ p]_i
            rao_entropy_flat = torch.sum(p_flat * dp, dim=-1)
            rao_entropy = rao_entropy_flat.reshape(B, L).unsqueeze(-1)            
            logits_BlV_emp = logits_BlV / rao_entropy

            softmax_BlV_emp = torch.softmax(logits_BlV_emp, dim = -1)
            sorted_softmax_BlV, sorted_indices = torch.sort(softmax_BlV_emp, dim=-1, descending=True) # B, L, V
            probs = torch.cumsum(sorted_softmax_BlV, dim=-1)  # shape B, l
            cal_scores = probs
            mask = probs <= qhats            
            #idx_Bl = sample_from_prediction_set(logits_BlV, mask=mask, rng=rng, num_samples=1)[:, :, 0]
            #idx_Bl = idx_Bl.unsqueeze(-1)
            
            count_set_l = mask.sum(dim=-1) 
            self.cp_count_sets.append(count_set_l)
            position_in_sorted = (sorted_indices == idx_Bl).nonzero(as_tuple=True)[-1].unsqueeze(-1).unsqueeze(0)
            scores_idxBl = torch.gather(probs, dim=-1, index=position_in_sorted) 
        
         # CPS Experimental: Scale the Logits based on expected distance per class --> similar approach to entropy weighting 
         # Intuition: The farer apart the most likely classes the smoother the softmax --> higher uncertainty    
        if self.cp_type == 9: 
            p_flat = softmax_BlV.reshape(-1, V)  # shape (B*L, V)
            # Matrix multiply to get expected distances per class: (B*L, V) @ (V, V^T)
            distance_prob_flat = torch.matmul(p_flat, distances)  # shape (B*L, V)
            # Reshape back to (B, L, V)
            distance_prob = distance_prob_flat.reshape(B, L, V)
            
            logits_BlV_scaled = logits_BlV/distance_prob
            softmax_BlV_scaled = torch.softmax(logits_BlV_scaled, dim = -1)
            
            max_values, _ = softmax_BlV_scaled.max(dim=-1, keepdim=True)  # Shape: (B, L, 1)
            # Create a mask for elements that are equal to the max
            mask = softmax_BlV_scaled == max_values  # Shape: (B, L, V)

            # Replace max values with -inf temporarily to get second max
            X_masked = softmax_BlV_scaled.clone()
            X_masked[mask] = float('-inf')
            second_max_values, _ = X_masked.max(dim=-1, keepdim = True)  # Shape: (B, L)
            max_excluding_self = torch.where(mask, second_max_values, max_values)
                                
            gamma = 0.00001
            a = max_excluding_self / (softmax_BlV + gamma)
            mask = a <= qhats
            count_set_l = mask.sum(dim=-1) 
            self.cp_count_sets.append(count_set_l)
        
        # CPS Experimental: Scale the Logits based on max. expected distance from other classes
        # Intuition: The farer apart the most likely classes the smoother the softmax --> higher uncertainty    
        if self.cp_type == 10: 
            p_flat = softmax_BlV.reshape(-1, V)  # shape (B*l, V)
            # # Matrix multiply: (B*l, V) x (V, V) → (B*l, V)
            # dp = torch.matmul(p_flat, distances)  # each row is D @ p
            # # Elementwise product + sum over last dim: (B*l,) = sum_i p_i * [D @ p]_i
            # rao_entropy_flat = torch.sum(p_flat * dp, dim=-1)
            # rao_entropy = rao_entropy_flat.reshape(B, L).unsqueeze(-1)
            
            distance_prob_flat = torch.matmul(p_flat, distances)
            distance_prob = distance_prob_flat.reshape(B, L, V)
            max_dist, _ = distance_prob.max(dim = -1, keepdim = True)
            dist_entropy = torch.sum(softmax_BlV * (max_dist), dim = -1).unsqueeze(-1) 
            logits_BlV_emp = logits_BlV / dist_entropy                
            softmax_BlV_emp = torch.softmax(logits_BlV_emp, dim = -1)
            max_values, _ = softmax_BlV_emp.max(dim=-1, keepdim=True)  # Shape: (B, L, 1)
            mask = softmax_BlV_emp == max_values  # Shape: (B, L, V)

            X_masked = softmax_BlV_emp.clone()
            X_masked[mask] = float('-inf')
            second_max_values, _ = X_masked.max(dim=-1, keepdim = True)  # Shape: (B, L)
            max_excluding_self = torch.where(mask, second_max_values, max_values)
                                
            gamma = 0.00001
            a = max_excluding_self / (softmax_BlV + gamma)
            mask = a <= qhats
            count_set_l = mask.sum(dim=-1) 
            self.cp_count_sets.append(count_set_l)
                        
            # additional analysis on how similar the elements in the prediciton sets actually are
            from scipy.cluster.hierarchy import linkage, fcluster
            from scipy.spatial.distance import squareform
            distances.fill_diagonal_(0)
            condensed_dist = squareform(distances)
            Z = linkage(condensed_dist, method='average')
            clusters = fcluster(Z, t=150, criterion='maxclust')
            clusters = torch.from_numpy(clusters)
            masked_clusters = clusters.unsqueeze(0).unsqueeze(0) * mask  # (1, 1, V) broadcast#
            unique_counts = torch.stack([
            torch.tensor([len(torch.unique(v)) for v in mat])
            for mat in masked_clusters])
            self.cluster_count_sets.append(unique_counts)
            
        # CPS Experimental: Scale scores based on entropy of top-k logits    
        # if self.cp_type == 3: 
        #     _, topk_idxs = distances.topk(1000, dim=-1, largest=False) # V , k           
        #     count_set_l = torch.empty((B, L), device=softmax_BlV.device)
            
        #     for b in range(B): 
        #         for l in range(L):
        #             classes = list()
        #             for v in range(V): 
        #                 mask = torch.zeros(V).bool()
        #                 mask.scatter_(dim=-1, index=topk_idxs[v, :], value=True)
        #                 softmax_bl = softmax_BlV[b, l, :] * mask # shape: 1, 1, V
        #                 #print(softmax_bl.shape)
        #                 #print(torch.count_nonzero(softmax_bl, dim = -1))
                        
        #                 softmax_bl_sorted, sorted_indices = torch.sort(softmax_bl, dim=-1, descending=True) # 1,1, V  
        #                 probs = torch.cumsum(softmax_bl_sorted, dim=-1)  # shape B, l
        #                 #print("probs", probs.shape)
                        
        #                 mask = probs <= qhats[0, l, 0]
        #                 mask.squeeze_()
                        
        #                 if sorted_indices[mask].shape[0] >= 1000: 
        #                     classes = classes + topk_idxs[v, :]
        #                 else: 
        #                     classes = classes + sorted_indices[mask].shape[0]
                         
        #             count_set_l[b, l] =  torch.unique(classes).shape[0]                             
        #     self.cp_count_sets.append(count_set_l)
             
        return count_set_l, idx_Bl.squeeze(-1), None
        
    
    @torch.no_grad()
    def autoregressive_infer_and_uq(
        self, B: int, label_B: Optional[Union[int, torch.LongTensor]],
        g_seed: Optional[int] = None, cfg=4, top_k=0, top_p=0.0,
        more_smooth=False, calc_pred_sets = False, generate_maps = False, input_img_tokens = None, edit_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:   # returns reconstructed image (B, 3, H, W) in [0, 1]
        """
        only used for inference, on autoregressive mode
        :param B: batch size
        :param label_B: imagenet label; if None, randomly sampled
        :param g_seed: random seed
        :param cfg: classifier-free guidance ratio
        :param top_k: top-k sampling
        :param top_p: top-p sampling
        :param more_smooth: smoothing the pred using gumbel softmax; only used in visualization, not used in FID/IS benchmarking
        :return: if returns_vemb: list of embedding h_BChw := vae_embed(idx_Bl), else: list of idx_Bl
        """
        if g_seed is None: rng = None
        else: self.rng.manual_seed(g_seed); rng = self.rng
        
        if label_B is None:
            label_B = torch.multinomial(self.uniform_prob, num_samples=B, replacement=True, generator=rng).reshape(B)
        elif isinstance(label_B, int):
            label_B = torch.full((B,), fill_value=self.num_classes if label_B < 0 else label_B, device=self.lvl_1L.device)
        
        sos = cond_BD = self.class_emb(torch.cat((label_B, torch.full_like(label_B, fill_value=self.num_classes)), dim=0))
        
        #sos = cond_BD = self.class_emb(label_B)
            
        lvl_pos = self.lvl_embed(self.lvl_1L) + self.pos_1LC
        next_token_map = sos.unsqueeze(1).expand(2 * B, self.first_l, -1) + self.pos_start.expand(2 * B, self.first_l, -1) + lvl_pos[:, :self.first_l]        
        cur_L = 0
        f_hat = sos.new_zeros(B, self.Cvae, self.patch_nums[-1], self.patch_nums[-1])
        self.cp_count_sets = []
        self.cluster_count_sets = []
        
        distances = self.vae_proxy[0].create_embedding_distances()
        
        if calc_pred_sets and (self.cp_type == 5 or self.cp_type == 6 or self.cp_type == 4):
            self.cp_count_sets_min = []
        for b in self.blocks: b.attn.kv_caching(True)
        for si, pn in enumerate(self.patch_nums):   # si: i-th segment
            ratio = si / self.num_stages_minus_1
            # last_L = cur_L
            cur_L += pn*pn
            # assert self.attn_bias_for_masking[:, :, last_L:cur_L, :cur_L].sum() == 0, f'AR with {(self.attn_bias_for_masking[:, :, last_L:cur_L, :cur_L] != 0).sum()} / {self.attn_bias_for_masking[:, :, last_L:cur_L, :cur_L].numel()} mask item'
            cond_BD_or_gss = self.shared_ada_lin(cond_BD)
            x = next_token_map
            AdaLNSelfAttn.forward
            for b in self.blocks:
                x = b(x=x, cond_BD=cond_BD_or_gss, attn_bias=None)
            logits_BlV = self.get_logits(x, cond_BD)
                        
            t = cfg * ratio
            logits_BlV = (1+t) * logits_BlV[:B] - t * logits_BlV[B:]
            
            begin = self.begin_ends[si][0]
            end = self.begin_ends[si][1]
            
            # get CPS or CCR
            if calc_pred_sets: 
                count_set_l, idx_Bl, ratio_q_scores = self.get_prediction_sets(si, begin, end, logits_BlV, rng=rng, top_k=top_k, top_p=top_p, num_samples=1, distances=distances)
                last_count_set = count_set_l
            
            if not more_smooth: # this is the default case
                # embedding of the class indices
                h_BChw = self.vae_quant_proxy[0].embedding(idx_Bl)   # B, l, Cvae
            else:   # not used when evaluating FID/IS/Precision/Recall
                gum_t = max(0.27 * (1 - ratio * 0.95), 0.005)   # refer to mask-git
                h_BChw = gumbel_softmax_with_rng(logits_BlV.mul(1 + ratio), tau=gum_t, hard=False, dim=-1, rng=rng) @ self.vae_quant_proxy[0].embedding.weight.unsqueeze(0)
            
            h_BChw = h_BChw.transpose_(1, 2).reshape(B, self.Cvae, pn, pn)
            
            if edit_mask is not None:
                gt_BChw = self.vae_quant_proxy[0].embedding(input_img_tokens[si]).transpose_(1, 2).reshape(B, self.Cvae, pn, pn)
                h_BChw = self.replace_embedding(edit_mask, h_BChw, gt_BChw, pn, pn)
            
            f_hat_old = f_hat
            f_hat, next_token_map = self.vae_quant_proxy[0].get_next_autoregressive_input(si, len(self.patch_nums), f_hat, h_BChw)
            if si != self.num_stages_minus_1:   # prepare for next stage
                next_token_map = next_token_map.view(B, self.Cvae, -1).transpose(1, 2)
                next_token_map = self.word_embed(next_token_map) + lvl_pos[:, cur_L:cur_L + self.patch_nums[si+1] ** 2]
                next_token_map = next_token_map.repeat(2, 1, 1)   # double the batch sizes due to CFG
        
        for b in self.blocks: b.attn.kv_caching(False)
        
        if edit_mask is not None and calc_pred_sets: 
            em = edit_mask.view(-1, edit_mask.shape[0]*edit_mask.shape[1]) > 0
            last_count_set[em] = 0 
            
        if generate_maps and calc_pred_sets: 
            map_mean, map_var = self.generate_variance_maps(count_set_l=last_count_set, idx_l=idx_Bl, logits_BlV=logits_BlV, pn=pn, si=si, f_hat=f_hat_old)
            #other_map, other_map_var = self.generate_variance_maps(ratio_q_scores, idx_l=idx_Bl, logits_BlV=logits_BlV, pn=pn, si=si, f_hat=f_hat_old)
            other_map, other_map_var = None, None
        else: 
            map_mean, map_var, other_map, other_map_var =None, None, None, None
        
        if calc_pred_sets and (self.cp_type == 5 or self.cp_type == 6 or self.cp_type == 4):
           self.cp_count_sets = [self.cp_count_sets, self.cp_count_sets_min]
        
        return self.vae_proxy[0].fhat_to_img(f_hat).add_(1).mul_(0.5), other_map_var, map_var, self.cp_count_sets, last_count_set, self.cluster_count_sets
    
    
    
    def generate_variance_maps(self, count_set_l, idx_l, logits_BlV, pn, si, f_hat):
        means = list()
        vars = list()
        uq_mask = self.select_high_uncertainty_embeddings(count_set_l, 4)[0, :].squeeze() #shape(L) of first element
        # get index i of High UQ postition
        indcs_uq = uq_mask.nonzero(as_tuple=True)[0]
        for idx_uq in indcs_uq.numpy():         
            logits_V = logits_BlV[0, idx_uq, :]
            samples_uq = sample_from_high_uq(logits_V=logits_V.squeeze())#, mask=mask_high_uq.squeeze())
            B = samples_uq.shape[0]
            f_hat = f_hat[0, :]
            f_hat = f_hat.unsqueeze(0).expand(B, -1, -1, -1).clone()
            new_idx_Bl = generate_high_uq_samples(samples_uq, idx_Bl=idx_l[0, :], index_uq=idx_uq)
            
            h_BChw = self.vae_quant_proxy[0].embedding(new_idx_Bl)
            h_BChw = h_BChw.transpose_(1, 2).reshape(B, self.Cvae, pn, pn)
            f_hat, next_token_map = self.vae_quant_proxy[0].get_next_autoregressive_input(si, len(self.patch_nums), f_hat, h_BChw)
            
            imgs = self.vae_proxy[0].fhat_to_img(f_hat).add_(1).mul_(0.5) # B, C, h, w, 
            
            mean_L = imgs.mean(dim=0)  # Shape: (L,)
            var_L = imgs.var(dim=0, unbiased=False) 
            
            mean_L = mean_L.mean(dim = 0)
            var_L = var_L.mean(dim = 0)
            
            means.append(mean_L)
            vars.append(var_L)
        
        return means, vars
    
    def replace_embedding(self, edit_mask: torch.Tensor, h_BChw: torch.Tensor, gt_BChw: torch.Tensor, ph: int, pw: int) -> torch.Tensor:
        B = h_BChw.shape[0]
        h, w = edit_mask.shape[-2:]
        if edit_mask.ndim == 2:
            edit_mask = edit_mask.unsqueeze(0).expand(B, h, w)
        
        force_gt_B1hw = F.interpolate(edit_mask.unsqueeze(1).to(dtype=torch.float, device=gt_BChw.device), size=(ph, pw), mode='bilinear', align_corners=False).gt(0.5).int()
        if ph * pw <= 3: force_gt_B1hw.fill_(1)
        return gt_BChw * force_gt_B1hw + h_BChw * (1 - force_gt_B1hw)
    
    
    def select_high_uncertainty_embeddings(self, uncertainty_map, top_k=5):
        B, L = uncertainty_map.shape  # (B, 256)
        assert L == 256, "Expected 256 latent embeddings per batch."
        topk_indices = torch.topk(uncertainty_map, k=top_k, dim=1)[1]  # (B, top_k)
        mask = torch.zeros(B, 256, dtype=torch.bool, device=uncertainty_map.device)
        mask[torch.arange(B).unsqueeze(1), topk_indices] = True  # Vectorized assignment
        #mask = mask.view(-1, 256)
        return mask  
    
    
    def forward(self, label_B: torch.LongTensor, x_BLCv_wo_first_l: torch.Tensor) -> torch.Tensor:  # returns logits_BLV
        """
        :param label_B: label_B
        :param x_BLCv_wo_first_l: teacher forcing input (B, self.L-self.first_l, self.Cvae)
        :return: logits BLV, V is vocab_size
        """
        bg, ed = self.begin_ends[self.prog_si] if self.prog_si >= 0 else (0, self.L)
        # batch size
        B = x_BLCv_wo_first_l.shape[0]
        with torch.cuda.amp.autocast(enabled=False):
            # for classifier free guidance, randomly choose from num_classes or concrete label B 
            label_B = torch.where(torch.rand(B, device=label_B.device) < self.cond_drop_rate, self.num_classes, label_B)
            sos = cond_BD = self.class_emb(label_B)
            sos = sos.unsqueeze(1).expand(B, self.first_l, -1) + self.pos_start.expand(B, self.first_l, -1)
            
            if self.prog_si == 0: x_BLC = sos
            # contains previous tokenmaps --> teacher forcing
            else: x_BLC = torch.cat((sos, self.word_embed(x_BLCv_wo_first_l.float())), dim=1)
            x_BLC += self.lvl_embed(self.lvl_1L[:, :ed].expand(B, -1)) + self.pos_1LC[:, :ed] # lvl: BLC;  pos: 1LC
        
        # mask for sucessive token maps 
        attn_bias = self.attn_bias_for_masking[:, :, :ed, :ed]
        cond_BD_or_gss = self.shared_ada_lin(cond_BD)
        
        # hack: get the dtype if mixed precision is used
        temp = x_BLC.new_ones(8, 8)
        main_type = torch.matmul(temp, temp).dtype
        
        x_BLC = x_BLC.to(dtype=main_type)
        cond_BD_or_gss = cond_BD_or_gss.to(dtype=main_type)
        attn_bias = attn_bias.to(dtype=main_type)
        
        AdaLNSelfAttn.forward
        for i, b in enumerate(self.blocks):
            # with mask for sucessive token maps 
            x_BLC = b(x=x_BLC, cond_BD=cond_BD_or_gss, attn_bias=attn_bias)
        x_BLC = self.get_logits(x_BLC.float(), cond_BD)
        
        if self.prog_si == 0:
            if isinstance(self.word_embed, nn.Linear):
                x_BLC[0, 0, 0] += self.word_embed.weight[0, 0] * 0 + self.word_embed.bias[0] * 0
            else:
                s = 0
                for p in self.word_embed.parameters():
                    if p.requires_grad:
                        s += p.view(-1)[0] * 0
                x_BLC[0, 0, 0] += s
        return x_BLC    # logits BLV, V is vocab_size
    
    def init_weights(self, init_adaln=0.5, init_adaln_gamma=1e-5, init_head=0.02, init_std=0.02, conv_std_or_gain=0.02):
        if init_std < 0: init_std = (1 / self.C / 3) ** 0.5     # init_std < 0: automated
        
        print(f'[init_weights] {type(self).__name__} with {init_std=:g}')
        for m in self.modules():
            with_weight = hasattr(m, 'weight') and m.weight is not None
            with_bias = hasattr(m, 'bias') and m.bias is not None
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight.data, std=init_std)
                if with_bias: m.bias.data.zero_()
            elif isinstance(m, nn.Embedding):
                nn.init.trunc_normal_(m.weight.data, std=init_std)
                if m.padding_idx is not None: m.weight.data[m.padding_idx].zero_()
            elif isinstance(m, (nn.LayerNorm, nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.SyncBatchNorm, nn.GroupNorm, nn.InstanceNorm1d, nn.InstanceNorm2d, nn.InstanceNorm3d)):
                if with_weight: m.weight.data.fill_(1.)
                if with_bias: m.bias.data.zero_()
            # conv: VAR has no conv, only VQVAE has conv
            elif isinstance(m, (nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.ConvTranspose1d, nn.ConvTranspose2d, nn.ConvTranspose3d)):
                if conv_std_or_gain > 0: nn.init.trunc_normal_(m.weight.data, std=conv_std_or_gain)
                else: nn.init.xavier_normal_(m.weight.data, gain=-conv_std_or_gain)
                if with_bias: m.bias.data.zero_()
        
        if init_head >= 0:
            if isinstance(self.head, nn.Linear):
                self.head.weight.data.mul_(init_head)
                self.head.bias.data.zero_()
            elif isinstance(self.head, nn.Sequential):
                self.head[-1].weight.data.mul_(init_head)
                self.head[-1].bias.data.zero_()
        
        if isinstance(self.head_nm, AdaLNBeforeHead):
            self.head_nm.ada_lin[-1].weight.data.mul_(init_adaln)
            if hasattr(self.head_nm.ada_lin[-1], 'bias') and self.head_nm.ada_lin[-1].bias is not None:
                self.head_nm.ada_lin[-1].bias.data.zero_()
        
        depth = len(self.blocks)
        for block_idx, sab in enumerate(self.blocks):
            sab: AdaLNSelfAttn
            sab.attn.proj.weight.data.div_(math.sqrt(2 * depth))
            sab.ffn.fc2.weight.data.div_(math.sqrt(2 * depth))
            if hasattr(sab.ffn, 'fcg') and sab.ffn.fcg is not None:
                nn.init.ones_(sab.ffn.fcg.bias)
                nn.init.trunc_normal_(sab.ffn.fcg.weight, std=1e-5)
            if hasattr(sab, 'ada_lin'):
                sab.ada_lin[-1].weight.data[2*self.C:].mul_(init_adaln)
                sab.ada_lin[-1].weight.data[:2*self.C].mul_(init_adaln_gamma)
                if hasattr(sab.ada_lin[-1], 'bias') and sab.ada_lin[-1].bias is not None:
                    sab.ada_lin[-1].bias.data.zero_()
            elif hasattr(sab, 'ada_gss'):
                sab.ada_gss.data[:, :, 2:].mul_(init_adaln)
                sab.ada_gss.data[:, :, :2].mul_(init_adaln_gamma)
    
    def extra_repr(self):
        return f'drop_path_rate={self.drop_path_rate:g}'


class VARHF(VAR, PyTorchModelHubMixin):
            # repo_url="https://github.com/FoundationVision/VAR",
            # tags=["image-generation"]):
    def __init__(
        self,
        vae_kwargs,
        num_classes=1000, depth=16, embed_dim=1024, num_heads=16, mlp_ratio=4., drop_rate=0., attn_drop_rate=0., drop_path_rate=0.,
        norm_eps=1e-6, shared_aln=False, cond_drop_rate=0.1,
        attn_l2_norm=False,
        patch_nums=(1, 2, 3, 4, 5, 6, 8, 10, 13, 16),   # 10 steps by default
        flash_if_available=True, fused_if_available=True,
    ):
        vae_local = VQVAE(**vae_kwargs)
        super().__init__(
            vae_local=vae_local,
            num_classes=num_classes, depth=depth, embed_dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, drop_rate=drop_rate, attn_drop_rate=attn_drop_rate, drop_path_rate=drop_path_rate,
            norm_eps=norm_eps, shared_aln=shared_aln, cond_drop_rate=cond_drop_rate,
            attn_l2_norm=attn_l2_norm,
            patch_nums=patch_nums,
            flash_if_available=flash_if_available, fused_if_available=fused_if_available,
        )
