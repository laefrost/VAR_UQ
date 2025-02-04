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
from utils.helper_calib import stochastic_simplex_sampling
import scipy



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
        flash_if_available=True, fused_if_available=True, cp_type = "global_vanilla_cp", alpha = 0.1, qhats = None
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
        self.cp_type = cp_type
        self.register_buffer('qhats', None if qhats is None else torch.FloatTensor(qhats), persistent=True)
        self.cp_count_sets = []
        self.alpha = alpha
        self.cfg = 1.5 # TODO: Maybe delete this
    
    def get_logits(self, h_or_h_and_residual: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]], cond_BD: Optional[torch.Tensor]):
        if not isinstance(h_or_h_and_residual, torch.Tensor):
            h, resi = h_or_h_and_residual   # fused_add_norm must be used
            h = resi + self.blocks[-1].drop_path(h)
        else:                               # fused_add_norm is not used
            h = h_or_h_and_residual
        return self.head(self.head_nm(h.float(), cond_BD).float()).float()
    
    #@torch.no_grad()
    def autoregressive_infer_cfg(
        self, B: int, label_B: Optional[Union[int, torch.LongTensor]],
        g_seed: Optional[int] = None, cfg=1.5, top_k=0, top_p=0.0,
        more_smooth=False, calc_pred_sets = False, 
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
            
        lvl_pos = self.lvl_embed(self.lvl_1L) + self.pos_1LC
        next_token_map = sos.unsqueeze(1).expand(2 * B, self.first_l, -1) + self.pos_start.expand(2 * B, self.first_l, -1) + lvl_pos[:, :self.first_l]        
        cur_L = 0
        f_hat = sos.new_zeros(B, self.Cvae, self.patch_nums[-1], self.patch_nums[-1])
        
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
            
            # TODO: 
            logits_BlV = self.get_logits(x, cond_BD)
            
            #  TODO: Maybe adapt this            
            t = self.cfg * ratio
            logits_BlV = (1+t) * logits_BlV[:B] - t * logits_BlV[B:]
            
            begin = self.begin_ends[si][0]
            end = self.begin_ends[si][1]
            
            # if calc_pred_sets: 
            #     # change this
            #     qhats = self.qhats[begin:end] # shape begin:end
            #     qhats = qhats.view(1, qhats.shape[0], 1)
            #     #qhat = self.qhats[si]
            #     #print(f"1-qhat: {len(qhats)}")
            #     softmax_BlV = logits_BlV.softmax(dim=-1) #shape[B, begin:end, V]
            #     #print(softmax_BlV.shape)
            #     mask = softmax_BlV >= (1-qhats)
                
            #     # max_values, _ = softmax_BlV.max(dim=2)
            #     #print("Max values along the last dimension (V):", max_values)
                
            #     #prediction_set_l = softmax_BlV[mask]
            #     count_set_l = mask.sum(dim=2)
            #     #print(prediction_set_l.shape)            
            #     #self.prediction_sets.append(prediction_set_l)
            #     self.cp_count_sets.append(count_set_l)
            if si == len(self.patch_nums)-1:
                print(si)
                # get sampled lambdas
                qhats = self.qhats[begin:end] # shape begin:end
                qhats = qhats.view(1, qhats.shape[0], 1)
                lambdas = stochastic_simplex_sampling(logits_BlV.shape[-1], 100) # shape [100, V]
                
                print(lambdas.shape)
                softmax_BlV = logits_BlV.softmax(dim=-1) #shape[B, begin:end, V]
                
                # TODO: Compute ws distance per element in lambdas and sofmax blv --> output 100, L, V 
                # for each b, compute ws distance between all lambdas and the resepctive L, V  
                # mask = ws >= 1- qhats
                B, L, V = softmax_BlV.shape
                for b in range(B): 
                    print(b)
                    for l in range(L): 
                        probs = softmax_BlV[b, l].cpu().numpy() # shape: l, v
                        for la in lambdas: 
                            ws = scipy.stats.wasserstein_distance(range(V), range(V), la , probs) 
                        
                #mask = softmax_BlV >= (1-qhats)
                
                # max_values, _ = softmax_BlV.max(dim=2)
                # print("Max values along the last dimension (V):", max_values)
                
                #prediction_set_l = softmax_BlV[mask]
                #count_set_l = mask.sum(dim=2)
                #print(prediction_set_l.shape)            
                #self.prediction_sets.append(prediction_set_l)
                #self.cp_count_sets.append(count_set_l)
            
            idx_Bl = sample_with_top_k_top_p_(logits_BlV, rng=rng, top_k=top_k, top_p=top_p, num_samples=1)[:, :, 0]
            if not more_smooth: # this is the default case
                # embedding of the class indices
                h_BChw = self.vae_quant_proxy[0].embedding(idx_Bl)   # B, l, Cvae
            else:   # not used when evaluating FID/IS/Precision/Recall
                gum_t = max(0.27 * (1 - ratio * 0.95), 0.005)   # refer to mask-git
                h_BChw = gumbel_softmax_with_rng(logits_BlV.mul(1 + ratio), tau=gum_t, hard=False, dim=-1, rng=rng) @ self.vae_quant_proxy[0].embedding.weight.unsqueeze(0)
            
            #print('idx_Bl')
            #print(idx_Bl.shape)
            #print('h_BChw')
            #print(h_BChw.shape)
            h_BChw = h_BChw.transpose_(1, 2).reshape(B, self.Cvae, pn, pn)
            #print(h_BChw.shape)
            f_hat, next_token_map = self.vae_quant_proxy[0].get_next_autoregressive_input(si, len(self.patch_nums), f_hat, h_BChw)
            #print(f_hat.shape)
            #print(next_token_map.shape)
            if si != self.num_stages_minus_1:   # prepare for next stage
                next_token_map = next_token_map.view(B, self.Cvae, -1).transpose(1, 2)
                next_token_map = self.word_embed(next_token_map) + lvl_pos[:, cur_L:cur_L + self.patch_nums[si+1] ** 2]
                next_token_map = next_token_map.repeat(2, 1, 1)   # double the batch sizes due to CFG
        
        for b in self.blocks: b.attn.kv_caching(False)
        #print("FHAT2")
        #print(f_hat.shape)
        # torch.Size([5, 32, 16, 16]) --> input to fhat_to_image
        #print("Count set")
        #print(count_set_l.shape)
        #uq_mask = self.select_high_uncertainty_embeddings(count_set_l, 5)
        #print(uq_mask)
        grads = None 
        #grads = self.compute_pixelwise_jacobian(f_hat_old, idx_Bl)
        return self.vae_proxy[0].fhat_to_img(f_hat).add_(1).mul_(0.5), grads   # de-normalize, from [-1, 1] to [0, 1]
    
    def jacobian_model(self, h_BChw, f_hat):
        B = h_BChw.shape[0]
        h_BChw = h_BChw.transpose(1, 2).reshape(B, self.Cvae, 16, 16) # shape : (B, 32, 16, 16])
        #h_BChw = h_BChw.clone().detach().requires_grad_(True)
        print("Before quant_resi: ", h_BChw.requires_grad, h_BChw.grad_fn)
        h = self.vae_quant_proxy[0].quant_resi[int(9/(10-1))](h_BChw)
        print("After quant_resi: ", h.requires_grad, h.grad_fn)
        f_hat = f_hat + h
        output = self.vae_proxy[0].fhat_to_img(f_hat)  # Output shape: (B, 1, 256, 256)
        
        # ----------------------------------------------------------------
        assert h_BChw.requires_grad, "h_BChw is not tracking gradients!"
        assert f_hat.requires_grad, "f_hat does not require gradients!"
        assert output.requires_grad, "Output does not require gradients!"
        return output 
        

    def compute_pixelwise_jacobian(self, f_hat, idx_Bl):
        print("Compute jacobian")
        B, L = idx_Bl.shape  # Latent space (B, 256) 
        idx_Bl = idx_Bl.clone().detach()
        
        h_BChw = self.vae_quant_proxy[0].embedding(idx_Bl) # shape: B, 256, 32
        h_BChw = h_BChw.requires_grad_(True)
        
        def model_wrapper(h):
            return self.jacobian_model(h, f_hat)
    
        jacobian_matrix = jacobian(model_wrapper, h_BChw, vectorize=True)
        jacobian_matrix = jacobian_matrix.detach()
        return jacobian_matrix
        
        # expanded_mask = mask.expand(B, 32, 16, 16)  # Shape: (B, 32, 16, 16)
        # print(mask.shape)
        # print(expanded_mask.shape)
        # print(expanded_mask)

        # # Use masked_select to extract only the selected embeddings
        # h_selected = h_BChw[expanded_mask]  # Flattened tensor
        # print("h selected")
        # print(h_selected.shape)
         
        # Initialize Jacobian storage
        # jacobian = torch.zeros(B, self.Cvae, 16, 16, 256, 256).to(h_BChw.device)  # (B, 256, 256, 256)
        # print("f hat ", f_hat.requires_grad, f_hat.grad_fn)
        # print("Output ", output.requires_grad, output.grad_fn)
        # num_samples = 10  # Reduce computation
        # sampled_pixels = torch.randint(0, 256, (num_samples, 2))  # Random (x, y) pairs

        # for idx in range(num_samples):
        #     print(idx)
        #     x, y = sampled_pixels[idx]

        #     grad_outputs = torch.zeros_like(output, requires_grad=True).clone()
        #     grad_outputs[:, :, x, y] = 1  # Isolate gradient for one output pixel
            
        #     # Compute gradient of f(x,y) w.r.t. latent space (16x16)
        #     grads = torch.autograd.grad(outputs=output, inputs=h_BChw,
        #                                 grad_outputs=grad_outputs,
        #                                 create_graph=True, retain_graph=True)[0]  # Shape: (B, 16, 16)
        #     print(grads.shape)

        #     jacobian[:, :, :, :, x, y] = grads  # Store the gradients

        #return jacobian  # Shape: (B, 16, 16, 256, 256)
    
    
    def select_high_uncertainty_embeddings(self, uncertainty_map, top_k=5):
        """
        Selects the top-k uncertain latent embeddings and returns a mask of shape [B, 1, 16, 16].

        Args:
            uncertainty_map: Tensor of shape [B, 256] containing uncertainty values.
            top_k: Number of highest-uncertainty embeddings to select.

        Returns:
            mask: Binary mask of shape [B, 1, 16, 16] where selected positions are True.
        """
        B, L = uncertainty_map.shape  # (B, 256)
        assert L == 256, "Expected 256 latent embeddings per batch."

        # Get the top-k uncertainty indices for each batch
        topk_indices = torch.topk(uncertainty_map, k=top_k, dim=1)[1]  # (B, top_k)

        # Compute (h, w) positions for the 16x16 spatial grid
        h = topk_indices // 16  # Compute row indices (B, top_k)
        w = topk_indices % 16   # Compute column indices (B, top_k)

        # Create an empty mask (B, 1, 16, 16)
        mask = torch.zeros(B, 1, 16, 16, dtype=torch.bool, device=uncertainty_map.device)

        # Use advanced indexing to set selected positions to True
        mask[torch.arange(B).unsqueeze(1), 0, h, w] = True  # Vectorized assignment

        return mask  # Shape: (B, 1, 16, 16)

    
    
    
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
