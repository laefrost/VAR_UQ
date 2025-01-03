import os
import os.path as osp
import torch, torchvision
import random
import numpy as np
import PIL.Image as PImage, PIL.ImageDraw as PImageDraw
setattr(torch.nn.Linear, 'reset_parameters', lambda self: None)     # disable default parameter init for faster speed
setattr(torch.nn.LayerNorm, 'reset_parameters', lambda self: None)  # disable default parameter init for faster speed
from models import VQVAE, build_vae_var
import dist
from utils import arg_util, misc
from utils.data import build_cal_dataset
from utils.data_sampler import DistInfiniteBatchSampler, EvalDistributedSampler
from utils.misc import auto_resume
from calibrator import VARCalibrator

import gc
import os
import shutil
import sys
import time
import warnings
from functools import partial

import torch
from torch.utils.data import DataLoader


def load_model(num_classes, args: arg_util.Args): 
        # build models
    from torch.nn.parallel import DistributedDataParallel as DDP
    from models import VAR, VQVAE, build_vae_var
    from trainer import VARTrainer
    from utils.amp_sc import AmpOptimizer
    from utils.lr_control import filter_params
        
    vae_local, var_wo_ddp = build_vae_var(
        V=4096, Cvae=32, ch=160, share_quant_resi=4,        # hard-coded VQVAE hyperparameters
        device=dist.get_device(), patch_nums=args.patch_nums,
        num_classes=num_classes, depth=args.depth, shared_aln=args.saln, attn_l2_norm=args.anorm,
        flash_if_available=args.fuse, fused_if_available=args.fuse,
        init_adaln=args.aln, init_adaln_gamma=args.alng, init_head=args.hd, init_std=args.ini,
    )
    
    vae_ckpt = 'vae_ch160v4096z32.pth'
    if dist.is_local_master():
        if not os.path.exists(vae_ckpt):
            os.system(f'wget https://huggingface.co/FoundationVision/var/resolve/main/{vae_ckpt}')
    dist.barrier()
    vae_local.load_state_dict(torch.load(vae_ckpt, map_location='cpu'), strict=True)
    
    vae_local: VQVAE = args.compile_model(vae_local, args.vfast)
    var_wo_ddp: VAR = args.compile_model(var_wo_ddp, args.tfast)
    
    #print(f'[INIT] VAR model = {var_wo_ddp}\n\n')
    count_p = lambda m: f'{sum(p.numel() for p in m.parameters())/1e6:.2f}'
    print(f'[INIT][#para] ' + ', '.join([f'{k}={count_p(m)}' for k, m in (('VAE', vae_local), ('VAE.enc', vae_local.encoder), ('VAE.dec', vae_local.decoder), ('VAE.quant', vae_local.quantize))]))
    print(f'[INIT][#para] ' + ', '.join([f'{k}={count_p(m)}' for k, m in (('VAR', var_wo_ddp),)]) + '\n\n')
    
    return var_wo_ddp, vae_local


def load_cal_data(args: arg_util.Args): 
    #auto_resume_info, start_ep, start_it, trainer_state, args_state = auto_resume(args, 'ar-ckpt*.pth')
    
    data_path = os.path.join(os.getcwd(), 'data')
    
    #### load it wo args 
    # TODO modify this
    if args.pn == '256':
        args.pn = '1_2_3_4_5_6_8_10_13_16'
    elif args.pn == '512':
        args.pn = '1_2_3_4_6_9_13_18_24_32'
    elif args.pn == '1024':
        args.pn = '1_2_3_4_5_7_9_12_16_21_27_36_48_64'
    # patch_size = 16
    # patch_nums = tuple(map(int, pn.replace('-', '_').split('_')))
    # resos = tuple(pn * patch_size for pn in patch_nums)
    # data_load_reso = max(resos)
        
    print(f'[build PT data] ...\n')
    # caution: modified data path to use the hardcoded one from above
    num_classes, dataset_val = build_cal_dataset(
        data_path, final_reso=args.data_load_reso,
    )
    # num_classes, dataset_val = build_cal_dataset(
    #     data_path, final_reso=data_load_reso
    # )
    types = str((type(dataset_val).__name__))

    #TODO: TBD
    # bs = 768
    # ac = 1
    # batch_size = round(bs / ac / dist.get_world_size() / 8) * 8
    # glb_batch_size = batch_size * dist.get_world_size() 
    
    # check whether it is okay to use this as calibration data set?
    # ld_val = DataLoader(
    #     dataset_val, num_workers=0, pin_memory=True,
    #     batch_size=round(batch_size*1.5), sampler=EvalDistributedSampler(dataset_val, num_replicas=dist.get_world_size(), rank=dist.get_rank()),
    #     shuffle=False, drop_last=False,
    # )
    # del dataset_val
    
    ld_val = DataLoader(
        dataset_val, num_workers=0, pin_memory=True,
        batch_size=round(args.batch_size*1.5), sampler=EvalDistributedSampler(dataset_val, num_replicas=dist.get_world_size(), rank=dist.get_rank()),
        shuffle=False, drop_last=False,
    )
    del dataset_val
    
    #[print(line) for line in auto_resume_info]
    print(f'[dataloader multi processing] ...', end='', flush=True)
    stt = time.time()
    # noinspection PyArgumentList
    print(f'     [dataloader multi processing](*) finished! ({time.time()-stt:.2f}s)')
    print(f'[dataloader] gbs={args.glb_batch_size}, lbs={args.batch_size}, types(tr, va)={types}')
    
    return num_classes, ld_val

def build_everything(args: arg_util.Args): 
    num_classes, ld_val = load_cal_data(args)
    var_wo_ddp, vae_local = load_model(num_classes, args)
    
    calibrator: VARCalibrator
    
    # create calibrator
    calibrator = VARCalibrator(device=args.device, patch_nums=args.patch_nums, resos=args.resos,
        vae_local=vae_local, var_wo_ddp=var_wo_ddp)
    del vae_local, var_wo_ddp
    
    return calibrator, ld_val
    
    
def main_calibrating():
    args: arg_util.Args = arg_util.init_dist_and_get_args()
    print('Hi')
    calibrator, ld_val = build_everything(args) 
    print(len(ld_val))
    #if args.calibration_mode == 'teacher_forced':
    if True: 
        #pass
        calibrator.teacher_enforced_cp(ld_val)
    else: 
        # TODO: include "dynamic cp"
        # sth. sth. similar to inference method
        pass
    

if __name__ == '__main__':
    main_calibrating()