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
from utils.data_cal import build_cal_dataset
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


def load_model(num_classes, depth, args: arg_util.Args): 
    # build models
    from torch.nn.parallel import DistributedDataParallel as DDP
    from models import build_vae_var

    
    #hf_home = 'https://huggingface.co/FoundationVision/var/resolve/main'
    vae_ckpt, var_ckpt = 'vae_ch160v4096z32.pth', f'var_d{depth}.pth'
    #if not osp.exists(vae_ckpt): os.system(f'wget {hf_home}/{vae_ckpt}')
    #if not osp.exists(var_ckpt): os.system(f'wget {hf_home}/{var_ckpt}')
    

    # build vae, var
    patch_nums = (1, 2, 3, 4, 5, 6, 8, 10, 13, 16)
    #patch_nums = (1, 2, 3, 4, 6, 9, 13, 18, 24, 32)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if 'vae' not in globals() or 'var' not in globals():
        vae, var = build_vae_var(
            V=4096, Cvae=32, ch=160, share_quant_resi=4,    # hard-coded VQVAE hyperparameters
            device=device, patch_nums=patch_nums,
            num_classes=1000, depth=depth, shared_aln= False,

        )
    # load checkpoints
    vae.load_state_dict(torch.load(vae_ckpt, map_location='cpu'), strict=True)
    var.load_state_dict(torch.load(var_ckpt, map_location='cpu'), strict=True)
    vae.eval(), var.eval()
    for p in vae.parameters(): p.requires_grad_(False)
    for p in var.parameters(): p.requires_grad_(False)
    print(f'prepare finished.')

    return var, vae


def load_cal_data(args: arg_util.Args): 
    #auto_resume_info, start_ep, start_it, trainer_state, args_state = auto_resume(args, 'ar-ckpt*.pth')
    
    data_path = os.path.join(os.getcwd(), 'data')
        
    print(f'[build PT data] ...\n')
    # caution: modified data path to use the hardcoded one from above
    num_classes, dataset_val = build_cal_dataset(
        data_path, final_reso=args.data_load_reso,
    )

    types = str((type(dataset_val).__name__))
    
    print(f'[dataloader multi processing] ...', end='', flush=True)
    stt = time.time()
    # noinspection PyArgumentList
    print(f'     [dataloader multi processing](*) finished! ({time.time()-stt:.2f}s)')
    print(f'[dataloader] gbs={args.glb_batch_size}, lbs={args.batch_size}, types(tr, va)={types}')
    
    return num_classes, dataset_val

def build_everything(args: arg_util.Args, depth): 
    num_classes, dataset_val = load_cal_data(args)
    
    ld_val = DataLoader(
            dataset_val, num_workers=0, pin_memory=True,
            batch_size= args.bs_calib, #round(args.batch_size*1.5),
            shuffle=True, drop_last=False
        )
    del dataset_val
    
    var_wo_ddp, vae_local = load_model(num_classes, depth, args)
        
    calibrator = VARCalibrator(device=args.device, patch_nums=args.patch_nums, resos=args.resos,
        vae_local=vae_local, var_wo_ddp=var_wo_ddp, num_classes=num_classes)
    del vae_local, var_wo_ddp
    
    return calibrator, ld_val
    
    
def main_calibrating():
    torch.cuda.empty_cache()
    MODEL_DEPTH = 16
    args: arg_util.Args = arg_util.init_dist_and_get_args()
    calibrator, ld_val = build_everything(args, MODEL_DEPTH) 
    calibrator.teacher_enforced_cp(ld_val=ld_val, cp_type=args.cp_type, alpha=args.alpha, autoregressive=args.autoregressive)
        
    vae_ckpt = 'vae_ch160v4096z32_calib_test.pth'
    var_ckpt = f'var_d{MODEL_DEPTH}_calib_test.pth'

    torch.save(calibrator.vae_local.state_dict(), vae_ckpt)
    torch.save(calibrator.var_wo_ddp.state_dict(), var_ckpt)
    print('DONE! Calibrated model saved!')

if __name__ == '__main__':
    main_calibrating()