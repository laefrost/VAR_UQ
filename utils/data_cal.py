import os.path as osp

import PIL.Image as PImage
from torchvision.datasets.folder import DatasetFolder, IMG_EXTENSIONS
from torchvision.transforms import InterpolationMode, transforms
import torch
import numpy as np 
from utils.data import normalize_01_into_pm1, pil_loader, print_aug

def build_cal_dataset(data_path: str, final_reso: int, mid_reso=1.125,): 
    # build augmentations
    mid_reso = round(mid_reso * final_reso)  # first resize to mid_reso, then crop to final_reso
    val_aug = [
        transforms.Resize(mid_reso, interpolation=InterpolationMode.LANCZOS), # transforms.Resize: resize the shorter edge to mid_reso
        transforms.CenterCrop((final_reso, final_reso)),
        transforms.ToTensor(), normalize_01_into_pm1,
    ]
    val_aug = transforms.Compose(val_aug)
    
    # build dataset
    val_set = DatasetFolder(root=osp.join(data_path, 'val'), loader=pil_loader, extensions=IMG_EXTENSIONS, transform=val_aug)
    #val_set_subset = torch.utils.data.Subset(val_set, np.random.choice(len(val_set), 10, replace=False))
    num_classes = 1000
    print(f'[Dataset] {len(val_set)=}, {num_classes=}')
    print_aug(val_aug, '[val]')
    
    return num_classes, val_set 