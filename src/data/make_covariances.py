import os
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent)) #
from src.utils.general import get_spd, get_center_slices
import json
from pathlib import Path
import numpy as np
from skimage.transform import resize
import torch
import timm
from src.utils.general import get_spd, get_center_slices
from time import sleep

#Initialize model
model = timm.create_model(
    'vgg19.tv_in1k',
    pretrained=True,
    features_only=True,
)
model = model.eval()

# model = timm.create_model(
#     'convnext_small.fb_in22k_ft_in1k_384',
#     pretrained=True,
#     features_only=True,
# )
# model = model.eval()

# File paths
PATH_DATA = "/Datasets/PICAI_64x64_patches/"
info = json.load(open(PATH_DATA + '8x64x64-CIspheres.json'))
PATH_VOLS = PATH_DATA + "patches/"
PATH_SAVE = PATH_DATA + "covariances/"

# Process data
i = 0
total_files = len(info)

for key in info:
    vol = np.load(PATH_VOLS + key + '.npy')
    vol1, vol2, vol3 = vol[0, :, :, :], vol[1, :, :, :], vol[2, :, :, :]

    # Standardize the volumes not necessary because they where previously standardized by olmos
    # vol1 = (vol1 - vol1.mean()) / vol1.std()
    # vol2 = (vol2 - vol2.mean()) / vol2.std()
    # vol3 = (vol3 - vol3.mean()) / vol3.std()

    vol_list = [vol1, vol2, vol3]
    s1_list, s2_list, s3_list = [], [], []

    for vol in vol_list:
        s1, s2, s3 = get_center_slices(vol)
        s1 = resize(s1, (64, 64), anti_aliasing=True)
        s2 = resize(s2, (64, 64), anti_aliasing=True)
        s3 = resize(s3, (64, 64), anti_aliasing=True)

        # Repeat slices along the channel axis to create RGB-like inputs
        s1 = np.repeat(s1[:, :, np.newaxis], 3, axis=2)
        s2 = np.repeat(s2[:, :, np.newaxis], 3, axis=2)
        s3 = np.repeat(s3[:, :, np.newaxis], 3, axis=2)
        s1_list.append(s1)
        s2_list.append(s2)
        s3_list.append(s3)

    # Stack slices and convert to tensors
    s1_l = torch.tensor(np.stack(s1_list, axis=0), dtype=torch.float32).permute(0, 3, 1, 2)  # Shape: [3, 3, 32, 32]
    s2_l = torch.tensor(np.stack(s2_list, axis=0), dtype=torch.float32).permute(0, 3, 1, 2)
    s3_l = torch.tensor(np.stack(s3_list, axis=0), dtype=torch.float32).permute(0, 3, 1, 2)

    # Model forward pass
    s1_acts = model(s1_l)[0]
    s2_acts = model(s2_l)[0]
    s3_acts = model(s3_l)[0]

    #Unify the 0 and 1 dimensions of the activations
    s1_reordered = s1_acts.reshape(-1, s1_acts.shape[2], s1_acts.shape[3]).cpu().detach().numpy().transpose(1, 2, 0)
    s2_reordered = s2_acts.reshape(-1, s2_acts.shape[2], s2_acts.shape[3]).cpu().detach().numpy().transpose(1, 2, 0)
    s3_reordered = s3_acts.reshape(-1, s3_acts.shape[2], s3_acts.shape[3]).cpu().detach().numpy().transpose(1, 2, 0)

    # Compute SPD matrices
    mat = "cov"
    
    s1_spd = get_spd(s1_reordered, mat)
    s2_spd = get_spd(s2_reordered, mat)
    s3_spd = get_spd(s3_reordered, mat)
    
    # Save SPD matrices
    np.save(PATH_SAVE + key + "_s1.npy", s1_spd)
    np.save(PATH_SAVE + key + "_s2.npy", s2_spd)
    np.save(PATH_SAVE + key + "_s3.npy", s3_spd)

    # Loading progress
    i += 1
    print(f"Process {i}/{total_files} - ({round((i / total_files) * 100, 3)}% done)", end="\r")
    sleep(0.05)
