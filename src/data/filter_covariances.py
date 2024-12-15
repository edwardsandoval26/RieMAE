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
from tqdm import tqdm  # Import tqdm for the loading bar



PATH_DATA = "/Datasets/PICAI_64x64_patches/"
info = json.load(open(PATH_DATA + '8x64x64-CIspheres.json'))
PATH_VOLS = PATH_DATA + "patches/"
PATH_COVS = PATH_DATA + "covariances/"

#Read all the elements in PATH_COVS and get a list of the diagonal of each covariance matrix

#First get the list of all the files in PATH_COVS
files = os.listdir(PATH_COVS)
covs_diags = []

print("Processing covariance matrices...")
for file in tqdm(files, desc="Reading Covariance Matrices"):
    cov = np.load(PATH_COVS + file)
    covs_diags.append(np.diag(cov))

#Now we have a list of the diagonal of each covariance matrix
#Get the mean and std of each element in the diagonal of the covariance matrix
    
covs_diags = np.array(covs_diags)
mean = np.mean(covs_diags, axis=0)
std = np.std(covs_diags, axis=0)

# Sort indices based on descending order of mean * std
sort_indices = np.argsort(mean * std)[::-1]

# Get the last 64 indices
filter_indices = sort_indices[-64:]

# Modify the covariance matrices to exclude the last 64 indices
filtered_cov_dir = os.path.join(PATH_DATA, "filtered_covariances_64/")
os.makedirs(filtered_cov_dir, exist_ok=True)  # Ensure the output directory exists

print("Filtering and saving covariance matrices...")
for file in tqdm(files, desc="Filtering Covariance Matrices"):
    # Load the covariance matrix
    cov = np.load(PATH_COVS + file)
    
    # Remove rows and columns corresponding to filter_indices
    filtered_cov = np.delete(cov, filter_indices, axis=0)  # Remove rows
    filtered_cov = 1000*np.delete(filtered_cov, filter_indices, axis=1)  # Remove columns
    
    # Save the filtered covariance matrix
    filtered_file_path = os.path.join(filtered_cov_dir, file)
    np.save(filtered_file_path, filtered_cov)

print(f"Filtered covariance matrices saved to {filtered_cov_dir}")

