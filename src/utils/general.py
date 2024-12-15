import os
import torch
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent)) #
from src.models.SPD_GeoNet.build_AESPDs import ASPDNet
import numpy as np

def enable_gpu(gpu_number=0):
    if gpu_number:
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_number #aca se pone nuemro de grafica libre

    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f'Using GPU: {torch.cuda.get_device_name()}')
        print('CUDA Visible devices:',os.getenv('CUDA_VISIBLE_DEVICES'))
    else:
        device = torch.device('cpu')
        print("Failed to find GPU, using CPU instead.")

    return device


def get_spd(acts,type="gramm"):
    h,w,d = acts.shape
    vect_acts = acts.reshape(h*w,d)
    if type == "gramm":
        spd = vect_acts.T@vect_acts
    elif type == "corr":
        spd = np.corrcoef(vect_acts.T)
    elif type == "cov":
        spd = np.cov(vect_acts.T)
    return spd

#make a function that given a volume returns the center slice in x,y and z
def get_center_slices(vol):
    x,y,z = vol.shape
    return vol[(x-1)//2,:,:], vol[:,(y-1)//2,:], vol[:,:,(z-1)//2]


def flatten_upper_triangular(arr, euclidean_projection = False):
    # Get the shape of the input array
    c, n, _ = arr.shape

    # Initialize an empty list to store the flattened upper triangular matrices
    flattened_matrices = []

    # Iterate over each c
    for i in range(c):
        
        mat = arr[i]
        
        if euclidean_projection:       
            eigenvals, eigenvecs = np.linalg.eig(mat)
            #If an eigenvalue is negative, we set it to a small positive value
            eigenvals = np.where(eigenvals < 1e-6, 1e-6, eigenvals)
            eigenvals = np.log(eigenvals)           
            mat = eigenvecs@np.diag(np.real(eigenvals),k=0)@eigenvecs.T

            
        # Get the upper triangular matrix of the ith element in the array
        upper_triangular = mat[np.triu_indices_from(mat)]

        # Flatten the upper triangular matrix and append it to the list
        flattened_matrices.append(upper_triangular)

    # Return the list of flattened upper triangular matrices
    return np.real(np.stack(flattened_matrices,axis = 0))