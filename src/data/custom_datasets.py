import os
import numpy as np
from torch.utils.data import Dataset
import torch

# Define the dataset class
class dataset_from_folder(Dataset):
    def __init__(self, filenames, PATH):
        """
        Args:
            filenames (list): List of filenames (base names without the _sX.npy extensions).
            PATH (str): Path to the directory containing .npy files.
        """
        self.filenames = filenames
        self.PATH = PATH

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        """
        Args:
            idx (int): Index of the data point to retrieve.
        
        Returns:
            torch.Tensor: A tensor where the three files (_s1, _s2, _s3) are stacked along the batch axis.
        """
        base_name = self.filenames[idx]
        file_paths = [
            os.path.join(self.PATH, f"{base_name}_s1.npy"),
            os.path.join(self.PATH, f"{base_name}_s2.npy"),
            os.path.join(self.PATH, f"{base_name}_s3.npy"),
        ]

        # Check if all files exist
        for path in file_paths:
            if not os.path.exists(path):
                raise FileNotFoundError(f"File {path} not found.")

        # Load each file and convert to PyTorch tensor
        tensors = [torch.tensor(np.load(path), dtype=torch.float32) for path in file_paths]
        #For each matrix in the tensors list add a diagonal matrix with values 1e-4
        tensors = [tensor + torch.eye(tensor.size(0), dtype=torch.float32) * 1e-4 for tensor in tensors]
        
        # Stack along a new dimension (e.g., batch axis, axis 0)
        stacked_tensor = torch.stack(tensors, dim=0)
        return stacked_tensor
