import os
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent.parent)) #
from src.models.SPD_GeoNet.build_AESPDs import ASPDNet
from src.utils.general import enable_gpu
from src.data.custom_datasets import dataset_from_folder
from src.models.SPD_GeoNet.build_SPDNets import ASPDNet
from torch.utils.data import DataLoader
from src.training.functions import train_autoencoder
import json
import torch

PATH_DATA = "/Datasets/PICAI_64x64_patches/"
PATH_COVS = PATH_DATA+"covariances/"

info = json.load(open(PATH_DATA + '8x64x64-CIspheres.json'))
folds_idxs = json.load(open(PATH_DATA + 'picai_patches_splits_5kf.json'))

device = enable_gpu()

conf = {'epochs': 50, 'lr': 1e-3, 'batch_size': 10}



#Compute conf_name using all the keys and values in conf
conf_name = "_".join([f"{k}:{v}" for k,v in conf.items()])

for i in range(5):
    fold_train = f"train_fold_{i}"
    fold_test = f"val_fold_{i}"

    idx_train = folds_idxs[fold_train]
    idx_test = folds_idxs[fold_test]

    dataloader_train = DataLoader(dataset_from_folder(idx_train, PATH_COVS),
                                   batch_size=conf['batch_size'], shuffle=True)
    dataloader_test = DataLoader(dataset_from_folder(idx_test, PATH_COVS),
                                  batch_size=conf['batch_size'], shuffle=False)
    print(f"Training fold {i}")

    os.makedirs(f"models/multiclass_grouped/{conf_name}", exist_ok=True)
    os.makedirs(f"outputs/multiclass_grouped/{conf_name}", exist_ok=True)
    
    model = ASPDNet(192,1).to(device)
    # break
    model, training_losses, testing_losses = train_autoencoder(model, dataloader_train,
                                                               dataloader_test, num_epochs=conf['epochs'], learning_rate=conf['lr'], device=device)
    
    #Create a folder if doesen exists in models/multiclass_grouped with the configuration name
    
    #Save the model weights in the folder models/multiclass_grouped/{conf_name}/fold_{i}.pth
    torch.save(model.state_dict(), f"models/multiclass_grouped/{conf_name}/fold_{i}.pth")
    #Save the training and testing losses per epoch in the folder outputs/multiclass_grouped/{conf_name}/fold_{i} as csv both in the same file
    with open(f"outputs/multiclass_grouped/{conf_name}/fold_{i}.csv", 'w') as f:
        f.write("epoch,train_loss,test_loss\n")
        for epoch, (train_loss, test_loss) in enumerate(zip(training_losses, testing_losses)):
            f.write(f"{epoch},{train_loss},{test_loss}\n")

    

    

