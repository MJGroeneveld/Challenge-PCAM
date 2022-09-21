#%% 
import torch 
import pandas as pd 
import h5py 
import os 
from os import path
from plots import plotPatches
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

# %% Dataloader with pytorch 

class Dataset(torch.utils.data.Dataset): 
    'Characterizes a dataset for PyTorch'
    def __init__ (self, images, labels):
        'Initialization'
        h5_images = h5py.File(images, 'r')
        h5_labels = h5py.File(labels, 'r')
        self.image = h5_images['x']
        self.labels = h5_labels['y']

    def __len__(self):
        'Denotes the total number of samples'

        return self.image.shape[0]
    
    def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        image = self.image[index]

        #Load data and get label
        label = int(self.labels[index])

        print("Hoi")
        return image, label

def Dataloaders(data_dir):
    # Training set:
    train_x = path.join(data_dir, "camelyonpatch_level_2_split_train_x.h5")
    train_y = path.join(data_dir, "camelyonpatch_level_2_split_train_y.h5")

    train_loader = DataLoader(
        Dataset(train_x, train_y), batch_size=64, shuffle=True
    )

    # Validation set:
    valid_x = path.join(data_dir, 'camelyonpatch_level_2_split_valid_x.h5')
    valid_y = path.join(data_dir, 'camelyonpatch_level_2_split_valid_y.h5')

    valid_loader = DataLoader(
        Dataset(valid_x, valid_y), batch_size=64, shuffle=True
    )

    return train_loader, valid_loader

def main():
    # Inladen van data 

    data_dir = path.join(path.dirname(__file__), "pcamv1")
    train_loader, valid_loader = Dataloaders(data_dir)

    for b in train_loader:
        print(type(b))
        break

if __name__ == '__main__':
    main()
# %%
