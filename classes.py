#%% 
import torch 
import pandas as pd 
import h5py 
import os 

# %% Dataloader with pytorch 

class Dataset(torch.utils.data.Dataset): 
    'Characterizes a dataset for PyTorch'
    def __init__ (self, list_IDs, labels): 
        'Initialization'
        h5_images = h5py.File(list_IDs, 'r')
        h5_labels = h5py.File(labels, 'r')
        self.image = h5_images['x']
        self.labels = h5_labels['y']

    def __len__(self):
        'Denotes the total number of samples'
        return self.image.shape[0]
    
    def __getitem__ (self, index): 
        'Generates one sample of data'
        # Select sample 
        image = self.image[index]

        #Load data and get label 
        label = int(self.lables[index])

        return image, label 


def Dataloaders(args, path): 

    # Training set:
    train_x = os.path.join(path, 'camelyonpath_level_2_split_train_x.h5')
    train_y = os.path.join(path, 'camelyonpath_level_2_split_train_y_h5')

    train_loader = torch.utils.data.DataLoader(
        Dataset(train_x, train_y, batch_size = args.batch_size, shuffle = True)
    )

    # Validation set 
    valid_x = os.path.join(path, 'camelyonpath_level_2_split_valid_x.h5')
    valid_y = os.path.join(path, 'camelyonpath_level_2_split_valid_y_h5')

    valid_loader = torch.utils.data.DataLoader(
        Dataset(valid_x, valid_y, batch_size = args.batch_size, shuffle = True)
    )
    return train_loader, valid_loader 