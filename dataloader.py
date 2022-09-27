#%% 
import torch 
import pandas as pd 
import h5py 
import os 
from os import path
from plots import plotPatches
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from network import Network
from resnet import ResNet
import torch.optim as optim
import torch.nn as nn
from torch.optim import Adam
from torch.autograd import Variable
import numpy as np

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
        image = np.transpose(image)

        #Load data and get label
        label = int(self.labels[index])

        return image, label

def Dataloaders(data_dir):
    # Training set:
    train_x = path.join(data_dir, "camelyonpatch_level_2_split_train_x.h5")
    train_y = path.join(data_dir, "camelyonpatch_level_2_split_train_y.h5")

    train_loader = DataLoader(
        Dataset(train_x, train_y), batch_size=20, shuffle=True
    )

    # Validation set:
    valid_x = path.join(data_dir, 'camelyonpatch_level_2_split_valid_x.h5')
    valid_y = path.join(data_dir, 'camelyonpatch_level_2_split_valid_y.h5')

    valid_loader = DataLoader(
        Dataset(valid_x, valid_y), batch_size=20, shuffle=True
    )

    return train_loader, valid_loader

def train(args):
    dropout = args['dropout']
    learning_rate = args['lr']
    num_epochs = args['epochs']
    emb_size = args['emb_size']
    aggregation_type = args['aggregation_type']
    bidirectional_encoder = args['bidirectional']
    seed = args['seed']
    steps = args['steps']
    data = args['data']
    notes = args['notes']
    timestep = args['timestep']
    normalizer_state = args['normalizer_state']

    # Inladen van data
    data_dir = path.join(path.dirname(__file__), "pcamv1")
    train_loader, valid_loader = Dataloaders(data_dir)

    for images, labels in train_loader:
        #images to tuple (batch size, 96,96,3)
        images = tuple(images)
        #labels to tuple (batch size, 1)
        labels = tuple(labels)
        break

    model = ResNet()

    # Check for device
    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    model.to(device)

    print("test hier")

    # Define stochastic gradient descent optimizer
    opt = torch.optim.SGD(model.parameters(), learning_rate)

    #Deze optimizer gebruiken we nu niet
    # optimizer = torch.optim.SGD(model.parameters(), lr=0.001, weight_decay=0.005, momentum=0.9)

    # Cross entropy loss function
    criterion = nn.CrossEntropyLoss()

    # define hyperparameters
    e_losses = []
    running_loss = 0.0

    print("for loop")
    for e in range(num_epochs):
        for i, (images, labels) in enumerate(train_loader, 0):
            # get the inputs
            images = Variable(images.to(device))
            labels = Variable(labels.to(device))

            #zero the parameter gradients
            opt.zero_grad()

            # predict classes using images from the training set
            outputs = model(images.float())

            # Compute the loss based on model output and real labels
            loss = criterion(outputs, labels)
            # backpropagate the loss
            loss.backward()
            # adjust parameters based on the calculated gradients
            opt.step()

            running_loss += loss.item()
            if i % 2000 == 1999:  # print every 2000 mini-batches
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
                running_loss = 0.0

            #Metrics printen
            #metrics()

def metrics():
    # defining metric
    self.train_acc = torchmetrics.Accuracy()
    self.val_acc = torchmetrics.Accuracy()

    self.train_f1 = torchmetrics.F1Score(multiclass=False)
    self.val_f1 = torchmetrics.F1Score(multiclass=False)

    self.train_prec = torchmetrics.Precision(multiclass=False)
    self.val_prec = torchmetrics.Precision(multiclass=False)

    self.train_rec = torchmetrics.Recall(multiclass=False)
    self.val_rec = torchmetrics.Recall(multiclass=False)

if __name__ == '__main__':
    args = {'dim': 16,
            'dropout': 0.3,
            'batch_size': 8,
            'lr': 1e-3,
            'epochs': 1,
            'emb_size': 16,
            'aggregation_type': 'mean',
            'bidirectional': False,  # we are not going to use biRNN
            'seed': 42,
            'steps': 50,  # print loss every num of steps
            'data': 'test_text_data_2/in-hospital-mortality',  # path to MIMIC data
            'notes': 'test_text_data_2/train',  # code ignores text
            'timestep': 1.0,  # observations every hour
            'imputation': 'previous',  # imputation method
            'normalizer_state': None}  # we use normalization config
    train(args)
# %%
