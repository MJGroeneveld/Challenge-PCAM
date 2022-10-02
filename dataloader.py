#%% 
from pyparsing import java_style_comment
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
import torchmetrics
import numpy as np
from sklearn import metrics

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

#%%  
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

# Metrics for train, test, validation set (accuracy)
# def metrics(total, correct, actual_labels, predicted_outputs):
#     ''' Set output to 0 or 1 to be able to calculate the accuracy later on'''
#     _, predicted = torch.max(predicted_outputs.data, 1)
#     total += actual_labels.size(0)
#     correct+= (predicted == actual_labels).sum().item()
#     print(f' Accuracy: {100 * correct // total} %')

#%% Runnen van de train 
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
    #device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu") 
    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    model.to(device)

    # Define stochastic gradient descent optimizer
    opt = torch.optim.SGD(model.parameters(), learning_rate)

    #Deze optimizer gebruiken we nu niet
    # optimizer = torch.optim.SGD(model.parameters(), lr=0.001, weight_decay=0.005, momentum=0.9)

    # Cross entropy loss function
    criterion = nn.CrossEntropyLoss()

    # define hyperparameters
    e_losses = []
    total = 0
    correct = 0

    auc = [] 
    for e in range(num_epochs):
        running_loss = 0.0

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

            loss.backward()
            # adjust parameters based on the calculated gradients
            opt.step()

            # and you can calculate the probabilities, but don't pass them to `nn.CrossEntropyLoss`
            #probs = torch.nn.functional.softmax(outputs, dim=1)
            #print("probs:", probs)

            running_loss += loss.item()
            
            #if i % 2000 == 1999:  # print every 2000 mini-batches
            print(f'[{e + 1}, {i + 1:5d}] loss: {running_loss / (i+1):.3f}')

        #Validaten met validation set
        #op basis van AUC, accuracy of loss (of gemiddelde ervan). Wij doen AUC
        #gebruik sklearn voor AUC
        #hiervoor eerst ROC plot per epoch maken (daaruit haal je 1 AUC per epoch) 
        #en op basis van die AUC (wat dus een aantal getallen zijn) kan je het beste model halen
        #check daarmee welk model het beste is, gebruik deze voor het testen
        #maak nog een plotje van je AUC om te zien hoe mooi die loopt

        #dit was beste model op basis van loss, maar gebruiken we ff niet
        # print(f'Epoch {e+1} \t\t Training Loss: {running_loss / len(train_loader)} \t\t Validation Loss: {valid_loss / len(valid_loader)}')
        # if min_valid_loss > valid_loss:
        #     print(f'Validation Loss Decreased({min_valid_loss:.6f}--->{valid_loss:.6f}) \t Saving The Model')
        #     min_valid_loss = valid_loss
        #     # Saving State Dict
        #     torch.save(model.state_dict(), 'saved_model.pth')

        #dit gebruiken we nu ook even niet
        # ''' Set output to 0 or 1 to be able to calculate the accuracy later on'''
        # _, predicted = torch.max(target.data, 1)
        # total += val_labels.size(0)
        # correct+= (predicted == val_labels).sum().item()
        # print(f' Accuracy: {100 * correct // total} %')

        valid_loss = 0.0
        model.eval()     # Optional when not using Model Specific layer

        for val_images, val_labels in valid_loader:
            val_images = val_images.to(device)
            val_labels = val_labels.to(device)
     
            target = model(val_images.type(torch.float32))
            loss = criterion(target,val_labels)
            valid_loss = loss.item() * val_images.size(0)
    
        print("target", target)
        print("target detach cpu", target.detach().cpu()) 
        target = target.detach().cpu()
        val_labels = val_labels.detach().cpu()
        _, predicted = torch.max(target.data, 1)
        print("predicted", predicted)
        fpr, tpr, _ = metrics.roc_curve(val_labels, predicted)
        print("fpr", fpr)
        print("tpr", tpr)
        plt.plot(fpr,tpr)
        plt.show() 
        auc.append(metrics.auc(fpr, tpr))
        # en dan hier nog je beste model opslaan die je gebruikt voor testen (mbv if min(auc) > auc: net als dat loss stukje hierboven in commentaar)
    
    print("auc", auc)
    plt.plot(auc)
    plt.title("AUC")
    plt.xlabel("Epoch")
    plt.ylabel("AUC")
    plt.show()


if __name__ == '__main__':
    args = {'dim': 16,
            'dropout': 0.3,
            'batch_size': 8,
            'lr': 1e-3,
            'epochs': 5,
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
