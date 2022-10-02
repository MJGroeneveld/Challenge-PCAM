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
import logging
from sklearn.calibration import calibration_curve
import matplotlib.lines as mlines
import matplotlib.transforms as mtransforms

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

    #Test set: 
    test_x = path.join(data_dir, 'camelyonpatch_level_2_split_test_x.h5')
    test_y = path.join(data_dir, 'camelyonpatch_level_2_split_test_y.h5')
    
    test_loader = DataLoader(
        Dataset(test_x, test_y), batch_size=20, shuffle=True
    )

    return train_loader, valid_loader, test_loader 

#%% 
def print_metrics_binary(y_true, predictions, logging, verbose=1):
    predictions = np.array(predictions)
    if len(predictions.shape) == 1:
        predictions = np.stack([1 - predictions, predictions]).transpose((1, 0))
    cf = metrics.confusion_matrix(y_true, predictions.argmax(axis=1))
    if verbose:
        logging.info("confusion matrix:")
        logging.info(cf)
    cf = cf.astype(np.float32)

    acc = (cf[0][0] + cf[1][1]) / np.sum(cf)
    prec0 = cf[0][0] / (cf[0][0] + cf[1][0])
    prec1 = cf[1][1] / (cf[1][1] + cf[0][1])
    rec0 = cf[0][0] / (cf[0][0] + cf[0][1])
    rec1 = cf[1][1] / (cf[1][1] + cf[1][0])
    auroc = metrics.roc_auc_score(y_true, predictions[:, 1])

    (precisions, recalls, thresholds) = metrics.precision_recall_curve(y_true, predictions[:, 1])
    auprc = metrics.auc(recalls, precisions)
    minpse = np.max([min(x, y) for (x, y) in zip(precisions, recalls)])

    if verbose:
        logging.info("accuracy = {0:.3f}".format(acc))
        logging.info("precision class 0 = {0:.3f}".format(prec0))
        logging.info("precision class 1 = {0:.3f}".format(prec1))
        logging.info("recall class 0 = {0:.3f}".format(rec0))
        logging.info("recall class 1 = {0:.3f}".format(rec1))
        logging.info("AUC of ROC = {0:.3f}".format(auroc))
        logging.info("AUC of PRC = {0:.3f}".format(auprc))
       

    return {"acc": acc,
            "prec0": prec0,
            "prec1": prec1,
            "rec0": rec0,
            "rec1": rec1,
            "auroc": auroc,
            "auprc": auprc}

#%% 
def eval_model(model, dataset, device):
    model.eval() # the model will not update parameters or perform dropout
    sigmoid = nn.Sigmoid()
    with torch.no_grad(): # the model will not update parameters
        y_true = []
        predictions = []
        for images, labels in dataset:
            images = images.to(device)
            labels = labels.to(device)
            logits = model(images.type(torch.float32))
            logits = logits.squeeze(1)
            probs = sigmoid(logits) #compute probabilities
            #_, predicted = torch.max(probs.data, 1)
            #y_hat_class = np.where(probs.data<0.5, 0, 1)
            predictions += [p.item() for p in probs] #concatenate all predictions
            y_true += [y.item() for y in labels] #concatenate all labels
    results = print_metrics_binary(y_true, predictions, logging)
    # return results, predictions (probs), and labels
    return results, predictions, y_true

#%% Runnen van de train 
def train(args):
    mode = 'train'
    learning_rate = args['lr']
    num_epochs = args['epochs']

    # Inladen van data
    data_dir = path.join(path.dirname(__file__), "pcamv1")
    train_loader, valid_loader, _ = Dataloaders(data_dir)

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

    # path to save model with extension .pt on disk
    save_model = 'resnet_model.pt'
    best_val_auc = 0.

    # define hyperparameters
    e_losses = []
    results = []

    auc = [] 
    for e in range(num_epochs):
        loss_batch = 0.0
        num_batches = 0 

        for i, (images, labels) in enumerate(train_loader, 0):
            num_batches += 1 
            # get the inputs
            images = Variable(images.to(device))
            labels = Variable(labels.to(device))

            #zero the parameter gradients
            opt.zero_grad()

            # predict classes using images from the training set
            outputs = model(images.float())

            # Compute the loss based on model output and real labels
            loss = criterion(outputs, labels)
            loss_batch += loss.item() 
            loss.backward()
            # adjust parameters based on the calculated gradients
            opt.step()

            # and you can calculate the probabilities, but don't pass them to `nn.CrossEntropyLoss`
            #probs = torch.nn.functional.softmax(outputs, dim=1)
            #print("probs:", probs)

             #if i % 2000 == 1999:  # print every 2000 mini-batches
            print(f'[{e + 1}, {i + 1:5d}] loss: {loss_batch / (i+1):.3f}')

        e_losses.append(loss_batch / num_batches)
        metrics_results, _, _ = eval_model(model,
                                  valid_loader,
                                  device)
    
        metrics_results['epoch'] = e 
        # save results of current epoch
        results.append(metrics_results)    
       
        #HIER DAN BEWAREN VAN MODEL EN KIEZEN VAN HET BESTE MODEL 

#%% 
def test(args): 
    # define training and validation datasets
    mode = 'test'

    data_dir = path.join(path.dirname(__file__), "pcamv1")
    _, _, test_loader = Dataloaders(data_dir)

    for images, labels in test_loader:
        #images to tuple (batch size, 96,96,3)
        images = tuple(images)
        #labels to tuple (batch size, 1)
        labels = tuple(labels)
        break
    
    model = ResNet()
    # model.load_state_dict(torch.load(best_model))

    # Check for device
    #device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu") 
    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    model.to(device)

    metrics_results, pred_probs, y_true = eval_model(model,
                                test_loader,
                                device)

    return metrics_results, pred_probs, y_true                         

#%% 
def main_train(): 
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
        'timestep': 1.0,  # observations every hour
        'imputation': 'previous',  # imputation method
        'normalizer_state': None}  # we use normalization config
    train(args)

#%% 
def main_test(): 
    args = {'dim': 16,
    'dropout': 0.3,
    'batch_size': 8,
    'lr': 1e-3,
    'epochs': 10,
    'emb_size': 16,
    'aggregation_type': 'mean',
    'bidirectional': False,  # we are not going to use biRNN
    'seed': 42,
    'steps': 50,  # print loss every num of steps
    'timestep': 1.0,  # observations every hour
    'imputation': 'previous',  # imputation method
    'normalizer_state': None}  # we use normalization config
    metrics_results, pred_probs, y_true = test(args)
    # Plot roc curve
    resnet_fpr, resnet_tpr, _ = metrics.roc_curve(y_true, pred_probs)
    # plot the roc curve for the model
    plt.figure()
    plt.ylim(0., 1.0)
    plt.xlim(0.,1.0)
    plt.plot(resnet_fpr, resnet_tpr, marker='.', label='resnet', color='darkorange')
    plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
    # axis labels
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    # show the legend
    plt.legend()
    plt.show()

#%%
if __name__ == '__main__':
   main_train() 
     
# %%
if __name__ =='__main__': 
    main_test()