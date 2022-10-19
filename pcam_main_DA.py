#%% 
from xml.etree.ElementTree import PI
from pyparsing import java_style_comment
import torch 
import pandas as pd 
import h5py 
import os 
from os import path
from plots import plotPatches
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from networknet import Network
from resnet import ResNet
import torch.optim as optim
import torch.nn as nn
from torch.autograd import Variable
#import torchmetrics
import numpy as np
from sklearn import metrics
import logging
import time
from torchvision import transforms
import torch.optim.lr_scheduler as lr_scheduler
from scipy.ndimage import rotate
from torchsummary import summary
from PIL import Image

start_time = time.time()
# %% Dataloader with pytorch 

class Dataset(torch.utils.data.Dataset): 
    'Characterizes a dataset for PyTorch'
    def __init__ (self, images, labels, transform=False):
        self.transform = transform
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
        #print("orgineel:")
        #image.save("origineel.png")
        
        #print("Original image", image.shape)
        #96,96,3
        
        #image = np.transpose(image)
        #plt.imshow(image)
        #plt.savefig('origineel.png')
        #print("Transposed before transform", image.shape)
        #3, 96, 96
     
        #Load data and get label
        label = int(self.labels[index])

        image = Image.fromarray(image.astype(np.uint8), 'RGB')

        if self.transform:
            image = self.transform(image)


        #transform = transforms.ToPILImage() 
        #img = transform(image)

        #plt.imshow(  image.permute(1, 2, 0))
        #plt.savefig('transformed.png')
        return image, label

#%%  
def Dataloaders(data_dir, batch_size):
    train_transforms = transforms.Compose([transforms.RandomApply([transforms.RandomRotation((90,90))], p=1.0), 
                                            transforms.RandomHorizontalFlip(0.5), 
                                            transforms.RandomVerticalFlip(0.5), 
                                            transforms.ToTensor()
                                            ]) 

    # Training set:
    train_x = path.join(data_dir, "camelyonpatch_level_2_split_train_x.h5")
    train_y = path.join(data_dir, "camelyonpatch_level_2_split_train_y.h5")

    train_dataset = Dataset(train_x, train_y, train_transforms)

    train_loader = DataLoader(train_dataset, batch_size, shuffle=True)

    # Validation set:
    valid_x = path.join(data_dir, 'camelyonpatch_level_2_split_valid_x.h5')
    valid_y = path.join(data_dir, 'camelyonpatch_level_2_split_valid_y.h5')

    valid_transforms = transforms.Compose([transforms.ToTensor()])

    valid_dataset = Dataset(valid_x, valid_y, valid_transforms)
    
    valid_loader = DataLoader(valid_dataset, batch_size, shuffle=True)

    #Test set: 
    test_x = path.join(data_dir, 'camelyonpatch_level_2_split_test_x.h5')
    test_y = path.join(data_dir, 'camelyonpatch_level_2_split_test_y.h5')
    
    test_transforms = transforms.Compose([transforms.ToTensor()])

    test_dataset = Dataset(test_x, test_y, test_transforms)
    
    test_loader = DataLoader(test_dataset, batch_size, shuffle=False)

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
#eval model
def eval_model(model, dataset, device):
    model.eval() # the model will not update parameters or perfom dropout
    sigmoid = nn.Sigmoid()
    with torch.no_grad(): # the model will not update parameters
        y_true = []
        predictions = []
        for images, labels in dataset:
            images = images.to(device)
            labels = labels.to(device)
            logits = model(images.float())
            logits_hat = logits.squeeze(1)

            probs = sigmoid(logits_hat) #compute probabilities
            
            #values, _ = torch.max(probs, 1)
            
            #y_hat_class = np.where(probs.data<0.5, 0, 1)
            predictions += [p.item() for p in probs] #concatenate all predictions
            y_true += [y.item() for y in labels] #concatenate all labels
        
    results = print_metrics_binary(y_true, predictions, logging)

    return results, predictions, y_true

#%% Runnen van de train 
def train(args):
    mode = 'train'
    learning_rate = args['lr']
    num_epochs = args['epochs']
    batch_size = args['batch_size']
    seed = args['seed']

    if seed:
        torch.manual_seed(seed)
        np.random.seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    # Inladen van data
    data_dir = path.join(path.dirname(__file__), "pcamv1")
    train_loader, valid_loader, _ = Dataloaders(data_dir, batch_size)

    model = ResNet()
    #print(model)
    
    # Check for device
    if torch.cuda.is_available():
        device = torch.device("cuda:0")

    model.to(device)

    print(summary(model, (3,96,96), batch_size = 32))

    # Define stochastic gradient descent optimizer
    opt = torch.optim.Adam(model.parameters(), learning_rate)
    #lambda1 = lambda epoch: num_epochs / 10 
    #scheduler = lr_scheduler.LambdaLR(opt, lambda1)

    # Cross entropy loss function
    criterion = nn.BCEWithLogitsLoss() #zit hier al een sigmoid in?

    # path to save model with extension .pt on disk
    save_model = 'resnet_model.pt'
    best_val_auc = 0.

    # define hyperparameters
    train_losses = []
    results = []
    val_losses = []

    for e in range(num_epochs*20):
        model.train()       
        loss_epoch = 0.0
        print("Steps: ", e+1)
        steps = 0 
        for i, (images, labels) in enumerate(train_loader, 0):
            steps = steps + 1
            if steps>25:
                  break

            # get the inputs
            images = Variable(images.to(device))
            labels = Variable(labels.to(device))

            #zero the parameter gradients
            opt.zero_grad()
  
            # predict classes using images from the training set
            outputs = model(images.float())
            outputs_hat = outputs.squeeze(1)
            
            # Compute the loss based on model output and real labels
            loss_batch = criterion(outputs_hat, labels.float())            
            loss_batch.backward()
            
            # adjust parameters based on the calculated gradients
            opt.step()
            #scheduler.step()
            
            loss_epoch += loss_batch.item() 
 
            #print(f'[{e + 1}, {i + 1:5d}] loss: {loss_batch / (i+1):.3f}')

        train_losses.append(loss_epoch / len(train_loader))
        
        loss_valid = 0.0
        model.eval()

        for i, (image_valid, label_valid) in enumerate(valid_loader, 0):
            image_valid = Variable(image_valid.to(device))
            label_valid = Variable(label_valid.to(device))

            output_valid = model(image_valid.float())
            output_valid = output_valid.squeeze(1)

            loss_val = criterion(output_valid, label_valid.float())
            loss_valid += loss_val.item()


        metrics_results, _, _ = eval_model(model,
                                  valid_loader,
                                  device)
        
        val_losses.append(loss_valid / len(valid_loader))
    
        metrics_results['epoch'] = e + 1 
        # save results of current epoch
        results.append(metrics_results)    

        max_best_val_auc = results[e]["auroc"]
        if max_best_val_auc > best_val_auc:
            print(f'Validation AUC Increased({best_val_auc:.6f}--->{max_best_val_auc:.6f}) \t Saving The Model')
            best_val_auc = max_best_val_auc
            # save best model to disk
            torch.save(model.state_dict(), save_model)
    
    plt.figure(figsize=(20,10))
    plt.xlabel('Steps')
    plt.ylabel('Loss')
    plt.plot(train_losses, label = "Train loss")
    plt.plot(val_losses, label = "Validation Loss")
    plt.legend()
    plt.show()
    plt.savefig("resnet_20epochs_batchsize512_losses.png")

#%% 
def test(args): 
    # define training and validation datasets
    mode = 'test'
    best_model = args['best_model']
    batch_size = args['batch_size']

    data_dir = path.join(path.dirname(__file__), "pcamv1")
    _, _, test_loader = Dataloaders(data_dir, batch_size)
    
    model = ResNet()
    model.load_state_dict(torch.load(best_model))

    # Check for device
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    model.to(device)

    metrics_results, pred_probs, y_true = eval_model(model,
                                test_loader,
                                device)

    return metrics_results, pred_probs, y_true                         

def valid(args): 
    # define training and validation datasets
    mode = 'test'
    best_model = args['best_model']
    batch_size = args['batch_size']

    data_dir = path.join(path.dirname(__file__), "pcamv1")
    _, valid_loader, _ = Dataloaders(data_dir, batch_size)
    
    model = ResNet()
    model.load_state_dict(torch.load(best_model))

    # Check for device
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    model.to(device)

    metrics_results, pred_probs, y_true = eval_model(model,
                                valid_loader,
                                device)

    return metrics_results, pred_probs, y_true                         


#%% 
def main_train(): 
    args = {'dropout': 0.3, #Misschien nog gebruiken
        'batch_size': 512,
        'lr': 1e-4,
        'epochs': 20,
        'seed': 42,
        'normalizer_state': None}  #Misschien nog gebruiken
    train(args)

def main_valid(): 
    args = {'best_model':'resnet_model.pt',
    'dropout': 0.3, #Misschien nog gebruiken
    'batch_size': 512,
    'normalizer_state': None}  #Misschien nog gebruiken

    metrics_results, pred_probs, y_true = valid(args)

    # uit metrics_results haal je de auc en die plot je 

     # # Plot roc curve
    fpr, tpr, _ = metrics.roc_curve(y_true, pred_probs)
    # plot the roc curve for the model
    plt.figure()
    plt.ylim(0., 1.0)
    plt.xlim(0.,1.0)
    plt.plot(fpr, tpr, marker='.', label='resnet', color='darkorange')
    plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
    # axis labels
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    # show the legend
    plt.legend()
    plt.show()
    plt.savefig("Resnet_20epochs_batchsize512_AUC.png")



#%% 
def main_test(): 
    args = {'best_model':'resnet_model.pt',
    'dropout': 0.3, #Misschien nog gebruiken
    'batch_size': 512,
    'normalizer_state': None}  #Misschien nog gebruiken

    metrics_results, pred_probs, _ = test(args)

    print(metrics_results)

    header = "case, prediction"
    with open('predictions_resnet_DA.csv', 'w') as file_handler:
        index = 0
        file_handler.write(header)
        file_handler.write("\n")
        for item in pred_probs:
            file_handler.write("{}, {}\n".format(index, item))
            index += 1

#%%
if __name__ == '__main__':
    main_train() 
     
if __name__ == '__main__':
    main_valid()
# %%
if __name__ =='__main__': 
    main_test()

difference_time = time.time() - start_time
print("--- %s minutes ---" % (difference_time//60))