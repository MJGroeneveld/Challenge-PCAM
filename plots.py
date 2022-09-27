#%% Needed packages 
import numpy as np 
import h5py 
import matplotlib.pyplot as plt 
import torch 

#%% Function: Plot the first 9 patches with labels
def plotPatches(filename_x, filename_y):
    first_9_x = filename_x[:9]
    first_9_y = filename_y[:9].flatten()

    print(first_9_y)

    plt.figure(figsize=(20, 8))

    for row in range(3):
        for col in range(3):

            i = row * 3 + col
            plt.subplot(3, 3, i+1)

            label = 'Yes' if first_9_y[i] else 'No'
            plt.title(label, fontsize=18)
            plt.imshow(first_9_x[i])

    plt.tight_layout(pad=0.4, w_pad=-60, h_pad=1.0)
    plt.savefig('examples_plot.png')
    plt.show()

#%% Function: Check number of positive and negative labels
def countLabels(filename):
    labelyes = 0
    labelno = 0
    for i in filename:
        if i == 1:
            labelyes += 1
        else:
            labelno += 1

    print("yes", filename, ":", labelyes)
    print("no", filename, ":", labelno)


#%% read from train files
def main(): 
    train_x_file = h5py.File('pcamv1/camelyonpatch_level_2_split_train_x.h5', 'r')
    train_y_file = h5py.File('pcamv1/camelyonpatch_level_2_split_train_y.h5', 'r')

    train_x = train_x_file['x']
    train_y = train_y_file['y']

    print('Shape train x: {}'.format(train_x.shape))  # the patches
    print('Shape train y: {}'.format(train_y.shape))  # labels with yes or no

    # read from valid files
    valid_x_file = h5py.File('pcamv1/camelyonpatch_level_2_split_valid_x.h5', 'r')
    valid_y_file = h5py.File('pcamv1/camelyonpatch_level_2_split_valid_y.h5', 'r')

    valid_x = valid_x_file['x']
    valid_y = valid_y_file['y']

    print('Shape valid x: {}'.format(valid_x.shape))  # the patches
    print('Shape valid y: {}'.format(valid_y.shape))  # labels with yes or no

    plotPatches(train_x, train_y)
    plotPatches(valid_x, valid_y)

    countLabels(train_y)
    countLabels(valid_y)

if __name__ == '__main__': 
    main()