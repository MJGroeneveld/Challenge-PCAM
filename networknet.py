import pytorch_lightning as pl
import torch.nn as nn
import torch.nn.functional as F

#Define a CNN
class Network(pl.LightningModule):

    def __init__(self):
        super(Network, self).__init__()

        self.layers = nn.Sequential(
            # input = 3 x 96 x 96, output = 32 x 96 x 96 
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            # input = 32 x 94 x 94, output = 64 x 48 x 48 
            nn.MaxPool2d(kernel_size=2, stride=2),

            #input = 32 x 48 x 48, output = 64 x 48 x 48
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            #input = 64 x 48 x 48, output = 64 x 24 x 24
            nn.MaxPool2d(kernel_size=2, stride=2),

            # input = 64 x 24 x 24, output = 128 x 24 x 24 
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            #input = 128 x 24 x 24, output = 128 x 12 x 12 
            nn.MaxPool2d(kernel_size=2, stride=2),

            #input = 128 x 12 x 12, output = 512 x 12 x 12
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            #input = 512 x 12 x 12, output = 512 x 6 x 6 
            nn.MaxPool2d(kernel_size=2, stride=2),

            #input = 256 x 6 x 6, output = 512 x 6 x 6 
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding = 1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            #input = 512 x 6 x 6, output = 512 x 3 x 3 
            nn.MaxPool2d(kernel_size=2, stride=2), 

        )
        
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((3,3)), 
            nn.Flatten(),
            nn.Linear(in_features = 3*3*512, out_features = 1),
            # nn.ReLU(),
            # nn.Linear(in_features = 128, out_features = 32), 
            # nn.ReLU(), 
            # nn.Linear(in_features = 32, out_features = 1)
        )

    def forward(self, x):
        x = self.layers(x)
        x = self.classifier(x)

        return x
