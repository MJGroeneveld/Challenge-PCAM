import pytorch_lightning as pl
import torch.nn as nn
import torch.nn.functional as F

#Define a CNN
class Network(pl.LightningModule):

    def __init__(self):
        super(Network, self).__init__()

        self.layers = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            #Even de vraag of we van 6400 terug naar 32 kunnen
            nn.Linear(in_features = 6400, out_features = 32),
            nn.ReLU(),
            nn.Linear(in_features = 32, out_features = 1)
        )

    def forward(self, x):
        x = self.layers(x)
        x = self.classifier(x)

        return x