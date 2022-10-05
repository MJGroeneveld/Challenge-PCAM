import torch 
import pytorch_lightning as pl
import torchvision
import torch.nn as nn

class DenseNet(pl.LightningModule): 
    def __init__(self): 
        super().__init__()
        self.model = torchvision.models.densenet201(pretrained=True)
        # We change the input and output layers to make the model compatible to our data
        #self.model.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=2, padding=3, bias=False)
        #self.model.fc = nn.Linear(in_features=512, out_features=2, bias=True)

    def forward(self, x):
        x = self.model(x)
        return x


