import pytorch_lightning as pl
import torchvision
import torch.nn as nn

class ResNet(pl.LightningModule):

  def __init__(self):
    print("init boven")
    super().__init__()
    self.model = torchvision.models.resnet18(pretrained=True)
    print("")
    # We change the input and output layers to make the model compatible to our data
    self.model.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=2, padding=3, bias=False)
    self.model.fc = nn.Linear(in_features=512, out_features=2, bias=True)

    print("init onder")

  def forward(self, x):
    print("boven de x")
    x = self.model(x)
    print("TESTTTTTTTTTTTTTTTT")
    return x
