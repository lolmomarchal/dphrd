import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision.models import ResNet34_Weights

class ResNet_dropout(nn.Module):
    def __init__(self, dropoutRate=0.5):
        super(ResNet_dropout, self).__init__()
        # Load backbone
        self.resnet = torchvision.models.resnet34(weights=ResNet34_Weights.DEFAULT)
        in_features = self.resnet.fc.in_features # 512
        self.resnet.fc = nn.Identity()

        self.head = nn.Sequential(
            nn.LayerNorm(in_features),
            nn.Dropout(dropoutRate),
            nn.Linear(in_features, 2)
        )

    def forward(self, x):
        features = self.resnet(x)

        logits = self.head(features)
        return logits, 0, features
