import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision.models import ResNet34_Weights
class ResNet_dropout(nn.Module):
    def __init__(self, dropoutRate=0.5, pretrained=True):
        super().__init__()
        self.resnet = torchvision.models.resnet34(
            weights=ResNet34_Weights.IMAGENET1K_V1 
        )

        in_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Sequential(
            nn.Dropout(dropoutRate),
            nn.Linear(in_features, 2)
        )
    def forward(self, x):
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)
        x = self.resnet.layer1(x)
        x = self.resnet.layer2(x)
        x = self.resnet.layer3(x)
        x = self.resnet.layer4(x)
        x = self.resnet.avgpool(x)
        features = torch.flatten(x, 1)
        logits = self.resnet.fc(features)
        return logits, features
