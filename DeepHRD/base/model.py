import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision.models import ResNet34_Weights


class ResNet_dropout(nn.Module):
    def __init__(self, dropoutRate=0.5):
        super(ResNet_dropout, self).__init__()
        self.dropoutRate = dropoutRate
        self.resnet = torchvision.models.resnet34(weights=ResNet34_Weights.IMAGENET1K_V1)

        for name, child in self.resnet.named_children():
            if name in ['conv1', 'bn1', 'layer1', 'layer2']:
                for param in child.parameters():
                    param.requires_grad = False
            else:
                for param in child.parameters():
                    param.requires_grad = True

        in_features = self.resnet.fc.in_features
        
        self.resnet.fc = nn.Identity() 
        self.classifier = nn.Sequential(
            nn.Dropout(self.dropoutRate),
            nn.Linear(in_features, 2)
        )

        # 3. Contrastive Projection Head
        projection_dim = 128
        self.projection_head = nn.Sequential(
            nn.Linear(in_features, projection_dim),
            nn.ReLU(inplace=True),
            nn.Linear(projection_dim, projection_dim)
        )

    def forward(self, x):
        features = self.resnet(x) 
        
        logits = self.classifier(features)
        if self.training:
            projected_features = self.projection_head(features)
            return logits, features, projected_features
        else:
            return logits
