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
            if name in ['conv1', 'bn1', 'layer1']:
                for param in child.parameters():
                    param.requires_grad = False
            else:
                for param in child.parameters():
                    param.requires_grad = True

        in_features = self.resnet.fc.in_features

        self.resnet.fc = nn.Identity()
        # task 1: Classification Head (HRD status +/-)
        self.classifier = nn.Sequential(
            nn.Dropout(dropoutRate),
            nn.Linear(in_features, 2)
        )

        # task 2: Regression Head (Continuous HRD Score)
        # Inspired by HRDPath's multi-task learning
        self.regression_head = nn.Sequential(
            nn.Dropout(dropoutRate),
            nn.Linear(in_features, 1))

    def forward(self, x):
        features = self.resnet(x)
        logits = self.classifier(features)
        hrd_score_pred = self.regression_head(features)


        return logits, hrd_score_pred, features
