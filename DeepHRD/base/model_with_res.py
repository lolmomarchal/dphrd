import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision.models import ResNet34_Weights

class ResNet_dropout(nn.Module):
    def __init__(self, dropoutRate=0.5):
        super(ResNet_dropout, self).__init__()
        self.resnet = torchvision.models.resnet34(weights=ResNet34_Weights.DEFAULT)
        in_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Identity()

        # Shared Neck (Feature Refiner)
        self.shared_neck = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropoutRate)
        )

        self.classifier = nn.Linear(256, 2)
        # self.regression_head = nn.Linear(256, 1)

    def forward(self, x):
        raw_features = self.resnet(x) # 512d for clustering
        refined = self.shared_neck(raw_features) # 256d for logic
        logits = self.classifier(refined)
        # hrd_score_pred = self.regression_head(refined)
        hrd_score_pred = 0
        return logits, hrd_score_pred, raw_features