import torch
import torch.nn as nn
import torchvision
from torchvision.models import ResNet18_Weights

class ResNet_dropout(nn.Module):
    def __init__(self, dropoutRate=0.5):
        super(ResNet_dropout, self).__init__()
        self.dropoutRate = dropoutRate
        self.resnet = torchvision.models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)

        in_features = self.resnet.fc.in_features
        
        # Replace FC with Dropout + Linear
        self.resnet.fc = nn.Sequential(
            nn.Dropout(self.dropoutRate),
            nn.Linear(in_features, 2)
        )

        # Contrastive Projection Head
        projection_dim = 128
        self.projection_head = nn.Sequential(
            nn.Linear(in_features, projection_dim),
            nn.ReLU(inplace=True),
            nn.Linear(projection_dim, projection_dim)
        )

    def forward(self, x):
        # To get features for SupCon, we need to hook into the layer before FC
        # ResNet34 usually provides 512 features before the FC layer.
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
        projected_features = self.projection_head(features)
        
        return logits, features, projected_features
