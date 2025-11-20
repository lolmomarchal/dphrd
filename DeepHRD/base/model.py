import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision.models import ResNet18_Weights


class ResNet_dropout(nn.Module):
	def __init__(self, dropoutRate):
		super(ResNet_dropout, self).__init__()

		self.dropoutRate = dropoutRate

		# Import the pretrained ResNet18 architecture
		# self.resnet = torchvision.models.resnet.resnet18(pretrained=True)
		self.resnet = torchvision.models.resnet.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)

		# # Freezes the first 6 layers of the ResNet architecture
		# count = 0
		# for child in self.resnet.children():
		# 	count += 1
		# 	if count < 7:
		# 		for param in child.parameters():
		# 			param.requires_grad=False
		in_features = self.resnet.fc.in_features

		# Adds dropout into the fully connected layer of the ResNet model and changes the final output to 2 nodes
		self.resnet.fc = nn.Sequential(nn.Dropout(self.dropoutRate), nn.Linear(self.resnet.fc.in_features, 2))

		# ============= CONTRASTIVE LEARNING PORTION =============
		projection_dim = 128 # The new feature space dimension for regularization
		self.projection_head = nn.Sequential(
			nn.Linear(in_features, projection_dim), # 512 -> 128
			nn.ReLU(inplace=True),
			nn.Linear(projection_dim, projection_dim) # Final output 128D
		)


	def forward(self, x):
		# print(f"[DEBUG] Input shape: {x.shape}")  # (B, C, H, W)
		x = self.resnet.conv1(x)
		# print(f"[DEBUG] After conv1: {x.shape}")
		x = self.resnet.bn1(x)
		# print(f"[DEBUG] After bn1: {x.shape}")

		x = self.resnet.relu(x)
		# print(f"[DEBUG] After relu: {x.shape}")

		x = self.resnet.maxpool(x)
		# print(f"[DEBUG] After maxpool: {x.shape}")
		x = self.resnet.layer1(x)
		# print(f"[DEBUG] After layer1: {x.shape}")
		x = self.resnet.layer2(x)
		# print(f"[DEBUG] After layer2: {x.shape}")
		x = self.resnet.layer3(x)
		# print(f"[DEBUG] After layer3: {x.shape}")
		x = self.resnet.layer4(x)
		# print(f"[DEBUG] After layer4: {x.shape}")
		x = self.resnet.avgpool(x)
		# print(f"[DEBUG] After avgpool: {x.shape}")
		features_512 = torch.flatten(x, 1)
		# print(f"[DEBUG] After flatten: {x.shape}")
		logits = self.resnet.fc(features_512)
		# print(f"[DEBUG] After final FC: {x.shape}")
		if self.training:
			projected_features = self.projection_head(features_512)
			return logits, features_512, projected_features
		else:
			return logits