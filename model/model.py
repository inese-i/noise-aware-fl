import torch
from torch import nn
import torch.nn.functional as F
from torchvision import models

class Net(nn.Module):
    def __init__(self) -> None:
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class CIFAR100Net(nn.Module):
    def __init__(self) -> None:
        super(CIFAR100Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.fc1 = nn.Linear(128 * 8 * 8, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 100)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv1(x))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def get_densenet121():
    # Load the pre-trained DenseNet-121 model
    model = models.densenet121(pretrained=True)
    
    # Modify the classifier for CIFAR-100 (100 classes)
    num_features = model.classifier.in_features
    model.classifier = nn.Linear(num_features, 100)
    
    return model

def get_efficientnet_b1():
    # Load the pre-trained EfficientNet-B1 model
    model = models.efficientnet_b1(pretrained=True)

    # Modify the classifier for Tiny ImageNet (200 classes)
    num_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(num_features, 200)

    return model

