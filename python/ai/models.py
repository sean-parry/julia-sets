import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt

class LinearModel(nn.Module):
    def __init__(self, width, height):
        super(LinearModel, self).__init__()
        self.layer1 = nn.Linear(2,256)
        self.layer2 = nn.Linear(256, 1024)
        self.layer3 = nn.Linear(1024, width*height)
        self.width = width
        self.height = height
        self.activation = nn.ReLU()

    def forward(self, x):
        x = self.layer1(x)
        x = self.activation(x)
        x = self.layer2(x)
        x = self.activation(x)
        x = self.layer3(x)
        x = x.view(-1, self.width, self.height)  # Reshape to match the desired output size
        return x
    

class CNNModel(nn.Module):
    def __init__(self, width, height):
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=2, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=256, out_channels=1024, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(in_channels=1024, out_channels=width*height, kernel_size=3, stride=1, padding=1)
        self.width = width
        self.height = height
        self.activation = nn.ReLU()

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.conv1(x)
        x = self.activation(x)
        x = self.conv2(x)
        x = self.activation(x)
        x = self.conv3(x)
        x = self.activation(x)
        x = x.view(-1, self.width, self.height)  # Reshape to match the desired output size
        return x