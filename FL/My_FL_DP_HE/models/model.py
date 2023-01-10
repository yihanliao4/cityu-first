import numpy as np
import torch
from torch import nn
import torch.nn.functional as functional


# class MLP(nn.Module):
#     def __init__(self, dim_in, dim_hidden, dim_out):
#         super(MLP, self).__init__()
#
#         self.layer_input = nn.Linear(dim_in, dim_out)
#         self.relu = nn.ReLU()
#         self.dropout = nn.Dropout()
#         self.layer_hidden = nn.Linear(dim_hidden, dim_out)
#
#     def forward(self, x):
#         x = x.view(-1, x.shape[1] * x.shape[-2] * x.shape[-1])
#
#         x = self.layer_input(x)
#         x = self.dropout(x)
#         x = self.relu(x)
#         x = self.layer_hidden(x)
#
#         return x


class Mnist_2NN(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 200)
        self.fc2 = nn.Linear(200, 200)
        self.fc3 = nn.Linear(200, 10)

    def forward(self, inputs):
        tensor = functional.relu(self.fc1(inputs))
        tensor = functional.relu(self.fc2(tensor))
        tensor = self.fc3(tensor)

        return tensor


class Mnist_CNN(nn.Module):
    def __init__(self):
        super(Mnist_CNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=0)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=0)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=0)
        self.fc1 = nn.Linear(3 * 3 * 64, 64)
        self.fc2 = nn.Linear(64, 10)

    def forward(self, x):
        x = x.view(-1, 1, 28, 28)
        x = functional.relu(self.conv1(x))
        x = self.pool1(x)
        x = functional.relu(self.conv2(x))
        x = self.pool2(x)
        x = functional.relu(self.conv3(x))
        x = x.view(-1, 3*3*64)
        x = functional.relu(self.fc1(x))
        x = self.fc2(x)

        return x
