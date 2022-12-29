import os
import numpy as np
import gzip
import torch
import torchvision
from torchvision import datasets, transforms


train_dataset = torchvision.datasets.FashionMNIST(root='../datasets/fmnist',
                                                  train=True,
                                                  transform=transforms.ToTensor(),
                                                  download=False)

test_dataset = torchvision.datasets.FashionMNIST(root='../datasets/fmnist',
                                                 train=False,
                                                 transform=transforms.ToTensor(),
                                                 download=False)


