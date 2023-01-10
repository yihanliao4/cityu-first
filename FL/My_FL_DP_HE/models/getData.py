import os
import numpy as np
import torch
import torchvision
from tensorflow import keras
from torchvision import datasets, transforms
from keras.utils import np_utils


class GetDataset(object):

    def __init__(self, dataset, is_iid):
        self.dataset = dataset
        self.is_iid = is_iid

        self.train_images = None
        self.train_labels = None
        self.test_images = None
        self.test_labels = None
        self.train_size = None
        self.test_size = None

        if self.dataset == 'fmnist':
            self.load_fashion_mnist_data(is_iid)
        elif self.dataset == 'mnist':
            self.load_mnist_data(is_iid)

    def load_mnist_data(self, is_iid):
        mnist = keras.datasets.mnist
        (train_images, train_labels), (test_images, test_labels) = mnist.load_data()
        train_images, test_images = train_images.astype(np.float32), test_images.astype(np.float32)
        train_images, test_images = train_images / 255.0, test_images / 255.0
        train_labels, test_labels = np_utils.to_categorical(train_labels), np_utils.to_categorical(test_labels)

        train_images = train_images.reshape(train_images.shape[0], train_images.shape[1] * train_images.shape[2])
        test_images = test_images.reshape(test_images.shape[0], test_images.shape[1] * test_images.shape[2])

        self.train_size = train_images.shape[0]

        if is_iid:
            self.iid(train_images, train_labels)

        else:
            self.non_iid(train_images, train_labels)

        self.test_images = test_images
        self.test_labels = test_labels

    def load_fashion_mnist_data(self, is_iid):
        fashion_mnist = keras.datasets.fashion_mnist
        (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

        train_images, test_images = train_images.astype(np.float32), test_images.astype(np.float32)
        # Shrink the values to float numbers between 0 and 1
        train_images, test_images = np.multiply(train_images, 1.0 / 255.0), np.multiply(test_images, 1.0 / 255.0)
        # train_labels, test_labels = train_labels.astype(np.int32), test_labels.astype(np.int32)

        self.train_size = train_images.shape[0]

        if is_iid:
            self.iid(train_images, train_labels)

        else:
            self.non_iid(train_images, train_labels)

        self.test_images = test_images
        self.test_labels = test_labels

    def iid(self, train_images, train_labels):
        order = np.arange(train_images.shape[0])
        np.random.shuffle(order)
        train_images = train_images[order]
        train_labels = train_labels[order]

        self.train_images = train_images
        self.train_labels = train_labels

    def non_iid(self, train_images, train_labels):
        labels = np.argmax(train_labels, axis=1)
        order = np.argsort(labels)
        train_images = train_images[order]
        train_labels = train_labels[order]

        self.train_images = train_images
        self.train_labels = train_labels


# if __name__ == "__main__":
#     data = GetDataset('mnist', 1)
#     print(data.train_labels)



