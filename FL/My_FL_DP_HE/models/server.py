import os
import argparse, json
import time
import numpy as np
import torch
import torch.nn.functional as functional
from torch import optim
from sklearn.metrics import accuracy_score
import tensorflow as tf
from tensorflow import keras
from tqdm import tqdm
import pysnooper
from functools import reduce
from joblib import Parallel, delayed
from utils.configure import args_parser
from model import Mnist_2NN, Mnist_CNN
from client import Client, ClientGroup
import matplotlib.pyplot as plt
from keras.utils import np_utils


tf.compat.v1.enable_eager_execution()

print(tf.__version__)
print(tf.executing_eagerly())
print(torch.cuda.is_available())


# def test_mkdir(path):
#     if not os.path.isdir(path):
#         os.mkdir(path)


def add_noise(parameters, dp, sigma, dev):
    noise = None
    if dp == 0:
        return parameters
    elif dp == 1:
        noise = torch.tensor(np.random.laplace(0, sigma, parameters.shape)).to(dev)
    else:
        noise = torch.cuda.FloatTensor(parameters.shape).normal_(0, sigma)

    return parameters.add_(noise)


def dense_to_one_hot(labels_dense, num_classes=10):
    num_labels = labels_dense.shape[0]
    index_offset = np.arange(num_labels) * num_classes
    labels_one_hot = np.zeros((num_labels, num_classes))
    labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
    return labels_one_hot




if __name__ == "__main__":

    args = args_parser()

    # args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')
    # print(torch.cuda.is_available())
    # print(torch.cuda.device_count())
    # dev = args.device

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    dev = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # build model
    net_global = None
    # if args.model == '2nn' and args.dataset == 'mnist':
    #     net_global = Mnist_2NN()
    # elif args.model == 'cnn' and args.dataset == 'mnist':
    #     net_global = Mnist_CNN()
    # else:
    #     exit('Error: unrecognized model')
    # print('net_global_first: {}'.format(net_global))

    if torch.cuda.device_count() > 1 or torch.cuda.device_count() == 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        net = torch.nn.DataParallel(net_global)
    net = net.to(dev)
    # print('net: {}'.format(net))

    loss_func = functional.cross_entropy
    optim = optim.Adam(net.parameters(), lr=args.learning_rate)

    # test_mkdir(args['save_path'])
    myClients = ClientGroup(args.dataset, args.iid, args.num_of_clients, args.batch_size, dev)
    trainDataLoader = myClients.train_dataloader
    testDataLoader = myClients.test_dataloader
    data_size = myClients.img_size

    client_num_in_comm = int(max(args.num_of_clients * args.frac, 1))

    global_parameters = {}
    for key, var in net.state_dict().items():
        # print('key: {}'.format(key))
        # print('var: '.format(var[0]))
        global_parameters[key] = var.clone()
        # print('len after: {}'.format(len(global_parameters)))
    # print('global_parameters_first: {}'.format(global_parameters))

    dp = args.noise
    sigma = args.sigma

    train_acc = []
    test_acc = []
    train_loss = []

    net_best = None
    best_loss = None
    net_list = []

    for i in range(args.num_of_comm):
        print('Communicate round {}'.format(i + 1))

        local_loss = []

        # clients_in_comm = np.random.choice(range(args.num_of_clients), client_num_in_comm, replace=False)

        order = np.random.permutation(args.num_of_clients)
        clients_in_comm = ['client{}'.format(i) for i in order[0:client_num_in_comm]]
        # print('number of clients in one communication: {}'.format(clients_in_comm))

        sum_parameters = None

        for client in tqdm(clients_in_comm):
            # print(myClients.dict_clients[client])
            # print(client)
            local_parameters, epoch_loss = myClients.dict_clients[client].localUpdate(
                args.local_epoch, args.local_batchsize, net, loss_func, optim, global_parameters)
            # print('local_param: {}'.format(local_parameters))
            # print('Loss of client: {}'.format(epoch_loss))

            # print('sum_param: '.format(sum_parameters))
            # print('local_param: '.format(local_parameters))
            # print('local_len: '.format(len(local_parameters)))

            if sum_parameters is None:

                sum_parameters = {}

                for key, var in local_parameters.items():
                    sum_parameters[key] = var
                    # print('test: {}'.format(sum_parameters))

            # if sum_parameters is None:
            #
            #     sum_parameters = {}
            #
            #     # print('sum_len: {}'.format(len(sum_parameters)))
            #
            #     for key, var in local_parameters.items():
            #
            #         print('local_len: {}'.format(len(local_parameters)))
            #         print('key: {}'.format(key))
            #         print('var: {}'.format(var))
            #         print('sum_len: {}'.format(len(sum_parameters)))
            #
            #         sum_parameters[key] = var
            #
            #         print('sum_param: {}'.format(sum_parameters))
            #         # print('local_len: {}'.format(len(local_parameters)))
            #         print('sum_parameters[key]: {}'.format(sum_parameters[key]))
            #         print('sum_len: {}'.format(len(sum_parameters)))
            #
            #         sum_parameters = add_noise(sum_parameters[key], dp, sigma, dev)

            else:
                for key in sum_parameters:
                    sum_parameters[key].add_(add_noise(local_parameters[key], dp, sigma, dev))

            # local_loss.append(epoch_loss)

        for var in global_parameters:
            global_parameters[var] = (sum_parameters[var] / client_num_in_comm)

        with torch.no_grad():
            net.load_state_dict(global_parameters, strict=True)
            num = 0
            sum_acc = 0

            for data, label in testDataLoader:
                data, label = data.to(dev), label.to(dev)
                predict = net(data)
                predict = torch.argmax(predict, dim=1)
                sum_acc += (predict == label).float().mean()
                num += 1
            print('test accuracy: {}'.format(sum_acc / num))
            test_acc.append((sum_acc / num).cpu())

            sum_loss = 0
            for data, label in trainDataLoader:
                data, label = data.to(dev), label.to(dev)
                predict = net(data)
                # true_label = dense_to_one_hot(label.cpu().numpy())
                true_label = np_utils.to_categorical(label.cpu().numpy())
                true_label = torch.tensor(true_label).to(dev)
                sum_loss += functional.cross_entropy(predict, true_label, reduction='sum').item()

                predict = torch.argmax(predict, dim=1)
                sum_acc += (predict == label).float().mean()
                num += 1
            print('train accuracy: {}'.format(sum_acc / num))
            print('train loss: {}'.format(sum_loss / num))
            train_loss.append(sum_loss / num)
            train_acc.append(sum_acc / num)

    # plot loss curve
    plt.figure()
    plt.plot(range(len(train_loss)), train_loss)
    plt.ylabel('train_loss')
    plt.savefig(
        './save/fed_{}_model{}_E{}_B{}_lr{}_C{}_iid{}.png'.format(args.dataset, args.model, args.local_epoch,
                                                                  args.local_batchsize, args.learning_rate,
                                                                  args.num_of_clients, args.iid))
















