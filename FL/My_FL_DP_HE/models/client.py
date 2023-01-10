import numpy as np
import torch
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
from getData import GetDataset


class Client(object):

    def __init__(self, train_dataset, dev):
        self.train_dataset = train_dataset
        self.dev = dev
        self.train_loader = None
        self.local_parameter = None

    def localUpdate(self, local_epoch, local_batchsize, net, loss_func, optimiser, global_parameters):
        net.load_state_dict(global_parameters, strict=True)
        # print('global_param: {}'.format(global_parameters))
        self.train_loader = DataLoader(self.train_dataset, batch_size=local_batchsize, shuffle=True)
        epoch_loss = []

        for epoch in range(local_epoch):
            batch_loss = []
            for data, label in self.train_loader:
                data, label = data.to(self.dev), label.to(self.dev)
                predict = net(data)
                loss = loss_func(predict, label)
                loss.backward()
                optimiser.step()
                optimiser.zero_grad()

                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss) / len(batch_loss))

        # print('local_param: {}'.format(net.state_dict()))
        # print('local_len: {}'.format(len(net.state_dict())))

        return net.state_dict(), epoch_loss


class ClientGroup(object):

    def __init__(self, dataset, is_iid, num_of_clients, batch_size, dev):

        self.dataset = dataset
        self.is_iid = is_iid
        self.num_of_clients = num_of_clients
        self.batch_size = batch_size
        self.dev = dev
        # The clients list in each communication
        self.dict_clients = {}
        self.train_dataloader = None
        self.test_dataloader = None
        self.img_size = None
        # Allocate data to clients
        self.datasetAllocation()

    def datasetAllocation(self):
        dataset = GetDataset(self.dataset, self.is_iid)
        self.img_size = dataset.train_images[0][0].shape

        test_images = torch.tensor(dataset.test_images)
        test_labels = torch.argmax(torch.tensor(dataset.test_labels), dim=1)
        self.test_dataloader = DataLoader(TensorDataset(test_images, test_labels),
                                          batch_size=self.batch_size, shuffle=False)

        train_images = dataset.train_images
        train_labels = dataset.train_labels
        self.train_dataloader = DataLoader(
            TensorDataset(torch.tensor(train_images), torch.argmax(torch.tensor(train_labels), dim=1)),
            batch_size=self.batch_size, shuffle=False)

        shard_size = dataset.train_size // self.num_of_clients // 2
        shard_index = np.random.permutation(dataset.train_size // shard_size)

        for i in range(self.num_of_clients):
            rand_set = np.random.choice(shard_index, 2, replace=False)

            image_part1 = train_images[rand_set[0] * shard_size:rand_set[0] * shard_size + shard_size]
            image_part2 = train_images[rand_set[1] * shard_size:rand_set[1] * shard_size + shard_size]
            label_part1 = train_labels[rand_set[0] * shard_size:rand_set[0] * shard_size + shard_size]
            label_part2 = train_labels[rand_set[1] * shard_size:rand_set[1] * shard_size + shard_size]

            # shards_id1 = shard_index[i * 2]
            # shards_id2 = shard_index[i * 2 + 1]
            # image_part1 = train_images[shards_id1 * shard_size: shards_id1 * shard_size + shard_size]
            # image_part2 = train_images[shards_id2 * shard_size: shards_id2 * shard_size + shard_size]
            # label_part1 = train_labels[shards_id1 * shard_size: shards_id1 * shard_size + shard_size]
            # label_part2 = train_labels[shards_id2 * shard_size: shards_id2 * shard_size + shard_size]

            local_images, local_labels = np.vstack((image_part1, image_part2)), np.vstack((label_part1, label_part2))
            local_labels = np.argmax(local_labels, axis=1)

            client = Client(TensorDataset(torch.tensor(local_images), torch.tensor(local_labels)), self.dev)
            self.dict_clients['client{}'.format(i)] = client


# if __name__=="__main__":
#     MyClients = ClientGroup('mnist', True, 100, 100, 1)
#     print(MyClients.dict_clients['client10'].train_dataset[0:100])
#     print(MyClients.dict_clients['client11'].train_dataset[400:500])

        # Split dataset
        # if self.is_iid == True:
        #     """
        #     Sample I.I.D clients data from dataset
        #     :param dataset:
        #     :param num_of_clients:
        #     :return dict of users
        #     """
        #     num_items = int(len(dataset.train_images) / self.num_of_clients)
        #     dict_users, all_index = {}, [i for i in range(len(dataset.train_images))]
        #     for i in range(self.num_of_clients):
        #         dict_users[i] = set(np.random.choice(all_index, num_items, replace=False))
        #         all_index = list(set(all_index) - dict_users[i])

        # elif self.is_iid == False:
        #     num_shards, num_shards_size = 200, 300
        #     shards_index = [i for i in range(num_shards)]
        #     dict_users = {i: np.array([], dtype='int64') for i in range(self.num_of_clients)}
        #
        #     shards_set = np.arange(num_shards * num_shards_size)
        #
        #     labels = dataset.train_labels
        #
        #     # sort labels
        #     shards_set_labels = np.vstack((shards_set, labels[1]))
        #     shards_set_labels = shards_set_labels[:, shards_set_labels[1, :].argsort()]
        #     shards_set = shards_set_labels[0, :]
        #
        #     # divide and assign
        #     for i in range(self.num_of_clients):
        #         rand_set = set(np.random.choice(shards_index, 2, replace=False))
        #         shards_index = list(set(shards_index) - rand_set)
        #         for rand in rand_set:
        #             dict_users[i] = np.concatenate((
        #                 dict_users[i], shards_set[rand * num_shards_size : (rand + 1) * num_shards_size]), axis=0)

        # print(dict_users[i])












