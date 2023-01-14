import torch.utils.data as data
from PIL import Image
import numpy as np
import torchvision
from sklearn.preprocessing import MinMaxScaler
from torch.distributions import Dirichlet
from torchvision.datasets import MNIST, EMNIST, CIFAR10, CIFAR100, SVHN, FashionMNIST, ImageFolder, DatasetFolder, utils
import os
import logging
import numpy as np
import torch
import torchvision.transforms as transforms
import torch.utils.data as data
from torch.autograd import Variable
import torch.nn.functional as F
import torch.nn as nn
import random
# from sklearn.metrics import confusion_matrix
import os
import os.path
import logging
import pandas as pd
import json

from options import features_dict, args, series_dict

logging.basicConfig()
logger = logging.getLogger()
logger.setLevel(logging.INFO)

IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', '.webp')

scalers = [MinMaxScaler() for i in range(series_dict[args.dataset])]


def mkdirs(dirpath):
    try:
        os.makedirs(dirpath)
    except Exception as _:
        pass

def generate_data_nums(num_client, num_data, beta=0.5):
    while True:
        data_num_each_client = Dirichlet(torch.tensor([beta] * num_client)).sample()
        data_num_each_client = torch.floor(num_data * data_num_each_client)
        data_num_each_client[-1] = num_data - torch.sum(data_num_each_client[:-1])
        if not (0 in data_num_each_client):
            break
    return data_num_each_client

class CIFAR10_truncated(data.Dataset):

    def __init__(self, root, dataidxs=None, train=True, transform=None, target_transform=None, download=False):

        self.root = root
        self.dataidxs = dataidxs
        self.train = train
        self.transform = transform
        self.target_transform = target_transform
        self.download = download

        self.data, self.target = self.__build_truncated_dataset__()

    def __build_truncated_dataset__(self):

        cifar_dataobj = CIFAR10(self.root, self.train, self.transform, self.target_transform, self.download)

        if torchvision.__version__ == '0.2.1':
            if self.train:
                data, target = cifar_dataobj.train_data, np.array(cifar_dataobj.train_labels)
            else:
                data, target = cifar_dataobj.test_data, np.array(cifar_dataobj.test_labels)
        else:
            data = cifar_dataobj.data
            target = np.array(cifar_dataobj.targets)

        if self.dataidxs is not None:
            data = data[self.dataidxs]
            target = target[self.dataidxs]

        return data, target

    def truncate_channel(self, index):
        for i in range(index.shape[0]):
            gs_index = index[i]
            self.data[gs_index, :, :, 1] = 0.0
            self.data[gs_index, :, :, 2] = 0.0

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.target[index]

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.data)


class CIFAR100_truncated(data.Dataset):

    def __init__(self, root, dataidxs=None, train=True, transform=None, target_transform=None, download=False):

        self.root = root
        self.dataidxs = dataidxs
        self.train = train
        self.transform = transform
        self.target_transform = target_transform
        self.download = download

        self.data, self.target = self.__build_truncated_dataset__()

    def __build_truncated_dataset__(self):

        cifar_dataobj = CIFAR100(self.root, self.train, self.transform, self.target_transform, self.download)

        if torchvision.__version__ == '0.2.1':
            if self.train:
                data, target = cifar_dataobj.train_data, np.array(cifar_dataobj.train_labels)
            else:
                data, target = cifar_dataobj.test_data, np.array(cifar_dataobj.test_labels)
        else:
            data = cifar_dataobj.data
            target = np.array(cifar_dataobj.targets)

        if self.dataidxs is not None:
            data = data[self.dataidxs]
            target = target[self.dataidxs]

        return data, target

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.target[index]
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.data)


class SMD_truncated(data.Dataset):

    def __init__(self, root, dataidxs=None, train=True, transform=None, target_transform=None, download=False, window_len=args.slide_win):

        self.root = root
        self.dataidxs = dataidxs
        self.train = train
        self.transform = transform
        self.target_transform = target_transform
        self.download = download
        if args.tsadalg == 'deep_svdd':
            self.window_len = 1
        else:
            self.window_len = window_len
        global scalers
        self.scalers = scalers

        self.data, self.target = self.__build_truncated_dataset__()

    def __build_truncated_dataset__(self):

        current_path = os.getcwd()

        if self.train:
            train_path = current_path + '/data/datasets/smd/SMD/raw/train'
            file_names = os.listdir(train_path)
            file_names.sort()
            data = []
            data_all = []
            data_starts = []
            for i in range(len(file_names)):
                file_name = file_names[i]
                with open(train_path + '/' + file_name) as f:
                    this_data = pd.read_csv(train_path + '/' + file_name, header=None)
                    this_data = this_data.values.astype(np.float32)
                    if len(data_starts) == 0:
                        data_starts.append(0)
                    else:
                        data_starts.append(data_starts[-1] + this_data.shape[0])
                    data_all.append(this_data)
            data_all = np.concatenate(data_all, axis=0)
            data_all = self.scalers[0].fit_transform(data_all)
            for i in range(len(data_starts)):
                if i != len(data_starts) - 1:
                    data.append(data_all[data_starts[i]: data_starts[i + 1]])
                else:
                    data.append(data_all[data_starts[-1]:])

            target = data.copy()
        else:
            test_path = current_path + '/data/datasets/smd/SMD/raw/test'
            file_names = os.listdir(test_path)
            file_names.sort()
            data = []
            data_all = []
            data_starts = []
            for i in range(len(file_names)):
                file_name = file_names[i]
                with open(test_path + '/' + file_name) as f:
                    this_data = pd.read_csv(test_path + '/' + file_name, header=None)
                    this_data = this_data.values.astype(np.float32)
                    if len(data_starts) == 0:
                        data_starts.append(0)
                    else:
                        data_starts.append(data_starts[-1] + this_data.shape[0])
                    data_all.append(this_data)
            data_all = np.concatenate(data_all, axis=0)
            data_all = self.scalers[0].transform(data_all)
            data = data_all
            test_target_path = current_path + '/data/datasets/smd/SMD/raw/test_label'
            file_names = os.listdir(test_target_path)
            file_names.sort()
            target = []
            for file_name in file_names:
                with open(test_target_path + '/' + file_name) as f:
                    this_target = []
                    for line in f.readlines():
                        this_target.append(line.split(','))
                    this_target = np.asarray(this_target)
                    this_target = this_target.astype(np.float32)
                target.append(this_target)
            target = np.concatenate(target, axis=0)

        if self.dataidxs is not None:
            data = data[self.dataidxs]
            target = target[self.dataidxs]

        return data, target

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        if index + 1 - self.window_len < 0:
            data = self.data[0: index + 1]
            delta = self.window_len - data.shape[0]
            data0 = data[0]
            if len(data[0].shape) == 1:
                data0 = data0[np.newaxis, :]
            data0 = np.repeat(data0, delta, axis=0)
            data = np.concatenate((data0, data), axis=0)
        else:
            data = self.data[index + 1 - self.window_len: index + 1]
        target = self.target[index]

        return data, target

    def __len__(self):
        return len(self.data)

class SMAP_truncated(data.Dataset):

    def __init__(self, root, dataidxs=None, train=True, transform=None, target_transform=None, download=False, window_len=args.slide_win):

        self.root = root
        self.dataidxs = dataidxs
        self.train = train
        self.transform = transform
        self.target_transform = target_transform
        self.download = download
        if args.tsadalg == 'deep_svdd':
            self.window_len = 1
        else:
            self.window_len = window_len
        global scalers
        self.scalers = scalers

        self.data, self.target = self.__build_truncated_dataset__()

    def __build_truncated_dataset__(self):

        current_path = os.getcwd()

        if self.train:
            train_path = current_path + '/data/datasets/smap/raw/train'
            file_names = os.listdir(train_path)
            file_names.sort()
            data = []
            data_all = []
            data_starts = []
            for i in range(len(file_names)):
                file_name = file_names[i]
                this_data = np.load(train_path + '/' + file_name)
                this_data = this_data.astype(np.float32)
                if len(data_starts) == 0:
                    data_starts.append(0)
                else:
                    data_starts.append(data_starts[-1] + this_data.shape[0])
                data_all.append(this_data)
            data_all = np.concatenate(data_all, axis=0)
            data_all = self.scalers[0].fit_transform(data_all)
            for i in range(len(data_starts)):
                if i != len(data_starts) - 1:
                    data.append(data_all[data_starts[i]: data_starts[i + 1]])
                else:
                    data.append(data_all[data_starts[-1]:])
            target = data.copy()
        else:
            test_path = current_path + '/data/datasets/smap/raw/test'
            file_names = os.listdir(test_path)
            file_names.sort()
            data = []
            data_all = []
            data_starts = []
            for i in range(len(file_names)):
                file_name = file_names[i]
                this_data = np.load(test_path + '/' + file_name)
                this_data = this_data.astype(np.float32)
                if len(data_starts) == 0:
                    data_starts.append(0)
                else:
                    data_starts.append(data_starts[-1] + this_data.shape[0])
                data_all.append(this_data)
            data_all = np.concatenate(data_all, axis=0)
            data_all = self.scalers[0].transform(data_all)
            data = data_all
            test_target_path = current_path + '/data/datasets/smap/raw/test_label'
            file_names = os.listdir(test_target_path)
            file_names.sort()
            target = []
            for file_name in file_names:
                with open(test_target_path + '/' + file_name) as f:
                    this_target = []
                    for line in f.readlines():
                        this_target.append(int(line.replace('\n', '')))
                    this_target = np.asarray(this_target)
                    this_target = this_target.astype(np.float32)
                target.append(this_target)
            target = np.concatenate(target, axis=0)

        if self.dataidxs  is not None:
            data = data[self.dataidxs]
            target = target[self.dataidxs]

        return data, target

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        if index + 1 - self.window_len < 0:
            data = self.data[0: index + 1]
            delta = self.window_len - data.shape[0]
            data0 = data[0]
            if len(data0.shape) == 1:
                data0 = data0[np.newaxis, :]
            data0 = np.repeat(data0, delta, axis=0)
            data = np.concatenate((data0, data), axis=0)
        else:
            data = self.data[index + 1 - self.window_len: index + 1]
        target = self.target[index]

        return data, target

    def __len__(self):
        return len(self.data)

class PSM_truncated(data.Dataset):

    def __init__(self, root, dataidxs=None, train=True, transform=None, target_transform=None, download=False, window_len=args.slide_win):

        self.root = root
        self.dataidxs = dataidxs
        self.train = train
        self.transform = transform
        self.target_transform = target_transform
        self.download = download
        if args.tsadalg == 'deep_svdd':
            self.window_len = 1
        else:
            self.window_len = window_len
        global scalers
        self.scalers = scalers

        self.data, self.target = self.__build_truncated_dataset__()

    def __build_truncated_dataset__(self):

        current_path = os.getcwd()

        if self.train:
            train_path = current_path + '/data/datasets/psm/raw'
            data = []
            this_data = pd.read_csv(train_path + '/train.csv')
            this_data.drop(columns=[r'timestamp_(min)'], inplace=True)
            data_length = this_data.values.shape[0]
            if args.beta >= 10000:
                each_length = data_length // args.num_clients
                this_data_values = this_data.values.astype(np.float32)
                this_data_values = np.nan_to_num(this_data_values)
                this_data_values = self.scalers[0].fit_transform(this_data_values)
                for i in range(args.num_clients):
                    data.append(this_data_values[i * each_length: (i + 1) * each_length])
            else:
                lengths = generate_data_nums(num_client=args.num_clients, num_data=data_length, beta=args.beta)
                lengths = lengths.detach().cpu().numpy()
                lengths = lengths.astype(np.int)
                lengths = lengths.tolist()
                start = 0
                this_data_values = this_data.values.astype(np.float32)
                this_data_values = np.nan_to_num(this_data_values)
                this_data_values = self.scalers[0].fit_transform(this_data_values)
                for li in range(len(lengths)):
                    l = lengths[li]
                    # if start + l <= this_data_values.shape[0] - 1:
                    if start + l <= this_data_values.shape[0]:
                        data.append(this_data_values[start: start + l])
                    else:
                        data.append(this_data_values[start:])
                    start += l
            target = data.copy()
        else:
            test_path = current_path + '/data/datasets/psm/raw'
            this_data = pd.read_csv(test_path + '/test.csv')
            this_data.drop(columns=[r'timestamp_(min)'], inplace=True)
            data = this_data.values
            data = data.astype(np.float32)
            data = np.nan_to_num(data)
            data = self.scalers[0].transform(data)
            data_length = this_data.values.shape[0]
            test_target_path = current_path + '/data/datasets/psm/raw/test_label.csv'
            # print(test_target_path)
            target_csv = pd.read_csv(test_target_path)
            target_csv.drop(columns=[r'timestamp_(min)'], inplace=True)
            target = target_csv.values
            target = target.astype(np.float32)

        if self.dataidxs is not None:
            data = data[self.dataidxs]
            target = target[self.dataidxs]

        return data, target

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        if index + 1 - self.window_len < 0:
            data = self.data[0: index + 1]
            delta = self.window_len - data.shape[0]
            data0 = np.repeat(data[0][np.newaxis, :], delta, axis=0)
            data = np.concatenate((data0, data), axis=0)
        else:
            data = self.data[index + 1 - self.window_len: index + 1]
        target = self.target[index]

        return data, target

    def __len__(self):
        return len(self.data)


def load_cifar10_data(datadir):
    cifar10_train_ds = CIFAR10_truncated(datadir, train=True, download=True)
    cifar10_test_ds = CIFAR10_truncated(datadir, train=False, download=True)

    X_train, y_train = cifar10_train_ds.data, cifar10_train_ds.target
    X_test, y_test = cifar10_test_ds.data, cifar10_test_ds.target
    return X_train, y_train, X_test, y_test


def load_cifar100_data(datadir):
    cifar100_train_ds = CIFAR100_truncated(datadir, train=True, download=True)
    cifar100_test_ds = CIFAR100_truncated(datadir, train=False, download=True)

    X_train, y_train = cifar100_train_ds.data, cifar100_train_ds.target
    X_test, y_test = cifar100_test_ds.data, cifar100_test_ds.target
    return X_train, y_train, X_test, y_test

def load_smd_data(datadir):
    smd_train_ds = SMD_truncated(datadir, train=True, download=True)
    smd_test_ds = SMD_truncated(datadir, train=False, download=True)

    X_train, y_train = smd_train_ds.data, smd_train_ds.target
    X_test, y_test = smd_test_ds.data, smd_test_ds.target
    return X_train, y_train, X_test, y_test

def load_smap_data(datadir):
    smap_train_ds = SMAP_truncated(datadir, train=True, download=True)
    smap_test_ds = SMAP_truncated(datadir, train=False, download=True)

    X_train, y_train = smap_train_ds.data, smap_train_ds.target
    X_test, y_test = smap_test_ds.data, smap_test_ds.target
    return X_train, y_train, X_test, y_test


def load_psm_data(datadir):
    psm_train_ds = PSM_truncated(datadir, train=True, download=True)
    psm_test_ds = PSM_truncated(datadir, train=False, download=True)

    X_train, y_train = psm_train_ds.data, psm_train_ds.target
    X_test, y_test = psm_test_ds.data, psm_test_ds.target
    return X_train, y_train, X_test, y_test


def partition_data(dataset, datadir, partition, n_parties, beta=0.5):
    if dataset == 'smd':
        X_train, y_train, X_test, y_test = load_smd_data(datadir)
    elif dataset == 'smap':
        X_train, y_train, X_test, y_test = load_smap_data(datadir)
    elif dataset == 'psm':
        X_train, y_train, X_test, y_test = load_psm_data(datadir)

    n_train = y_train.shape[0]

    if partition == "iid":
        idxs = np.random.permutation(n_train)
        batch_idxs = np.array_split(idxs, n_parties)
        net_dataidx_map = {i: batch_idxs[i] for i in range(n_parties)}


    elif partition == "noniid":
        min_size = 0
        min_require_size = 10
        K = 10

        N = y_train.shape[0]
        net_dataidx_map = {}

        while min_size < min_require_size:
            idx_batch = [[] for _ in range(n_parties)]
            for k in range(K):
                idx_k = np.where(y_train == k)[0]
                np.random.shuffle(idx_k)
                proportions = np.random.dirichlet(np.repeat(beta, n_parties))
                proportions = np.array([p * (len(idx_j) < N / n_parties) for p, idx_j in zip(proportions, idx_batch)])
                proportions = proportions / proportions.sum()
                proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
                idx_batch = [idx_j + idx.tolist() for idx_j, idx in zip(idx_batch, np.split(idx_k, proportions))]
                min_size = min([len(idx_j) for idx_j in idx_batch])
                # if K == 2 and n_parties <= 10:
                #     if np.min(proportions) < 200:
                #         min_size = 0
                #         break

        for j in range(n_parties):
            np.random.shuffle(idx_batch[j])
            net_dataidx_map[j] = idx_batch[j]

    # traindata_cls_counts = record_net_data_stats(y_train, net_dataidx_map, logdir)
    return net_dataidx_map


def get_dataset(dataset, datadir, dataidxs=None, noise_level=0):
    if dataset in ('smd', 'smap', "psm"):
        if dataset == 'smd':
            dl_obj = SMD_truncated
        elif dataset == 'smap':
            dl_obj = SMAP_truncated
        elif dataset == 'psm':
            dl_obj = PSM_truncated

        train_ds = dl_obj(datadir, dataidxs=dataidxs, train=True, transform=None, download=True)
        test_ds = dl_obj(datadir, train=False, transform=None, download=True)

    return train_ds, test_ds


if __name__ == '__main__':
    X_train, y_train, X_test, y_test, net_dataidx_map, traindata_cls_counts = partition_data(
        'cifar10',
        "../data",
        "../",
        "noniid",
        10
    )
    print(1)
