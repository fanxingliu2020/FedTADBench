import pandas as pd
import torch.utils.data as data
import os
import numpy as np
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler(feature_range=(0, 1))

class SMD_Dataset(data.Dataset):

    def __init__(self, dataidxs=None, train=True, transform=None, target_transform=None, download=False, window_len=5):

        self.dataidxs = dataidxs
        self.train = train
        self.transform = transform
        self.target_transform = target_transform
        self.download = download
        self.window_len = window_len
        global scaler
        self.scaler = scaler

        self.data, self.target = self.__build_truncated_dataset__()

    def __build_truncated_dataset__(self):

        current_path = os.path.abspath(os.path.dirname(os.path.abspath(os.path.dirname(os.getcwd()))))

        if self.train:
            train_path = current_path + '/data/datasets/smd/SMD/raw/train'
            file_names = os.listdir(train_path)
            file_names.sort()
            data = []
            for i in range(len(file_names)):
                file_name = file_names[i]
                with open(train_path + '/' + file_name) as f:
                    this_data = pd.read_csv(train_path + '/' + file_name, header=None)
                    this_data = this_data.values.astype(np.float32)
                    data.append(this_data)
            data = np.concatenate(data, axis=0)
            data = self.scaler.fit_transform(data)
            target = data.copy()
        else:
            test_path = current_path + '/data/datasets/smd/SMD/raw/test'
            file_names = os.listdir(test_path)
            file_names.sort()
            data = []
            for file_name in file_names:
                with open(test_path + '/' + file_name) as f:
                    this_data = []
                    for line in f.readlines():
                        this_data.append(line.split(','))
                    this_data = np.asarray(this_data)
                    this_data = this_data.astype(np.float32)
                data.append(this_data)
            data = np.concatenate(data, axis=0)
            data = self.scaler.transform(data)
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

        if self.dataidxs:
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
            # print(data0.shape, data.shape)
            data = np.concatenate((data0, data), axis=0)
        else:
            data = self.data[index + 1 - self.window_len: index + 1]
        target = self.target[index]

        return data, target

    def __len__(self):
        return self.data.shape[0]

class SMAP_Dataset(data.Dataset):

    def __init__(self, dataidxs=None, train=True, transform=None, target_transform=None, download=False, window_len=5):

        self.dataidxs = dataidxs
        self.train = train
        self.transform = transform
        self.target_transform = target_transform
        self.download = download
        self.window_len = window_len
        global scaler
        self.scaler = scaler

        self.data, self.target = self.__build_truncated_dataset__()

    def __build_truncated_dataset__(self):

        current_path = os.path.abspath(os.path.dirname(os.path.abspath(os.path.dirname(os.getcwd()))))

        if self.train:
            train_path = current_path + '/data/datasets/smap/raw/train'
            file_names = os.listdir(train_path)
            file_names.sort()
            data = []
            for file_name in file_names:
                this_data = np.load(train_path + '/' + file_name)
                this_data = this_data.astype(np.float32)
                # print(this_data.shape)
                data.append(this_data)
            data = np.concatenate(data, axis=0)
            data = self.scaler.fit_transform(data)
            target = data.copy()
        else:
            test_path = current_path + '/data/datasets/smap/raw/test'
            file_names = os.listdir(test_path)
            file_names.sort()
            data = []
            for file_name in file_names:
                this_data = np.load(test_path + '/' + file_name)
                this_data = this_data.astype(np.float32)
                data.append(this_data)
            data = np.concatenate(data, axis=0)
            data = self.scaler.transform(data)
            test_target_path = current_path + '/data/datasets/smap/raw/test_label'
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

        if self.dataidxs:
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
            # print(data0.shape, data.shape)
            data = np.concatenate((data0, data), axis=0)
        else:
            data = self.data[index + 1 - self.window_len: index + 1]
        target = self.target[index]

        return data, target

    def __len__(self):
        return self.data.shape[0]

class PSM_Dataset(data.Dataset):

    def __init__(self, dataidxs=None, train=True, transform=None, target_transform=None, download=False, window_len=5):

        self.dataidxs = dataidxs
        self.train = train
        self.transform = transform
        self.target_transform = target_transform
        self.download = download
        self.window_len = window_len
        global scaler
        self.scaler = scaler

        self.data, self.target = self.__build_truncated_dataset__()

    def __build_truncated_dataset__(self):

        current_path = os.path.abspath(os.path.dirname(os.path.abspath(os.path.dirname(os.getcwd()))))

        if self.train:
            train_path = current_path + '/data/datasets/psm/raw'
            data = pd.read_csv(train_path + '/train.csv')
            data.drop(columns=[r'timestamp_(min)'], inplace=True)
            data = data.values.astype(np.float32)
            data = np.nan_to_num(data)
            data = self.scaler.fit_transform(data)
            target = data.copy()
        else:
            test_path = current_path + '/data/datasets/psm/raw'
            file_names = os.listdir(test_path)
            file_names.sort()
            data = pd.read_csv(test_path + '/test.csv')
            data.drop(columns=[r'timestamp_(min)'], inplace=True)
            data = data.values.astype(np.float32)
            data = np.nan_to_num(data)
            data = self.scaler.transform(data)
            test_target_path = current_path + '/data/datasets/psm/raw/test_label.csv'
            target_csv = pd.read_csv(test_target_path)
            target_csv.drop(columns=[r'timestamp_(min)'], inplace=True)
            target = target_csv.values
            target = target.astype(np.float32)

        if self.dataidxs:
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
            # print(data0.shape, data.shape)
            data = np.concatenate((data0, data), axis=0)
        else:
            data = self.data[index + 1 - self.window_len: index + 1]
        target = self.target[index]

        return data, target

    def __len__(self):
        return self.data.shape[0]


if __name__ == '__main__':
    smd_train = SMD_Dataset()
    smd_test = SMD_Dataset(train=False)
