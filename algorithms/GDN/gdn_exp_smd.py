# -*- coding: utf-8 -*-
import os
import sys


sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "../..")))


from convert_time import convert_time

import math
import time

import pandas
import pandas as pd
import numpy as np
import torch
from sklearn.metrics import roc_auc_score, average_precision_score
from torch import nn
import torch.nn.functional as F

from torch.utils.data import DataLoader, random_split, Subset, Dataset

from sklearn.preprocessing import MinMaxScaler

from algorithms.read_datasets import SMD_Dataset, SMAP_Dataset, PSM_Dataset
from algorithms.GDN.train_federated_learning import train, timeSincePlus
from algorithms.GDN.evaluate import get_err_scores, get_best_performance_data, get_val_performance_data, \
    get_full_err_scores, get_full_err_scores_only_test

import sys
from datetime import datetime

import os
import argparse
from pathlib import Path

import json
import random

import torch
from torch.nn import Parameter, Linear, Sequential, BatchNorm1d, ReLU
import torch.nn.functional as F
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import remove_self_loops, add_self_loops, softmax

from torch_geometric.nn.inits import glorot, zeros

from scores_adjusted import roc_curve_adjusted_for_time_series


def test(model, dataloader):
    # test
    loss_func = nn.MSELoss(reduction='mean')
    device = get_device()

    test_loss_list = []
    now = time.time()

    test_predicted_list = []
    test_ground_list = []
    test_labels_list = []

    t_test_predicted_list = []
    t_test_ground_list = []
    t_test_labels_list = []

    test_len = len(dataloader)

    model.eval()

    i = 0
    acu_loss = 0
    for x, y, labels, edge_index in dataloader:
        x, y, labels, edge_index = [item.to(device).float() for item in [x, y, labels, edge_index]]

        with torch.no_grad():

            labels = labels.unsqueeze(1).repeat(1, x.shape[1])
            _, predicted, loss = model(x, edge_index, labels)
            # predicted = predicted.float().to(device)

            # loss = loss_func(predicted, y)

            # labels = labels.unsqueeze(1).repeat(1, predicted.shape[1])

            if len(t_test_predicted_list) <= 0:
                t_test_predicted_list = predicted
                t_test_ground_list = y
                t_test_labels_list = labels
            else:
                t_test_predicted_list = torch.cat((t_test_predicted_list, predicted), dim=0)
                t_test_ground_list = torch.cat((t_test_ground_list, y), dim=0)
                t_test_labels_list = torch.cat((t_test_labels_list, labels), dim=0)

        test_loss_list.append(loss.item())
        acu_loss += loss.item()

        i += 1

        if i % 10000 == 1 and i > 1:
            print(timeSincePlus(now, i / test_len))

    test_predicted_list = t_test_predicted_list.tolist()
    test_ground_list = t_test_ground_list.tolist()
    test_labels_list = t_test_labels_list.tolist()

    avg_loss = sum(test_loss_list) / len(test_loss_list)

    return avg_loss, [test_predicted_list, test_ground_list, test_labels_list]

_device = torch.device('cuda:0')

def get_device():
    # return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return _device

def set_device(dev):
    global _device
    _device = dev

def build_loc_net(struc, all_features, feature_map=[]):
    index_feature_map = feature_map
    edge_indexes = [
        [],
        []
    ]
    for node_name, node_list in struc.items():
        if node_name not in all_features:
            continue

        if node_name not in index_feature_map:
            index_feature_map.append(node_name)

        p_index = index_feature_map.index(node_name)
        for child in node_list:
            if child not in all_features:
                continue

            if child not in index_feature_map:
                print(f'error: {child} not in index_feature_map')
                # index_feature_map.append(child)

            c_index = index_feature_map.index(child)
            # edge_indexes[0].append(p_index)
            # edge_indexes[1].append(c_index)
            edge_indexes[0].append(c_index)
            edge_indexes[1].append(p_index)

    return edge_indexes

def construct_data(data, feature_map, labels=0):
    res = []

    for feature in feature_map:
        if feature in data.columns:
            res.append(data.loc[:, feature].values.tolist())
        else:
            print(feature, 'not exist in data')
    # append labels as last
    sample_n = len(res[0])

    if type(labels) == int:
        res.append([labels]*sample_n)
    elif len(labels) == sample_n:
        res.append(labels)

    return res

def get_feature_map(dataset):
    if 'FedTADBench' in os.path.abspath(os.path.dirname(os.path.dirname(os.getcwd()))):
        if dataset == 'smd':
            feature_file = open(os.path.abspath(os.path.dirname(os.path.dirname(os.getcwd()))) + '/data/datasets/smd/SMD/raw/list.txt', 'r')
        elif dataset == 'smap':
            feature_file = open(os.path.abspath(os.path.dirname(os.path.dirname(os.getcwd()))) + '/data/datasets/smap/raw/list.txt', 'r')
        elif dataset == 'psm':
            feature_file = open(os.path.abspath(os.path.dirname(os.path.dirname(os.getcwd()))) + '/data/datasets/psm/raw/list.txt', 'r')
    else:
        if dataset == 'smd':
            feature_file = open(
                os.path.abspath(os.getcwd()) + '/data/datasets/smd/SMD/raw/list.txt',
                'r')
        elif dataset == 'smap':
            feature_file = open(
                os.path.abspath(os.getcwd()) + '/data/datasets/smap/raw/list.txt', 'r')
        elif dataset == 'psm':
            feature_file = open(
                os.path.abspath(os.getcwd()) + '/data/datasets/psm/raw/list.txt', 'r')

    feature_list = []
    for ft in feature_file:
        feature_list.append(ft.strip())

    return feature_list

def get_fc_graph_struc(dataset):
    if 'FedTADBench' in os.path.abspath(os.path.dirname(os.path.dirname(os.getcwd()))):
        if dataset == 'smd':
            feature_file = open(os.path.abspath(os.path.dirname(os.path.dirname(os.getcwd()))) + '/data/datasets/smd/SMD/raw/list.txt', 'r')
        elif dataset == 'smap':
            feature_file = open(os.path.abspath(os.path.dirname(os.path.dirname(os.getcwd()))) + '/data/datasets/smap/raw/list.txt', 'r')
        elif dataset == 'psm':
            feature_file = open(os.path.abspath(os.path.dirname(os.path.dirname(os.getcwd()))) + '/data/datasets/psm/raw/list.txt', 'r')
    else:
        if dataset == 'smd':
            feature_file = open(
                os.path.abspath(os.getcwd()) + '/data/datasets/smd/SMD/raw/list.txt',
                'r')
        elif dataset == 'smap':
            feature_file = open(
                os.path.abspath(os.getcwd()) + '/data/datasets/smap/raw/list.txt', 'r')
        elif dataset == 'psm':
            feature_file = open(
                os.path.abspath(os.getcwd()) + '/data/datasets/psm/raw/list.txt', 'r')

    struc_map = {}
    feature_list = []
    for ft in feature_file:
        feature_list.append(ft.strip())

    for ft in feature_list:
        if ft not in struc_map:
            struc_map[ft] = []

        for other_ft in feature_list:
            if other_ft is not ft:
                struc_map[ft].append(other_ft)

    return struc_map

class TimeDataset(Dataset):
    def __init__(self, raw_data, edge_index, mode='train', config=None):
        self.raw_data = raw_data

        self.config = config
        self.edge_index = edge_index
        self.mode = mode

        x_data = raw_data[:-1]
        labels = raw_data[-1]

        data = x_data

        # to tensor
        data = torch.tensor(data).double()
        labels = torch.tensor(labels).double()

        self.x, self.y, self.labels = self.process(data, labels)

    def __len__(self):
        return len(self.x)

    def process(self, data, labels):
        x_arr, y_arr = [], []
        labels_arr = []

        slide_win, slide_stride = [self.config[k] for k
                                   in ['slide_win', 'slide_stride']
                                   ]
        is_train = self.mode == 'train'

        node_num, total_time_len = data.shape

        rang = range(slide_win, total_time_len, slide_stride) if is_train else range(slide_win, total_time_len)

        for i in rang:
            ft = data[:, i - slide_win:i]
            tar = data[:, i]

            x_arr.append(ft)
            y_arr.append(tar)

            labels_arr.append(labels[i])

        x = torch.stack(x_arr).contiguous()
        y = torch.stack(y_arr).contiguous()

        labels = torch.Tensor(labels_arr).contiguous()

        return x, y, labels

    def __getitem__(self, idx):
        feature = self.x[idx].double()
        y = self.y[idx].double()

        edge_index = self.edge_index.long()

        label = self.labels[idx].double()

        return feature, y, label, edge_index

def loss_func(y_pred, y_true):
    loss = F.mse_loss(y_pred, y_true, reduction='mean')

    return loss


def get_batch_edge_index(org_edge_index, batch_num, node_num):
    # org_edge_index:(2, edge_num)
    edge_index = org_edge_index.clone().detach()
    edge_num = org_edge_index.shape[1]
    batch_edge_index = edge_index.repeat(1, batch_num).contiguous()

    for i in range(batch_num):
        batch_edge_index[:, i * edge_num:(i + 1) * edge_num] += i * node_num

    return batch_edge_index.long()


class OutLayer(nn.Module):
    def __init__(self, in_num, node_num, layer_num, inter_num=512):
        super(OutLayer, self).__init__()

        modules = []

        for i in range(layer_num):
            # last layer, output shape:1
            if i == layer_num - 1:
                modules.append(nn.Linear(in_num if layer_num == 1 else inter_num, 1))
            else:
                layer_in_num = in_num if i == 0 else inter_num
                modules.append(nn.Linear(layer_in_num, inter_num))
                modules.append(nn.BatchNorm1d(inter_num))
                modules.append(nn.ReLU())

        self.mlp = nn.ModuleList(modules)

    def forward(self, x):
        out = x

        for mod in self.mlp:
            if isinstance(mod, nn.BatchNorm1d):
                out = out.permute(0, 2, 1)
                out = mod(out)
                out = out.permute(0, 2, 1)
            else:
                out = mod(out)

        return out


class GNNLayer(nn.Module):
    def __init__(self, in_channel, out_channel, inter_dim=0, heads=1, node_num=100):
        super(GNNLayer, self).__init__()

        self.gnn = GraphLayer(in_channel, out_channel, inter_dim=inter_dim, heads=heads, concat=False)

        self.bn = nn.BatchNorm1d(out_channel)
        self.relu = nn.ReLU()
        self.leaky_relu = nn.LeakyReLU()

    def forward(self, x, edge_index, embedding=None, node_num=0):
        out, (new_edge_index, att_weight) = self.gnn(x, edge_index, embedding, return_attention_weights=True)
        self.att_weight_1 = att_weight
        self.edge_index_1 = new_edge_index

        out = self.bn(out)

        return self.relu(out)


class GraphLayer(MessagePassing):
    def __init__(self, in_channels, out_channels, heads=1, concat=True,
                 negative_slope=0.2, dropout=0, bias=True, inter_dim=-1, **kwargs):
        super(GraphLayer, self).__init__(aggr='add', **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.concat = concat
        self.negative_slope = negative_slope
        self.dropout = dropout

        self.__alpha__ = None

        self.lin = Linear(in_channels, heads * out_channels, bias=False)

        self.att_i = Parameter(torch.Tensor(1, heads, out_channels))
        self.att_j = Parameter(torch.Tensor(1, heads, out_channels))
        self.att_em_i = Parameter(torch.Tensor(1, heads, out_channels))
        self.att_em_j = Parameter(torch.Tensor(1, heads, out_channels))

        if bias and concat:
            self.bias = Parameter(torch.Tensor(heads * out_channels))
        elif bias and not concat:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.lin.weight)
        glorot(self.att_i)
        glorot(self.att_j)

        zeros(self.att_em_i)
        zeros(self.att_em_j)

        zeros(self.bias)

    def forward(self, x, edge_index, embedding, return_attention_weights=False):
        """"""
        if torch.is_tensor(x):
            x = self.lin(x)
            x = (x, x)
        else:
            x = (self.lin(x[0]), self.lin(x[1]))

        edge_index, _ = remove_self_loops(edge_index)
        edge_index, _ = add_self_loops(edge_index,
                                       num_nodes=x[1].size(self.node_dim))

        out = self.propagate(edge_index, x=x, embedding=embedding, edges=edge_index,
                             return_attention_weights=return_attention_weights)

        if self.concat:
            out = out.view(-1, self.heads * self.out_channels)
        else:
            out = out.mean(dim=1)

        if self.bias is not None:
            out = out + self.bias

        if return_attention_weights:
            alpha, self.__alpha__ = self.__alpha__, None
            return out, (edge_index, alpha)
        else:
            return out

    def message(self, x_i, x_j, edge_index_i, size_i,
                embedding,
                edges,
                return_attention_weights):

        x_i = x_i.view(-1, self.heads, self.out_channels)
        x_j = x_j.view(-1, self.heads, self.out_channels)

        if embedding is not None:
            embedding_i, embedding_j = embedding[edge_index_i], embedding[edges[0]]
            embedding_i = embedding_i.unsqueeze(1).repeat(1, self.heads, 1)
            embedding_j = embedding_j.unsqueeze(1).repeat(1, self.heads, 1)

            key_i = torch.cat((x_i, embedding_i), dim=-1)
            key_j = torch.cat((x_j, embedding_j), dim=-1)

        cat_att_i = torch.cat((self.att_i, self.att_em_i), dim=-1)
        cat_att_j = torch.cat((self.att_j, self.att_em_j), dim=-1)

        alpha = (key_i * cat_att_i).sum(-1) + (key_j * cat_att_j).sum(-1)

        alpha = alpha.view(-1, self.heads, 1)

        alpha = F.leaky_relu(alpha, self.negative_slope)
        alpha = softmax(alpha, edge_index_i, size_i)

        if return_attention_weights:
            self.__alpha__ = alpha

        alpha = F.dropout(alpha, p=self.dropout, training=self.training)

        return x_j * alpha.view(-1, self.heads, 1)

    def __repr__(self):
        return '{}({}, {}, heads={})'.format(self.__class__.__name__,
                                             self.in_channels,
                                             self.out_channels, self.heads)


class GDN(nn.Module):
    def __init__(self, edge_index_sets, node_num, dim=64, out_layer_inter_dim=256, input_dim=10, out_layer_num=1,
                 topk=20):

        super(GDN, self).__init__()

        self.edge_index_sets = edge_index_sets

        device = get_device()

        edge_index = edge_index_sets[0]

        embed_dim = dim
        self.embedding = nn.Embedding(node_num, embed_dim)
        self.bn_outlayer_in = nn.BatchNorm1d(embed_dim)

        edge_set_num = len(edge_index_sets)
        self.gnn_layers = nn.ModuleList([
            GNNLayer(input_dim, dim, inter_dim=dim + embed_dim, heads=1) for i in range(edge_set_num)
        ])

        self.node_embedding = None
        self.topk = topk
        self.learned_graph = None

        self.out_layer = OutLayer(dim * edge_set_num, node_num, out_layer_num, inter_num=out_layer_inter_dim)

        self.cache_edge_index_sets = [None] * edge_set_num
        self.cache_embed_index = None

        self.dp = nn.Dropout(0.2)

        self.init_params()

    def init_params(self):
        nn.init.kaiming_uniform_(self.embedding.weight, a=math.sqrt(5))

    def forward(self, data, org_edge_index, labels):

        x = data.clone().detach()  # (批量大小, 维度, 窗口长度)
        edge_index_sets = self.edge_index_sets

        device = data.device

        batch_num, node_num, all_feature = x.shape
        x = x.view(-1, all_feature).contiguous()

        gcn_outs = []
        for i, edge_index in enumerate(edge_index_sets):
            edge_num = edge_index.shape[1]
            cache_edge_index = self.cache_edge_index_sets[i]

            if cache_edge_index is None or cache_edge_index.shape[1] != edge_num * batch_num:
                self.cache_edge_index_sets[i] = get_batch_edge_index(edge_index, batch_num, node_num).to(device)

            batch_edge_index = self.cache_edge_index_sets[i]

            all_embeddings = self.embedding(torch.arange(node_num).to(device))

            weights_arr = all_embeddings.detach().clone()
            all_embeddings = all_embeddings.repeat(batch_num, 1)

            weights = weights_arr.view(node_num, -1)

            cos_ji_mat = torch.matmul(weights, weights.T)
            normed_mat = torch.matmul(weights.norm(dim=-1).view(-1, 1), weights.norm(dim=-1).view(1, -1))
            cos_ji_mat = cos_ji_mat / normed_mat

            dim = weights.shape[-1]
            topk_num = self.topk

            topk_indices_ji = torch.topk(cos_ji_mat, topk_num, dim=-1)[1]

            self.learned_graph = topk_indices_ji

            gated_i = torch.arange(0, node_num).T.unsqueeze(1).repeat(1, topk_num).flatten().to(device).unsqueeze(0)
            gated_j = topk_indices_ji.flatten().unsqueeze(0)
            gated_edge_index = torch.cat((gated_j, gated_i), dim=0)

            batch_gated_edge_index = get_batch_edge_index(gated_edge_index, batch_num, node_num).to(device)
            gcn_out = self.gnn_layers[i](x, batch_gated_edge_index, node_num=node_num * batch_num,
                                         embedding=all_embeddings)

            gcn_outs.append(gcn_out)

        x = torch.cat(gcn_outs, dim=1)
        x = x.view(batch_num, node_num, -1)

        indexes = torch.arange(0, node_num).to(device)
        out_features = torch.mul(x, self.embedding(indexes))

        out = out_features.permute(0, 2, 1)
        out = F.relu(self.bn_outlayer_in(out))
        out = out.permute(0, 2, 1)

        out = self.dp(out)
        out = self.out_layer(out)
        out = out.view(-1, node_num)

        out = out.float().to(device)

        loss = loss_func(out, labels)

        return out_features, out, loss  # out_features: (批量大小, 维度数量, dims), out: 预测结果 (批量大小, 维度数量)


class Main():

    def __init__(self, batch: int=128, epoch: int=100, slide_win: int=15, dim: int=64, slide_stride: int=5,
                 save_path_pattern: str='', dataset: str='smd', device: str='cuda', random_seed: int=0,
                 comment: str='', out_layer_num: int=1, out_layer_inter_dim: int=256, decay: float=0,
                 val_ratio: float=0, topk: int=20, report: str='best', load_model_path: str='', debug=False):

        random.seed(random_seed)
        np.random.seed(random_seed)
        torch.manual_seed(random_seed)
        torch.cuda.manual_seed(random_seed)
        torch.cuda.manual_seed_all(random_seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        os.environ['PYTHONHASHSEED'] = str(random_seed)
        self.dataset = dataset

        train_config = {
            'batch': batch,
            'epoch': epoch,
            'slide_win': slide_win,
            'dim': dim,
            'slide_stride': slide_stride,
            'comment': comment,
            'seed': random_seed,
            'out_layer_num': out_layer_num,
            'out_layer_inter_dim': out_layer_inter_dim,
            'decay': decay,
            'val_ratio': val_ratio,
            'topk': topk,
        }

        env_config = {
            'save_path': save_path_pattern,
            'dataset': dataset,
            'report': report,
            'device': device,
            'load_model_path': load_model_path
        }

        self.train_config = train_config
        self.env_config = env_config
        self.datestr = None

        dataset = self.env_config['dataset']
        if dataset == 'smd' or dataset == 'smap' or dataset == 'psm':
            if dataset == 'smd':
                train_data = SMD_Dataset(train=True)
                test_data = SMD_Dataset(train=False)
            elif dataset == 'smap':
                train_data = SMAP_Dataset(train=True)
                test_data = SMAP_Dataset(train=False)
            elif dataset == 'psm':
                train_data = PSM_Dataset(train=True)
                test_data = PSM_Dataset(train=False)
            columns = [str(i) for i in range(train_data.data.shape[-1])]
            train, test = train_data.data, test_data.data
            train = pd.DataFrame(train, columns=columns)
            test = pd.DataFrame(test, columns=columns)
            labels = test_data.target

        if 'attack' in train.columns:
            train = train.drop(columns=['attack'])

        feature_map = get_feature_map(dataset)
        fc_struc = get_fc_graph_struc(dataset)

        set_device(env_config['device'])
        self.device = get_device()

        fc_edge_index = build_loc_net(fc_struc, list(train.columns), feature_map=feature_map)
        fc_edge_index = torch.tensor(fc_edge_index, dtype=torch.long)

        self.feature_map = feature_map
        train_dataset_indata = construct_data(train, feature_map, labels=0)
        test_dataset_indata = construct_data(test, feature_map, labels=labels)

        cfg = {
            'slide_win': train_config['slide_win'],
            'slide_stride': train_config['slide_stride'],
        }

        train_dataset = TimeDataset(train_dataset_indata, fc_edge_index, mode='train', config=cfg)
        test_dataset = TimeDataset(test_dataset_indata, fc_edge_index, mode='test', config=cfg)

        train_dataloader, val_dataloader = self.get_loaders(train_dataset, train_config['seed'], train_config['batch'],
                                                            val_ratio=train_config['val_ratio'])

        self.train_dataset = train_dataset
        self.test_dataset = test_dataset

        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.test_dataloader = DataLoader(test_dataset, batch_size=train_config['batch'],
                                          shuffle=False, num_workers=0)

        edge_index_sets = []
        edge_index_sets.append(fc_edge_index)

        self.model = GDN(edge_index_sets, len(feature_map),
                         dim=train_config['dim'],
                         input_dim=train_config['slide_win'],
                         out_layer_num=train_config['out_layer_num'],
                         out_layer_inter_dim=train_config['out_layer_inter_dim'],
                         topk=train_config['topk']
                         ).to(self.device)

        print(self.model)

    def run(self):

        # if len(self.env_config['load_model_path']) > 0:
        #     model_save_path = self.env_config['load_model_path']
        # else:
        #     model_save_path = self.get_save_path()[0]

        model_save_path = os.path.abspath(os.path.join(os.getcwd(), "../..")) + '/fltsad/pths/gdn_' + self.dataset + '.pth'
        score_save_path = os.path.abspath(os.path.join(os.getcwd(), "../..")) + '/fltsad/scores/gdn_' + self.dataset + '.npy'

        time_start = time.time()

        self.train_log, best_auc_roc, best_ap = train(self.model, model_save_path,
                               config=self.train_config,
                               train_dataloader=self.train_dataloader,
                               val_dataloader=self.val_dataloader,
                               feature_map=self.feature_map,
                               test_dataloader=self.test_dataloader,
                               test_dataset=self.test_dataset,
                               train_dataset=self.train_dataset,
                               dataset_name=self.env_config['dataset'],
                               score_save_path=score_save_path)

        time_end = time.time()

        # test
        self.model.load_state_dict(torch.load(model_save_path))
        best_model = self.model.to(self.device)

        # _, self.test_result = test(best_model, self.test_dataloader)
        # if self.val_dataloader is None:
        #     self.val_result = None
        #     scores = self.get_score_only_test(self.test_result)
        #     np.save(score_save_path, scores)
        # else:
        #     _, self.val_result = test(best_model, self.val_dataloader)
        #     scores = self.get_score(self.test_result, self.val_result)
        #     np.save(score_save_path, scores)

        print('Best auc_roc:', best_auc_roc)
        print('Best ap:', best_ap)
        print('Total time:', convert_time(time_end - time_start))

    def get_loaders(self, train_dataset, seed, batch, val_ratio=0.1):
        dataset_len = int(len(train_dataset))
        train_use_len = int(dataset_len * (1 - val_ratio))
        val_use_len = int(dataset_len * val_ratio)
        val_start_index = random.randrange(train_use_len)
        indices = torch.arange(dataset_len)

        train_sub_indices = torch.cat([indices[:val_start_index], indices[val_start_index + val_use_len:]])
        train_subset = Subset(train_dataset, train_sub_indices)

        val_sub_indices = indices[val_start_index:val_start_index + val_use_len]
        val_subset = Subset(train_dataset, val_sub_indices)

        train_dataloader = DataLoader(train_subset, batch_size=batch,
                                      shuffle=True)
        if val_ratio != 0:
            val_dataloader = DataLoader(val_subset, batch_size=batch,
                                        shuffle=False)
        else:
            val_dataloader = None

        return train_dataloader, val_dataloader

    def get_score(self, test_result, val_result):

        feature_num = len(test_result[0][0])
        np_test_result = np.array(test_result)
        np_val_result = np.array(val_result)

        test_labels = np_test_result[2, :, 0].tolist()

        test_scores, normal_scores = get_full_err_scores(test_result, val_result)

        top1_best_info = get_best_performance_data(test_scores, test_labels, topk=1)
        top1_val_info = get_val_performance_data(test_scores, normal_scores, test_labels, topk=1)

        print('=========================** Result **============================\n')

        info = None
        if self.env_config['report'] == 'best':
            info = top1_best_info
        elif self.env_config['report'] == 'val':
            info = top1_val_info

        # print(f'F1 score: {info[0]}')
        # print(f'precision: {info[1]}')
        # print(f'recall: {info[2]}\n')

        precision = info[1]
        recall = info[2]
        f1 = 2 * precision * recall / (precision + recall)

        print(f'F1 score: {f1}')
        print(f'precision: {precision}')
        print(f'recall: {recall}\n')

        topk = 1
        total_features = test_scores.shape[0]
        topk_indices = np.argpartition(test_scores, range(total_features - topk - 1, total_features), axis=0)[
                       -topk:]
        total_topk_err_scores = np.sum(np.take_along_axis(test_scores, topk_indices, axis=0), axis=0)

        auc_roc = roc_auc_score(test_labels, total_topk_err_scores)
        print('auc_roc:', auc_roc)
        ap = average_precision_score(test_labels, total_topk_err_scores)
        print('ap:', ap)

        return total_topk_err_scores

    def get_score_only_test(self, test_result):

        feature_num = len(test_result[0][0])
        np_test_result = np.array(test_result)

        test_labels = np_test_result[2, :, 0].tolist()

        test_scores = get_full_err_scores_only_test(test_result)

        top1_best_info = get_best_performance_data(test_scores, test_labels, topk=1)

        print('=========================** Result **============================\n')

        info = None
        if self.env_config['report'] == 'best':
            info = top1_best_info

        # print(f'F1 score: {info[0]}')
        # print(f'precision: {info[1]}')
        # print(f'recall: {info[2]}\n')

        precision = info[1]
        recall = info[2]
        f1 = 2 * precision * recall / (precision + recall)

        print(f'F1 score: {f1}')
        print(f'precision: {precision}')
        print(f'recall: {recall}\n')

        topk = 1
        total_features = test_scores.shape[0]
        topk_indices = np.argpartition(test_scores, range(total_features - topk - 1, total_features), axis=0)[
                       -topk:]
        total_topk_err_scores = np.sum(np.take_along_axis(test_scores, topk_indices, axis=0), axis=0)

        auc_roc = roc_auc_score(test_labels, total_topk_err_scores)
        print('auc_roc:', auc_roc)
        ap = average_precision_score(test_labels, total_topk_err_scores)
        print('ap:', ap)
        return total_topk_err_scores

    def get_save_path(self, feature_name=''):

        dir_path = self.env_config['save_path']

        if self.datestr is None:
            now = datetime.now()
            self.datestr = now.strftime('%m|%d-%H:%M:%S')
        datestr = self.datestr

        paths = [
            f'./pretrained/{dir_path}/best_{datestr}.pt',
            f'./results/{dir_path}/{datestr}.csv',
        ]

        for path in paths:
            dirname = os.path.dirname(path)
            Path(dirname).mkdir(parents=True, exist_ok=True)

        return paths


if __name__ == "__main__":

    main = Main(dataset='smd', save_path_pattern='gdn_smd', slide_stride=1, slide_win=5, batch=64, epoch=100,
                comment='smd', random_seed=42, decay=0, dim=128, out_layer_num=1, out_layer_inter_dim=128,
                val_ratio=0, report='best', topk=30, debug=False)

    main.run()





