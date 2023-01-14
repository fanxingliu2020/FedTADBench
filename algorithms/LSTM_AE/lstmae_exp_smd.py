import os
import sys
import time


sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "../..")))


from convert_time import convert_time

import numpy as np
import torch
from sklearn.metrics import roc_auc_score, average_precision_score
from torch import nn
from torch.utils.data import DataLoader

from algorithms.LSTM_AE.LSTMAE import LSTMAE
from algorithms.read_datasets import SMD_Dataset, SMAP_Dataset, PSM_Dataset

import random
import os

dataset_dims_dict = {'smd': 38,
                     'smap': 25,
                     'psm': 25}

def lstmae_exp(args=None):

    model_save_path = os.path.abspath(os.path.join(os.getcwd(), "../..")) + '/fltsad/pths/lstmae_' + args['dataset_name'] + '.pth'
    score_save_path = os.path.abspath(os.path.join(os.getcwd(), "../..")) + '/fltsad/scores/lstmae_' + args['dataset_name'] + '.npy'

    random_seed = args['random_seed']
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    os.environ['PYTHONHASHSEED'] = str(random_seed)

    if args['dataset_name'] == 'smd':
        train_data = SMD_Dataset(train=True, window_len=args['window_len'])
        test_data = SMD_Dataset(train=False, window_len=args['window_len'])
    elif args['dataset_name'] == 'smap':
        train_data = SMAP_Dataset(train=True, window_len=args['window_len'])
        test_data = SMAP_Dataset(train=False, window_len=args['window_len'])
    elif args['dataset_name'] == 'psm':
        train_data = PSM_Dataset(train=True, window_len=args['window_len'])
        test_data = PSM_Dataset(train=False, window_len=args['window_len'])

    trainloader = DataLoader(
        train_data,
        batch_size=args['batch_size'],
        shuffle=True,
        pin_memory=True,
        # persistent_workers=True,
        # prefetch_factor=4,
        num_workers=2,
        # drop_last=True
        drop_last=False
    )

    testloader = DataLoader(
        test_data,
        batch_size=args['batch_size'],
        shuffle=False,
        pin_memory=True,
        # persistent_workers=True,
        # prefetch_factor=4,
        num_workers=2,
        drop_last=False
    )

    model = LSTMAE(n_features=args['dataset_dims_dict'][args['dataset_name']], hidden_size=args['hidden_size'],
                   n_layers=args['n_layers'], use_bias=args['use_bias'], dropout=args['dropout'], device=args['device']).to(args['device'])

    optimizer = torch.optim.Adam(model.parameters(), lr=args['lr'])
    # 训练

    best_auc_roc = 0
    best_ap = 0

    time_start = time.time()

    for e in range(args['epochs']):
        model.train()
        epoch_loss = 0
        for i, (x, y) in enumerate(trainloader):
            x, y = x.to(args['device']), y.to(args['device'])
            # print(x.detach().cpu().numpy().max(), x.detach().cpu().numpy().min())
            optimizer.zero_grad()
            feature, logits, others = model(x)
            if 'output' in others.keys():
                pred_y = others['output']
            loss = args['criterion'](pred_y, y)
            loss.backward()
            optimizer.step()
            # print(loss.item())
            epoch_loss += loss.item()
        epoch_loss /= len(trainloader)
        print('epoch', e + 1, 'loss', epoch_loss, end=' ')

        scores = []
        ys = []
        model.eval()
        with torch.no_grad():
            for i, (x, y) in enumerate(testloader):
                x, y = x.to(args['device']), y.to(args['device'])
                #
                optimizer.zero_grad()
                feature, logits, others = model(x)
                scores.append(logits.detach().cpu().numpy())
                ys.append(y.detach().cpu().numpy())
        scores = np.concatenate(scores, axis=0)
        ys = np.concatenate(ys, axis=0)
        # print(scores.shape, ys.shape)
        if len(scores.shape) == 2:
            scores = np.squeeze(scores, axis=1)
        if len(ys.shape) == 2:
            ys = np.squeeze(ys, axis=1)
        auc_roc = roc_auc_score(ys, scores)
        ap = average_precision_score(ys, scores)
        print('auc-roc: ' + str(auc_roc) + ' auc_pr: ' + str(ap), end='')

        if auc_roc > best_auc_roc:
            best_auc_roc = auc_roc
            best_ap = ap
            torch.save(model.state_dict(), model_save_path)
            np.save(score_save_path, scores)
            print(' update')
        else:
            print('\n', end='')

    time_end = time.time()

    print('Best auc_roc:', best_auc_roc)
    print('Best ap:', best_ap)
    print('Total time:', convert_time(time_end - time_start))



    # 测试
    # model = LSTMAE(n_features=args['dataset_dims_dict'][args['dataset_name']], hidden_size=args['hidden_size'],
    #                n_layers=args['n_layers'], use_bias=args['use_bias'], dropout=args['dropout'], device=args['device']).to(args['device'])
    #
    # model.load_state_dict(torch.load(model_save_path))
    # model.eval()
    #
    # scores = []
    # ys = []
    # for i, (x, y) in enumerate(testloader):
    #     x, y = x.to(args['device']), y.to(args['device'])
    #     #
    #     optimizer.zero_grad()
    #     feature, logits, others = model(x)
    #     scores.append(logits.detach().cpu().numpy())
    #     ys.append(y.detach().cpu().numpy())
    # scores = np.concatenate(scores, axis=0)
    # ys = np.concatenate(ys, axis=0)
    # # print(scores.shape, ys.shape)
    # if len(scores.shape) == 2:
    #     scores = np.squeeze(scores, axis=1)
    # if len(ys.shape) == 2:
    #     ys = np.squeeze(ys, axis=1)
    # np.save(score_save_path, scores)
    # auc_roc = roc_auc_score(ys, scores)
    # ap = average_precision_score(ys, scores)
    # print('auc_roc:', auc_roc)
    # print('ap:', ap)

if __name__ == '__main__':
    args = {'dataset_name': 'smd', 'epochs': 100, 'batch_size': 64, 'lr': 0.001, 'dataset_dims_dict': dataset_dims_dict, 'hidden_size': 128,
            'n_layers': (2, 2), 'use_bias': (True, True), 'dropout': (0, 0), 'criterion': nn.MSELoss(), 'device': torch.device('cuda:1'), 'random_seed': 42,
            'window_len': 30}
    lstmae_exp(args)
