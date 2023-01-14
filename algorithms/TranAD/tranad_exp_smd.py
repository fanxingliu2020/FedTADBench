import os
import time

import numpy as np
import torch
from sklearn.metrics import roc_auc_score, average_precision_score
from torch import nn
from torch.utils.data import DataLoader
import random

import sys


sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "../..")))


from convert_time import convert_time

from algorithms.TranAD.TranAD import TranAD
from algorithms.read_datasets import SMD_Dataset, SMAP_Dataset, PSM_Dataset

def tranad_main(args=None):
    device = torch.device("cuda:0")
    random_seed = args['seed']
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

    feats = trainloader.dataset.data.shape[-1]
    model = TranAD(feats).to(device)
    model.lr = args['lr']
    model.n_window = args['window_len']

    optimizer = torch.optim.AdamW(model.parameters(), lr=model.lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 5, 0.9)

    epochs = args['epoch']
    w_size = model.n_window
    l1s, l2s = [], []
    l = nn.MSELoss(reduction='none')
    best_auc_roc = 0
    best_ap = 0
    model_save_path = os.path.abspath(os.path.join(os.getcwd(), "../..")) + '/fltsad/pths/tranad_' + args['dataset_name'] + '.pth'
    score_save_path = os.path.abspath(os.path.join(os.getcwd(), "../..")) + '/fltsad/scores/tranad_' + args['dataset_name'] + '.npy'

    time_start = time.time()

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        epoch_items = 0
        n = epoch + 1
        for d, _ in trainloader:
            local_bs = d.shape[0]
            window = d.permute(1, 0, 2).to(device)
            elem = window[-1, :, :].view(1, local_bs, feats).to(device)
            features, logits, others = model(window, elem)
            z = (others['x1'], others['x2'])
            l1 = l(z, elem) if not isinstance(z, tuple) else (1 / n) * l(z[0], elem) + (1 - 1 / n) * l(z[1], elem)
            if isinstance(z, tuple): z = z[1]
            l1s.append(torch.mean(l1).item())
            loss = torch.mean(l1)
            optimizer.zero_grad()
            loss.backward(retain_graph=True)
            optimizer.step()
            epoch_loss += loss.item() * local_bs
            epoch_items += local_bs
        scheduler.step()
        print('epoch', n, 'loss:', (epoch_loss / epoch_items), end='')

        model.eval()
        test_losses = []
        for d, _ in testloader:
            window = d.permute(1, 0, 2).to(device)
            elem = window[-1, :, :].view(1, d.shape[0], feats).to(device)
            features, logits, others = model(window, elem)
            z = (others['x1'], others['x2'])
            if isinstance(z, tuple): z = z[1]
            test_losses.append(z[0].detach().cpu().numpy())
        test_losses = np.concatenate(test_losses, axis=0)
        test_losses = np.mean(test_losses, axis=1)

        labels = testloader.dataset.target
        auc_roc = roc_auc_score(labels, test_losses)
        ap = average_precision_score(labels, test_losses)
        print(' auc_roc:', auc_roc, 'auc_pr:', ap)

        if auc_roc > best_auc_roc:
            best_auc_roc = auc_roc
            best_ap = ap
            torch.save(model.state_dict(), model_save_path)
            np.save(score_save_path, test_losses)

    time_end = time.time()

    print('Best auc_roc:', best_auc_roc)
    print('Best ap:', best_ap)
    print('Total time:', convert_time(time_end - time_start))


if __name__ == '__main__':

    tranad_main(args={'dataset_name': 'smd', 'lr': 0.0001, 'batch_size': 64, 'epoch': 100, 'window_len': 10, 'seed': 42})