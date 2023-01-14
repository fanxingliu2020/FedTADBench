import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "../..")))

from algorithms.LSTM_AE.lstmae_exp_smd import *

if __name__ == '__main__':
    args = {'dataset_name': 'psm', 'epochs': 100, 'batch_size': 64, 'lr': 0.001, 'dataset_dims_dict': dataset_dims_dict, 'hidden_size': 128,
            'n_layers': (2, 2), 'use_bias': (True, True), 'dropout': (0, 0), 'criterion': nn.MSELoss(), 'device': torch.device('cuda:1'), 'random_seed': 42,
            'window_len': 30}
    lstmae_exp(args)
