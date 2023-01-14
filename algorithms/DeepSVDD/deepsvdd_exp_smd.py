import json
import os
import time

import torch
import logging
import random
import numpy as np
from matplotlib import pyplot as plt
from torchvision.utils import make_grid

import sys


sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "../..")))

from convert_time import convert_time

from algorithms.DeepSVDD.DeepSVDD import DeepSVDD
from algorithms.read_datasets import SMD_Dataset, SMAP_Dataset, PSM_Dataset


class Config(object):
    """Base class for experimental setting/configuration."""

    def __init__(self, settings):
        self.settings = settings

    def load_config(self, import_json):
        """Load settings dict from import_json (path/filename.json) JSON-file."""

        with open(import_json, 'r') as fp:
            settings = json.load(fp)

        for key, value in settings.items():
            self.settings[key] = value

    def save_config(self, export_json):
        """Save settings dict to export_json (path/filename.json) JSON-file."""

        with open(export_json, 'w') as fp:
            json.dump(self.settings, fp)

def plot_images_grid(x: torch.tensor, export_img, title: str = '', nrow=8, padding=2, normalize=False, pad_value=0):
    """Plot 4D Tensor of images of shape (B x C x H x W) as a grid."""

    grid = make_grid(x, nrow=nrow, padding=padding, normalize=normalize, pad_value=pad_value)
    npgrid = grid.cpu().numpy()

    plt.imshow(np.transpose(npgrid, (1, 2, 0)), interpolation='nearest')

    ax = plt.gca()
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)

    if not (title == ''):
        plt.title(title)

    plt.savefig(export_img, bbox_inches='tight', pad_inches=0.1)
    plt.clf()

################################################################################
# Settings
################################################################################

def main(dataset_name: str='smd', net_name: str='smd_mlp', xp_path: str='.', data_path: str='../data', load_config=None, load_model=None, objective: str='one-class', nu: float=0.1, device: str='cuda:0', seed: int=42,
         optimizer_name: str='adam', lr: float=0.001, n_epochs: int=200, lr_milestone: tuple=(50,), batch_size: int=200, weight_decay: float=5e-7, pretrain: bool=True, ae_optimizer_name: str='adam', ae_lr: int=0.001,
         ae_n_epochs: int=200, ae_lr_milestone: tuple=(50,), ae_batch_size: int=200, ae_weight_decay: float=0.0005, n_jobs_dataloader: int=0, normal_class: int=3, random_seed: int=42):
    """
    Deep SVDD, a fully deep method for anomaly detection.

    :arg DATASET_NAME: Name of the dataset to load.
    :arg NET_NAME: Name of the neural network to use.
    :arg XP_PATH: Export path for logging the experiment.
    :arg DATA_PATH: Root path of data.
    """

    model_save_path = os.path.abspath(os.path.join(os.getcwd(), "../..")) + '/fltsad/pths/deepsvdd_' + dataset_name + '.pth'
    score_save_path = os.path.abspath(os.path.join(os.getcwd(), "../..")) + '/fltsad/scores/deepsvdd_' + dataset_name + '.npy'
    rc_save_path = os.path.abspath(os.path.join(os.getcwd(), "../..")) + '/fltsad/pths/deepsvdd_rc_' + dataset_name + '.npy'

    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    os.environ['PYTHONHASHSEED'] = str(random_seed)

    # Get configuration
    cfg = Config(locals().copy())

    # Set up logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    log_file = xp_path + '/' + str(dataset_name) + '_log.txt'
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # Print arguments
    logger.info('Log file is %s.' % log_file)
    logger.info('Data path is %s.' % data_path)
    logger.info('Export path is %s.' % xp_path)

    logger.info('Dataset: %s' % dataset_name)
    logger.info('Normal class: %d' % normal_class)
    logger.info('Network: %s' % net_name)

    # If specified, load experiment config from JSON-file
    if load_config:
        cfg.load_config(import_json=load_config)
        logger.info('Loaded configuration from %s.' % load_config)

    # Print configuration
    logger.info('Deep SVDD objective: %s' % cfg.settings['objective'])
    logger.info('Nu-paramerter: %.2f' % cfg.settings['nu'])

    # Default device to 'cpu' if cuda is not available
    if not torch.cuda.is_available():
        device = 'cpu'
    logger.info('Computation device: %s' % device)
    logger.info('Number of dataloader workers: %d' % n_jobs_dataloader)

    # Load data
    dataset = dataset_name
    if dataset == 'smd' or dataset == 'smap' or dataset == 'psm':
        if dataset == 'smd':
            train_data = SMD_Dataset(train=True, window_len=1)
            test_data = SMD_Dataset(train=False, window_len=1)
        elif dataset == 'smap':
            train_data = SMAP_Dataset(train=True, window_len=1)
            test_data = SMAP_Dataset(train=False, window_len=1)
        elif dataset == 'psm':
            train_data = PSM_Dataset(train=True, window_len=1)
            test_data = PSM_Dataset(train=False, window_len=1)

    # Initialize DeepSVDD model and set neural network \phi
    deep_SVDD = DeepSVDD(cfg.settings['objective'], cfg.settings['nu'])
    deep_SVDD.set_network(net_name)
    # If specified, load Deep SVDD model (radius R, center c, network weights, and possibly autoencoder weights)
    if load_model:
        deep_SVDD.load_model(model_path=load_model, load_ae=True)
        logger.info('Loading model from %s.' % load_model)

    time_start = time.time()

    logger.info('Pretraining: %s' % pretrain)
    if pretrain:
        # Log pretraining details
        logger.info('Pretraining optimizer: %s' % cfg.settings['ae_optimizer_name'])
        logger.info('Pretraining learning rate: %g' % cfg.settings['ae_lr'])
        logger.info('Pretraining epochs: %d' % cfg.settings['ae_n_epochs'])
        logger.info('Pretraining learning rate scheduler milestones: %s' % (cfg.settings['ae_lr_milestone'],))
        logger.info('Pretraining batch size: %d' % cfg.settings['ae_batch_size'])
        logger.info('Pretraining weight decay: %g' % cfg.settings['ae_weight_decay'])

        # Pretrain model on dataset (via autoencoder)
        deep_SVDD.pretrain(train_data, test_data=test_data,
                           optimizer_name=cfg.settings['ae_optimizer_name'],
                           lr=cfg.settings['ae_lr'],
                           n_epochs=cfg.settings['ae_n_epochs'],
                           lr_milestones=cfg.settings['ae_lr_milestone'],
                           batch_size=cfg.settings['ae_batch_size'],
                           weight_decay=cfg.settings['ae_weight_decay'],
                           device=device,
                           n_jobs_dataloader=n_jobs_dataloader)

    # Log training details
    logger.info('Training optimizer: %s' % cfg.settings['optimizer_name'])
    logger.info('Training learning rate: %g' % cfg.settings['lr'])
    logger.info('Training epochs: %d' % cfg.settings['n_epochs'])
    logger.info('Training learning rate scheduler milestones: %s' % (cfg.settings['lr_milestone'],))
    logger.info('Training batch size: %d' % cfg.settings['batch_size'])
    logger.info('Training weight decay: %g' % cfg.settings['weight_decay'])

    # Train model on dataset
    best_auc_roc, best_ap = deep_SVDD.train(train_data,
                    optimizer_name=cfg.settings['optimizer_name'],
                    lr=cfg.settings['lr'],
                    n_epochs=cfg.settings['n_epochs'],
                    lr_milestones=cfg.settings['lr_milestone'],
                    batch_size=cfg.settings['batch_size'],
                    weight_decay=cfg.settings['weight_decay'],
                    device=device,
                    n_jobs_dataloader=n_jobs_dataloader,
                    model_save_path = model_save_path,
                    test_data=test_data,
                    score_save_path=score_save_path,
                    rc_save_path=rc_save_path)

    time_end = time.time()

    print('Best auc_roc:', best_auc_roc)
    print('Best ap:', best_ap)
    print('Total time:', convert_time(time_end - time_start))


if __name__ == '__main__':
    main(dataset_name='smd', random_seed=42, ae_n_epochs=100, n_epochs=100)

