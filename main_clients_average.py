# -*- coding: utf-8 -*-
import os
import sys
import time

import numpy as np
from tqdm import tqdm

from algorithms.GDN.gdn_exp_smd import get_feature_map, get_fc_graph_struc, build_loc_net, GDN
from convert_time import convert_time
from options import args, features_dict, gdn_topk_dict

from typing import List
import torch
from logger import logger
from general_tools import mean, set_random_seed

set_random_seed(args.seed)


torch.backends.cudnn.benchmark = False
os.environ['CUBLAS_WORKSPACE_CONFIG']=':16:8'
torch.use_deterministic_algorithms(True)

from task import test_dataset, model_fun, client_datasets
from clients.client_mix_average import generate_clients, test_inference


def average_main():
    logger.tik()
    clients = generate_clients(client_datasets)

    model = model_fun().cpu()
    global_state_dict = model.state_dict()
    print(os.getcwd())
    data_nums = []
    last_auc_rocs = []
    last_auc_prs = []
    best_auc_rocs = []
    best_auc_prs = []
    times = []
    for cc in range(len(clients)):
        client = clients[cc]
        data_nums.append(len(client.dataset))
        model, best_auc_roc, best_auc_pr, last_auc_roc, last_auc_pr, client_times = client.local_train(model_fun, client_idx=cc, test_dataset=test_dataset)
        time_client_end = time.time()
        best_auc_rocs.append(best_auc_roc)
        best_auc_prs.append(best_auc_pr)
        last_auc_rocs.append(last_auc_roc)
        last_auc_prs.append(last_auc_pr)
        times.append(mean(client_times))
        print('Client', cc, 'best_auc_roc', best_auc_roc, 'best_auc_pr', best_auc_pr, 'last_auc_roc', last_auc_roc, 'last_auc_pr', last_auc_pr)
        print('max time till now:', max(times))

    logger.print(f' \n Last Results:')

    print(f"Best Average AUC-ROC: {mean(best_auc_rocs)}")
    print(f"Best Average AP: {mean(best_auc_prs)}")
    print(f"Last Average AUC-ROC: {mean(last_auc_rocs)}")
    print(f"Last Average AP: {mean(last_auc_prs)}")
    print(f"Max time:", convert_time(max(times)))
    print(f"Min time:", convert_time(min(times)))
    print(f"Average time:", convert_time(mean(times)))

    logger.tok()
    try:
        logger.save()
    except:
        pass


if __name__ == '__main__':
    average_main()
