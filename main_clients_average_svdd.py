# -*- coding: utf-8 -*-
import os
import time

from convert_time import convert_time
from options import args

import torch
from logger import logger
from general_tools import mean, set_random_seed

set_random_seed(args.seed)

torch.backends.cudnn.benchmark = False
os.environ['CUBLAS_WORKSPACE_CONFIG']=':16:8'
torch.use_deterministic_algorithms(True)

from task import client_datasets
from clients.client_mix_average import generate_clients

if args.dataset == 'smd':
    from task.smd_MOON import test_dataset
elif args.dataset == 'smap':
    from task.smap_MOON import test_dataset
elif args.dataset == 'psm':
    from task.psm_MOON import test_dataset
from task.SVDD import config_svdd, Model_first_stage, Model_second_stage, client_datasets, load_model, switch_config


def average_main():
    logger.tik()
    clients = generate_clients(client_datasets)
    model = Model_first_stage().cpu()
    global_state_dict = model.state_dict()
    model2 = Model_second_stage().cpu()
    global_state_dict2 = model2.state_dict()
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
        time_client_start = time.time()
        config_svdd["stage"] = "first"
        model_stage_1, _, _, _, _, _ = client.local_train(Model_first_stage, client_idx=cc, test_dataset=test_dataset)
        switch_config()
        model_stage_2 = load_model(global_state_dict2)
        model_stage_2.load_first_stage_model(model_stage_1.state_dict())
        global_state_dict_stage2 = model_stage_2.state_dict()
        model, best_auc_roc, best_auc_pr, last_auc_roc, last_auc_pr, _ = client.local_train(Model_second_stage,
                                                                                         client_idx=cc,
                                                                                         test_dataset=test_dataset)
        time_client_end = time.time()
        times.append(time_client_end - time_client_start)
        best_auc_rocs.append(best_auc_roc)
        best_auc_prs.append(best_auc_pr)
        last_auc_rocs.append(last_auc_roc)
        last_auc_prs.append(last_auc_pr)
        print('Client', cc, 'best_auc_roc', best_auc_roc, 'best_auc_pr', best_auc_pr, 'last_auc_roc', last_auc_roc, 'last_auc_pr', last_auc_pr, 'time', convert_time(time_client_end - time_client_start))


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
