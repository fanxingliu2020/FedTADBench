# -*- coding: utf-8 -*-
import os
import sys
import time

import numpy as np
from tqdm import tqdm

from algorithms.GDN.gdn_exp_smd import get_feature_map, get_fc_graph_struc, build_loc_net, GDN
from options import args, features_dict, gdn_topk_dict

from typing import List
import torch
from logger import logger
from general_tools import mean, set_random_seed

set_random_seed(args.seed)

torch.backends.cudnn.benchmark = False
os.environ['CUBLAS_WORKSPACE_CONFIG']=':16:8'
torch.use_deterministic_algorithms(True)

from task import config, test_dataset, model_fun, client_datasets, load_model
from clients.client_mix import test_inference, get_init_grad_correct, generate_clients

def average_weights(state_dicts: List[dict], fed_avg_freqs: torch.Tensor):
    # init
    avg_state_dict = {}
    for key in state_dicts[0].keys():
        avg_state_dict[key] = state_dicts[0][key] * fed_avg_freqs[0]

    state_dicts = state_dicts[1:]
    fed_avg_freqs = fed_avg_freqs[1:]
    for state_dict, freq in zip(state_dicts, fed_avg_freqs):
        for key in state_dict.keys():
            avg_state_dict[key] += state_dict[key] * freq
    return avg_state_dict


def update_global_grad_correct(old_correct: dict, grad_correct_deltas: List[dict], fed_avg_freqs: torch.Tensor, num_chosen_client, num_total_client):
    assert (len(grad_correct_deltas) == num_chosen_client)
    total_delta = average_weights(grad_correct_deltas, [1 / num_chosen_client] * num_chosen_client)
    for key in old_correct.keys():
        if key in total_delta.keys():
            old_correct[key] = old_correct[key] + total_delta[key]
    return old_correct


def fed_main():
    logger.tik()
    clients = generate_clients(client_datasets)

    model = model_fun().cpu()
    global_state_dict = model.state_dict()
    global_correct = get_init_grad_correct(model_fun().cpu())

    # endregion

    best_auc_roc = 0
    best_ap = 0
    print(os.getcwd())
    model_save_path = os.path.abspath(os.getcwd()) + '/fltsad/pths/' + args.alg + '_' + args.tsadalg + '_' + args.dataset + '_beta_' + str(args.beta).replace('.', '') + '.pth'
    score_save_path = os.path.abspath(os.getcwd()) + '/fltsad/scores/' + args.alg + '_' + args.tsadalg + '_' + args.dataset + '_beta_' + str(args.beta).replace('.', '') + '.npy'

    # Training
    times = []
    for global_round in tqdm(range(config["epochs"]), file=sys.stdout):
        logger.print(f'\n | Global Training Round : {global_round + 1} |\n')


        num_active_client = int((len(clients) * args.client_rate))

        ind_active_clients = np.random.choice(range(len(clients)), num_active_client, replace=False)
        active_clients = [clients[i] for i in ind_active_clients]
        # endregion

        active_state_dict = []
        data_nums = []
        train_accuracies = []
        train_losses = []
        grad_correct_deltas = []
        client_times = []
        for client in active_clients:
            client_start = time.time()
            data_nums.append(len(client.dataset))
            loss, accuracy, grad_correct_delta = client.local_train(
                global_state_dict,
                global_round,
                global_correct,
                )
            client_times.append(time.time() - client_start)
            grad_correct_deltas.append(grad_correct_delta)

            train_losses.append(loss)
            active_state_dict.append(client.state_dict_prev)

        # endregion

        this_time = max(client_times)
        time_start = time.time()
        fed_freq = torch.tensor(data_nums, dtype=torch.float) / sum(data_nums)
        global_state_dict = average_weights(active_state_dict, fed_freq)
        if args.alg == 'scaffold':
            global_correct = update_global_grad_correct(
                global_correct, grad_correct_deltas,
                fed_freq, num_active_client, len(clients)
                )
        # endregion
        time_end = time.time()
        this_time += ((time_end - time_start) / 5)
        times.append(this_time)

        logger.add_record("train_loss", float(mean(train_losses)), global_round)
        if (global_round + 1) % args.save_every == 0:

            auc_roc, ap, test_loss, scores = test_inference(
                load_model(global_state_dict).to(args.device),
                test_dataset
            )
            logger.add_record("test_auc_roc", auc_roc, global_round + 1)
            logger.add_record("test_ap", ap, global_round + 1)
            logger.add_record("test_loss", test_loss, global_round + 1)
            logger.print(f' \n Results after {global_round + 1} global rounds of training:')
            print('average time:', mean(times))
            print(f"Test AUC-ROC: {auc_roc}")
            print(f"Test AP: {ap}")

            if auc_roc > best_auc_roc:
                best_auc_roc = auc_roc
                best_ap = ap
                torch.save(global_state_dict, model_save_path)
                np.save(score_save_path, scores)
            try:
                print(f"Test loss (full sample rate): {test_loss:.2f}")
            except:
                pass
        # endregion

    print('average time:', mean(times))

    logger.print(f' \n Last Results:')

    print(f"Test AUC-ROC: {best_auc_roc}")
    print(f"Test AP: {best_ap}")
    try:
        print(f"Test loss (full sample rate): {test_loss:.2f}")
    except:
        pass

    logger.tok()
    try:
        logger.save()
    except:
        pass


if __name__ == '__main__':
    fed_main()
