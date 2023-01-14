# -*- coding: utf-8 -*-
import sys
import numpy as np
from tqdm import tqdm

from algorithms.DeepSVDD.DeepSVDD import SMD_MLP, SMAP_MLP, PSM_MLP
from options import args
import os
from typing import List
import torch
from logger import logger
from general_tools import set_random_seed

set_random_seed(args.seed)

torch.backends.cudnn.benchmark = False
os.environ['CUBLAS_WORKSPACE_CONFIG']=':16:8'
torch.use_deterministic_algorithms(True)

from task.SVDD import switch_config
from clients.client_mix import test_inference, get_init_grad_correct, generate_clients

if args.dataset == 'smd':
    from task.smd_MOON import test_dataset
elif args.dataset == 'smap':
    from task.smap_MOON import test_dataset
elif args.dataset == 'psm':
    from task.psm_MOON import test_dataset
from task.SVDD import config_svdd, Model_first_stage, Model_second_stage, client_datasets, load_model

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
    clients = generate_clients(client_datasets)
    model_stage_1 = train_stage_1(clients)

    switch_config()
    train_stage_2(clients, model_stage_1)


def train_stage_1(clients):
    model = Model_first_stage().cpu()
    config_svdd["stage"] = "first"
    global_state_dict = model.state_dict()
    global_correct = get_init_grad_correct(Model_first_stage().cpu())


    # Training
    best_auc_roc = 0
    for global_round in tqdm(range(config_svdd["epochs"]), file=sys.stdout):
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
        for client in active_clients:
            data_nums.append(len(client.dataset))
            loss, accuracy, grad_correct_delta = client.local_train(
                    global_state_dict,
                    global_round,
                    global_correct,
            )
            grad_correct_deltas.append(grad_correct_delta)
            #
            train_accuracies.append(accuracy)
            train_losses.append(loss)
            active_state_dict.append(client.state_dict_prev)
        # endregion

        fed_freq = torch.tensor(data_nums, dtype=torch.float) / sum(data_nums)
        global_state_dict = average_weights(active_state_dict, fed_freq)
        if args.alg == 'scaffold':
            global_correct = update_global_grad_correct(
                    global_correct, grad_correct_deltas,
                    fed_freq, num_active_client, len(clients)
            )
        # endregion

    model = load_model(global_state_dict).cuda()
    return model


def train_stage_2(clients, model_stage_1):

    model_save_path = os.path.abspath(os.getcwd()) + '/fltsad/pths/' + args.alg + '_deepsvdd_stage_2_' + args.dataset + '_new.pth'
    score_save_path = os.path.abspath(os.getcwd()) + '/fltsad/scores/' + args.alg + '_deepsvdd_' + args.dataset + '_new.npy'

    model = Model_second_stage().cpu()

    model.load_first_stage_model(model_stage_1.state_dict())

    global_state_dict = model.state_dict()

    # for k, v in global_state_dict.items():
    #     print(k, v)

    global_c = model.c

    global_correct = get_init_grad_correct(Model_second_stage().cpu())

    best_auc_roc = 0
    best_ap = 0

    cs = []

    # Training
    for global_round in tqdm(range(config_svdd["epochs"]), file=sys.stdout):
        logger.print(f'\n | Global Training Round : {global_round + 1} |\n')


        num_active_client = int((len(clients) * args.client_rate))

        ind_active_clients = np.random.choice(range(len(clients)), num_active_client, replace=False)
        active_clients = [clients[i] for i in ind_active_clients]
        # endregion

        # region 各节点进行训练，并返回权重和loss以及梯度修正向量
        active_state_dict = []
        data_nums = []
        train_accuracies = []
        train_losses = []
        grad_correct_deltas = []
        for client in active_clients:
            data_nums.append(len(client.dataset))
            if args.dataset == 'smd':
                model_fun = lambda: SMD_MLP()
            elif args.dataset == 'smap':
                model_fun = lambda: SMAP_MLP()
            elif args.dataset == 'psm':
                model_fun = lambda: PSM_MLP()
            client.grad_correct = get_init_grad_correct(model_fun())
            if global_round == 0:
                client.global_state_dict = None
            loss, accuracy, grad_correct_delta, client_c = client.local_train(
                    global_state_dict,
                    global_round,
                    global_correct,
                    global_c
            )
            grad_correct_deltas.append(grad_correct_delta)
            #
            train_accuracies.append(accuracy)
            train_losses.append(loss)
            active_state_dict.append(client.state_dict_prev)
            cs.append(client_c.detach().cpu().numpy())
        # endregion


        # region 平均权重 更新global 模型
        fed_freq = torch.tensor(data_nums, dtype=torch.float) / sum(data_nums)
        global_state_dict = average_weights(active_state_dict, fed_freq)
        if args.alg == 'scaffold':
            global_correct = update_global_grad_correct(
                    global_correct, grad_correct_deltas,
                    fed_freq, num_active_client, len(clients)
            )
        # endregion

        mdl = load_model(global_state_dict).cuda()
        cs_now = torch.tensor(np.asarray(cs))
        mdl.c = torch.mean(cs_now, dim=0)
        if (global_round + 1) % args.save_every == 0:
            auc_roc, ap, test_loss, scores = test_inference(
                    mdl,
                    test_dataset
            )
            logger.add_record("test_auc_roc", auc_roc, global_round + 1)
            if test_loss is not None:
                logger.add_record("test_ap", ap, global_round + 1)
            logger.add_record("test_loss", test_loss, global_round + 1)
            logger.print(f' \n Last Results:')
            # logger.print(f"Test Accuracy (full sample rate): {100 * test_acc:.2f}%")
            print(f"Test AUC-ROC: {auc_roc}")
            print(f"Test AP: {ap}")
            if test_loss is not None:
                print(f"Test loss (full sample rate): {test_loss:.2f}")

            if auc_roc > best_auc_roc:
                best_auc_roc = auc_roc
                best_ap = ap
                torch.save(global_state_dict, model_save_path)
                np.save(score_save_path, scores)

        # endregion

    logger.print(f' \n Last Results:')

    print(f"Test AUC-ROC: {best_auc_roc}")
    print(f"Test AP: {best_ap}")
    if test_loss is not None:
        print(f"Test loss (full sample rate): {test_loss:.2f}")


if __name__ == '__main__':
    set_random_seed(args.seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.cuda.set_device(args.device)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device[-1]
    torch.multiprocessing.set_sharing_strategy("file_system")
    torch.backends.cudnn.benchmark = False
    fed_main()
