import os
import time
from typing import List

import numpy as np
import pandas
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from algorithms.DeepSVDD.DeepSVDD import BaseNet
from algorithms.GDN.evaluate import get_err_median_and_iqr
from algorithms.GDN.gdn_exp_smd import TimeDataset, get_feature_map, get_fc_graph_struc, build_loc_net, construct_data
from options import args, seed_worker
from task import config, model_fun

if args.tsadalg == 'deep_svdd':
    from task.SVDD import config_svdd
from sklearn.metrics import roc_auc_score, average_precision_score


def get_init_grad_correct(model: nn.Module):
    correct = {}
    for name, _ in model.named_parameters():
        correct[name] = torch.tensor(0, dtype=torch.float, device="cpu")
    return correct


class Client(object):

    def __init__(self, dataset):
        self.dataset = dataset
        self.state_dict_prev = None
        self.moon_mu = config["moon_mu"]
        self.prox_mu = config["prox_mu"]
        self.local_bs = config["local_bs"]
        self.grad_correct = get_init_grad_correct(model_fun())
        self.local_ep = config["local_ep"]
        self.criterion = nn.MSELoss().to(args.device)
        self.cos_sim = torch.nn.CosineSimilarity(dim=-1).to(args.device)
        self.temperature = config["tau"]
        self.trainloader = None

    def set_local_optimizer(self, model: nn.Module) -> torch.optim.Optimizer:
        return config["optimizer_fun"](model.parameters())

    def local_train(
            self, model_fun, global_c: torch.Tensor = None, client_idx=0, test_dataset=None
    ):

        last_auc_roc = 0
        last_auc_pr = 0
        best_auc_roc = 0
        best_auc_pr = 0

        client_times = []


        model_save_path_best = os.path.abspath(
            os.path.join(os.getcwd(), "")) + '/fltsad/pths_average/' + args.tsadalg  + '_' + args.dataset + '_' + str(client_idx) + '_best.pth'
        score_save_path_best = os.path.abspath(
            os.path.join(os.getcwd(), "")) + '/fltsad/scores_average/' + args.tsadalg + '_' + args.dataset + '_' + str(client_idx) + '_best.npy'

        model_save_path_last = os.path.abspath(
            os.path.join(os.getcwd(), "")) + '/fltsad/pths_average/' + args.tsadalg + '_' + args.dataset + '_' + str(client_idx) + '_last.pth'
        score_save_path_last = os.path.abspath(
            os.path.join(os.getcwd(), "")) + '/fltsad/scores_average/' + args.tsadalg + '_' + args.dataset + '_' + str(client_idx) + '_last.npy'

        scheduler = None
        #
        model_current = model_fun()
        model_current.requires_grad_(True)
        # if args.tsadalg == 'deep_svdd' and config_svdd['stage'] == 'second':
        #     model_current.c = global_c
        model_current.train()
        model_current.to(args.device)
        # endregion

        epoch_loss = []
        train_acc = []

        # region Set optimizer and dataloader
        optimizer = self.set_local_optimizer(model_current)
        if args.tsadalg == 'tran_ad':
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 5, 0.9)
        if args.tsadalg == 'deep_svdd':
            scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=(50,), gamma=0.1)
        if args.tsadalg == 'gdn':
            feature_map = get_feature_map(args.dataset)
            fc_struc = get_fc_graph_struc(args.dataset)
            fc_edge_index = build_loc_net(fc_struc, feature_map, feature_map=feature_map)
            fc_edge_index = torch.tensor(fc_edge_index, dtype=torch.long)
            edge_index_sets = []
            edge_index_sets.append(fc_edge_index)
            train_scaled = self.dataset.data
            train = pandas.DataFrame(train_scaled, columns=feature_map, dtype=np.float32)
            train_dataset_indata = construct_data(train, feature_map, labels=0)
            cfg = {
                    'slide_win': args.slide_win,
                    'slide_stride': 1,
            }
            train_dataset = TimeDataset(train_dataset_indata, fc_edge_index, mode='train', config=cfg)

            trainloader = DataLoader(
                    train_dataset, batch_size=self.local_bs,
                    shuffle=True,
                    drop_last=False
            )
        else:
            trainloader = DataLoader(
                    self.dataset,
                    batch_size=self.local_bs,
                    shuffle=True,
                    pin_memory=True,
                    num_workers=args.num_workers,
                    drop_last=False
            )

        if args.tsadalg == 'deep_svdd' and config_svdd['stage'] == 'second':

            def init_center_c(train_loader: DataLoader, net: BaseNet, eps=0.1):

                """Initialize hypersphere center c as the mean from an initial forward pass on the data."""
                n_samples = 0
                c = torch.zeros(net.rep_dim).to(args.device)

                net.eval()
                with torch.no_grad():
                    for data in train_loader:
                        inputs, _ = data
                        inputs = inputs.to(net.device)
                        outputs = net(inputs)
                        n_samples += outputs[0].shape[0]
                        outputs0 = outputs[0]
                        outputs0 = torch.sum(outputs0, dim=0)
                        outputs0 = torch.squeeze(outputs0, 0)
                        c += outputs0

                c /= n_samples

                # If c_i is too close to 0, set to +-eps. Reason: a zero unit can be trivially matched with zero weights.
                c[(abs(c) < eps) & (c < 0)] = -eps
                c[(abs(c) < eps) & (c > 0)] = eps

                return c

            model_current.c = init_center_c(trainloader, model_current)
            self.state_dict_prev = None
        # endregion
        if args.tsadalg != 'gdn' and args.tsadalg != 'usad':
            l1s = []
            for local_epoch in range(config['epochs']):
                client_time_epoch_start = time.time()
                model_current.train()
                loss1_list = []
                batch_loss = []
                correct = 0
                num_data = 0
                for i, (x, y) in enumerate(trainloader):
                    x, y = x.to(args.device), y.to(args.device)
                    optimizer.zero_grad()
                    if args.tsadalg == 'tran_ad':
                        local_bs = x.shape[0]
                        feats = x.shape[-1]
                        window = x.permute(1, 0, 2)
                        elem = window[-1, :, :].view(1, local_bs, feats)
                        feature, logits, others = model_current(window, elem)
                    else:
                        feature, logits, others = model_current(x)

                    if 'output' in others.keys() and not (args.tsadalg == 'deep_svdd' and config_svdd['stage'] == 'second'):
                        pred_y = others['output']
                        if np.any(np.isnan(pred_y.detach().cpu().numpy())) or np.any(np.isnan(y.detach().cpu().numpy())):
                            print('nan exists in y_pred or y')
                        if len(pred_y.shape) == 3:
                            loss = self.criterion(pred_y[:, -1, :], y)
                        else:
                            loss = self.criterion(pred_y, y)
                    elif 'x1' in others.keys() and not (args.tsadalg == 'deep_svdd' and config_svdd['stage'] == 'second'):
                        l = nn.MSELoss(reduction='none')
                        n = local_epoch + 1
                        z = (others['x1'], others['x2'])
                        l1 = l(z, elem) if not isinstance(z, tuple) else (1 / n) * l(z[0], elem) + (1 - 1 / n) * l(
                                z[1],
                                elem
                        )
                        if isinstance(z, tuple): z = z[1]
                        l1s.append(torch.mean(l1).item())
                        loss = torch.mean(l1)
                    elif 'loss' in others.keys() and not (args.tsadalg == 'deep_svdd' and config_svdd['stage'] == 'second'):
                        loss = others['loss']
                    elif args.tsadalg == 'deep_svdd' and config_svdd['stage'] == 'second':
                        dist = torch.sum((logits[:, -1, :] - model_current.c) ** 2, dim=1)
                        loss = torch.mean(dist)


                    loss.backward()
                    optimizer.step()

                client_time_epoch_end = time.time()
                client_times.append(client_time_epoch_end - client_time_epoch_start)

                model_current.eval()
                if args.tsadalg == 'deep_svdd' and config_svdd['stage'] != 'second':
                    print('client', client_idx, 'pretrain epoch', local_epoch, 'loss:', loss.item())
                else:
                    auc_roc, auc_pr, _, scores = test_inference(model_current, test_dataset)
                    print('client', client_idx, 'epoch', local_epoch, 'auc_roc:', auc_roc, 'auc_pr', auc_pr)

                    if auc_roc > best_auc_roc:
                        best_auc_roc = auc_roc
                        best_auc_pr = auc_pr
                        torch.save(model_current.state_dict(), model_save_path_best)
                        np.save(score_save_path_best, scores)

                    if local_epoch == 2 - 1:
                        last_auc_roc = auc_roc
                        last_auc_pr = auc_pr
                        torch.save(model_current.state_dict(), model_save_path_last)
                        np.save(score_save_path_last, scores)


                if scheduler is not None:
                    scheduler.step()
        elif args.tsadalg == 'usad':
            losses1 = []
            losses2 = []
            for local_epoch in range(config['epochs']):
                client_time_epoch_start = time.time()
                batch_loss = []
                correct = 0
                num_data = 0
                opt_func = torch.optim.Adam
                optimizer1 = opt_func(list(model_current.encoder.parameters()) + list(model_current.decoder1.parameters()))
                optimizer2 = opt_func(list(model_current.encoder.parameters()) + list(model_current.decoder2.parameters()))
                model_current.train()
                for i, (x, y) in enumerate(trainloader):
                    x, y = x.to(args.device), y.to(args.device)
                    x = x.view([x.shape[0], x.shape[1] * x.shape[2]])
                    loss_stat = 0
                    #
                    optimizer1.zero_grad()
                    _, _, others = model_current.training_step(x, local_epoch + 1)
                    loss1, _ = others['loss1'], others['loss2']
                    loss1.backward()
                    losses1.append(loss1.item())
                    optimizer1.step()
                    optimizer1.zero_grad()
                    loss_stat += float(loss1.item())
                    #
                    # Train AE2
                    feature, logits, others = model_current.training_step(x, local_epoch + 1)
                    _, loss2 = others['loss1'], others['loss2']

                    loss2.backward()
                    losses2.append(loss2.item())
                    loss_stat += float(loss2.item())
                    optimizer2.step()
                    optimizer2.zero_grad()
                    # correct grad

                client_time_epoch_end = time.time()
                client_times.append(client_time_epoch_end - client_time_epoch_start)
                model_current.eval()
                auc_roc, auc_pr, _, scores = test_inference(model_current, test_dataset)

                print('client', client_idx, 'epoch', local_epoch, 'auc_roc:', auc_roc, 'auc_pr', auc_pr)

                if auc_roc > best_auc_roc:
                    best_auc_roc = auc_roc
                    best_auc_pr = auc_pr
                    torch.save(model_current.state_dict(), model_save_path_best)
                    np.save(score_save_path_best, scores)

                if local_epoch == 2 - 1:
                    last_auc_roc = auc_roc
                    last_auc_pr = auc_pr
                    torch.save(model_current.state_dict(), model_save_path_last)
                    np.save(score_save_path_last, scores)


        else:
            for local_epoch in range(config['epochs']):
                batch_loss = []
                correct = 0
                num_data = 0
                model_current.train()
                for i, (x, labels, attack_labels, edge_index) in enumerate(trainloader):
                    x, labels, edge_index = [item.float().to(args.device) for item in [x, labels, edge_index]]
                    #
                    optimizer.zero_grad()
                    feature, logits, loss = model_current(x, edge_index, labels)

                    loss.backward()
                    optimizer.step()

                model_current.eval()
                auc_roc, auc_pr, _, scores = test_inference(model_current, test_dataset)

                print('client', client_idx, 'epoch', local_epoch, 'auc_roc:', auc_roc, 'auc_pr:', auc_pr)

                if auc_roc > best_auc_roc:
                    best_auc_roc = auc_roc
                    best_auc_pr = auc_pr
                    torch.save(model_current.state_dict(), model_save_path_best)
                    np.save(score_save_path_best, scores)

                if local_epoch == 2 - 1:
                    last_auc_roc = auc_roc
                    last_auc_pr = auc_pr
                    torch.save(model_current.state_dict(), model_save_path_last)
                    np.save(score_save_path_last, scores)



        return model_current, best_auc_roc, best_auc_pr, last_auc_roc, last_auc_pr, client_times


def generate_clients(datasets: List[Dataset]) -> List[Client]:
    clients = []
    for dataset in datasets:
        clients.append(Client(dataset))
    return clients


@torch.no_grad()
def test_inference(model, dataset, score_save_path=''):
    model.eval()
    num_data = 0
    correct = 0
    if args.tsadalg == 'gdn':
        feature_map = get_feature_map(args.dataset)
        fc_struc = get_fc_graph_struc(args.dataset)
        fc_edge_index = build_loc_net(fc_struc, feature_map, feature_map=feature_map)
        fc_edge_index = torch.tensor(fc_edge_index, dtype=torch.long)
        edge_index_sets = []
        edge_index_sets.append(fc_edge_index)
        test_scaled = dataset.data
        test = pandas.DataFrame(test_scaled, columns=feature_map, dtype=np.float32)
        test_dataset_indata = construct_data(test, feature_map, labels=0)
        cfg = {
                'slide_win': args.slide_win,
                'slide_stride': 1,
        }
        test_dataset = TimeDataset(test_dataset_indata, fc_edge_index, mode='test', config=cfg)
        if len(dataset.target.shape) == 1:
            test_dataset.labels = torch.tensor(dataset.target[:])
        elif len(dataset.target.shape) == 2:
            test_dataset.labels = torch.tensor(dataset.target[:, 0])

        testloader = DataLoader(
                test_dataset, batch_size=128,
                shuffle=False,
                drop_last=False
        )
    else:
        testloader = DataLoader(
                dataset,
                batch_size=128,
                shuffle=False,
                num_workers=args.num_workers,
                worker_init_fn=seed_worker,
                drop_last=False
        )
    criterion = nn.MSELoss()
    loss = []
    anomaly_scores_all = []
    labels_all = []
    if args.tsadalg != 'gdn':
        for i, (x, labels) in enumerate(testloader):
            x = x.to(args.device)
            labels = labels.to(args.device)
            if args.tsadalg == 'tran_ad':
                local_bs = x.shape[0]
                feats = x.shape[-1]
                window = x.permute(1, 0, 2)
                elem = window[-1, :, :].view(1, local_bs, feats)
                feature, logits, others = model(window, elem)
            else:
                if args.tsadalg == 'usad':
                    x = x.view(x.shape[0], x.shape[1] * x.shape[2])
                feature, logits, others = model(x)
            if args.tsadalg == 'deep_svdd':
                if len(feature.shape) == 3:
                    feature = feature[:, -1, :]
                logits = torch.sum((feature - model.c.to(torch.device(args.device))) ** 2, dim=1)
            elif args.tsadalg == 'tran_ad':
                z = (others['x1'], others['x2'])
                if isinstance(z, tuple): z = z[1]
                logits = z[0]
            anomaly_scores_all.append(logits)
            if isinstance(labels, list):
                labels = np.asarray(labels)
            if len(labels.shape) == 1:
                labels_all.append(labels)
            else:
                labels_all.append(torch.squeeze(labels, dim=1))
            if others is not None and 'output' in others.keys():
                loss.append(criterion(others['output'], x[:, -1, :]).item())
        anomaly_scores_all = torch.cat(anomaly_scores_all, dim=0)
    else:
        def get_err_scores_gdn_federated(test_predict, test_gt):

            n_err_mid, n_err_iqr = get_err_median_and_iqr(test_predict, test_gt)

            test_delta = np.abs(
                    np.subtract(
                            np.array(test_predict).astype(np.float64),
                            np.array(test_gt).astype(np.float64)
                    )
            )
            epsilon = 1e-2

            err_scores = (test_delta - n_err_mid) / (np.abs(n_err_iqr) + epsilon)

            smoothed_err_scores = np.zeros(err_scores.shape)
            before_num = 3
            for i in range(before_num, len(err_scores)):
                smoothed_err_scores[i] = np.mean(err_scores[i - before_num:i + 1])

            return smoothed_err_scores

        def get_full_err_scores_gdn_federated(pred, x, labels):

            all_scores = None
            all_normals = None
            feature_num = pred.shape[-1]

            for i in range(feature_num):
                predi = pred[:, i]
                xi = x[:, i]

                scores = get_err_scores_gdn_federated(predi, xi)

                if all_scores is None:
                    all_scores = scores
                else:
                    all_scores = np.vstack(
                            (
                                    all_scores,
                                    scores
                            )
                    )

            return all_scores

        xs = []
        preds = []
        for i, (x, y, labels, edge_index) in enumerate(testloader):
            x, y, labels = [item.float().to(args.device) for item in [x, y, labels]]
            feature, logits, loss_this = model(x, edge_index, y)
            preds.append(logits)
            xs.append(y)
            labels_all.append(labels)
            loss.append(loss_this)
        preds = torch.cat(preds, dim=0)
        xs = torch.cat(xs, dim=0)
        anomaly_scores_all = get_full_err_scores_gdn_federated(preds.detach().cpu().numpy(), xs.detach().cpu().numpy(), labels)
        total_features = anomaly_scores_all.shape[0]
        topk = 1
        topk_indices = np.argpartition(anomaly_scores_all, range(total_features - topk - 1, total_features), axis=0)[
                       -topk:]
        anomaly_scores_all = np.sum(np.take_along_axis(anomaly_scores_all, topk_indices, axis=0), axis=0)
    labels_all = torch.cat(labels_all, dim=0)
    labels_all_numpy = labels_all.detach().cpu().numpy()
    if isinstance(anomaly_scores_all, torch.Tensor):
        anomaly_scores_all_numpy = anomaly_scores_all.detach().cpu().numpy()
    else:
        anomaly_scores_all_numpy = anomaly_scores_all
    if len(anomaly_scores_all_numpy.shape) == 2:
        anomaly_scores_all_numpy = np.mean(anomaly_scores_all_numpy, axis=1)
    if len(score_save_path) != 0:
        np.save(score_save_path, anomaly_scores_all_numpy)
    auc_roc = roc_auc_score(labels_all_numpy, anomaly_scores_all_numpy)
    ap = average_precision_score(labels_all_numpy, anomaly_scores_all_numpy)
    if len(loss) != 0:
        return auc_roc, ap, float(sum(loss) / len(loss)), anomaly_scores_all_numpy
    else:
        return auc_roc, ap, None, anomaly_scores_all_numpy


@torch.no_grad()
def test_inference_pretrain(model, dataset):
    model.eval()
    num_data = 0
    correct = 0
    if args.tsadalg == 'gdn':
        feature_map = get_feature_map(args.dataset)
        fc_struc = get_fc_graph_struc(args.dataset)
        fc_edge_index = build_loc_net(fc_struc, feature_map, feature_map=feature_map)
        fc_edge_index = torch.tensor(fc_edge_index, dtype=torch.long)
        edge_index_sets = []
        edge_index_sets.append(fc_edge_index)
        test_scaled = dataset.data
        test = pandas.DataFrame(test_scaled, columns=feature_map, dtype=np.float32)
        test_dataset_indata = construct_data(test, feature_map, labels=0)
        cfg = {
                'slide_win': args.slide_win,
                'slide_stride': 1,
        }
        test_dataset = TimeDataset(test_dataset_indata, fc_edge_index, mode='test', config=cfg)
        if len(dataset.target.shape) == 1:
            test_dataset.labels = torch.tensor(dataset.target[:])
        elif len(dataset.target.shape) == 2:
            test_dataset.labels = torch.tensor(dataset.target[:, 0])
        testloader = DataLoader(
                test_dataset, batch_size=128,
                shuffle=False,
                drop_last=False
        )
    else:
        testloader = DataLoader(
                dataset,
                batch_size=128,
                shuffle=False,
                num_workers=args.num_workers,
                worker_init_fn=seed_worker,
                drop_last=False
                # generator=g,
        )
    criterion = nn.MSELoss()
    loss = []
    anomaly_scores_all = []
    labels_all = []
    if args.tsadalg != 'gdn':
        for i, (x, labels) in enumerate(testloader):
            x = x.to(args.device)
            labels = labels.to(args.device)
            feature, logits, others = model(x)
            anomaly_scores_all.append(logits)
            if isinstance(labels, list):
                labels = np.asarray(labels)
            if len(labels.shape) == 1:
                labels_all.append(labels)
            else:
                labels_all.append(torch.squeeze(labels, dim=1))
            if 'output' in others.keys():
                loss.append(criterion(others['output'], x[:, -1, :]).item())
        anomaly_scores_all = torch.cat(anomaly_scores_all, dim=0)
    else:
        def get_err_scores_gdn_federated(test_predict, test_gt):

            n_err_mid, n_err_iqr = get_err_median_and_iqr(test_predict, test_gt)

            test_delta = np.abs(
                    np.subtract(
                            np.array(test_predict).astype(np.float64),
                            np.array(test_gt).astype(np.float64)
                    )
            )
            epsilon = 1e-2

            err_scores = (test_delta - n_err_mid) / (np.abs(n_err_iqr) + epsilon)

            smoothed_err_scores = np.zeros(err_scores.shape)
            before_num = 3
            for i in range(before_num, len(err_scores)):
                smoothed_err_scores[i] = np.mean(err_scores[i - before_num:i + 1])

            return smoothed_err_scores

        def get_full_err_scores_gdn_federated(pred, x, labels):

            all_scores = None
            all_normals = None
            feature_num = pred.shape[-1]

            for i in range(feature_num):
                predi = pred[:, i]
                xi = x[:, i]

                scores = get_err_scores_gdn_federated(predi, xi)

                if all_scores is None:
                    all_scores = scores
                else:
                    all_scores = np.vstack(
                            (
                                    all_scores,
                                    scores
                            )
                    )

            return all_scores

        xs = []
        preds = []
        for i, (x, y, labels, edge_index) in enumerate(testloader):
            x, y, labels = [item.float().to(args.device) for item in [x, y, labels]]
            feature, logits, loss_this = model(x, edge_index, y)
            preds.append(logits)
            xs.append(y)
            labels_all.append(labels)
            loss.append(loss_this)
        preds = torch.cat(preds, dim=0)
        xs = torch.cat(xs, dim=0)
        anomaly_scores_all = get_full_err_scores_gdn_federated(preds.detach().cpu().numpy(), xs.detach().cpu().numpy(), labels)
        total_features = anomaly_scores_all.shape[0]
        topk = 1
        topk_indices = np.argpartition(anomaly_scores_all, range(total_features - topk - 1, total_features), axis=0)[
                       -topk:]
        anomaly_scores_all = np.sum(np.take_along_axis(anomaly_scores_all, topk_indices, axis=0), axis=0)
    labels_all = torch.cat(labels_all, dim=0)
    labels_all_numpy = labels_all.detach().cpu().numpy()
    if isinstance(anomaly_scores_all, torch.Tensor):
        anomaly_scores_all_numpy = anomaly_scores_all.detach().cpu().numpy()
    else:
        anomaly_scores_all_numpy = anomaly_scores_all
    auc_roc = roc_auc_score(labels_all_numpy, anomaly_scores_all_numpy)
    ap = average_precision_score(labels_all_numpy, anomaly_scores_all_numpy)
    return auc_roc, ap, float(sum(loss) / len(loss))
