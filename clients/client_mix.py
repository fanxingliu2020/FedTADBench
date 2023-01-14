from typing import List

import numpy as np
import pandas
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from algorithms.DeepSVDD.DeepSVDD import BaseNet
from algorithms.GDN.evaluate import get_err_median_and_iqr
from algorithms.GDN.gdn_exp_smd import TimeDataset, get_feature_map, get_fc_graph_struc, build_loc_net, construct_data
from logger import logger
from options import args, seed_worker
from task import load_model, config, model_fun

if args.tsadalg == 'deep_svdd':
    from task.SVDD import config_svdd
from sklearn.metrics import roc_auc_score, average_precision_score
import torch.nn.functional as F


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
            self, global_state_dict, global_round, global_grad_correct: dict, global_c: torch.Tensor = None
    ):

        scheduler = None

        # region 准备 model model_prev model_global
        model_global = load_model(global_state_dict)
        model_global.requires_grad_(False)
        model_global.eval()
        model_global.to(args.device)
        #
        model_current = load_model(global_state_dict)
        model_current.requires_grad_(True)
        if args.tsadalg == 'deep_svdd' and config_svdd['stage'] == 'second':
            model_current.c = global_c
        model_current.train()
        model_current.to(args.device)
        # endregion

        epoch_loss = []
        train_acc = []  # 返回的loss和acc都是local round上的平均

        # region Set optimizer and dataloader
        optimizer = self.set_local_optimizer(model_current)
        if args.tsadalg == 'tran_ad':
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 5, 0.9)
        if args.tsadalg == 'deep_svdd':
            scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=(50,), gamma=0.1)
        # if self.dataset.data.shape[0] < self.local_bs:
        #     self.local_bs = self.dataset.data.shape[0]
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
            # trainloader = DataLoader(
            #     train_dataset,
            #     batch_size=self.local_bs,
            #     shuffle=True,
            #     pin_memory=True,
            #     num_workers=args.num_workers,
            #     drop_last=True if args.alg == 'scaffold' else False
            # )

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
                        # get the inputs of the batch
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
            for local_epoch in range(self.local_ep):
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
                    if args.alg == 'moon' and self.state_dict_prev is not None:
                        model_prev = load_model(
                                self.state_dict_prev,
                        )
                        model_prev.requires_grad_(False)
                        model_prev.eval()
                        model_prev.to(args.device)
                        if args.tsadalg == 'tran_ad':
                            local_bs = x.shape[0]
                            feats = x.shape[-1]
                            window = x.permute(1, 0, 2)
                            elem = window[-1, :, :].view(1, local_bs, feats)
                            feature_prev, _, _ = model_prev(window, elem)
                            feature_global, _, _ = model_global(window, elem)
                        else:
                            feature_prev, _, _ = model_prev(x)
                            feature_global, _, _ = model_global(x)
                        featture_flatten = torch.flatten(feature, start_dim=1)
                        feature_global_flatten = torch.flatten(feature_global, start_dim=1)
                        feature_prev_flatten = torch.flatten(feature_prev, start_dim=1)
                        posi = self.cos_sim(featture_flatten, feature_global_flatten)
                        logits_moon = posi.reshape(-1, 1)
                        nega = self.cos_sim(featture_flatten, feature_prev_flatten)
                        logits_moon = torch.cat((logits_moon, nega.reshape(-1, 1)), dim=1)
                        logits_moon /= self.temperature
                        loss_con = F.cross_entropy(
                                logits_moon,
                                torch.zeros(x.size(0), device=args.device, dtype=torch.long)
                        )
                        loss = loss + self.moon_mu * loss_con

                    if args.alg == 'fedprox':
                        proximal_term = 0
                        for w, w_0 in zip(model_current.parameters(), model_global.parameters()):
                            proximal_term += (w - w_0).norm(2)
                        loss += proximal_term * self.prox_mu / 2

                    loss.backward()
                    optimizer.step()

                    # correct grad
                    if args.alg == 'scaffold':
                        state_dict_current = model_current.state_dict()
                        lr = optimizer.state_dict()['param_groups'][-1]['lr']
                        for key in state_dict_current:
                            # if not state_dict_current[key].requires_grad:
                            #     continue
                            if key == 'pos_encoder.pe':
                                continue
                            c_global = global_grad_correct[key].to(args.device)
                            c_local = self.grad_correct[key].to(args.device)
                            state_dict_current[key] -= lr * (c_global - c_local)
                        model_current.load_state_dict(state_dict_current)
                batch_loss.append(loss.item())
                # endregion

                # region log
                epoch_loss.append(sum(batch_loss) / len(batch_loss))
                if args.verbose:
                    logger.print(
                            f'| Global Round : {global_round} | Local Epoch : {local_epoch} |' +
                            f' Training Loss: {loss.item():.6f}\t'
                            # + f' Training Accuracy: {train_acc[-1]:.6f}'
                    )
                # endregion

                if scheduler is not None:
                    scheduler.step()
        elif args.tsadalg == 'usad':
            losses1 = []
            losses2 = []
            for local_epoch in range(self.local_ep):
                batch_loss = []
                correct = 0
                num_data = 0
                opt_func = torch.optim.Adam
                optimizer1 = opt_func(list(model_current.encoder.parameters()) + list(model_current.decoder1.parameters()))
                optimizer2 = opt_func(list(model_current.encoder.parameters()) + list(model_current.decoder2.parameters()))
                for i, (x, y) in enumerate(trainloader):
                    x, y = x.to(args.device), y.to(args.device)
                    x = x.view([x.shape[0], x.shape[1] * x.shape[2]])
                    loss_stat = 0

                    optimizer1.zero_grad()
                    _, _, others = model_current.training_step(x, local_epoch + 1)
                    loss1, _ = others['loss1'], others['loss2']
                    if args.alg == 'fedprox':
                        proximal_term = 0
                        for w, w_0 in zip(model_current.parameters(), model_global.parameters()):
                            proximal_term = proximal_term + (w - w_0).norm(2)
                        loss1 = loss1 + proximal_term * self.prox_mu / 2
                    loss1.backward()
                    losses1.append(loss1.item())
                    optimizer1.step()
                    optimizer1.zero_grad()
                    loss_stat += float(loss1.item())

                    feature, logits, others = model_current.training_step(x, local_epoch + 1)
                    _, loss2 = others['loss1'], others['loss2']
                    if args.alg == 'fedprox':
                        proximal_term = 0
                        for w, w_0 in zip(model_current.parameters(), model_global.parameters()):
                            proximal_term = proximal_term + (w - w_0).norm(2)
                        loss2 = loss2 + proximal_term * self.prox_mu / 2

                    if args.alg == 'moon' and self.state_dict_prev is not None:
                        model_prev = load_model(
                                self.state_dict_prev,
                        )
                        model_prev.requires_grad_(False)
                        model_prev.eval()
                        model_prev.to(args.device)
                        feature_prev, _, _ = model_prev(x)
                        feature_global, _, _ = model_global(x)
                        posi = self.cos_sim(feature, feature_global)
                        logits_moon = posi.reshape(-1, 1)
                        nega = self.cos_sim(feature, feature_prev)
                        logits_moon = torch.cat((logits_moon, nega.reshape(-1, 1)), dim=1)
                        logits_moon /= self.temperature
                        loss_con = F.cross_entropy(
                            logits_moon,
                            torch.zeros(x.size(0), device=args.device, dtype=torch.long)
                        )
                        loss2 = loss2 + self.moon_mu * loss_con
                    loss2.backward()
                    losses2.append(loss2.item())
                    loss_stat += float(loss2.item())
                    optimizer2.step()
                    optimizer2.zero_grad()
                    # correct grad
                    if args.alg == 'scaffold':
                        state_dict_current = model_current.state_dict()
                        lr = optimizer.state_dict()['param_groups'][-1]['lr']
                        for key in state_dict_current:
                            c_global = global_grad_correct[key].to(args.device)
                            c_local = self.grad_correct[key].to(args.device)
                            state_dict_current[key] -= lr * (c_global - c_local)
                        model_current.load_state_dict(state_dict_current)
                mean_loss1 = np.mean(losses1)
                mean_loss2 = np.mean(losses2)
                if scheduler is not None:
                    scheduler.step()
                # endregion
                if args.verbose:
                    logger.print(
                            f'| Global Round : {global_round} | Local Epoch : {local_epoch} |' +
                            f' Training Loss 1: {loss1.item():.6f} |' + f' Training Loss 2: {loss2.item():.6f}\t'
                    )
                # endregion
        else:
            for local_epoch in range(self.local_ep):
                batch_loss = []
                correct = 0
                num_data = 0
                for i, (x, labels, attack_labels, edge_index) in enumerate(trainloader):
                    x, labels, edge_index = [item.float().to(args.device) for item in [x, labels, edge_index]]
                    #
                    optimizer.zero_grad()
                    feature, logits, loss = model_current(x, edge_index, labels)

                    if args.alg == 'moon' and self.state_dict_prev is not None:
                        model_prev = load_model(
                                self.state_dict_prev,
                        )
                        model_prev.requires_grad_(False)
                        model_prev.eval()
                        model_prev.to(args.device)
                        if args.tsadalg == 'gdn':
                            feature_prev, _, _ = model_prev(x, edge_index, labels)
                            feature_global, _, _ = model_global(x, edge_index, labels)
                        else:
                            feature_prev, _, _ = model_prev(x)
                            feature_global, _, _ = model_global(x)
                        feature_flatten = torch.flatten(feature, start_dim=1)
                        feature_global_flatten = torch.flatten(feature_global, start_dim=1)
                        feature_prev_flatten = torch.flatten(feature_prev, start_dim=1)
                        posi = self.cos_sim(feature_flatten, feature_global_flatten)
                        logits_moon = posi.reshape(-1, 1)
                        nega = self.cos_sim(feature_flatten, feature_prev_flatten)
                        logits_moon = torch.cat((logits_moon, nega.reshape(-1, 1)), dim=1)
                        logits_moon /= self.temperature
                        loss_con = F.cross_entropy(
                                logits_moon,
                                torch.zeros(x.size(0), device=args.device, dtype=torch.long)
                        )
                        loss = loss + self.moon_mu * loss_con

                    if args.alg == 'fedprox':
                        proximal_term = 0
                        for w, w_0 in zip(model_current.parameters(), model_global.parameters()):
                            proximal_term += (w - w_0).norm(2)
                        loss += proximal_term * self.prox_mu / 2

                    loss.backward()
                    optimizer.step()
                    # correct grad
                    if args.alg == 'scaffold':
                        state_dict_current = model_current.state_dict()
                        lr = optimizer.state_dict()['param_groups'][-1]['lr']
                        for key in state_dict_current:

                            try:
                                c_global = global_grad_correct[key].to(args.device)
                                c_local = self.grad_correct[key].to(args.device)
                                state_dict_current[key] -= lr * (c_global - c_local)
                            except:
                                continue
                        model_current.load_state_dict(state_dict_current)
                batch_loss.append(loss.item())
                # endregion

                # region log
                epoch_loss.append(sum(batch_loss) / len(batch_loss))
                if scheduler is not None:
                    scheduler.step()
                if args.verbose:
                    logger.print(
                            f'| Global Round : {global_round} | Local Epoch : {local_epoch} |' +
                            f' Training Loss: {loss.item():.6f}\t'
                    )
                # endregion

        if args.tsadalg != 'usad':
            mean_loss = sum(epoch_loss) / len(epoch_loss)

        # region
        if args.alg == 'scaffold':

            c_delta_local = {}
            state_global = model_global.state_dict()
            state_current = model_current.state_dict()

            for key in self.grad_correct.keys():
                old_c = self.grad_correct[key].to(args.device)
                new_c = old_c - global_grad_correct[key].to(args.device) + \
                        (state_global[key].to(args.device) - state_current[key].to(args.device)) / (len(trainloader) * self.local_ep * lr)
                self.grad_correct[key] = new_c.cpu()
                c_delta_local[key] = (new_c - old_c).cpu()
        else:
            c_delta_local = None
        # endregion

        model_current.cpu()
        self.state_dict_prev = model_current.state_dict()
        # endregion

        if args.tsadalg == 'deep_svdd' and config_svdd['stage'] == 'second':
            return float(mean_loss), None, c_delta_local, model_current.c
        elif args.tsadalg == 'usad':
            return float(mean_loss1), float(mean_loss2), c_delta_local
        else:
            return float(mean_loss), None, c_delta_local


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
            test_dataset.labels = torch.tensor(dataset.target[args.slide_win:, 0])

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
    print(anomaly_scores_all_numpy.shape)
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
            test_dataset.labels = torch.tensor(dataset.target[args.slide_win:, 0])
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
            # print(labels.shape)
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
