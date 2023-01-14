from sklearn.metrics import mean_squared_error, average_precision_score
from algorithms.GDN.evaluate import get_best_performance_data, get_val_performance_data, get_full_err_scores, \
    get_full_err_scores_only_test
from sklearn.metrics import precision_score, recall_score, roc_auc_score, f1_score
import math
from datetime import datetime

import torch
import torch.nn as nn
import time
import torch.nn.functional as F

from scipy.stats import rankdata, iqr, trim_mean
from sklearn.metrics import f1_score, mean_squared_error
from numpy import percentile

# preprocess data
import numpy as np

def get_most_common_features(target, all_features, max=3, min=3):
    res = []
    main_keys = target.split('_')

    for feature in all_features:
        if target == feature:
            continue

        f_keys = feature.split('_')
        common_key_num = len(list(set(f_keys) & set(main_keys)))

        if common_key_num >= min and common_key_num <= max:
            res.append(feature)

    return res


def build_net(target, all_features):
    # get edge_indexes, and index_feature_map
    main_keys = target.split('_')
    edge_indexes = [
        [],
        []
    ]
    index_feature_map = [target]

    # find closest features(nodes):
    parent_list = [target]
    graph_map = {}
    depth = 2

    for i in range(depth):
        for feature in parent_list:
            children = get_most_common_features(feature, all_features)

            if feature not in graph_map:
                graph_map[feature] = []

            # exclude parent
            pure_children = []
            for child in children:
                if child not in graph_map:
                    pure_children.append(child)

            graph_map[feature] = pure_children

            if feature not in index_feature_map:
                index_feature_map.append(feature)
            p_index = index_feature_map.index(feature)
            for child in pure_children:
                if child not in index_feature_map:
                    index_feature_map.append(child)
                c_index = index_feature_map.index(child)

                edge_indexes[1].append(p_index)
                edge_indexes[0].append(c_index)

        parent_list = pure_children

    return edge_indexes, index_feature_map


def construct_data(data, feature_map, labels=0):
    res = []

    for feature in feature_map:
        if feature in data.columns:
            res.append(data.loc[:, feature].values.tolist())
        else:
            print(feature, 'not exist in data')
    # append labels as last
    sample_n = len(res[0])

    if type(labels) == int:
        res.append([labels] * sample_n)
    elif len(labels) == sample_n:
        res.append(labels)

    return res


def build_loc_net(struc, all_features, feature_map=[]):
    index_feature_map = feature_map
    edge_indexes = [
        [],
        []
    ]
    for node_name, node_list in struc.items():
        if node_name not in all_features:
            continue

        if node_name not in index_feature_map:
            index_feature_map.append(node_name)

        p_index = index_feature_map.index(node_name)
        for child in node_list:
            if child not in all_features:
                continue

            if child not in index_feature_map:
                print(f'error: {child} not in index_feature_map')
                # index_feature_map.append(child)

            c_index = index_feature_map.index(child)
            # edge_indexes[0].append(p_index)
            # edge_indexes[1].append(c_index)
            edge_indexes[0].append(c_index)
            edge_indexes[1].append(p_index)

    return edge_indexes


def get_attack_interval(attack):
    heads = []
    tails = []
    for i in range(len(attack)):
        if attack[i] == 1:
            if attack[i - 1] == 0:
                heads.append(i)

            if i < len(attack) - 1 and attack[i + 1] == 0:
                tails.append(i)
            elif i == len(attack) - 1:
                tails.append(i)
    res = []
    for i in range(len(heads)):
        res.append((heads[i], tails[i]))
    # print(heads, tails)
    return res


# calculate F1 scores
def eval_scores(scores, true_scores, th_steps, return_thresold=False):
    padding_list = [0] * (len(true_scores) - len(scores))
    # print(padding_list)

    if len(padding_list) > 0:
        scores = padding_list + scores

    scores_sorted = rankdata(scores, method='ordinal')
    th_steps = th_steps
    # th_steps = 500
    th_vals = np.array(range(th_steps)) * 1.0 / th_steps
    fmeas = [None] * th_steps
    thresholds = [None] * th_steps
    for i in range(th_steps):
        cur_pred = scores_sorted > th_vals[i] * len(scores)

        fmeas[i] = f1_score(true_scores, cur_pred)

        score_index = scores_sorted.tolist().index(int(th_vals[i] * len(scores) + 1))
        thresholds[i] = scores[score_index]

    if return_thresold:
        return fmeas, thresholds
    return fmeas


def eval_mseloss(predicted, ground_truth):
    ground_truth_list = np.array(ground_truth)
    predicted_list = np.array(predicted)

    # return loss
    loss = mean_squared_error(predicted_list, ground_truth_list)

    return loss


def get_err_median_and_iqr(predicted, groundtruth):
    np_arr = np.abs(np.subtract(np.array(predicted), np.array(groundtruth)))

    err_median = np.median(np_arr)
    err_iqr = iqr(np_arr)

    return err_median, err_iqr


def get_err_median_and_quantile(predicted, groundtruth, percentage):
    np_arr = np.abs(np.subtract(np.array(predicted), np.array(groundtruth)))

    err_median = np.median(np_arr)
    # err_iqr = iqr(np_arr)
    err_delta = percentile(np_arr, int(percentage * 100)) - percentile(np_arr, int((1 - percentage) * 100))

    return err_median, err_delta


def get_err_mean_and_quantile(predicted, groundtruth, percentage):
    np_arr = np.abs(np.subtract(np.array(predicted), np.array(groundtruth)))

    err_median = trim_mean(np_arr, percentage)
    # err_iqr = iqr(np_arr)
    err_delta = percentile(np_arr, int(percentage * 100)) - percentile(np_arr, int((1 - percentage) * 100))

    return err_median, err_delta


def get_err_mean_and_std(predicted, groundtruth):
    np_arr = np.abs(np.subtract(np.array(predicted), np.array(groundtruth)))

    err_mean = np.mean(np_arr)
    err_std = np.std(np_arr)

    return err_mean, err_std


def get_f1_score(scores, gt, contamination):
    padding_list = [0] * (len(gt) - len(scores))
    # print(padding_list)

    threshold = percentile(scores, 100 * (1 - contamination))

    if len(padding_list) > 0:
        scores = padding_list + scores

    pred_labels = (scores > threshold).astype('int').ravel()

    return f1_score(gt, pred_labels)


def test(model, dataloader):
    # test
    loss_func = nn.MSELoss(reduction='mean')
    device = get_device()

    test_loss_list = []
    now = time.time()

    test_predicted_list = []
    test_ground_list = []
    test_labels_list = []

    t_test_predicted_list = []
    t_test_ground_list = []
    t_test_labels_list = []

    test_len = len(dataloader)

    model.eval()

    i = 0
    acu_loss = 0
    for x, y, labels, edge_index in dataloader:
        x, y, labels, edge_index = [item.to(device).float() for item in [x, y, labels, edge_index]]

        with torch.no_grad():

            labels = labels.unsqueeze(1).repeat(1, x.shape[1])
            _, predicted, loss = model(x, edge_index, labels)

            if len(t_test_predicted_list) <= 0:
                t_test_predicted_list = predicted
                t_test_ground_list = y
                t_test_labels_list = labels
            else:
                t_test_predicted_list = torch.cat((t_test_predicted_list, predicted), dim=0)
                t_test_ground_list = torch.cat((t_test_ground_list, y), dim=0)
                t_test_labels_list = torch.cat((t_test_labels_list, labels), dim=0)

        test_loss_list.append(loss.item())
        acu_loss += loss.item()

        i += 1

        if i % 10000 == 1 and i > 1:
            print(timeSincePlus(now, i / test_len))

    test_predicted_list = t_test_predicted_list.tolist()
    test_ground_list = t_test_ground_list.tolist()
    test_labels_list = t_test_labels_list.tolist()

    avg_loss = sum(test_loss_list) / len(test_loss_list)

    return avg_loss, [test_predicted_list, test_ground_list, test_labels_list]


_device = torch.device('cuda:0')

def get_device():
    return _device

def set_device(dev):
    global _device
    _device = dev

def init_work(worker_id, seed):
    np.random.seed(seed + worker_id)


def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def timeSincePlus(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (asMinutes(s), asMinutes(rs))


def timeSince(since):
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

def timestamp2str(sec, fmt, tz):
    return datetime.fromtimestamp(sec).astimezone(tz).strftime(fmt)


def loss_func(y_pred, y_true):
    loss = F.mse_loss(y_pred, y_true, reduction='mean')

    return loss



def train(model = None, save_path = '', config={},  train_dataloader=None, val_dataloader=None, feature_map={}, test_dataloader=None, test_dataset=None, dataset_name='swat', train_dataset=None, score_save_path=''):

    seed = config['seed']

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=config['decay'], betas=(0.9, 0.99))

    now = time.time()
    
    train_loss_list = []
    cmp_loss_list = []

    device = get_device()

    best_auc_roc = 0
    best_ap = 0


    acu_loss = 0
    min_loss = 1e+8
    min_f1 = 0
    min_pre = 0
    best_prec = 0

    i = 0
    epoch = config['epoch']
    # early_stop_win = 15
    early_stop_win = 10

    model.train()

    log_interval = 1000
    stop_improve_count = 0

    dataloader = train_dataloader

    for i_epoch in range(epoch):

        acu_loss = 0
        model.train()

        for x, labels, attack_labels, edge_index in dataloader:
            _start = time.time()

            x, labels, edge_index = [item.float().to(device) for item in [x, labels, edge_index]]

            optimizer.zero_grad()
            _, out, loss = model(x, edge_index, labels)
            
            loss.backward()
            optimizer.step()

            
            train_loss_list.append(loss.item())
            acu_loss += loss.item()
                
            i += 1


        # each epoch
        print('epoch ({} / {}) (Loss:{:.8f}, ACU_loss:{:.8f})'.format(
                        i_epoch, epoch, 
                        acu_loss/len(dataloader), acu_loss), flush=True
             , end=' ')

        _, test_result = test(model, test_dataloader)

        def get_score_only_test(test_result):

            feature_num = len(test_result[0][0])
            np_test_result = np.array(test_result)

            test_labels = np_test_result[2, :, 0].tolist()

            test_scores = get_full_err_scores_only_test(test_result)

            top1_best_info = get_best_performance_data(test_scores, test_labels, topk=1)

            info = top1_best_info

            topk = 1
            total_features = test_scores.shape[0]
            topk_indices = np.argpartition(test_scores, range(total_features - topk - 1, total_features), axis=0)[
                           -topk:]
            total_topk_err_scores = np.sum(np.take_along_axis(test_scores, topk_indices, axis=0), axis=0)

            auc_roc = roc_auc_score(test_labels, total_topk_err_scores)
            ap = average_precision_score(test_labels, total_topk_err_scores)
            return auc_roc, ap, total_topk_err_scores

        auc_roc, ap, scores = get_score_only_test(test_result)
        print('auc_roc:', auc_roc, 'auc_pr:', ap)

        if auc_roc > best_auc_roc:
            best_auc_roc = auc_roc
            best_ap = ap
            torch.save(model.state_dict(), save_path)
            np.save(score_save_path, scores)

    return train_loss_list, best_auc_roc, best_ap
