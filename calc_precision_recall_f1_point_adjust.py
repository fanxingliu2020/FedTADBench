import math

import numpy as np
import os
import sys

import pandas as pd
import sklearn
from numpy import mean
from sklearn.metrics import roc_curve, roc_auc_score

sys.path.append(os.path.abspath(os.path.join(os.getcwd())))

from algorithms.TranAD.adjust_predicts import adjust_predicts_from_tranad

def scores_to_dict():
    scores_main_exp = {}
    scores_beta_exp = {}
    scores_average_exp = {}
    scores_centralized_exp = {}
    scores_path = os.path.abspath(os.path.join(os.getcwd())) + '/fltsad/scores/'
    score_files = os.listdir(scores_path)

    for score_file in score_files:
        names = score_file.split('.')[0]
        names = names.replace('lstm_ae', 'lstmae')
        names = names.replace('tran_ad', 'tranad')
        names = names.replace('deep_svdd', 'deepsvdd')
        names = names.split('_')
        # print(names)
        key = score_file
        value = np.load(scores_path + score_file)
        if len(names) == 2:
            scores_centralized_exp[key] = value
        elif len(names) == 3:
            scores_main_exp[key] = value
        elif len(names) == 5:
            scores_beta_exp[key] = value
    scores_path = os.path.abspath(os.path.join(os.getcwd())) + '/fltsad/scores_average/'
    score_files = os.listdir(scores_path)
    score_files.sort()

    for score_file in score_files:
        if not 'best' in score_file:
            continue
        if 'fedavg' in score_file or 'fedprox' in score_file or 'scaffold' in score_file or 'moon' in score_file:
            continue
        value = np.load(scores_path + score_file)
        score_file = score_file.split('.')[0]
        score_file = score_file.replace('lstm_ae', 'lstmae')
        score_file = score_file.replace('tran_ad', 'tranad')
        score_file = score_file.replace('deep_svdd', 'deepsvdd')
        names = score_file.split('_')
        if names[0] + '_' + names[1] in scores_average_exp.keys():
            scores_average_exp[names[0] + '_' + names[1]].append(value)
        else:
            scores_average_exp[names[0] + '_' + names[1]] = [value]

    # SMD labels
    test_target_path = os.path.abspath(os.path.join(os.getcwd())) + '/data_fl/datasets/smd/SMD/raw/test_label'
    file_names = os.listdir(test_target_path)
    file_names.sort()
    target = []
    for file_name in file_names:
        with open(test_target_path + '/' + file_name) as f:
            this_target = []
            for line in f.readlines():
                this_target.append(line.split(','))
            this_target = np.asarray(this_target)
            this_target = this_target.astype(np.float32)
        target.append(this_target)
    SMD_labels = np.concatenate(target, axis=0)

    # SMAP labels
    test_target_path = os.path.abspath(os.path.join(os.getcwd())) + '/data_fl/datasets/smap/raw/test_label'
    file_names = os.listdir(test_target_path)
    file_names.sort()
    target = []
    for file_name in file_names:
        with open(test_target_path + '/' + file_name) as f:
            this_target = []
            for line in f.readlines():
                this_target.append(line.split(','))
            this_target = np.asarray(this_target)
            this_target = this_target.astype(np.float32)
        target.append(this_target)
    SMAP_labels = np.concatenate(target, axis=0)

    # PSM labels
    test_target_path = os.path.abspath(os.path.join(os.getcwd())) + '/data_fl/datasets/psm/raw/test_label.csv'
    target_csv = pd.read_csv(test_target_path)
    target_csv.drop(columns=[r'timestamp_(min)'], inplace=True)
    target = target_csv.values
    PSM_labels = target.astype(np.float32)

    labels_dict = {
        'smd': SMD_labels,
        'smap': SMAP_labels,
        'psm': PSM_labels
    }

    for k, v in scores_centralized_exp.items():
        try:
            k = k.replace('lstm_ae', 'lstmae').replace('tran_ad', 'tranad').replace('deep_svdd', 'deepsvdd')
            names = k.split('.')[0].split('_')
            tsadalg = names[0]
            dataset = names[1]
            labels = labels_dict[dataset.lower()]
            print(tsadalg, dataset, end=' ')
            if tsadalg == 'gdn':
                get_threshold_2(labels[5:, 0], v)
            else:
                get_threshold_2(labels[:, 0], v)
        except:
            print(1)

    for k, v in scores_main_exp.items():
        try:
            k = k.replace('lstm_ae', 'lstmae').replace('tran_ad', 'tranad').replace('deep_svdd', 'deepsvdd')
            names = k.split('.')[0].split('_')
            alg = names[0]
            tsadalg = names[1]
            dataset = names[2]
            labels = labels_dict[dataset.lower()]
            # print(k, v)
            print(tsadalg, dataset, alg, end=' ')
            if tsadalg == 'gdn':
                get_threshold_2(labels[5:, 0], v)
            else:
                get_threshold_2(labels[:, 0], v)
        except:
            print(1)

    for k, v in scores_beta_exp.items():
        try:
            k = k.replace('lstm_ae', 'lstmae').replace('tran_ad', 'tranad').replace('deep_svdd', 'deepsvdd')
            names = k.split('.')[0].split('_')
            alg = names[0]
            tsadalg = names[1]
            dataset = names[2]
            beta = names[4]
            labels = labels_dict[dataset.lower()]
            # print(k, v)
            print(tsadalg, dataset, alg, 'beta_' + str(beta), end=' ')
            if tsadalg == 'gdn':
                get_threshold_2(labels[5:, 0], v)
            else:
                get_threshold_2(labels[:, 0], v)
        except:
            print(1)

    for k, v in scores_average_exp.items():
        try:
            k = k.replace('lstm_ae', 'lstmae').replace('tran_ad', 'tranad').replace('deep_svdd', 'deepsvdd')
            names = k.split('.')[0].split('_')
            tsadalg = names[0]
            dataset = names[1]
            precisions, recalls, f1s, precisions_adjusted, recalls_adjusted, f1s_adjusted = [], [], [], [], [], []
            for s in v:
                labels = labels_dict[dataset.lower()]
                if tsadalg == 'gdn':
                    auc, precision, recall, f1, precision_adjusted, recall_adjusted, f1_adjusted = get_threshold_2(labels[5:, 0], s, print_or_not=False)
                else:
                    auc, precision, recall, f1, precision_adjusted, recall_adjusted, f1_adjusted = get_threshold_2(labels[:, 0], s, print_or_not=False)

                precisions.append(precision), recalls.append(recall), f1s.append(f1)
                precisions_adjusted.append(precision_adjusted), recalls_adjusted.append(recall_adjusted), f1s_adjusted.append(f1_adjusted)
            precision, recall, f1, precision_adjusted, recall_adjusted, f1_adjusted = mean(precisions), mean(recalls), mean(f1s),\
                                                                                      mean(precisions_adjusted), mean(recalls_adjusted), mean(f1s_adjusted)
            print(tsadalg, dataset, 'average', end=' ')
            print('auc:', auc, 'precision:', precision, 'recall:', recall, 'f1:', f1, 'precision_adjusted:', precision_adjusted, 'recall_adjusted:', recall_adjusted, 'f1_adjusted:', f1_adjusted)
        except:
            print(1)





def get_threshold_2(labels, scores, print_or_not=True):
    auc = roc_auc_score(labels, scores)
    thresholds_0 = scores.copy()
    thresholds_0.sort()
    thresholds = []
    for i in range(thresholds_0.shape[0]):
        if i % 1000 == 0 or i == thresholds_0.shape[0] - 1:
            thresholds.append(thresholds_0[i])

    best_precision = 0
    best_recall = 0
    best_f1 = 0
    best_threshold = math.inf
    best_f1_adjusted = 0
    best_precision_adjusted = 0
    best_recall_adjusted = 0


    for threshold in thresholds:
        y_pred_from_threshold = [1 if scores[i] >= threshold else 0 for i in range(scores.shape[0])]
        y_pred_from_threshold = np.asarray(y_pred_from_threshold)
        precision = sklearn.metrics.precision_score(labels, y_pred_from_threshold)
        recall = sklearn.metrics.recall_score(labels, y_pred_from_threshold)
        f1 = sklearn.metrics.f1_score(labels, y_pred_from_threshold)

        y_pred_adjusted = adjust_predicts_from_tranad(labels, scores, pred=y_pred_from_threshold, threshold=threshold)
        precision_adjusted = sklearn.metrics.precision_score(labels, y_pred_adjusted)
        recall_adjusted = sklearn.metrics.recall_score(labels, y_pred_adjusted)
        f1_adjusted = sklearn.metrics.f1_score(labels, y_pred_adjusted)

        if f1_adjusted > best_f1_adjusted:
            best_precision = precision
            best_recall = recall
            best_f1 = f1
            best_f1_adjusted = f1_adjusted
            best_precision_adjusted = precision_adjusted
            best_recall_adjusted = recall_adjusted
            best_threshold = threshold


    if print_or_not:
        print('auc:', auc, 'f1:', best_f1, 'precision_adjusted:', best_precision_adjusted, 'recall_adjusted:', best_recall_adjusted, 'f1_adjusted:', best_f1_adjusted, 'threshold:',  best_threshold)

    return auc, best_precision, best_recall, best_f1, best_precision_adjusted, best_recall_adjusted, best_f1_adjusted




if __name__ == '__main__':
    scores_to_dict()