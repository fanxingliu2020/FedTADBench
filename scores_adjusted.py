import warnings

from sklearn.exceptions import UndefinedMetricWarning
from sklearn.metrics import roc_auc_score, average_precision_score, auc, confusion_matrix
import numpy as np

def adjust_scores_for_time_series(y_true, y_score, threshold):

    pred = np.where(y_score >= threshold, 1, 0)
    starts_ends = []
    if isinstance(y_true, list):
        y_true = np.asarray(y_true)
    for i in range(y_true.shape[0]):
        if (i == 0 or y_true[i - 1] == 0) and y_true[i] == 1:
            starts_ends.append([i,])
        if (not i == 0) and y_true[i - 1] == 1 and y_true[i] == 0:
            starts_ends[-1].append(i)
        if len(starts_ends) != 0 and len(starts_ends[-1]) == 1:
            starts_ends[-1].append(y_true.shape[0])
    for i in range(len(starts_ends)):
        if np.sum(pred[starts_ends[i][0]: starts_ends[i][1]]) > 0:
            pred[starts_ends[i][0]: starts_ends[i][1]] = 1

    cnf_matrix = confusion_matrix(y_true, pred)
    FP = cnf_matrix[0, 1]
    FN = cnf_matrix[1, 0]
    TP = cnf_matrix[1, 1]
    TN = cnf_matrix[0, 0]
    FP = FP.astype(float)
    FN = FN.astype(float)
    TP = TP.astype(float)
    TN = TN.astype(float)
    TPR = TP / (TP + FN)
    FPR = FP / (FP + TN)
    precision = TP / (TP + FP)
    recall = TPR
    return FPR, TPR, precision, recall

def roc_curve_adjusted_for_time_series(y_true, y_score):

    desc_score_indices = np.argsort(y_score, kind="mergesort")[::-1]
    y_score_desc = y_score[desc_score_indices]

    fprs = []
    tprs = []
    precisions = []
    recalls = []
    thresholds = []

    for score in y_score_desc:
        fpr, tpr, precision, recall = adjust_scores_for_time_series(y_true, y_score, score)
        fprs.append(fpr)
        tprs.append(tpr)
        precisions.append(precision)
        recalls.append(recall)
        thresholds.append(score)

    return auc(fprs, tprs), -np.sum(np.diff(recalls) * np.array(precisions)[:-1])


if __name__ == '__main__':
    scores = np.asarray([1, 2, 3, 4])
    ys = np.asarray([0, 0, 0, 1])

    aps = average_precision_score(ys, scores)
    print(aps)
    auc_roc, ap = roc_curve_adjusted_for_time_series(ys, scores)
    print(auc_roc, ap)