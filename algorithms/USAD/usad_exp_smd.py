import os
import random
import time

import sklearn
from torch.utils.data import DataLoader

import sys



sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "../..")))
from convert_time import convert_time

from algorithms.USAD.USAD import *

import matplotlib.pyplot as plt
import seaborn as sns
import sklearn.metrics
import torch

from sklearn.metrics import roc_curve, roc_auc_score, average_precision_score

from algorithms.read_datasets import *


def get_default_device():
    """Pick GPU if available, else CPU"""
    if torch.cuda.is_available():
        return torch.device('cuda:0')
    else:
        return torch.device('cpu')


def to_device(data, device):
    """Move tensor(s) to chosen device"""
    if isinstance(data, (list, tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)


def plot_history(history):
    losses1 = [x['val_loss1'] for x in history]
    losses2 = [x['val_loss2'] for x in history]
    plt.plot(losses1, '-x', label="loss1")
    plt.plot(losses2, '-x', label="loss2")
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend()
    plt.title('Losses vs. No. of epochs')
    plt.grid()
    plt.show()


def histogram(y_test, y_pred):
    plt.figure(figsize=(12, 6))
    plt.hist([y_pred[y_test == 0],
              y_pred[y_test == 1]],
             bins=20,
             color=['#82E0AA', '#EC7063'], stacked=True)
    plt.title("Results", size=20)
    plt.grid()
    plt.show()


def ROC(y_test, y_pred):
    fpr, tpr, tr = roc_curve(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred)
    idx = np.argwhere(np.diff(np.sign(tpr - (1 - fpr)))).flatten()

    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.plot(fpr, tpr, label="AUC=" + str(auc))
    plt.plot(fpr, 1 - fpr, 'r:')
    plt.plot(fpr[idx], tpr[idx], 'ro')
    plt.legend(loc=4)
    plt.grid()
    plt.show()
    print('AUC:', auc)
    plt.savefig('auc.jpg')

    threshold = tr[idx]

    y_pred_from_threshold = [1 if y_pred[i] >= threshold else 0 for i in range(y_pred.shape[0])]
    y_pred_from_threshold = np.asarray(y_pred_from_threshold)
    precision = sklearn.metrics.precision_score(y_test, y_pred_from_threshold)
    recall = sklearn.metrics.recall_score(y_test, y_pred_from_threshold)
    f1 = sklearn.metrics.f1_score(y_test, y_pred_from_threshold)

    return tr[idx], auc, precision, recall, f1


def confusion_matrix(target, predicted, perc=False):
    data = {'y_Actual': target,
            'y_Predicted': predicted
            }
    df = pd.DataFrame(data, columns=['y_Predicted', 'y_Actual'])
    confusion_matrix = pd.crosstab(df['y_Predicted'], df['y_Actual'], rownames=['Predicted'], colnames=['Actual'])

    if perc:
        sns.heatmap(confusion_matrix / np.sum(confusion_matrix), annot=True, fmt='.2%', cmap='Blues')
    else:
        sns.heatmap(confusion_matrix, annot=True, fmt='d')
    plt.show()

device = get_default_device()

def main(dataset='psm', random_seed=42, n_epochs=100, hidden_size=100, w_size=5):

    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    os.environ['PYTHONHASHSEED'] = str(random_seed)

    if dataset == 'smd' or dataset == 'smap' or dataset == 'psm':
        if dataset == 'smd':
            train_data = SMD_Dataset(train=True, window_len=w_size)
            test_data = SMD_Dataset(train=False, window_len=w_size)
        elif dataset == 'smap':
            train_data = SMAP_Dataset(train=True, window_len=w_size)
            test_data = SMAP_Dataset(train=False, window_len=w_size)
        elif dataset == 'psm':
            train_data = PSM_Dataset(train=True, window_len=w_size)
            test_data = PSM_Dataset(train=False, window_len=w_size)

    normal = train_data
    attack = test_data

    train_loader = DataLoader(normal, batch_size=64, shuffle=False, num_workers=0)
    test_loader = DataLoader(attack, batch_size=64, shuffle=False, num_workers=0)

    model = UsadModel(w_size * normal.data.shape[-1], hidden_size)
    model = to_device(model, device)

    optimizer1 = torch.optim.Adam(list(model.encoder.parameters()) + list(model.decoder1.parameters()), lr=0.001)
    optimizer2 = torch.optim.Adam(list(model.encoder.parameters()) + list(model.decoder2.parameters()), lr=0.001)

    best_auc_roc = 0
    best_ap = 0

    model_save_path = os.path.abspath(os.path.join(os.getcwd(), "../..")) + '/fltsad/pths/usad_' + dataset + '.pth'
    score_save_path = os.path.abspath(os.path.join(os.getcwd(), "../..")) + '/fltsad/scores/usad_' + dataset + '.npy'



    for epoch in range(n_epochs):
        time_start = time.time()
        model.train()
        for x, y in train_loader:
            x = to_device(x, device)
            y = to_device(y, device)

            x = x.view([x.shape[0], x.shape[1] * x.shape[2]])

            # Train AE1
            z, score, others = model.training_step(x, epoch + 1)
            loss1, loss2 = others['loss1'], others['loss2']
            loss1.backward()
            optimizer1.step()
            optimizer1.zero_grad()

            # Train AE2
            z, score, others = model.training_step(x, epoch + 1)
            loss1, loss2 = others['loss1'], others['loss2']
            loss2.backward()
            optimizer2.step()
            optimizer2.zero_grad()


        time_end = time.time()

        if len(test_loader.dataset.target.shape) > 1:
            labels = test_loader.dataset.target[:, 0]
        else:
            labels = test_loader.dataset.target[:]
        results = []
        alpha = 0.5
        beta = 0.5
        model.eval()
        for x, y in test_loader:
            batch = x
            batch = to_device(batch, device)
            batch = batch.view(batch.shape[0], batch.shape[1] * batch.shape[2])
            w1 = model.decoder1(model.encoder(batch))
            w2 = model.decoder2(model.encoder(w1))
            results.append(alpha * torch.mean((batch - w1) ** 2, axis=1) + beta * torch.mean((batch - w2) ** 2, axis=1))

        y_pred = np.concatenate([torch.stack(results[:-1]).flatten().detach().cpu().numpy(),
                                 results[-1].flatten().detach().cpu().numpy()])

        auc_roc = roc_auc_score(labels, y_pred)
        ap = average_precision_score(labels, y_pred)
        print('epoch', epoch, 'auc_roc:', auc_roc, 'auc_pr:', ap, 'time:', str(convert_time(time_end - time_start)))

        # print('ap:', ap)

        if auc_roc > best_auc_roc:
            best_auc_roc = auc_roc
            best_ap = ap
            torch.save(model.state_dict(), model_save_path)
            np.save(score_save_path, y_pred)



    print('Best auc_roc:', best_auc_roc)
    print('Best ap:', best_ap)
    print('Total time:', convert_time(time_end - time_start))


if __name__ == '__main__':

    main(dataset='smd', random_seed=42, n_epochs=100, hidden_size=100, w_size=12)