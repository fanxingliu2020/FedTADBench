import numpy as np

from datasets.MOON_util import partition_data, get_dataset
import os

current_path = os.getcwd()
data_dir = current_path + '/data/datasets/smd/SMD/raw'
train_path = current_path + '/data/datasets/smd/SMD/raw/train'
test_path = current_path + '/data/datasets/smd/SMD/raw/test'
test_labels_path = current_path + '/data/datasets/smd/SMD/raw/test_label'

def smd_noniid():
    train_ds_locals, test_ds_locals = [None] * 28, [None] * 28
    chosen_idxes = [i for i in range(28)]
    for i in range(len(chosen_idxes)):
        dataidxs = chosen_idxes[i]
        train_ds_locals[i], test_ds_locals[i] = get_dataset(
            "smd", data_dir, dataidxs
        )
    return train_ds_locals


def smd_iid():
    train_ds_locals, test_ds_locals = [None] * 28, [None] * 28
    chosen_idxes = [i for i in range(28)]
    for i in range(len(chosen_idxes)):
        dataidxs = chosen_idxes[i]
        train_ds_locals[i], test_ds_locals[i] = get_dataset(
            "smd", data_dir, dataidxs
        )
    return train_ds_locals


_, test_dataset = get_dataset(
    "smd",
    data_dir,
    None,
    32
)

client_datasets_non_iid = smd_noniid()
client_datasets_iid = smd_iid()
