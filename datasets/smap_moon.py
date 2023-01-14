import numpy as np

from datasets.MOON_util import partition_data, get_dataset
import os

current_path = os.getcwd()
data_dir = current_path + '/data/datasets/smap/raw'
train_path = current_path + '/data/datasets/smap/raw/train'
test_path = current_path + '/data/datasets/smap/raw/test'
test_labels_path = current_path + '/data/datasets/smap/raw/test_label'

def smap_noniid():
    train_ds_locals, test_ds_locals = [None] * 54, [None] * 54
    # chosen_idxes = np.random.choice([i for i in range(54)], 10)
    chosen_idxes = [i for i in range(54)]
    for i in range(len(chosen_idxes)):
        dataidxs = chosen_idxes[i]
        train_ds_locals[i], test_ds_locals[i] = get_dataset(
            "smap", data_dir, dataidxs
        )
    return train_ds_locals


def smap_iid():
    train_ds_locals, test_ds_locals = [None] * 54, [None] * 54
    chosen_idxes = [i for i in range(54)]
    for i in range(len(chosen_idxes)):
        dataidxs = chosen_idxes[i]
        train_ds_locals[i], test_ds_locals[i] = get_dataset(
            "smap", data_dir, dataidxs
        )
    return train_ds_locals


_, test_dataset = get_dataset(
    "smap",
    data_dir,
    None,
    32
)

client_datasets_non_iid = smap_noniid()
client_datasets_iid = smap_iid()
