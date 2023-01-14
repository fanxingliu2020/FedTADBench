import random
from typing import List

from torch.utils.data import DataLoader, Dataset, Subset


class Dataset_wrapper(Dataset):
    def __init__(self, original_dataset: Dataset, avliable_idxs: List[int]):
        self.original_dataset = original_dataset
        self.avliable_idxs = avliable_idxs

    def __len__(self):
        return len(self.avliable_idxs)

    def __getitem__(self, idx):
        x, y = self.original_dataset[self.avliable_idxs[idx]]
        return x, y


def uniformly_split_list(l: list, num_sublist) -> List[list]:
    num_item_per_shard = round(len(l) / num_sublist)
    if len(l) - num_sublist * num_item_per_shard > 0:
        shard_intervals = list(range(0, len(l), num_item_per_shard))
        shard_intervals[-1] = len(l)
    else:
        shard_intervals = list(range(0, len(l), num_item_per_shard))
        shard_intervals.append(len(l))
    sublists = []
    for i in range(len(shard_intervals) - 1):
        sublists.append(l[shard_intervals[i]:shard_intervals[i + 1]])
    return sublists
