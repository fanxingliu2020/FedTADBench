import random

import numpy as np
import torch


def mean(a_list):
    return sum(a_list) / len(a_list)


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
