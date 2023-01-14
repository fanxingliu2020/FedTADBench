from torch import nn
from algorithms.DeepSVDD.SVDD import Model_first_stage, Model_second_stage
from options import args
if args.dataset == 'psm':
    from datasets.psm_moon import client_datasets_non_iid, client_datasets_iid
if args.dataset == 'smap':
    from datasets.smap_moon import client_datasets_non_iid, client_datasets_iid
if args.dataset == 'smd':
    from datasets.smd_moon import client_datasets_non_iid, client_datasets_iid
import warnings

warnings.filterwarnings("ignore")

config_svdd = {
        "epochs": 100,
        "iid": True,
        "stage": "first",
}

config_stage_2 = {
        "stage": "second",
}

if config_svdd["iid"]:
    client_datasets = client_datasets_iid
else:
    client_datasets = client_datasets_non_iid


def switch_config():
    for key in config_stage_2.keys():
        config_svdd[key] = config_stage_2[key]


def load_model(state_dict) -> nn.Module:

    if config_svdd["stage"] == "first":
        model = Model_first_stage()
    elif config_svdd["stage"] == "second":
        model = Model_second_stage()
    else:
        raise NotImplementedError
    model.load_state_dict(state_dict)
    return model

