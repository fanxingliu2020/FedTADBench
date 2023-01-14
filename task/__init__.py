from torch import nn

from algorithms.DeepSVDD.SVDD import Model_first_stage, Model_second_stage
from options import args
if args.tsadalg == 'deep_svdd':
    from task.SVDD import config_svdd

if args.dataset == 'smd':
    from task.smd_MOON import *
elif args.dataset == 'smap':
    from task.smap_MOON import *
elif args.dataset == 'psm':
    from task.psm_MOON import *

def load_model(state_dict) -> nn.Module:
    if args.tsadalg != 'deep_svdd':
        model = model_fun()
        model.load_state_dict(state_dict)
        return model
    else:
        if config_svdd["stage"] == "first":
            model = Model_first_stage()
        elif config_svdd["stage"] == "second":
            model = Model_second_stage()
        else:
            raise NotImplementedError
        model.load_state_dict(state_dict)
        return model


logger.log_config(config)
print(args)
print(config)
logger.print(f"local_optimizer:\n{config['optimizer_fun'](model_fun().parameters())} ")
