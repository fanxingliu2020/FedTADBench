import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "../..")))

from algorithms.TranAD.tranad_exp_smd import tranad_main

if __name__ == '__main__':

    tranad_main(args={'dataset_name': 'smap', 'lr': 0.001, 'batch_size': 64, 'epoch': 100, 'window_len': 10, 'seed': 42})