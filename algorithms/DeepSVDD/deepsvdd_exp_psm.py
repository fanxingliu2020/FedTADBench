import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "../..")))
from algorithms.DeepSVDD.deepsvdd_exp_smd import main

if __name__ == '__main__':
    main(dataset_name='psm', random_seed=42, net_name='psm_mlp', ae_n_epochs=100, n_epochs=100)