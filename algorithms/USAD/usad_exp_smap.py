import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "../..")))

from algorithms.USAD.usad_exp_smd import main


if __name__ == '__main__':

    main(dataset='smap', random_seed=42, n_epochs=100, hidden_size=100, w_size=12)