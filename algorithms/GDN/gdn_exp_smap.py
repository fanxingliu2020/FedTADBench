import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "../..")))
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "../..")))

from algorithms.GDN.gdn_exp_smd import Main

if __name__ == "__main__":

    main = Main(dataset='smap', save_path_pattern='gdn_smap', slide_stride=1, slide_win=5, batch=64, epoch=100,
                comment='smap', random_seed=42, decay=0, dim=128, out_layer_num=1, out_layer_inter_dim=128,
                val_ratio=0, report='best', topk=20, debug=False)

    main.run()





