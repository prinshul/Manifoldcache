import pandas as pd
from p_tqdm import p_map
import numpy as np
from pymatgen.core.structure import Structure
from matminer.featurizers.site.fingerprint import CrystalNNFingerprint
CrystalNNFP = CrystalNNFingerprint.from_preset("ops")
from xenonpy.descriptor import Compositions
import pickle as pkl
from scipy.spatial.distance import pdist,squareform
import argparse


def get_fps(c, ret_struct_fp=True):
    try:
        s = Structure.from_str(c,fmt='cif')
        if ret_struct_fp:
            site_fps = [CrystalNNFP.featurize(s, i) for i in range(len(s))]
            np_fps = np.array(site_fps)
            struct_fp = np.concatenate([np_fps.mean(axis=0),np_fps.std(axis=0),np_fps.min(axis=0),np_fps.max(axis=0)])
            return struct_fp, s.composition
        else:
            return s.composition
    except:
        if ret_struct_fp:
            return None, None
        else:
            return None
        
def get_fps_dataset(raw_data_dir, fp_data_dir):

    for split in ['train', 'val', 'test']:
        csv_file = pd.read_csv(f'{raw_data_dir}/{split}.csv')
        c_lis = []
        struct_fps = []
        res = p_map(get_fps, csv_file.cif, [split!='test'] * len(csv_file.cif))
        if split!='test':
            struct_fps = [_[0] for _ in res if _[0] is not None]
            c_lis = [_[1] for _ in res if _[1] is not None]
            struct_fps = np.array(struct_fps)
            comp_fps = Compositions().transform(c_lis)
            with open(f'{fp_data_dir}/{split}_struct_fps.pkl','wb') as f:
                pkl.dump(struct_fps, f)
            comp_fps.to_csv(f'{fp_data_dir}/{split}_comp_fps.csv')
        else:
            c_lis = res
            comp_fps = Compositions().transform(c_lis)
            comp_fps.to_csv(f'{fp_data_dir}/{split}_comp_fps.csv')

def get_pairs(fp_data_dir, train_size, val_size):

    with open(f'{fp_data_dir}/train_struct_fps.pkl','rb') as f:
        train_s_fps = pkl.load(f)

    dis_train = pdist(train_s_fps)
    dis_train_s = squareform(dis_train)
    D = dis_train_s
    n = D.shape[0]  
    indices = np.triu_indices(n, k=1)  
    valid_indices = np.where(D[indices] <= 0.3)[0]  

    chosen_indices = np.random.choice(valid_indices, size=train_size, replace=False)

    chosen_pairs = list(zip(indices[0][chosen_indices], indices[1][chosen_indices]))

    valid_indices_2 = np.where(D[indices] > 0.3)[0]  

    chosen_indices_2 = np.random.choice(valid_indices_2, size=train_size, replace=False)

    chosen_pairs_2 = list(zip(indices[0][chosen_indices_2], indices[1][chosen_indices_2]))

    train_pairs = np.concatenate([np.array(chosen_pairs),np.array(chosen_pairs_2)], axis=0)

    train_label = np.concatenate([np.ones(train_size),np.zeros(train_size)], axis=0).reshape(-1,1)

    with open(f'{fp_data_dir}/val_struct_fps.pkl','rb') as f:
        val_s_fps = pkl.load(f)
    dis_val = pdist(val_s_fps)
    dis_val_s = squareform(dis_val)
    D = dis_val_s
    n = D.shape[0] 
    indices = np.triu_indices(n, k=1) 
    valid_indices = np.where(D[indices] <= 0.3)[0]  
    
    chosen_indices = np.random.choice(valid_indices, size=val_size, replace=False)
    
    chosen_pairs = list(zip(indices[0][chosen_indices], indices[1][chosen_indices]))

    valid_indices_2 = np.where(D[indices] > 0.3)[0]  

    chosen_indices_2 = np.random.choice(valid_indices_2, size=val_size, replace=False)

    chosen_pairs_2 = list(zip(indices[0][chosen_indices_2], indices[1][chosen_indices_2]))

    val_pairs = np.concatenate([np.array(chosen_pairs),np.array(chosen_pairs_2)], axis=0)

    val_label = np.concatenate([np.ones(val_size),np.zeros(val_size)], axis=0).reshape(-1,1)

    with open(f'{fp_data_dir}/ml_train.pkl','wb') as f:
        pkl.dump((train_pairs, train_label, val_pairs, val_label), f)

def main(args):

    fp_data_dir = args.raw_data_dir if args.fp_data_dir == '' else args.fp_data_dir

    get_fps_dataset(args.raw_data_dir, fp_data_dir)

    get_pairs(fp_data_dir, args.train_size, args.val_size)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--raw_data_dir', required=True)
    parser.add_argument('--fp_data_dir', default='', type=str)
    parser.add_argument('--train_size', default=500000, type=int)
    parser.add_argument('--val_size', default=5000, type=int)
    args = parser.parse_args()


    main(args)
