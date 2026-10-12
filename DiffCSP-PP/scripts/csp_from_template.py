import time
import argparse
import torch
import torch.nn as nn

import hydra

from tqdm import tqdm
from torch.optim import Adam
from pathlib import Path
from types import SimpleNamespace
from torch_geometric.data import Data, Batch, DataLoader
from torch.utils.data import Dataset

from eval_utils import load_model, lattices_to_params_shape

from collections import Counter

from pymatgen.core.structure import Structure
from pymatgen.core.lattice import Lattice
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pyxtal.symmetry import Group

import copy

from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist

import numpy as np
import pandas as pd



props = pd.read_csv('./refinement/data/elem_prop.csv').values[:,1:].astype(np.float32)
props = (props - np.mean(props,axis=0)) / np.std(props,axis=0)


# Define the two-layer MLP model
class MLP(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(MLP, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 2)
        )
        self.softmax = nn.Softmax(dim=-1)
        
    def forward(self, x):
        x = self.fc(x)
        x = self.softmax(x)
        return x[:,:1]

def cspml(train_set, test_set, train_csv, test_csv, model_dir):

    model = MLP(290, 50)
    ckpts = list(Path(model_dir).glob('*.ckpt'))
    ckpt_epochs = np.array(
        [int(ckpt.parts[-1].split('-')[1].split('.')[0]) for ckpt in ckpts])
    ckpt = str(ckpts[ckpt_epochs.argsort()[-1]])
    print(ckpt)
    state = torch.load(ckpt)
    model.load_state_dict(state['model'], strict=True)
    model.cuda()

    # get crystal fingerprints

    train_cps = pd.read_csv(train_csv).values[:,1:]
    test_cps = pd.read_csv(test_csv).values[:,1:]
    mean_cps = train_cps.mean(axis=0)
    std_cps = train_cps.std(axis=0)

    train_cps_norm = (train_cps - mean_cps) / (std_cps + 1e-6)
    test_cps_norm = (test_cps - mean_cps) / (std_cps + 1e-6)

    # find composition templates

    templates = {}
    for idx,data in tqdm(enumerate(train_set), total=len(train_set)):
        cmp = tuple([int(_) for _ in sorted(Counter(data['atom_types'].tolist()).values())])
        if cmp not in templates:
            templates[cmp] = [idx]
        else:
            templates[cmp].append(idx)

    queries = []
    for idx,data in tqdm(enumerate(test_set), total=len(test_set)):
        cmp = tuple([int(_) for _ in sorted(Counter(data['atom_types'].tolist()).values())])
        queries.append(cmp)

    res_index = []
    res_replace = []

    for idx in tqdm(range(len(test_set))):

        if queries[idx] in templates:
            key_templates = templates[queries[idx]]
        else:
            res_index.append(-1)
            res_replace.append({})
            continue

        query = test_cps_norm[idx]

        keys = train_cps_norm[key_templates]

        inputs = np.abs(query - keys)

        model_inputs = torch.Tensor(inputs).cuda()

        sim = model(model_inputs).reshape(-1).detach().cpu()

        top = key_templates[sim.argsort().tolist()[-1]]

        comp_q = [(k - 1, int(v)) for k,v in Counter(test_set[idx]['atom_types'].tolist()).items()]
        comp_q = sorted(comp_q,key=lambda x:x[1])
        comp_k = [(k - 1, int(v)) for k,v in Counter(train_set[top]['atom_types'].tolist()).items()]
        comp_k = sorted(comp_k,key=lambda x:x[1])

        comp = queries[idx]
        comp_uni = sorted(list(set(comp)))
        q_e, k_e = [], []
        for num in comp_uni:
            k_e.append([_[0] for _ in comp_k if _[1] == num])
            q_e.append([_[0] for _ in comp_q if _[1] == num])
        subs = {}
        for q, k in zip(q_e, k_e):
            if len(q) == 1:
                subs[k[0] + 1] = q[0] + 1
            else:
                q_emb = props[q]
                k_emb = props[k]
                cost = cdist(k_emb, q_emb)
                assignment = linear_sum_assignment(cost)[1].astype(np.int32)
                q_align = np.array(q)[assignment]
                for src, dst in zip(k, q_align):
                    subs[src + 1] = dst + 1
        res_index.append(top)
        res_replace.append(subs)
    return res_index, res_replace

class SampleDataset(Dataset):

    def __init__(self, train_set, test_set, match_algo = 'cspml', seed = 9999, algo_config={}):
        super().__init__()
        self.train_set = train_set
        self.test_set = test_set
        self.seed = seed
        self.match_algo = match_algo
        self.algo_config = algo_config
        self.indexes, self.replace = self.find_template(train_set, test_set, match_algo, algo_config)


    def __len__(self) -> int:
        return len(self.test_set)

    def __getitem__(self, index):

        idx = self.indexes[index]
        if idx != -1:
            data = self.train_set[idx]
            replace = self.replace[index] 
            data.atom_types = torch.LongTensor([replace[k.item()] for k in data.atom_types])
            data.found = torch.LongTensor([1])
        else: # Not found
            data = self.test_set[index]
            num_node = data.atom_types.shape[0]
            data.spacegroup = torch.LongTensor([1])
            data.ops_inv = torch.eye(3).unsqueeze(0).repeat(num_node,1,1).long()
            data.ops = torch.zeros(num_node, 4, 4).long()
            data.ops[:,:3,:3] = torch.eye(3).unsqueeze(0).repeat(num_node,1,1).long()
            data.anchor_index = torch.arange(num_node).long()
            data.frac_coords = torch.rand_like(data.frac_coords)
            lengths, angles = lattices_to_params_shape(torch.randn(1,3,3))
            data.lengths, data.angles = torch.Tensor(lengths).view(1, -1), torch.Tensor(angles).view(1, -1)
            data.found = torch.LongTensor([0])
        return data

    def find_template(self, train_set, test_set, match_algo, algo_config):

        if match_algo == 'cspml':
            index, replace = cspml(train_set, test_set, **algo_config)
        else:
            raise NotImplementedError(f"{match_algo} has not been implemented currently.")

        return index, replace

def diffusion(loader, model, step_lr, diff_ratio=1.0):

    frac_coords = []
    num_atoms = []
    atom_types = []
    lattices = []
    input_data_list = []
    for idx, batch in enumerate(loader):

        if torch.cuda.is_available():
            batch.cuda()
        outputs, traj = model.sample(batch, step_lr = step_lr, diff_ratio=diff_ratio)
        frac_coords.append(outputs['frac_coords'].detach().cpu())
        num_atoms.append(outputs['num_atoms'].detach().cpu())
        atom_types.append(outputs['atom_types'].detach().cpu())
        lattices.append(outputs['lattices'].detach().cpu())
        input_data_list = input_data_list + batch.to_data_list()

    frac_coords = torch.cat(frac_coords, dim=0)
    num_atoms = torch.cat(num_atoms, dim=0)
    atom_types = torch.cat(atom_types, dim=0)
    lattices = torch.cat(lattices, dim=0)
    lengths, angles = lattices_to_params_shape(lattices)
    input_data_batch = Batch.from_data_list(input_data_list)

    return (
        frac_coords, atom_types, lattices, lengths, angles, num_atoms, input_data_batch
    )

def find_tmp(loader, model, step_lr):

    input_data_list = []
    for idx, batch in enumerate(loader):
        input_data_list = input_data_list + batch.to_data_list()

    input_data_batch = Batch.from_data_list(input_data_list)

    return input_data_batch


def main(args):
    # load_data if do reconstruction.
    model_path = Path(args.model_path)
    model, _, cfg = load_model(
        model_path, load_data=False)


    if torch.cuda.is_available():
        model.to('cuda')

    print('Evaluate the diffusion model.')

    datamodule = hydra.utils.instantiate(
        cfg.data.datamodule, _recursive_=False, scaler_path=model_path
    )

    datamodule.setup('test')
    test_set = datamodule.test_dataloader()[0].dataset
    datamodule.setup()
    train_set = datamodule.train_dataloader(shuffle=False).dataset

    data_config = {
        "train_csv": f'{args.csv_path}/train_comp_fps.csv',
        "test_csv": f'{args.csv_path}/test_comp_fps.csv',
        "model_dir": args.finder_model_path
    }


    test_set = SampleDataset(train_set, test_set, algo_config = data_config)
    test_loader = DataLoader(test_set, batch_size = args.batch_size)

    start_time = time.time()
    (frac_coords, atom_types, lattices, lengths, angles, num_atoms, input_data_batch) = diffusion(
        test_loader, model, args.step_lr, args.diff_ratio)

    if args.label == '':
        gen_out_name = 'eval_diff_template.pt'
    else:
        gen_out_name = f'eval_diff_template_{args.label}.pt'

    torch.save({
        'eval_setting': args,
        'frac_coords': frac_coords.reshape(1,-1,3),
        'num_atoms': num_atoms.reshape(1,-1),
        'atom_types': atom_types.reshape(1,-1),
        'lengths': lengths.reshape(1,-1,3),
        'angles': angles.reshape(1,-1,3),
    }, model_path / gen_out_name)


      


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', required=True)
    parser.add_argument('--csv_path', required=True)
    parser.add_argument('--finder_model_path', required=True)
    parser.add_argument('--step_lr', default=1e-5, type=float)
    parser.add_argument('--diff_ratio', default=0.1, type=float)
    parser.add_argument('--batch_size', default=500, type=int)
    parser.add_argument('--label', default='')
    args = parser.parse_args()


    main(args)
