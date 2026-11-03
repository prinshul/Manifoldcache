import time
import os
import json
from collections import defaultdict
from functools import partial
from dataclasses import dataclass, field

from tqdm import tqdm
from tqdm.contrib.concurrent import process_map
import multiprocessing as mp
from multiprocessing import Pool, cpu_count

from diffusion.distributions import DistributionNodes
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem

import torch
import numpy as np
import pandas as pd
from torch_geometric.data import Data

def expand_sparse_tensor(sparse_tensor, new_size):
    """
    Expands a 3D sparse tensor by adding new rows and columns with [1,0,...,0] pattern.
    
    Args:
        sparse_tensor (torch.sparse_coo_tensor): Input sparse tensor of shape [N,N,D]
        new_size (tuple): Target size (N+1, N+1, D)
        
    Returns:
        torch.sparse_coo_tensor: Expanded sparse tensor
    """
    old_indices = sparse_tensor.indices()
    old_values = sparse_tensor.values()
    depth = new_size[2]
    new_idx = new_size[0] - 1  # The index for new row/column
    
    # Create new entries for the last row and column
    # For row new_idx
    new_row_indices = torch.stack([
        torch.tensor([new_idx] * depth),  # row index
        torch.tensor([0] * depth),        # column index
        torch.arange(depth)               # depth index
    ])
    new_row_values = torch.zeros(depth)
    new_row_values[0] = 1.0  # Set first element to 1

    # For column new_idx
    new_col_indices = torch.stack([
        torch.tensor([0] * depth),        # row index
        torch.tensor([new_idx] * depth),  # column index
        torch.arange(depth)               # depth index
    ])
    new_col_values = torch.zeros(depth)
    new_col_values[0] = 1.0  # Set first element to 1

    # Combine all indices and values
    combined_indices = torch.cat([old_indices, new_row_indices, new_col_indices], dim=1)
    combined_values = torch.cat([old_values, new_row_values, new_col_values])

    # Create new sparse tensor
    return torch.sparse_coo_tensor(
        combined_indices,
        combined_values,
        size=new_size
    )

def dict_to_sparse_tensor(sparse_dict):
    """Convert a dictionary back to a sparse tensor"""
    return torch.sparse_coo_tensor(
        indices=torch.LongTensor(sparse_dict['indices']),
        values=torch.FloatTensor(sparse_dict['values']),
        size=sparse_dict['size']
    ).coalesce()
    
class DataInfos:
    def __init__(self, cfg, tokenizer, data_dir):
        meta_name = f'{cfg.task_name}.meta.json'

        vocab_len = len(tokenizer.vocab_node)
        vocab_ring_len = len(tokenizer.initial_rings)
        meta_filename = os.path.join(data_dir, 'processed', f'vocab{vocab_len}ring{vocab_ring_len}', meta_name)
        
        if os.path.exists(meta_filename):
            with open(meta_filename, 'r') as f:
                meta_dict = json.load(f)

        self.max_node_type = tokenizer.max_node_type + 1 # plus unknown
        self.max_bond_type = tokenizer.max_bond_type + 1 # plus null edge
        self.max_position_type = tokenizer.max_atom_in_token + 1 # plus null edge

        self.max_n_nodes = meta_dict['max_graph_size']
        self.node_types = torch.Tensor(meta_dict['node_dist'] + [0]) # plus unknwon
        self.edge_types = torch.Tensor(meta_dict['edge_dist'])
        self.pos_types = torch.Tensor(meta_dict['pos_dist'])[:self.max_position_type]
        self.pos_types = self.pos_types / self.pos_types.sum()
        self.co_occur_dist = dict_to_sparse_tensor(meta_dict['co_occur_dist'])
        self.co_occur_dist = expand_sparse_tensor(
            self.co_occur_dist, 
            (self.co_occur_dist.shape[0] + 1, 
            self.co_occur_dist.shape[1] + 1, 
            self.co_occur_dist.shape[2])
        )
        self.node_num_frequency = torch.Tensor(meta_dict['node_num_frequency'])
        self.nodes_dist = DistributionNodes(self.node_num_frequency)

        self.input_dims = {"X": self.max_node_type, "E": self.max_bond_type, "y": 0}
        self.output_dims = {"X": self.max_node_type, "E": self.max_bond_type, "y": 0}

def default_relation_func(x):
    return 1 - x

def convert_smiles_to_pyg_list(df, tokenizer, max_node=None, num_processors=None, relation_func=default_relation_func, verbose=False):
    RDLogger.DisableLog('rdApp.*')

    items = [(i, row, tokenizer, max_node, relation_func) for i, row in df.iterrows()]

    if num_processors is None:
        # num_processors = max(1, mp.cpu_count() - 2)
        num_processors = 10

    if verbose:
        print(f"Processing molecules using {num_processors} processors")
    with mp.Pool(processes=num_processors) as pool:
        results = list(tqdm(
            pool.imap(process_single_item, items),
            total=len(items),
            desc="Processing molecules",
            disable=not verbose
        ))

    data_list = []
    for result in tqdm(results, desc="Converting to PyG Data objects", disable=not verbose):
        # print('frequency', result['frequency'])
        data = Data(
            x=torch.LongTensor(result['x']),
            edge_index=torch.from_numpy(result['edge_index']).long(),
            edge_attr=torch.LongTensor(result['edge_attr']),
            edge_pos=torch.LongTensor(result['edge_pos']),
            relation=torch.FloatTensor(result['relation']),
            frequency=torch.FloatTensor(result['frequency']),
            idx=result['idx']
        )
        data_list.append(data)

    # print(f"Processed {len(data_list)} molecules")
    return data_list

def process_single_item(args):
    i, row, tokenizer, max_node, relation_func = args
    smiles = row['smiles']
    score = row['score']
    freq = row['frequency']
    relation = relation_func(score)
    
    def create_empty_data_dict(idx):
        return {
            'x': np.zeros(1, dtype=np.int64),
            'edge_index': np.zeros((2, 0), dtype=np.int64),
            'edge_attr': np.zeros(0, dtype=np.int64),
            'edge_pos': np.zeros(0, dtype=np.int64),
            'relation': np.array([1], dtype=np.float32),
            'idx': idx,
            'frequency': np.array([0], dtype=np.float32)
        }

    if pd.isna(smiles) or str(smiles).strip() == "":
        return create_empty_data_dict(i)

    node_type, adj = tokenizer.encode(smiles, update_vocab_edge=False)
    x = np.array(node_type, dtype=np.int64)

    if max_node is not None and len(x) > max_node:
        return create_empty_data_dict(i)

    adj = np.array(adj)
    adj += 1  # -1 to 0
    rows, cols = np.nonzero(adj)

    edge_combined = adj[rows, cols]
    edge_attr, edge_pos = [], []
    for edge_idx in edge_combined.tolist():
        pos, bond_type = tokenizer.vocab_edge[edge_idx - 1]
        if pos >= tokenizer.max_atom_in_token:
            pos = 0
        edge_pos.append(pos + 1)
        edge_attr.append(bond_type + 1)

    edge_index = np.stack([rows, cols], axis=0)

    return {
        'x': x,
        'edge_index': edge_index,
        'edge_attr': np.array(edge_attr, dtype=np.int64),
        'edge_pos': np.array(edge_pos, dtype=np.int64),
        'relation': np.array([relation], dtype=np.float32),
        'frequency': np.array([freq], dtype=np.float32),
        'idx': i
    }


##### about new data info class

def compute_dataset_metainfo_simple(tot_smiles, tokenizer, num_processes=None):
    max_node_type = tokenizer.max_node_type
    max_bond_type = tokenizer.max_bond_type
    # print('max_node_type', max_node_type, 'max_bond_type', max_bond_type)

    node_count_list = [0] * len(tokenizer.vocab_node)
    edge_count_list = [0] + [0] * max_bond_type
    pos_count_list = [0] + [0] * tokenizer.max_atom_in_token
    occurence_matrix = np.zeros((max_node_type, max_node_type, max_bond_type + 1))
    num_node_per_mol = []
    
    for smiles in tot_smiles:
        node_types, adj = tokenizer.encode(smiles, update_vocab_edge=False)

        adj, pos = tokenizer.get_bond_position_by_vocab(adj)
        adj = np.array(adj) + 1
        pos = np.array(pos)
        # Get edges
        rows, cols = np.nonzero(adj)
        edge_types = adj[rows, cols]
        pos_types = pos[rows, cols]

        cur_node_count = [0] * len(tokenizer.vocab_node)
        for node in node_types:
            node_count_list[node] += 1
            cur_node_count[node] += 1
        for edge in edge_types:
            edge_count_list[edge] += 1
        for pos in pos_types:
            pos_count_list[pos] += 1

        occurence_matrix_temp = np.zeros((max_node_type, max_node_type, max_bond_type + 1))
        for row, col, edge in zip(rows, cols, edge_types):
            node1, node2 = node_types[row], node_types[col]
            occurence_matrix_temp[node1, node2, edge] += 2
            occurence_matrix[node1, node2, edge] += 2
        
        num_node_per_mol.append(len(node_types))

        node_counts = np.bincount(node_types, minlength=max_node_type)
        potential_edges = node_counts.reshape(-1,1) * node_counts.reshape(1,-1)
        potential_edges = potential_edges - np.diag(node_counts)
        occurence_matrix[:,:,0] = potential_edges * 2 - occurence_matrix_temp.sum(axis=-1)

        null_edge_count = (adj.shape[0] ** 2 - adj.shape[0] - 2 * len(edge_types))
        edge_count_list[0] += null_edge_count
        pos_count_list[0] += null_edge_count

    node_frequency = np.array(node_count_list, dtype=np.int64)
    edge_frequency = np.array(edge_count_list, dtype=np.int64)
    pos_frequency = np.array(pos_count_list, dtype=np.int64)

    node_frequency = node_frequency / node_frequency.sum()
    edge_frequency = edge_frequency / edge_frequency.sum()
    pos_frequency = pos_frequency / pos_frequency.sum()
    
    no_edge = np.sum(occurence_matrix, axis=-1) == 0
    first_elt = occurence_matrix[:, :, 0]
    first_elt[no_edge] = 1
    occurence_matrix[:, :, 0] = first_elt
    occurence_matrix = occurence_matrix / np.sum(occurence_matrix, axis=-1, keepdims=True)

    node_num_frequency = np.bincount(num_node_per_mol, minlength=max(num_node_per_mol))
    
    # Expand occurrence matrix to include an additional row and column
    expanded_matrix = np.zeros((max_node_type + 1, max_node_type + 1, max_bond_type + 1))
    expanded_matrix[:max_node_type, :max_node_type, :] = occurence_matrix
    # Set the first element in the last dimension to 1 for the new row and column
    expanded_matrix[max_node_type, :, 0] = 1
    expanded_matrix[:, max_node_type, 0] = 1

    datainfo = ContextInfo(
        node_types=torch.Tensor(node_frequency.tolist() + [0]),
        edge_types=torch.from_numpy(edge_frequency),
        pos_types=torch.from_numpy(pos_frequency),
        co_occur_dist=torch.from_numpy(expanded_matrix),
        nodes_dist = DistributionNodes(torch.from_numpy(node_num_frequency))
    )

    return datainfo


@dataclass
class ContextInfo:
    node_types: torch.Tensor
    edge_types: torch.Tensor
    pos_types: torch.Tensor
    co_occur_dist: torch.Tensor
    nodes_dist: DistributionNodes

def process_single_smiles(smiles_data, tokenizer, max_node_type, max_edge_type):
    """Process single SMILES returning numpy arrays for sparse representation"""
    smiles, = smiles_data
    if pd.isna(smiles) or str(smiles).strip() == "":
        return None, None, None
    
    node_types, adj = tokenizer.encode(smiles, update_vocab_edge=False)
    # Get adj for bond types only
    adj, _ = tokenizer.get_bond_position_by_vocab(adj)
    adj = np.array(adj) + 1
    # Get edges
    rows, cols = np.nonzero(adj)
    edge_type = adj[rows, cols]
    node_types = np.array(node_types, dtype=np.int64)
    unknown_id = len(tokenizer.vocab_node)
    
    # First create dense array for current edges
    current_occur = np.zeros((max_node_type, max_node_type, max_edge_type + 1))
    for row, col, edge in zip(rows, cols, edge_type):
        node1, node2 = node_types[row], node_types[col]
        if node1 == unknown_id or node2 == unknown_id:
            continue
        current_occur[node1, node2, edge] += 2
    
    # Count potential edges and null edges
    node_types_valid = node_types[node_types != unknown_id]
    node_counts = np.bincount(node_types_valid, minlength=max_node_type)
    null_edge_count = (adj.shape[0] ** 2 - adj.shape[0] - 2 * len(edge_type))
    potential_edges = node_counts.reshape(-1,1) * node_counts.reshape(1,-1)
    potential_edges = potential_edges - np.diag(node_counts)
    
    # Update null edges in the dense array
    current_occur[:,:,0] = potential_edges * 2 - current_occur.sum(axis=-1)
    
    # Convert dense to sparse representation
    nonzero_indices = np.nonzero(current_occur)
    values = current_occur[nonzero_indices]
    indices = np.stack(nonzero_indices).astype(np.int64)
    values = values.astype(np.float32)
    
    return len(node_types), indices, values

def arrays_to_sparse_tensor(indices, values, size):
    """Convert numpy arrays to sparse tensor"""
    return torch.sparse_coo_tensor(
        torch.LongTensor(indices),
        torch.FloatTensor(values),
        size=size
    ).coalesce()

def compute_dataset_metainfo(tot_smiles, tokenizer, num_processes=None, compute_coexistence=False):
    """Compute meta information preserving sparsity"""
    
    node_frequency = [v[1] for v in tokenizer.vocab_node_stats.values()]
    # edge_frequency = [v[1] for v in tokenizer.vocab_edge_stats.values()]
    bond_frequencies = defaultdict(int)
    pos_frequencies = defaultdict(int)
    for (pos_type, bond_type), (_, freq) in tokenizer.vocab_edge_stats.items():
        bond_frequencies[bond_type] += freq
        pos_frequencies[pos_type] += freq
    max_bond = max(bond_frequencies.keys())
    max_bond = max(3, max_bond) # set mininum value to 2: single/double/triple bond
    edge_frequency = [bond_frequencies[i] if i in bond_frequencies else 0 
                    for i in range(max_bond)]
    
    max_pos = max(pos_frequencies.keys())
    max_pos = max(tokenizer.max_atom_in_token, max_pos)
    pos_frequency = [pos_frequencies[i] if i in pos_frequencies else 0 
                    for i in range(max_pos)]

    max_node_type = tokenizer.max_node_type
    max_bond_type = tokenizer.max_bond_type
    tensor_size = (max_node_type, max_node_type, max_bond_type + 1)
    
    # Prepare multiprocessing
    smiles_data = [(smiles,) for smiles in tot_smiles]
    process_func = partial(process_single_smiles, 
                        tokenizer=tokenizer,
                        max_node_type=max_node_type,
                        max_edge_type=max_bond_type)
    
    # Process SMILES with multiprocessing 
    if num_processes == 1:
        results = [process_func(data) for data in tqdm(smiles_data)]
    else:
        num_processes = num_processes or cpu_count()
        # with Pool(processes=num_processes) as pool:
        #     results = list(pool.map(process_func, smiles_data))
        results = process_map(process_func, 
                            smiles_data,
                            max_workers=num_processes,
                            desc='Processing SMILES for meta info')

    # Combine results
    num_node_per_mol = []
    all_indices = []
    all_values = []
    
    for num_nodes, indices, values in results:
        if num_nodes is not None:
            num_node_per_mol.append(num_nodes)
            all_indices.append(indices)
            all_values.append(values)
    
    # Combine arrays
    combined_indices = np.concatenate(all_indices, axis=1)
    combined_values = np.concatenate(all_values)
    
    # Create sparse tensor
    tot_co_occurrence = arrays_to_sparse_tensor(
        combined_indices,
        combined_values,
        tensor_size
    )
    dense_co_occur = tot_co_occurrence.to_dense()
    null_edge_count = dense_co_occur[:,:,0].sum().item()

    # Calculate statistics
    node_num_frequency = np.bincount(num_node_per_mol, minlength=max(num_node_per_mol))
    normalized_node_dist = np.array(node_frequency) / np.sum(node_frequency)
    
    # Get null edge count from sparse tensor (values at index 0)
    edge_frequency = [null_edge_count] + edge_frequency
    normalized_edge_dist = np.array(edge_frequency) / np.sum(edge_frequency)
    pos_frequency = [null_edge_count] + pos_frequency
    normalized_pos_dist = np.array(pos_frequency) / np.sum(pos_frequency)
    
    # Normalize co-occurrence distribution
    no_edge = torch.sum(dense_co_occur, dim=-1) == 0
    first_elt = dense_co_occur[:, :, 0].clone()
    first_elt[no_edge] = 1
    dense_co_occur[:, :, 0] = first_elt
    
    sum_per_pair = torch.sum(dense_co_occur, dim=-1, keepdim=True)
    normalized_co_occur = (dense_co_occur / sum_per_pair).to_sparse().coalesce()
    
    co_occur_dist = {
            'indices': normalized_co_occur.indices().tolist(),
            'values': normalized_co_occur.values().tolist(),
            'size': list(normalized_co_occur.size())
    }
    
    max_position_type = tokenizer.max_atom_in_token + 1
    pos_types = torch.Tensor(normalized_pos_dist.tolist())[:max_position_type]
    pos_types = pos_types / pos_types.sum()
    co_occur_dist = dict_to_sparse_tensor(co_occur_dist)
    co_occur_dist = expand_sparse_tensor(
        co_occur_dist, 
        (co_occur_dist.shape[0] + 1, 
        co_occur_dist.shape[1] + 1, 
        co_occur_dist.shape[2])
    )
    
    datainfo = ContextInfo(
        node_types=torch.Tensor(normalized_node_dist.tolist() + [0]),
        edge_types=torch.Tensor(normalized_edge_dist.tolist()),
        pos_types=pos_types,
        co_occur_dist=co_occur_dist,
        nodes_dist = DistributionNodes(torch.from_numpy(node_num_frequency))
    )

    return datainfo