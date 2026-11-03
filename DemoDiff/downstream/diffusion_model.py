import os
import time
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.transformer import Transformer

from diffusion.noise_schedule import NoiseScheduleDiscrete, MarginalTransition
from diffusion.diffusion_utils import sample_discrete_features, sample_discrete_feature_noise, reverse_diffusion
from dataset import convert_smiles_to_pyg_list
from dataset import compute_dataset_metainfo_simple as compute_dataset_metainfo

from utils.data_io import load_config, PlaceHolder, to_dense, replace_first_indices
from utils.molecule import convert_graph_to_smiles, convert_to_canonical
from torch_geometric.utils import to_dense_batch

import random

class GraphDiffusionTransformer:
    def __init__(self, path_to_model, device, data_dir='../data', guide_scale=2, sample_timesteps=None):
        super().__init__()
        
        save_path_base = os.path.splitext(path_to_model)[0]
        config_name = save_path_base.replace("model", "config")
        config_path = f"{config_name}.yaml"
        dm_cfg, data_info, tokenizer = load_config(config_path, data_dir)
        
        self.device = device
        self.tokenizer = tokenizer
        # datset info
        self.node_dist = data_info.nodes_dist
        self.max_node_type = data_info.max_node_type
        self.max_bond_type = data_info.max_bond_type
        self.max_position_type = data_info.max_position_type

        # other configs
        if sample_timesteps is not None:
            self.T = sample_timesteps
        else:
            self.T = dm_cfg.diffusion_steps
        self.max_n_nodes = min(data_info.max_n_nodes, dm_cfg.context_length)
        self.guide_scale = guide_scale
        self.model = Transformer(
            max_n_nodes=self.max_n_nodes,
            hidden_size=dm_cfg.hidden_size,
            depth=dm_cfg.depth,
            num_heads=dm_cfg.num_heads,
            mlp_ratio=dm_cfg.mlp_ratio,
            X_dim=self.max_node_type, 
            E_dim=self.max_bond_type,
            pos_dim=self.max_position_type,
            )
        
        self.load_model(path_to_model, map_location=device, verbose=True)
        self.noise_schedule = NoiseScheduleDiscrete('cosine', timesteps=self.T)
        self.noise_schedule.to(self.device)
        
        # Compute marginals and conditional distributions
        x_marginals, e_marginals, pos_marginals, xe_conditions, ex_conditions = self.compute_marginals_and_conditionals(data_info)

        self.transition_model = MarginalTransition(
            x_marginals=x_marginals,
            e_marginals=e_marginals,
            xe_conditions=xe_conditions,
            ex_conditions=ex_conditions,
            pos_marginals=pos_marginals,
            n_nodes=self.max_n_nodes
        )

        self.limit_dist = PlaceHolder(
            X=x_marginals,
            E=e_marginals,
            y=None,
            pos=pos_marginals
        )
        self.excluded_smiles = convert_to_canonical(tokenizer.vocab_node)
    
    def load_model(self, path_to_model, map_location='cpu', verbose=False):
        if os.path.exists(path_to_model):
            self.model.load_state_dict(torch.load(path_to_model, map_location=map_location, weights_only=True))
        else:
            raise FileNotFoundError(f"Model file not found: {path_to_model}")
        self.model.to(self.device)        
        print('GraphDiT Loaded to device:', self.device)

    def _forward(self, noisy_target, dense_context_dict, unconditioned=False):
        if unconditioned:
            noisy_X, noisy_E, noisy_pos = noisy_target['X_t'].float(), noisy_target['E_t'].float(), noisy_target['pos_t'].float()
            bs, n, _, de = noisy_E.size()
            noisy_E = noisy_E.view(bs, n, n * self.max_bond_type)
            noisy_pos = noisy_pos.view(bs, n, n * self.max_position_type)
            node_mask_dict = {'target': noisy_target['node_mask'], 'all': noisy_target['node_mask']}
            relation_to_target = torch.zeros(noisy_X.size(0), noisy_X.size(1), 1).type_as(noisy_X)
        else:
            indicator_mol_to_ctx = dense_context_dict['indicator_mol_to_ctx']
            indicator_node_to_mol = dense_context_dict['indicator_node_to_mol']
            relation_to_target = dense_context_dict['relation_to_target']
            node_mask_context = dense_context_dict['node_mask_context']
            node_mask_target = dense_context_dict['node_mask_target']
            dense_data = dense_context_dict['dense_data']

            noisy_data = replace_first_indices(dense_data, node_mask_target, noisy_target)
            indicator_node_to_ctx = indicator_mol_to_ctx[indicator_node_to_mol]
            # relation at the node-level
            relation_to_target = relation_to_target[indicator_node_to_mol].view(-1, 1)

            noisy_X = noisy_data.X
            bs, n, _, de = noisy_data.E.size()
            noisy_E = noisy_data.E.view(bs, n, n * self.max_bond_type)
            noisy_pos = noisy_data.pos.view(bs, n, n * self.max_position_type)

            relation_to_target, _ = to_dense_batch(x=relation_to_target, batch=indicator_node_to_ctx, max_num_nodes=self.max_n_nodes)
            noisy_X, noisy_E, noisy_pos = noisy_X.float(), noisy_E.float(), noisy_pos.float()
            node_mask_dict = {'target': node_mask_target, 'all': node_mask_context}

        target_n_nodes = noisy_target['X_t'].size(1)
        t = noisy_target['t'] * self.T
        relation_to_target[~node_mask_dict['all']] = 1

        pred_X, pred_E, pred_y, pred_pos = self.model(noisy_X, noisy_E, noisy_pos, relation_to_target, node_mask_dict, target_n_nodes, t=t)
        pred = PlaceHolder(X=pred_X, E=pred_E, y=pred_y, pos=pred_pos)
        return pred

    def construct_context(self, z_T, pos_pyg_list, neg_pyg_list, med_pyg_list, batch_size, max_len, node_mask_target):
        """
        Construct context by combining target molecules with positive, negative, and medium examples.
        
        Args:
            z_T: Target molecule features (X, E, pos)
            pos_pyg_list: List of positive examples (PyG data objects)
            neg_pyg_list: List of negative examples (PyG data objects)
            med_pyg_list: List of medium examples (PyG data objects)
            batch_size: Number of molecules in batch
            max_len: Maximum length of context
            node_mask_target: Node mask for target molecule
        Returns:
            dense_context_dict: Dictionary containing context data and masks
        """
        device = z_T.X.device
        
        # Initialize context tensors
        context_X = torch.zeros(batch_size, max_len, self.max_node_type, device=device)
        context_E = torch.zeros(batch_size, max_len, max_len, self.max_bond_type, device=device)
        context_pos = torch.zeros(batch_size, max_len, max_len, self.max_position_type, device=device)
        
        # Initialize masks
        context_mask = torch.zeros(batch_size, max_len, dtype=torch.bool, device=device)
        target_mask = torch.zeros(batch_size, max_len, dtype=torch.bool, device=device)
        
        # Initialize indicators
        indicator_mol_to_ctx = []  # Will track which molecule belongs to which context
        indicator_node_to_mol = []  # Will track which node belongs to which molecule
        relation_to_target = []  # Will track relation of each molecule to target
        
        # Track indices of sampled molecules
        context_indices = {
            'pos': [[] for _ in range(batch_size)],
            'med': [[] for _ in range(batch_size)],
            'neg': [[] for _ in range(batch_size)]
        }
        
        def get_remaining_length(target_nodes, max_len):
            remaining_len = max_len - target_nodes
            neg_max_len = remaining_len // 4
            med_max_len = remaining_len // 4
            pos_max_len = remaining_len - neg_max_len - med_max_len
            return pos_max_len, med_max_len, neg_max_len
        
        def add_molecules_to_context(mol_list, max_nodes, current_idx, b, context_type):
            """Helper function to add molecules to context"""
            idx = current_idx
            mol_count = 0
            
            if len(mol_list) == 0 or idx >= max_nodes:
                return idx, mol_count
            
            # frequency of molecules
            if hasattr(mol_list[0], 'frequency'):
                frequencies = torch.tensor([mol.frequency.item() for mol in mol_list], dtype=torch.float)
            else:
                frequencies = torch.ones(len(mol_list), dtype=torch.float)

            # Sample and add molecules one by one until we reach max_nodes
            while idx < max_nodes:
                # Sample a random molecule
                mol_idx = torch.multinomial(frequencies, 1).item()

                if mol_idx >= len(mol_list):
                    break
                # print('batch', b, 'addding mol_idx', mol_idx, 'from', context_type)
                mol_data = mol_list[mol_idx]
                mol_nodes = len(mol_data.x)
                
                # Record the index of the sampled molecule
                context_indices[context_type][b].append(mol_idx)
                
                # Check if adding this molecule would exceed max_len (global) or max_nodes (local)
                if idx + mol_nodes > max_len:
                    break
                if idx + mol_nodes > max_nodes:
                    break
                
                mol_x = F.one_hot(mol_data.x, num_classes=self.max_node_type).float().to(device)
                mol_edge_index = mol_data.edge_index.to(device)
                mol_edge_attr = F.one_hot(mol_data.edge_attr, num_classes=self.max_bond_type).float().to(device)
                mol_edge_pos = F.one_hot(mol_data.edge_pos, num_classes=self.max_position_type).float().to(device)
                
                # Create node-to-molecule indicator for this molecule
                local_indicator = torch.ones(mol_nodes, dtype=torch.long, device=device) * (len(indicator_mol_to_ctx))
                indicator_node_to_mol.append(local_indicator)
                
                # Add molecule-to-context indicator
                indicator_mol_to_ctx.append(b)
                
                # Add relation to target
                relation = mol_data.relation.item()
                relation_to_target.append(relation)
                assert len(relation_to_target) == len(indicator_mol_to_ctx)
                
                dense_data, _ = to_dense(mol_x, mol_edge_index, mol_edge_attr, 
                                         torch.zeros(mol_nodes, dtype=torch.long, device=device), 
                                         edge_pos=mol_edge_pos, max_num_nodes=mol_nodes)
                
                # Add to context
                context_X[b, idx:idx+mol_nodes] = dense_data.X
                context_E[b, idx:idx+mol_nodes, idx:idx+mol_nodes] = dense_data.E
                context_pos[b, idx:idx+mol_nodes, idx:idx+mol_nodes] = dense_data.pos
                
                # Update mask
                context_mask[b, idx:idx+mol_nodes] = True
                
                idx += mol_nodes
                mol_count += 1
                mol_idx += 1
            
            return idx, mol_count
        

        for b in range(batch_size):
            target_nodes = (z_T.X[b].sum(dim=1) > 0).sum()            
            context_X[b, :target_nodes] = z_T.X[b, :target_nodes]
            context_E[b, :target_nodes, :target_nodes] = z_T.E[b, :target_nodes, :target_nodes]
            context_pos[b, :target_nodes, :target_nodes] = z_T.pos[b, :target_nodes, :target_nodes]
            
            # Mark target nodes in mask
            context_mask[b, :target_nodes] = True
            target_mask[b, :target_nodes] = True

            local_indicator = torch.ones(target_nodes, dtype=torch.long, device=device) * len(indicator_mol_to_ctx)
            indicator_node_to_mol.append(local_indicator)
            
            indicator_mol_to_ctx.append(b)
            relation_to_target.append(0.0)
            
            # Calculate remaining lengths for context examples
            pos_max_len, med_max_len, neg_max_len = get_remaining_length(target_nodes, max_len)
            current_idx = target_nodes

            current_idx, pos_count = add_molecules_to_context(pos_pyg_list, pos_max_len, current_idx, b, 'pos')
            
            current_idx, med_count = add_molecules_to_context(med_pyg_list, pos_max_len+med_max_len, current_idx, b, 'med')

            current_idx, neg_count = add_molecules_to_context(neg_pyg_list, pos_max_len+med_max_len+neg_max_len, current_idx, b, 'neg')

        # Concatenate all indicators and relations
        indicator_node_to_mol = torch.cat(indicator_node_to_mol, dim=0)
        indicator_mol_to_ctx = torch.tensor(indicator_mol_to_ctx, device=device)
        relation_to_target = torch.tensor(relation_to_target, device=device).float()
        
        dense_data = PlaceHolder(X=context_X, E=context_E, pos=context_pos)
        
        dense_context_dict = {
            'dense_data': dense_data,
            'target_data': z_T,
            'node_mask_target': node_mask_target,
            'node_mask_context': context_mask,
            'indicator_mol_to_ctx': indicator_mol_to_ctx,
            'indicator_node_to_mol': indicator_node_to_mol,
            'relation_to_target': relation_to_target,
            'context_indices': context_indices  # Add the indices to the return dict
        }
        
        return dense_context_dict

    def prepare_data(
        self,
        pos_df,
        neg_df,
        med_df,
        batch_size=32,
        num_nodes=None,
    ):
        """
        Prepare data for generation by constructing the context.
        
        Args:
            pos_df: DataFrame containing positive examples
            neg_df: DataFrame containing negative examples
            med_df: DataFrame containing medium examples
            batch_size: Number of molecules in batch
            num_nodes: Number of nodes for each molecule (int, list, or None)
            
        Returns:
            dense_context_dict: Dictionary containing context data and masks
            node_mask: Node mask for target molecules
            n_nodes: Number of nodes for each molecule
        """
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Sample initial noise
        if num_nodes is None:
            n_nodes = self.node_dist.sample_n(batch_size, device)
        elif isinstance(num_nodes, int):
            n_nodes = torch.tensor([num_nodes] * batch_size, device=device)
        elif isinstance(num_nodes, list):
            assert len(num_nodes) == batch_size
            n_nodes = torch.tensor(num_nodes, device=device)
        else:
            raise ValueError(f"Invalid num_nodes: {num_nodes}")

        # Create node mask
        arange = torch.arange(self.max_n_nodes, device=device).unsqueeze(0).expand(batch_size, -1)
        node_mask = arange < n_nodes.unsqueeze(1)
        
        # Convert SMILES to PyG graphs
        pos_pyg_list = convert_smiles_to_pyg_list(pos_df, self.tokenizer, max_node=self.max_n_nodes)
        neg_pyg_list = convert_smiles_to_pyg_list(neg_df, self.tokenizer, max_node=self.max_n_nodes)
        med_pyg_list = convert_smiles_to_pyg_list(med_df, self.tokenizer, max_node=self.max_n_nodes)
        
        # Sample initial noise
        sampled_s = sample_discrete_feature_noise(
            limit_dist=self.limit_dist, node_mask=node_mask
        ).to(self.device)
        
        assert (sampled_s.E == torch.transpose(sampled_s.E, 1, 2)).all()

        # Construct context with target, positive, medium, and negative examples
        dense_context_dict = self.construct_context(
            sampled_s, pos_pyg_list, neg_pyg_list, med_pyg_list, batch_size, self.max_n_nodes, 
            node_mask, 
        )

        self.dense_context_dict = dense_context_dict
        self.target_node_mask = node_mask
        self.target_n_nodes = n_nodes
        self.pos_df = pos_df
        self.neg_df = neg_df
        self.med_df = med_df
        self.batch_size = batch_size

    @torch.no_grad()
    def generate(
        self,
        task_name=None,
        deterministic=False,
    ):
        dense_context_dict = self.dense_context_dict
        node_mask = self.target_node_mask
        n_nodes = self.target_n_nodes
        batch_size = self.batch_size

        node_mask = dense_context_dict['node_mask_target']
        # Sample initial noise
        sampled_s = sample_discrete_feature_noise(
            limit_dist=self.limit_dist, node_mask=node_mask
        ).to(self.device)
        assert (sampled_s.E == torch.transpose(sampled_s.E, 1, 2)).all()        
        
        # Iteratively sample p(z_s | z_t) for t = 1, ..., T, with s = t - 1.
        for s_int in reversed(range(0, self.T)):
            s_array = s_int * torch.ones((batch_size, 1)).to(self.device)
            t_array = s_array + 1
            s_norm = s_array / self.T
            t_norm = t_array / self.T

            # Sample z_s
            sampled_s = self.sample_p_zs_given_zt(
                s_norm, t_norm, sampled_s.X, sampled_s.E, sampled_s.pos, node_mask, dense_context_dict, deterministic
            )
        
        # Sample
        sampled_s = sampled_s.mask(node_mask, collapse=True)
        molecule_list = []
        for i in range(batch_size):
            n = n_nodes[i]
            atom_types = sampled_s.X[i, :n].cpu()
            edge_types = sampled_s.E[i, :n, :n].cpu()
            positions = sampled_s.pos[i, :n, :n].cpu()
            molecule_list.append([atom_types, edge_types, positions])

        _, valid_unique_list, all_smiles_list = convert_graph_to_smiles(molecule_list, self.tokenizer, self.excluded_smiles)

        # Extract context molecules' SMILES strings
        context_smiles = {
            'pos': [],
            'med': [],
            'neg': []
        }
                
        # Get SMILES strings and scores for context molecules
        pos_df, med_df, neg_df = self.pos_df, self.med_df, self.neg_df
        for b in range(batch_size):
            pos_smiles = [(pos_df.iloc[idx]['smiles'], pos_df.iloc[idx]['score']) 
                         for idx in dense_context_dict['context_indices']['pos'][b]]
            med_smiles = [(med_df.iloc[idx]['smiles'], med_df.iloc[idx]['score']) 
                         for idx in dense_context_dict['context_indices']['med'][b]]
            neg_smiles = [(neg_df.iloc[idx]['smiles'], neg_df.iloc[idx]['score']) 
                         for idx in dense_context_dict['context_indices']['neg'][b]]
            
            context_smiles['pos'].append(pos_smiles)
            context_smiles['med'].append(med_smiles)
            context_smiles['neg'].append(neg_smiles)
        
        # Create mapping between generated molecules and their context
        context_target_map = {}
        for i, smiles in enumerate(valid_unique_list):
            key = smiles if smiles is not None else f"None_{i}"
            context_target_map[key] = {
                'pos': context_smiles['pos'][i],
                'med': context_smiles['med'][i],
                'neg': context_smiles['neg'][i]
            }
        

        return context_target_map
    
    def sample_p_zs_given_zt(
        self, s, t, X_t, E_t, pos_t, node_mask, dense_context_dict, deterministic=False
    ):
        """Samples from zs ~ p(zs | zt). Only used during sampling.
        if last_step, return the graph prediction as well"""
        bs, n, _ = X_t.shape
        beta_t = self.noise_schedule(t_normalized=t)  # (bs, 1)
        alpha_s_bar = self.noise_schedule.get_alpha_bar(t_normalized=s)
        alpha_t_bar = self.noise_schedule.get_alpha_bar(t_normalized=t)

        # Neural net predictions
        noisy_data = {
            "X_t": X_t,
            "E_t": E_t,
            "pos_t": pos_t,
            "t": t,
            "node_mask": node_mask,
        }

        def get_prob(noisy_data, unconditioned=False):
            pred = self._forward(noisy_data, dense_context_dict, unconditioned=unconditioned)

            # Normalize predictions
            pred_X = F.softmax(pred.X, dim=-1)  # bs, n, d0
            pred_E = F.softmax(pred.E, dim=-1)  # bs, n, n, d0
            pred_pos = F.softmax(pred.pos, dim=-1)  # bs, n, n, dpos

            # Retrieve transitions matrix
            n_nodes = pred.X.size(1)
            Qtb = self.transition_model.get_Qt_bar(alpha_t_bar, self.device, n_nodes=n_nodes)
            Qsb = self.transition_model.get_Qt_bar(alpha_s_bar, self.device, n_nodes=n_nodes)
            Qt = self.transition_model.get_Qt(beta_t, self.device, n_nodes=n_nodes)

            Xt_all = torch.cat([X_t, E_t.reshape(bs, n, -1)], dim=-1)
            pred_all = torch.cat([pred_X, pred_E.reshape(bs, n, -1)], dim=-1)

            unnormalized_probX_all = reverse_diffusion(
                predX_0=pred_all, X_t=Xt_all, Qt=Qt.X, Qsb=Qsb.X, Qtb=Qtb.X
            )
            unnormalized_prob_X = unnormalized_probX_all[:, :, : self.max_node_type]
            unnormalized_prob_E = unnormalized_probX_all[:, :, self.max_node_type :].reshape(bs, n * n, -1)
            unnormalized_prob_pos = reverse_diffusion(
                predX_0=pred_pos.flatten(start_dim=1, end_dim=-2), X_t=pos_t.flatten(start_dim=1, end_dim=-2), Qt=Qt.pos, Qsb=Qsb.pos, Qtb=Qtb.pos
            )
            unnormalized_prob_pos = unnormalized_prob_pos.reshape(bs, n, n, pred_pos.shape[-1])

            unnormalized_prob_X[torch.sum(unnormalized_prob_X, dim=-1) == 0] = 1e-5
            unnormalized_prob_E[torch.sum(unnormalized_prob_E, dim=-1) == 0] = 1e-5
            unnormalized_prob_pos[torch.sum(unnormalized_prob_pos, dim=-1) == 0] = 1e-5

            prob_X = unnormalized_prob_X / torch.sum(
                unnormalized_prob_X, dim=-1, keepdim=True
            )  # bs, n, d_t-1
            prob_E = unnormalized_prob_E / torch.sum(
                unnormalized_prob_E, dim=-1, keepdim=True
            )  # bs, n, d_t-1
            prob_E = prob_E.reshape(bs, n, n, pred_E.shape[-1])
            prob_pos = unnormalized_prob_pos / torch.sum(
                unnormalized_prob_pos, dim=-1, keepdim=True
            )
            prob_pos = prob_pos.reshape(bs, n, n, pred_pos.shape[-1])
            return prob_X, prob_E, prob_pos

        prob_X, prob_E, prob_pos = get_prob(noisy_data, unconditioned=False)

        ### Guidance
        if self.guide_scale != 1:
            uncon_prob_X, uncon_prob_E, uncon_prob_pos = get_prob(noisy_data, unconditioned=True)
            prob_X = uncon_prob_X * (prob_X / uncon_prob_X.clamp_min(1e-10)) ** self.guide_scale  
            prob_E = uncon_prob_E * (prob_E / uncon_prob_E.clamp_min(1e-10)) ** self.guide_scale  
            prob_pos = uncon_prob_pos * (prob_pos / uncon_prob_pos.clamp_min(1e-10)) ** self.guide_scale  
            prob_X = prob_X / prob_X.sum(dim=-1, keepdim=True).clamp_min(1e-10)
            prob_E = prob_E / prob_E.sum(dim=-1, keepdim=True).clamp_min(1e-10)
            prob_pos = prob_pos / prob_pos.sum(dim=-1, keepdim=True).clamp_min(1e-10)

        assert ((prob_X.sum(dim=-1) - 1).abs() < 1e-4).all()
        assert ((prob_E.sum(dim=-1) - 1).abs() < 1e-4).all()
        assert ((prob_pos.sum(dim=-1) - 1).abs() < 1e-4).all()

        sampled_s = sample_discrete_features(prob_X, prob_E, prob_pos, node_mask=node_mask, deterministic=deterministic)

        X_s = F.one_hot(sampled_s.X, num_classes=self.max_node_type).float()
        E_s = F.one_hot(sampled_s.E, num_classes=self.max_bond_type).float()
        pos_s = F.one_hot(sampled_s.pos, num_classes=self.max_position_type).float()

        assert (E_s == torch.transpose(E_s, 1, 2)).all()
        assert (X_t.shape == X_s.shape) and (E_t.shape == E_s.shape)

        out_one_hot = PlaceHolder(X=X_s, E=E_s, y=None, pos=pos_s)

        return out_one_hot.mask(node_mask)

    def compute_marginals_and_conditionals(self, data_info):
        """
        Compute marginal and conditional distributions from data_info.
        
        Args:
            data_info: Object containing dataset information
            
        Returns:
            x_marginals: Node type marginal distribution
            e_marginals: Edge type marginal distribution
            pos_marginals: Position type marginal distribution
            xe_conditions: Node-to-edge conditional distribution
            ex_conditions: Edge-to-node conditional distribution
        """
        # Compute marginal distributions
        x_marginals = data_info.node_types.float()
        e_marginals = data_info.edge_types.float()
        pos_marginals = data_info.pos_types.float()
        
        x_marginals = x_marginals / x_marginals.sum()
        e_marginals = e_marginals / e_marginals.sum()
        pos_marginals = pos_marginals / pos_marginals.sum()
        
        # Compute conditional distributions
        # xe_conditions = data_info.co_occur_dist.to_dense().float()
        # check if co_occur_dist is dense tensor or sparse tensor
        if not hasattr(data_info.co_occur_dist, 'to_dense'):
            xe_conditions = data_info.co_occur_dist.float()
        else:
            xe_conditions = data_info.co_occur_dist.to_dense().float()

        # Make the co-occurrence matrix symmetric
        xe_conditions = (xe_conditions + xe_conditions.transpose(0, 1)) / 2
        
        # Sum over rows to get node-to-edge conditionals
        xe_conditions = xe_conditions.sum(dim=1)
        
        # Transpose to get edge-to-node conditionals
        ex_conditions = xe_conditions.t()
        
        # Normalize conditionals with small epsilon to avoid division by zero
        epsilon = 1e-10
        xe_conditions = xe_conditions / (xe_conditions.sum(dim=-1, keepdim=True) + epsilon)
        ex_conditions = ex_conditions / (ex_conditions.sum(dim=-1, keepdim=True) + epsilon)
        
        return x_marginals, e_marginals, pos_marginals, xe_conditions, ex_conditions