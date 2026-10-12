"""
Usage:
  python scripts/sample_Manifoldcache.py --model_path checkpoints/mp_csp/ --save_path output_manifoldcache --json_file queries.json --shallow_stride 2 --deep_stride 4
  python scripts/sample_Manifoldcache.py --model_path checkpoints/mp_csp/ --save_path output_manifoldcache --json_file queries.json --warmup tstar
  python scripts/sample_Manifoldcache.py --model_path checkpoints/mp_csp/ --save_path output_baseline --json_file queries.json --no_cache
"""

import argparse
import math
import os
import sys
import time
from pathlib import Path

import torch
from p_tqdm import p_map
from pymatgen.io.cif import CifWriter
from torch_geometric.data import DataLoader
from torch_scatter import scatter
from tqdm import tqdm

sys.path.append('.')

from eval_utils import load_model, get_crystals_list, lattices_to_params_shape
from sample_api import construct_dataset_from_json, construct_dataset_from_syminfo, get_pymatgen
from diffcsp.common.data_utils import lattice_params_to_matrix_torch


def compute_tstar(beta_scheduler):
    safe = torch.nonzero(beta_scheduler.alphas_cumprod >= 0.5).flatten()
    return int(safe.max()) if safe.numel() > 0 else 0


class BlockStrideCache:
    def __init__(self, num_layers, total_steps, shallow_stride=2, deep_stride=4,
                 warmup_frac=0.5045, cooldown_frac=0.05, mid_boost=1.5, tstar_step=None):
        self.num_layers = num_layers
        self.total_steps = total_steps
        self.shallow_stride = shallow_stride
        self.deep_stride = deep_stride
        self.warmup_frac = warmup_frac
        self.cooldown_frac = cooldown_frac
        self.mid_boost = mid_boost
        self.tstar_step = tstar_step
        self.cache = {}
        self.last_computed = {}
        self.current_step = 0
        self.current_tag = "corrector"
        self.total_evals = 0
        self.cached_evals = 0

    def get_stride(self, layer_idx):
        progress = self.current_step / max(self.total_steps, 1)
        if self.tstar_step is not None:
            if self.current_step < self.tstar_step:
                return 1
        elif progress < self.warmup_frac:
            return 1
        if progress > 1.0 - self.cooldown_frac:
            return 1
        depth = layer_idx / max(self.num_layers - 1, 1)
        base = self.shallow_stride + (self.deep_stride - self.shallow_stride) * depth
        boost = 1.0 + (self.mid_boost - 1.0) * math.exp(-0.5 * ((progress - 0.5) / 0.2) ** 2)
        return max(1, int(base * boost))

    def set_step(self, step_idx, tag="corrector"):
        self.current_step = step_idx
        self.current_tag = tag

    def should_reuse(self, layer_idx):
        key = (layer_idx, self.current_tag)
        if key not in self.last_computed:
            return False
        return self.current_step - self.last_computed[key] < self.get_stride(layer_idx)

    def get_cached(self, layer_idx):
        self.total_evals += 1
        self.cached_evals += 1
        return self.cache[(layer_idx, self.current_tag)]

    def store(self, layer_idx, tensor):
        key = (layer_idx, self.current_tag)
        self.total_evals += 1
        self.cache[key] = tensor
        self.last_computed[key] = self.current_step

    def reset(self):
        self.cache.clear()
        self.last_computed.clear()
        self.total_evals = 0
        self.cached_evals = 0

    def print_stats(self):
        rate = self.cached_evals / max(self.total_evals, 1) * 100
        computed = self.total_evals - self.cached_evals
        print(f"  ManifoldCache: {rate:.1f}% blocks cached "
              f"({self.cached_evals}/{self.total_evals} evals, {computed} computed)")


def cached_decoder_forward(decoder, cache, t, atom_types, frac_coords,
                           lattices, num_atoms, node2graph):
    edges, frac_diff = decoder.gen_edges(num_atoms, frac_coords)
    edge2graph = node2graph[edges[0]]

    if decoder.smooth:
        node_features = decoder.node_embedding(atom_types)
    else:
        node_features = decoder.node_embedding(atom_types - 1)

    t_per_atom = t.repeat_interleave(num_atoms, dim=0)
    node_features = torch.cat([node_features, t_per_atom], dim=1)
    node_features = decoder.atom_latent_emb(node_features)

    h_list = [node_features]
    for i in range(decoder.num_layers):
        if cache.should_reuse(i):
            node_features = cache.get_cached(i)
        else:
            node_features = decoder._modules["csp_layer_%d" % i](
                node_features, frac_coords, lattices, edges, edge2graph, frac_diff=frac_diff,
            )
            cache.store(i, node_features)
        if i != decoder.num_layers - 1:
            h_list.append(node_features)

    if decoder.ln:
        node_features = decoder.final_layer_norm(node_features)
    h_list.append(node_features)

    if decoder.dense:
        node_features = torch.cat(h_list, dim=-1)

    graph_features = scatter(node_features, node2graph, dim=0, reduce=decoder.pooling)
    coord_out = decoder.coord_out(node_features)
    lattice_out = decoder.lattice_out(graph_features)

    if decoder.pred_type:
        type_out = decoder.type_out(node_features)
        return lattice_out, coord_out, type_out
    return lattice_out, coord_out


@torch.no_grad()
def sample_cached(model, batch, diff_ratio=1.0, step_lr=1e-5, cache=None):
    batch_size = batch.num_graphs

    x_T = torch.rand([batch.num_nodes, 3]).to(model.device)
    crys_fam_T = torch.randn([batch_size, 6]).to(model.device)
    crys_fam_T = model.crystal_family.proj_k_to_spacegroup(crys_fam_T, batch.spacegroup)

    if diff_ratio < 1:
        time_start = int(model.beta_scheduler.timesteps * diff_ratio)
        lattices = lattice_params_to_matrix_torch(batch.lengths, batch.angles)
        lattices = model.crystal_family.de_so3(lattices)
        ori_crys_fam = model.crystal_family.m2v(lattices)
        ori_crys_fam = model.crystal_family.proj_k_to_spacegroup(ori_crys_fam, batch.spacegroup)
        frac_coords = batch.frac_coords
        rand_crys_fam = torch.randn_like(ori_crys_fam)
        rand_x = torch.randn_like(frac_coords)

        alphas_cumprod = model.beta_scheduler.alphas_cumprod[time_start]
        c0 = torch.sqrt(alphas_cumprod)
        c1 = torch.sqrt(1.0 - alphas_cumprod)
        sigmas = model.sigma_scheduler.sigmas[time_start]

        rand_x_anchor = rand_x[batch.anchor_index]
        rand_x_anchor = (batch.ops[batch.anchor_index, :3, :3] @ rand_x_anchor.unsqueeze(-1)).squeeze(-1)
        rand_x = (batch.ops[:, :3, :3] @ rand_x_anchor.unsqueeze(-1)).squeeze(-1)

        crys_fam_T = c0 * ori_crys_fam + c1 * rand_crys_fam
        x_T = (frac_coords + sigmas * rand_x) % 1.0
    else:
        time_start = model.beta_scheduler.timesteps - 1

    l_T = model.crystal_family.v2m(crys_fam_T)

    x_T_all = torch.cat(
        [x_T[batch.anchor_index], torch.ones(batch.ops.size(0), 1).to(x_T.device)], dim=-1,
    ).unsqueeze(-1)
    x_T = (batch.ops @ x_T_all).squeeze(-1)[:, :3] % 1.0

    traj = {
        time_start: {
            'num_atoms': batch.num_atoms,
            'atom_types': batch.atom_types,
            'frac_coords': x_T % 1.0,
            'lattices': l_T,
            'crys_fam': crys_fam_T,
        }
    }

    def call_decoder(time_emb, atom_types, frac_coords, crys_fam, num_atoms, node2graph):
        if cache is not None:
            return cached_decoder_forward(model.decoder, cache, time_emb, atom_types,
                                          frac_coords, crys_fam, num_atoms, node2graph)
        return model.decoder(time_emb, atom_types, frac_coords, crys_fam, num_atoms, node2graph)

    t_wall_start = time.time()

    for t in tqdm(range(time_start, 0, -1), desc="Sampling"):
        step_idx = time_start - t

        times = torch.full((batch_size,), t, device=model.device)
        time_emb = model.time_embedding(times)

        alphas = model.beta_scheduler.alphas[t]
        alphas_cumprod = model.beta_scheduler.alphas_cumprod[t]

        sigmas = model.beta_scheduler.sigmas[t]
        sigma_x = model.sigma_scheduler.sigmas[t]
        sigma_norm = model.sigma_scheduler.sigmas_norm[t]

        c0 = 1.0 / torch.sqrt(alphas)
        c1 = (1 - alphas) / torch.sqrt(1 - alphas_cumprod)

        x_t = traj[t]['frac_coords']
        crys_fam_t = traj[t]['crys_fam']

        rand_x = torch.randn_like(x_T) if t > 1 else torch.zeros_like(x_T)

        step_size = step_lr / (sigma_norm * (model.sigma_scheduler.sigma_begin) ** 2)
        std_x = torch.sqrt(2 * step_size)

        rand_x_anchor = rand_x[batch.anchor_index]
        rand_x_anchor = (batch.ops_inv[batch.anchor_index] @ rand_x_anchor.unsqueeze(-1)).squeeze(-1)
        rand_x = (batch.ops[:, :3, :3] @ rand_x_anchor.unsqueeze(-1)).squeeze(-1)

        if cache is not None:
            cache.set_step(step_idx, "corrector")

        pred_crys_fam, pred_x = call_decoder(
            time_emb, batch.atom_types, x_t, crys_fam_t, batch.num_atoms, batch.batch,
        )[:2]

        pred_x = pred_x * torch.sqrt(sigma_norm)

        pred_x_proj = torch.einsum('bij, bj-> bi', batch.ops_inv, pred_x)
        pred_x_anchor = scatter(pred_x_proj, batch.anchor_index, dim=0, reduce='mean')[batch.anchor_index]
        pred_x = (batch.ops[:, :3, :3] @ pred_x_anchor.unsqueeze(-1)).squeeze(-1)

        x_t_minus_05 = x_t - step_size * pred_x + std_x * rand_x

        frac_coords_all = torch.cat(
            [x_t_minus_05[batch.anchor_index], torch.ones(batch.ops.size(0), 1).to(x_t_minus_05.device)],
            dim=-1,
        ).unsqueeze(-1)
        x_t_minus_05 = (batch.ops @ frac_coords_all).squeeze(-1)[:, :3] % 1.0

        rand_crys_fam = torch.randn_like(crys_fam_T)
        rand_crys_fam = model.crystal_family.proj_k_to_spacegroup(rand_crys_fam, batch.spacegroup)
        ori_crys_fam = crys_fam_t
        rand_x = torch.randn_like(x_T) if t > 1 else torch.zeros_like(x_T)

        adjacent_sigma_x = model.sigma_scheduler.sigmas[t - 1]
        step_size = sigma_x ** 2 - adjacent_sigma_x ** 2
        std_x = torch.sqrt((adjacent_sigma_x ** 2 * (sigma_x ** 2 - adjacent_sigma_x ** 2)) / (sigma_x ** 2))

        rand_x_anchor = rand_x[batch.anchor_index]
        rand_x_anchor = (batch.ops_inv[batch.anchor_index] @ rand_x_anchor.unsqueeze(-1)).squeeze(-1)
        rand_x = (batch.ops[:, :3, :3] @ rand_x_anchor.unsqueeze(-1)).squeeze(-1)

        if cache is not None:
            cache.set_step(step_idx, "predictor")

        pred_crys_fam, pred_x = call_decoder(
            time_emb, batch.atom_types, x_t_minus_05, crys_fam_t, batch.num_atoms, batch.batch,
        )[:2]

        pred_x = pred_x * torch.sqrt(sigma_norm)

        crys_fam_t_minus_1 = c0 * (ori_crys_fam - c1 * pred_crys_fam) + sigmas * rand_crys_fam
        crys_fam_t_minus_1 = model.crystal_family.proj_k_to_spacegroup(crys_fam_t_minus_1, batch.spacegroup)

        pred_x_proj = torch.einsum('bij, bj-> bi', batch.ops_inv, pred_x)
        pred_x_anchor = scatter(pred_x_proj, batch.anchor_index, dim=0, reduce='mean')[batch.anchor_index]
        pred_x = (batch.ops[:, :3, :3] @ pred_x_anchor.unsqueeze(-1)).squeeze(-1)

        x_t_minus_1 = x_t_minus_05 - step_size * pred_x + std_x * rand_x

        l_t_minus_1 = model.crystal_family.v2m(crys_fam_t_minus_1)

        frac_coords_all = torch.cat(
            [x_t_minus_1[batch.anchor_index], torch.ones(batch.ops.size(0), 1).to(x_t_minus_1.device)],
            dim=-1,
        ).unsqueeze(-1)
        x_t_minus_1 = (batch.ops @ frac_coords_all).squeeze(-1)[:, :3] % 1.0

        traj[t - 1] = {
            'num_atoms': batch.num_atoms,
            'atom_types': batch.atom_types,
            'frac_coords': x_t_minus_1 % 1.0,
            'lattices': l_t_minus_1,
            'crys_fam': crys_fam_t_minus_1,
        }

    print(f"\n[ManifoldCache] Sampling time: {time.time() - t_wall_start:.2f}s")
    if cache is not None:
        cache.print_stats()

    traj_stack = {
        'num_atoms': batch.num_atoms,
        'atom_types': batch.atom_types,
        'all_frac_coords': torch.stack([traj[i]['frac_coords'] for i in range(time_start, -1, -1)]),
        'all_lattices': torch.stack([traj[i]['lattices'] for i in range(time_start, -1, -1)]),
    }
    return traj[0], traj_stack


def diffusion_cached(loader, model, step_lr, cache):
    frac_coords, num_atoms, atom_types, lattices = [], [], [], []

    for batch in loader:
        if torch.cuda.is_available():
            batch.cuda()
        if cache is not None:
            cache.reset()
        outputs, _ = sample_cached(model, batch, step_lr=step_lr, cache=cache)
        frac_coords.append(outputs['frac_coords'].detach().cpu())
        num_atoms.append(outputs['num_atoms'].detach().cpu())
        atom_types.append(outputs['atom_types'].detach().cpu())
        lattices.append(outputs['lattices'].detach().cpu())

    frac_coords = torch.cat(frac_coords, dim=0)
    num_atoms = torch.cat(num_atoms, dim=0)
    atom_types = torch.cat(atom_types, dim=0)
    lattices = torch.cat(lattices, dim=0)
    lengths, angles = lattices_to_params_shape(lattices)
    return frac_coords, atom_types, lattices, lengths, angles, num_atoms


def generate_structures_cached(model_path, dataset, batch_size=128, step_lr=1e-5,
                               shallow_stride=2, deep_stride=4, warmup_frac=0.5045,
                               cooldown_frac=0.05, mid_boost=1.5, warmup="frac", no_cache=False):
    model, _, cfg = load_model(Path(model_path), load_data=False)
    if torch.cuda.is_available():
        model.to('cuda')
    model.eval()

    num_layers = model.decoder.num_layers
    total_steps = model.beta_scheduler.timesteps - 1

    if no_cache:
        cache = None
        print("\n[ManifoldCache] Baseline mode, no caching")
    else:
        tstar = compute_tstar(model.beta_scheduler) if warmup == "tstar" else None
        tstar_step = total_steps - tstar if tstar is not None else None
        cache = BlockStrideCache(num_layers, total_steps, shallow_stride, deep_stride,
                                 warmup_frac, cooldown_frac, mid_boost, tstar_step)
        warmup_desc = (f"T*={tstar} (caching from step {tstar_step}/{total_steps})"
                       if tstar is not None else f"warmup_frac={warmup_frac}")
        print(f"\n[ManifoldCache] CSPNet layers={num_layers}, steps={total_steps}, "
              f"shallow_stride={shallow_stride}, deep_stride={deep_stride}, {warmup_desc}, "
              f"cooldown_frac={cooldown_frac}, mid_boost={mid_boost}, step_lr={step_lr}\n")

    loader = DataLoader(dataset, batch_size=min(batch_size, len(dataset)))
    frac_coords, atom_types, lattices, lengths, angles, num_atoms = diffusion_cached(
        loader, model, step_lr, cache,
    )

    crystal_list = get_crystals_list(frac_coords, atom_types, lengths, angles, num_atoms)
    return p_map(get_pymatgen, crystal_list)


def main(args):
    tar_dir = args.save_path
    os.makedirs(tar_dir, exist_ok=True)

    if args.json_file != '':
        dataset = construct_dataset_from_json(args.json_file)
    else:
        assert args.spacegroup > 0 and args.wyckoff_letters != ''
        if ',' in args.wyckoff_letters:
            wyckoff_letters = args.wyckoff_letters.split(',')
        else:
            wyckoff_letters = args.wyckoff_letters
        atom_types = args.atom_types.split(',') if args.atom_types != '' else None
        dataset = construct_dataset_from_syminfo(args.spacegroup, wyckoff_letters, atom_types)

    structure_list = generate_structures_cached(
        model_path=args.model_path,
        dataset=dataset,
        batch_size=args.batch_size,
        step_lr=args.step_lr,
        shallow_stride=args.shallow_stride,
        deep_stride=args.deep_stride,
        warmup_frac=args.warmup_frac,
        cooldown_frac=args.cooldown_frac,
        mid_boost=args.mid_boost,
        warmup="tstar" if args.warmup in ("tstar", "t*") else "frac",
        no_cache=args.no_cache,
    )

    print("Saving structures.")
    for i, structure in enumerate(structure_list):
        tar_file = os.path.join(tar_dir, f"{i + 1}.cif")
        if structure is not None:
            CifWriter(structure).write_file(tar_file)
        else:
            print(f"{i + 1} Error Structure.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', required=True)
    parser.add_argument('--save_path', required=True)
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--step_lr', default=1e-5, type=float)
    parser.add_argument('--spacegroup', default=0, type=int)
    parser.add_argument('--wyckoff_letters', default='', type=str)
    parser.add_argument('--atom_types', default='', type=str)
    parser.add_argument('--json_file', default='', type=str)
    parser.add_argument('--shallow_stride', default=2, type=int)
    parser.add_argument('--deep_stride', default=4, type=int)
    parser.add_argument('--warmup', type=str.lower, default='frac', choices=['frac', 'tstar', 't*'])
    parser.add_argument('--warmup_frac', default=0.5045, type=float)
    parser.add_argument('--cooldown_frac', default=0.05, type=float)
    parser.add_argument('--mid_boost', default=1.5, type=float)
    parser.add_argument('--no_cache', action='store_true', default=False)
    args = parser.parse_args()
    main(args)
