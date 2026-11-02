"""
Usage:
  python generate_confs_Manifoldcache.py --workdir checkpoints/qm9/apeB --shallow_stride 2 --deep_stride 4
  python generate_confs_Manifoldcache.py --workdir checkpoints/qm9/apeB --warmup tstar
  python generate_confs_Manifoldcache.py --workdir checkpoints/qm9/apeB --no_cache
"""

import jax
import jraph
import jax.numpy as jnp
import numpy as np
import pickle
import time as time_module
import os
import argparse

from copy import deepcopy
from rdkit.Geometry import Point3D
from rdkit.Chem.rdchem import ChiralType, Conformer
from tqdm.auto import tqdm
from omegaconf import OmegaConf

import jax.random as random
import hydra

from dit_mc.jraph_utils import (
    compute_batch_statistics,
    get_number_of_nodes,
    duplicate_graph,
    move_graph_positions,
)
from dit_mc.training.checkpoint import load_params_from_workdir
from dit_mc.training.utils import dynamically_batch_extended
from dit_mc.data_loader.utils import save_pkl


def compute_tstar_step(num_steps):
    # x_tau = (1 - tau) x0 + tau x1, so log-SNR(tau) = log(tau / (1 - tau))^2 >= 0 iff tau >= 1/2
    taus = np.asarray(jnp.linspace(0.0, 1.0, num_steps + 1)[:-1])
    safe = np.nonzero(taus >= 0.5)[0]
    return int(safe.min()) if safe.size > 0 else num_steps


class StrideCache:
    def __init__(self, num_steps, shallow_stride=2, deep_stride=4,
                 warmup_frac=0.10, cooldown_frac=0.05, tstar_step=None):
        self.num_steps = num_steps
        self.eff_stride = max(1, int((shallow_stride + deep_stride) / 2.0))
        self.warmup_end = tstar_step if tstar_step is not None else int(num_steps * warmup_frac)
        self.cooldown_start = int(num_steps * (1.0 - cooldown_frac))
        self.cached = None
        self.last_step = -999
        self.n_cached = 0
        self.n_computed = 0

    def reset(self):
        self.cached = None
        self.last_step = -999

    def should_reuse(self, n):
        if self.cached is None:
            return False
        if n < self.warmup_end or n >= self.cooldown_start:
            return False
        return (n - self.last_step) < self.eff_stride

    def reuse(self):
        self.n_cached += 1
        return self.cached

    def store(self, n, out):
        self.n_computed += 1
        self.cached = out
        self.last_step = n


def center_data(x, batch_segments, num_graphs):
    x_mean = jraph.segment_mean(x, batch_segments, num_graphs)
    return x - x_mean[batch_segments]


def sample_cached(process, params, graph, graph_prior, graph_cond, rng,
                  num_steps=50, stride_cache=None):
    rng, rng_prior = jax.random.split(rng, num=2)
    batch = compute_batch_statistics(graph)
    x1 = graph.nodes['x1']

    xtau = process.prior.sample(rng_prior, shape=x1.shape,
                                graph_prior=graph_prior)
    xtau = center_data(xtau, batch.batch_segments, batch.num_graphs)
    xtau = jnp.where(batch.node_mask_expanded, xtau, 0.)

    graph_latent = duplicate_graph(graph)
    graph_latent.nodes['positions'] = xtau
    graph_latent.nodes['cond_scaling'] = jnp.ones(shape=(len(x1),))
    graph_latent.edges['cond_scaling'] = jnp.ones(shape=len(graph.senders,))
    if process.self_conditioning_bool:
        graph_latent.nodes['self_cond'] = jnp.zeros_like(x1)

    taus = jnp.linspace(0.0, 1.0, num_steps + 1)
    ones = jnp.ones(shape=(batch.num_graphs,))
    ones = jnp.take(ones, batch.batch_segments)
    ones = jnp.where(batch.node_mask, ones, 0.)

    if stride_cache is not None:
        stride_cache.reset()

    x1_pred = None

    for n in range(num_steps):
        tau = taus[n]
        dtau = taus[n + 1] - taus[n]
        time_latent = ones * tau

        if stride_cache is not None and stride_cache.should_reuse(n):
            nn_out = stride_cache.reuse()
        else:
            if n > 0 and process.self_conditioning_bool and x1_pred is not None:
                graph_latent = duplicate_graph(graph_latent)
                graph_latent.nodes['self_cond'] = x1_pred

            nn_out = process.fwd_jit(
                params, time_latent=time_latent,
                graph_latent=graph_latent, graph_cond=graph_cond,
            )

            if stride_cache is not None:
                stride_cache.store(n, nn_out)

        # Clean prediction (always computed, cheap)
        nn_out_c = center_data(nn_out, batch.batch_segments, batch.num_graphs)
        nn_out_c = jnp.where(batch.node_mask_expanded, nn_out_c, 0.)

        if process.regress_x1_bool:
            x1_pred = nn_out_c
            vt = jnp.einsum('t, tn... -> tn...',
                            1 / (1. - time_latent),
                            nn_out_c - graph_latent.nodes['positions'])
        else:
            x1_pred = graph_latent.nodes['positions'] + jnp.einsum(
                't, tn... -> tn...', 1. - time_latent, nn_out_c)
            vt = nn_out_c

        if process.free_guidance_bool:
            gl_u = duplicate_graph(graph_latent)
            gl_u.nodes['cond_scaling'] = jnp.zeros(shape=(len(x1),))
            gl_u.edges['cond_scaling'] = jnp.zeros(shape=len(graph.senders,))
            nn_u = process.fwd_jit(params, time_latent=time_latent,
                                   graph_latent=gl_u, graph_cond=graph_cond)
            nn_u_c = center_data(nn_u, batch.batch_segments, batch.num_graphs)
            nn_u_c = jnp.where(batch.node_mask_expanded, nn_u_c, 0.)
            if process.regress_x1_bool:
                vtu = jnp.einsum('t, tn... -> tn...',
                                 1 / (1. - time_latent),
                                 nn_u_c - gl_u.nodes['positions'])
            else:
                vtu = nn_u_c
            vt = vtu + 1.0 * (vt - vtu)
            graph_latent = duplicate_graph(graph_latent)
            graph_latent.nodes['cond_scaling'] = jnp.ones(shape=(len(x1),))
            graph_latent.edges['cond_scaling'] = jnp.ones(shape=len(graph.senders,))

        graph_latent = move_graph_positions(graph_latent,
                                            delta_positions=dtau * vt)

    return graph_latent


chirality = {
    ChiralType.CHI_TETRAHEDRAL_CW: -1.0,
    ChiralType.CHI_TETRAHEDRAL_CCW: 1.0,
    ChiralType.CHI_UNSPECIFIED: 0,
    ChiralType.CHI_OTHER: 0,
}

def build_conformer(pos):
    if isinstance(pos, jnp.ndarray) or isinstance(pos, np.ndarray):
        pos = pos.tolist()
    conformer = Conformer()
    for i, atom_pos in enumerate(pos):
        conformer.SetAtomPosition(i, Point3D(*atom_pos))
    return conformer

def set_rdmol_positions(rdkit_mol, pos):
    mol = deepcopy(rdkit_mol)
    for conf_id in range(mol.GetNumConformers()):
        mol.RemoveConformer(conf_id)
    mol.AddConformer(build_conformer(pos))
    return mol

def get_chiral_centers_from_mol(mol):
    atoms = [(i, atom) for i, atom in enumerate(mol.GetAtoms())
             if chirality[atom.GetChiralTag()] != 0 and len(atom.GetNeighbors()) == 4]
    if not atoms:
        return np.array([], dtype=np.int32), np.array([], dtype=np.int32), np.array([], dtype=np.float32)
    chiral_index = np.array([i for i, _ in atoms], dtype=np.int32)
    chiral_nbr_index = np.array([[n.GetIdx() for n in a.GetNeighbors()] for _, a in atoms], dtype=np.int32).flatten()
    chiral_tag = np.array([chirality[a.GetChiralTag()] for _, a in atoms], dtype=np.float32)
    return chiral_index, chiral_nbr_index, chiral_tag

def signed_volume(lc):
    v1, v2, v3 = lc[:, 0] - lc[:, 3], lc[:, 1] - lc[:, 3], lc[:, 2] - lc[:, 3]
    return np.sign(np.sum(v1 * np.cross(v2, v3), axis=-1))

def switch_parity_of_pos(pos, nbr_idx, tag):
    nbr_idx = nbr_idx.reshape(-1, 4)
    sv = signed_volume(pos[nbr_idx])
    node_factor = -1. if (sv * tag == -1.0).all() else 1.
    return jnp.matmul(pos, jnp.diag(jnp.array([1., 1., node_factor])).T)


def main(workdir, batch_size=128, seed=-1, ckpt_dir_name="checkpoints",
         step=-1, use_correction=False,
         enable_cache=True, shallow_stride=2, deep_stride=4,
         warmup_frac=0.10, cooldown_frac=0.05, warmup="frac"):

    cfg = OmegaConf.load(os.path.join(workdir, ".hydra", "config.yaml"))
    rng = random.PRNGKey(cfg.globals.seed if seed == -1 else seed)

    # Data
    num_repeats = 2
    cfg_r = OmegaConf.to_container(cfg, resolve=True)
    data_cfg = cfg_r["data_loader"]["data_cfg"]
    data_cfg["n_proc"] = 0
    data_module = hydra.utils.instantiate(cfg.data_loader, data_cfg=data_cfg)
    dataset, num_test_samples = data_module(split="test_small")
    dataset = dataset.repeat(num_repeats)
    process = hydra.utils.instantiate(cfg.generative_process)
    params, loaded_step = load_params_from_workdir(
        workdir, ckpt_dir_name=ckpt_dir_name, step=step)
    step = loaded_step
    num_atoms = cfg.data_loader.num_atoms_mean
    num_steps = 50

    # ── PHASE 1: PRELOAD ──
    print("[ManifoldCache] Phase 1: Preloading batches + pre-deserializing mols...")
    t0 = time_module.time()

    batch_iter = dynamically_batch_extended(
        dataset.as_numpy_iterator(),
        n_graph=batch_size,
        n_node=(batch_size - 1) * num_atoms + 1,
        n_edge=(batch_size - 1) * num_atoms * num_atoms + 1,
        n_edge_cond=(batch_size - 1) * num_atoms * 4 + 1,
        n_edge_prior=(batch_size - 1) * num_atoms * num_atoms + 1,
    )

    all_batches = []
    for g, gc, gp in batch_iter:
        if not (get_number_of_nodes(g) == get_number_of_nodes(gc) == get_number_of_nodes(gp)):
            continue
        smiles_raw = g.globals['smiles']
        pkl_raw = g.globals['rdkit_pickle']
        smiles = []
        mols = []
        for i in range(len(smiles_raw)):
            s = smiles_raw[i]
            p = pkl_raw[i]
            # jraph.pad_with_graphs fills globals with zeros (ints)
            # Skip padding entries
            if isinstance(s, (int, np.integer)):
                continue
            try:
                smiles.append(s.decode('utf-8') if isinstance(s, bytes) else str(s))
                mols.append(pickle.loads(p))
            except Exception:
                continue
        chirals = [get_chiral_centers_from_mol(m) for m in mols] if use_correction else []
        del g.globals['smiles'], g.globals['rdkit_pickle']
        all_batches.append(dict(g=g, gc=gc, gp=gp, smiles=smiles, mols=mols, chirals=chirals))

    t_preload = time_module.time() - t0
    n_mols = sum(len(b['mols']) for b in all_batches)
    print(f"[ManifoldCache] Preloaded {len(all_batches)} batches, {n_mols} mols in {t_preload:.1f}s")

    # ── PHASE 2: JIT WARMUP ──
    print("[ManifoldCache] Phase 2: JIT warmup...")
    t0 = time_module.time()
    rng_w, rng = random.split(rng)
    b0 = all_batches[0]
    dummy = sample_cached(process, params, b0['g'], b0['gp'], b0['gc'], rng_w,
                          num_steps=num_steps)
    dummy.nodes['positions'].block_until_ready()
    t_jit = time_module.time() - t0
    print(f"[ManifoldCache] JIT warmup: {t_jit:.1f}s")

    # ── PHASE 3: INFERENCE ──
    tstar_step = None
    if enable_cache:
        if warmup == "tstar":
            tstar_step = compute_tstar_step(num_steps)
        warmup_tag = "tstar" if tstar_step is not None else f"frac{warmup_frac:g}"
        cache_label = f"{warmup_tag}_{shallow_stride}_{deep_stride}"
        warmup_desc = (f"T* step={tstar_step} (caching from step {tstar_step}/{num_steps})"
                       if tstar_step is not None else f"warmup_frac={warmup_frac}")
        print(f"[ManifoldCache] shallow_stride={shallow_stride}, deep_stride={deep_stride}, "
              f"{warmup_desc}, cooldown={cooldown_frac}")
    else:
        cache_label = "baseline"
    print(f"[ManifoldCache] Phase 3: Inference ({cache_label})...")
    t0 = time_module.time()

    predicted_mols = {}
    total_cached = 0
    total_computed = 0

    for bd in tqdm(all_batches, desc=f"ManifoldCache({cache_label})"):
        rng_s, rng = random.split(rng)

        sc = (StrideCache(num_steps, shallow_stride, deep_stride, warmup_frac, cooldown_frac,
                          tstar_step) if enable_cache else None)

        sample = sample_cached(process, params, bd['g'], bd['gp'], bd['gc'],
                               rng_s, num_steps=num_steps, stride_cache=sc)

        if sc:
            total_cached += sc.n_cached
            total_computed += sc.n_computed

        bs = compute_batch_statistics(sample)
        n_in_batch = int(sum(bs.graph_mask))

        for idx in range(n_in_batch):
            smi = bd['smiles'][idx]
            mol = bd['mols'][idx]
            pos = sample.nodes["positions"][bs.batch_segments == idx]

            if use_correction and bd['chirals'] and idx < len(bd['chirals']):
                _, nbr_idx, ctag = bd['chirals'][idx]
                if len(nbr_idx) > 0:
                    pos = switch_parity_of_pos(pos, nbr_idx, ctag)

            if smi not in predicted_mols:
                predicted_mols[smi] = []
            predicted_mols[smi].append(set_rdmol_positions(mol, pos))

    t_inf = time_module.time() - t0

    # ── REPORT ──
    print("\n" + "=" * 60)
    print("ManifoldCache RESULTS")
    print("=" * 60)
    print(f"  Preload:        {t_preload:.1f}s")
    print(f"  JIT warmup:     {t_jit:.1f}s")
    print(f"  Inference:      {t_inf:.1f}s  <- fair comparison metric")
    print(f"  Total wall:     {t_preload + t_jit + t_inf:.1f}s")
    if enable_cache:
        total = total_cached + total_computed
        print(f"  Cached steps:   {total_cached}/{total} "
              f"({total_cached / max(total, 1) * 100:.1f}%)")
    print(f"  Per-batch avg:  {t_inf / max(len(all_batches), 1) * 1000:.0f}ms")
    print("=" * 60)

    cc_tag = "CC" if use_correction else "noCC"
    fn = f"pred_mols_epoch{step}_{cache_label}_{cc_tag}.pkl"
    save_pkl(os.path.join(workdir, fn), predicted_mols)
    print(f"Saved: {fn}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--workdir", type=str, required=True)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--seed", type=int, default=-1)
    p.add_argument("--ckpt_dir_name", type=str, default="last_checkpoint")
    p.add_argument("--step", type=int, default=-1)
    p.add_argument("--use_correction", action="store_true", default=True)
    p.add_argument("--no_cache", action="store_true", default=False, help="Run the uncached baseline")
    p.add_argument("--shallow_stride", type=int, default=2)
    p.add_argument("--deep_stride", type=int, default=4)
    p.add_argument("--warmup", type=str.lower, default="frac", choices=["frac", "tstar", "t*"],
                   help="'frac': no caching for the first --warmup_frac of steps (default). "
                        "'tstar': no caching before T*, where log-SNR(t) >= 0.")
    p.add_argument("--warmup_frac", type=float, default=0.10)
    p.add_argument("--cooldown_frac", type=float, default=0.05)
    a = p.parse_args()
    main(a.workdir, a.batch_size, a.seed, a.ckpt_dir_name, a.step,
         a.use_correction, not a.no_cache, a.shallow_stride, a.deep_stride,
         a.warmup_frac, a.cooldown_frac,
         "tstar" if a.warmup in ("tstar", "t*") else "frac")
