"""
Usage:
  python generate_Manifoldcache.py --config configs/test/test_pep.yaml --ckpt checkpoints/model.ckpt --gpu 0 --save_dir ./results/fast --eps_stride 3 --geo_stride 4 --deep_stride 4
  python generate_Manifoldcache.py --config configs/test/test_pep.yaml --ckpt checkpoints/model.ckpt --gpu 0 --save_dir ./results/fast --warmup tstar
  python generate_Manifoldcache.py --config configs/test/test_pep.yaml --ckpt checkpoints/model.ckpt --gpu 0 --save_dir ./results/baseline --no_cache
"""

import argparse
import os
import time
from copy import deepcopy

import torch
import yaml

from utils.config_utils import overwrite_values
from data.bioparse.writer.complex_to_pdb import complex_to_pdb
from data import create_dataset
from utils.logger import print_log
from utils.random_seed import setup_seed
from models.LDM.data_utils import Recorder, _get_item
from generate import get_best_ckpt, to_device, generate_wrapper, overwrite, format_id
from manifoldcache_unimomo import patch_unimomo


def main(args, opt_args):
    config = yaml.safe_load(open(args.config))
    config = overwrite_values(config, opt_args)
    mode = config.get('sample_opt', {}).get('mode', 'codesign')
    struct_only = mode == 'fixseq'

    b_ckpt = args.ckpt if args.ckpt.endswith('.ckpt') else get_best_ckpt(args.ckpt)
    ckpt_dir = os.path.split(os.path.split(b_ckpt)[0])[0]
    print(f'Using checkpoint {b_ckpt}')
    model = torch.load(b_ckpt, map_location='cpu')
    device = torch.device('cpu' if args.gpu == -1 else f'cuda:{args.gpu}')
    model.to(device)
    model.eval()

    caches = {}
    if args.no_cache:
        print('[No cache] Baseline mode')
    else:
        caches = patch_unimomo(
            model,
            eps_stride=args.eps_stride, eps_warmup_frac=args.eps_warmup, eps_cooldown_frac=args.eps_cooldown,
            geo_stride=args.geo_stride, displacement_threshold=args.displacement_threshold,
            shallow_stride=args.shallow_stride, deep_stride=args.deep_stride,
            warmup_frac=args.warmup_frac, cooldown_frac=args.cooldown_frac,
            vae_shallow_stride=args.vae_shallow_stride, vae_deep_stride=args.vae_deep_stride,
            vae_geo_stride=args.vae_geo_stride, vae_displacement_threshold=args.vae_displacement_threshold,
            warmup='tstar' if args.warmup in ('tstar', 't*') else 'frac',
        )

    _, _, test_set = create_dataset(config['dataset'])
    save_dir = args.save_dir or os.path.join(ckpt_dir, 'results')
    ref_dir = os.path.join(save_dir, 'references')
    cand_dir = os.path.join(save_dir, 'candidates')
    tmp_dir = os.path.join(save_dir, 'tmp_candidates')
    for d in [ref_dir, cand_dir, tmp_dir]:
        os.makedirs(d, exist_ok=True)

    n_samples = config.get('n_samples', 1)
    n_cycles = config.get('n_cycles', 0)
    recorder = Recorder(test_set, n_samples, save_dir)
    batch_size = config['dataloader']['batch_size']
    t0 = time.time()
    batch_times = []

    while not recorder.is_finished():
        batch_list = recorder.get_next_batch_list(batch_size)
        batch = test_set.collate_fn([test_set[i] for i, _ in batch_list])
        batch = to_device(batch, device)
        bt = time.time()
        with torch.no_grad():
            outputs = generate_wrapper(model, deepcopy(config.get('sample_opt', {})))(batch)
        batch_times.append(time.time() - bt)

        vae_batch_list = []
        for S, X, A, ll, bonds, intra_bonds, (idx, n) in zip(*outputs, batch_list):
            cplx = deepcopy(test_set.get_raw_data(idx))
            summary = deepcopy(test_set.get_summary(idx))
            format_id(summary)
            summary.ref_pdb = os.path.join(ref_dir, summary.ref_pdb)
            if n == 0:
                os.makedirs(os.path.dirname(summary.ref_pdb), exist_ok=True)
                complex_to_pdb(cplx, summary.ref_pdb, summary.target_chain_ids + summary.ligand_chain_ids)
                os.makedirs(os.path.join(cand_dir, summary.id), exist_ok=True)
                os.makedirs(os.path.join(tmp_dir, summary.id), exist_ok=True)
                complex_to_pdb(cplx, os.path.join(tmp_dir, summary.id, 'pocket.pdb'), summary.target_chain_ids)
            save_path = os.path.join(cand_dir if n_cycles == 0 else tmp_dir, summary.id, f'{n}.pdb')
            log = overwrite(cplx, summary, S, X, A, ll, bonds, intra_bonds, save_path, check_validity=False)
            if n_cycles == 0:
                recorder.check_and_save(log, idx, n, struct_only)
            else:
                vae_batch_list.append(_get_item(os.path.join(tmp_dir, summary.id, 'pocket.pdb'),
                                                save_path.rstrip('.pdb') + '.sdf', summary.target_chain_ids))

        for cyc_i in range(n_cycles):
            final_cycle = cyc_i == n_cycles - 1
            batch = to_device(test_set.collate_fn(vae_batch_list), device)
            vae_batch_list = []
            autoencoder = getattr(model, 'autoencoder', model)
            with torch.no_grad():
                if final_cycle:
                    batch['topo_generate_mask'] = torch.zeros_like(batch['generate_mask'])
                outputs = generate_wrapper(autoencoder, deepcopy(config.get('sample_opt', {})))(batch)
            for S, X, A, ll, bonds, intra_bonds, (idx, n) in zip(*outputs, batch_list):
                cplx = deepcopy(test_set.get_raw_data(idx))
                summary = deepcopy(test_set.get_summary(idx))
                format_id(summary)
                summary.ref_pdb = os.path.join(ref_dir, summary.ref_pdb)
                if final_cycle:
                    save_path = os.path.join(cand_dir, summary.id, f'{n}.pdb')
                else:
                    save_path = os.path.join(tmp_dir, summary.id, f'{n}_cyc{cyc_i}.pdb')
                expect_atom_num = test_set.get_expected_atom_num(idx) if hasattr(test_set, 'get_expected_atom_num') else None
                log = overwrite(cplx, summary, S, X, A, ll, bonds, intra_bonds, save_path,
                                check_validity=final_cycle, expect_atom_num=expect_atom_num)
                if final_cycle:
                    recorder.check_and_save(log, idx, n, struct_only)
                else:
                    vae_batch_list.append(_get_item(os.path.join(tmp_dir, summary.id, 'pocket.pdb'),
                                                    save_path.rstrip('.pdb') + '.sdf', summary.target_chain_ids))

        print_log(f'Failed rate: {recorder.num_failed / recorder.num_generated}', level='DEBUG')

    total = time.time() - t0
    print(f"\n{'=' * 60}")
    print(f"  Total: {total:.1f}s ({total / 60:.1f}min)")
    if batch_times:
        print(f"  Avg batch: {sum(batch_times) / len(batch_times):.2f}s")
    for c in caches.values():
        c.print_stats()
    print(f"{'=' * 60}\n")


def parse():
    p = argparse.ArgumentParser()
    p.add_argument('--config', type=str, required=True)
    p.add_argument('--ckpt', type=str, required=True)
    p.add_argument('--save_dir', type=str, default=None)
    p.add_argument('--gpu', type=int, default=0, help='GPU to use, -1 for cpu')
    p.add_argument('--n_cpu', type=int, default=4)
    p.add_argument('--no_cache', action='store_true', help='Run the uncached baseline')
    p.add_argument('--warmup', type=str.lower, default='frac', choices=['frac', 'tstar', 't*'],
                   help="'frac': use --eps_warmup / --warmup_frac (default). "
                        "'tstar': no L1/L3 caching for t > T* = max{t : alpha_bar_t >= 1/2}.")
    p.add_argument('--eps_stride', type=int, default=2)
    p.add_argument('--eps_warmup', type=float, default=0.12)
    p.add_argument('--eps_cooldown', type=float, default=0.08)
    p.add_argument('--geo_stride', type=int, default=3)
    p.add_argument('--displacement_threshold', type=float, default=0.5)
    p.add_argument('--shallow_stride', type=int, default=2)
    p.add_argument('--deep_stride', type=int, default=3)
    p.add_argument('--warmup_frac', type=float, default=0.10)
    p.add_argument('--cooldown_frac', type=float, default=0.05)
    p.add_argument('--vae_shallow_stride', type=int, default=2)
    p.add_argument('--vae_deep_stride', type=int, default=2)
    p.add_argument('--vae_geo_stride', type=int, default=2)
    p.add_argument('--vae_displacement_threshold', type=float, default=0.3)
    return p.parse_known_args()


if __name__ == '__main__':
    args, opt_args = parse()
    print_log(f'Overwritting args: {opt_args}')
    setup_seed(12)
    main(args, opt_args)
