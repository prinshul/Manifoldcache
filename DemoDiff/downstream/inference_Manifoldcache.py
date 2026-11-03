"""
Usage:
  python inference_Manifoldcache.py --task_list Albuterol_Similarity --num_samples 32 --shallow_stride 2 --deep_stride 3
  python inference_Manifoldcache.py --task_list Albuterol_Similarity --num_samples 32 --warmup tstar
  python inference_Manifoldcache.py --task_list Albuterol_Similarity --num_samples 32 --no_cache
"""

# ── All original imports ──
import torch
import pandas as pd
import numpy as np
import os
from tdc import Oracle
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs
import argparse
import time
import multiprocessing
from functools import partial

from docking_oracle import DockingOracle
from polymer_oracle import PolymerOracle

# ── Import BOTH versions ──
from diffusion_model import GraphDiffusionTransformer
from manifoldcache_demodiff import CachedGraphDiffusionTransformer

# ── All original helper functions (unchanged) ──
from inference import (
    standardize_smiles,
    find_most_similar,
    calculate_relationship_score,
    calculate_context_similarities,
    process_molecule,
)


def main():
    parser = argparse.ArgumentParser(description='DemoDiff inference with ManifoldCache')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_samples', type=int, default=32)
    parser.add_argument('--task_list', nargs='+', default=['Albuterol_Similarity'])
    parser.add_argument('--sample_steps', type=int, default=500)
    parser.add_argument('--deterministic', action='store_true')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Default: log/exports_<warmup>_<shallow_stride>_<deep_stride>')
    parser.add_argument('--model_path', type=str, default='pretrained/model.pt')
    parser.add_argument('--top_k', type=int, default=1)
    parser.add_argument('--data_dir', type=str, default='context_data')

    # ── ManifoldCache args ──
    parser.add_argument('--shallow_stride', type=int, default=2,
                        help='Cache reuse interval for early transformer blocks')
    parser.add_argument('--deep_stride', type=int, default=3,
                        help='Cache reuse interval for deep transformer blocks')
    parser.add_argument('--warmup', type=str.lower, default='frac', choices=['frac', 'tstar', 't*'],
                        help="'frac': no caching for the first --warmup_frac of steps (default). "
                             "'tstar': no caching before T*, where alpha_bar_t >= 0.5.")
    parser.add_argument('--warmup_frac', type=float, default=0.10,
                        help='Fraction of steps with no caching at start')
    parser.add_argument('--cooldown_frac', type=float, default=0.05,
                        help='Fraction of steps with no caching at end')
    parser.add_argument('--no_cache', action='store_true',
                        help='Disable caching (use original model for timing comparison)')

    args = parser.parse_args()
    warmup = 'tstar' if args.warmup in ('tstar', 't*') else 'frac'

    if args.output_dir is None:
        if args.no_cache:
            args.output_dir = 'log/exports_baseline'
        else:
            warmup_tag = 'tstar' if warmup == 'tstar' else f'frac{args.warmup_frac:g}'
            args.output_dir = f'log/exports_{warmup_tag}_{args.shallow_stride}_{args.deep_stride}'

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(args.output_dir, exist_ok=True)

    # ── Model creation: cached or baseline ──
    if args.no_cache:
        print("Running WITHOUT caching (baseline)")
        graph_dit = GraphDiffusionTransformer(
            args.model_path, device, guide_scale=2, sample_timesteps=args.sample_steps
        )
    else:
        print(f"Running WITH ManifoldCache (shallow={args.shallow_stride}, deep={args.deep_stride})")
        graph_dit = CachedGraphDiffusionTransformer(
            args.model_path, device, guide_scale=2, sample_timesteps=args.sample_steps,
            shallow_stride=args.shallow_stride,
            deep_stride=args.deep_stride,
            warmup_frac=args.warmup_frac,
            cooldown_frac=args.cooldown_frac,
            warmup=warmup,
        )

    # ── Everything below is IDENTICAL to original inference.py ──
    for task in args.task_list:
        if task == 'Median_1': task = 'Median 1'
        elif task == 'Median_2': task = 'Median 2'
        print(f"\n=== Processing task: {task} ===")

        try: pos_df = pd.read_csv(f'{args.data_dir}/{task}/positive.csv')
        except: pos_df = pd.DataFrame(columns=['smiles', 'score'])
        try: neg_df = pd.read_csv(f'{args.data_dir}/{task}/negative.csv')
        except: neg_df = pd.DataFrame(columns=['smiles', 'score'])
        try: med_df = pd.read_csv(f'{args.data_dir}/{task}/medium.csv')
        except: med_df = pd.DataFrame(columns=['smiles', 'score'])

        if med_df.empty:
            med_df = neg_df.head(len(neg_df) // 2)
            neg_df = neg_df.tail(len(neg_df) // 2)
        if pos_df.empty:
            pos_df = med_df.head(len(med_df) // 2)
            med_df = med_df.tail(len(med_df) // 2)

        pos_df['frequency'] = 1
        neg_df['frequency'] = 1
        med_df['frequency'] = 1

        all_valid_results = []
        unique_list = set()

        train_smiles_list = set()
        for df in [pos_df, med_df, neg_df]:
            for smiles in df['smiles']:
                std = standardize_smiles(smiles)
                if std: train_smiles_list.add(std)

        pos_df = pos_df.sort_values(by='score', ascending=False)
        neg_df = neg_df.sort_values(by='score', ascending=False)
        med_df = med_df.sort_values(by='score', ascending=False)

        context_smiles_standardized = {
            'pos': [standardize_smiles(s) for s in list(pos_df['smiles'])],
            'med': [standardize_smiles(s) for s in list(med_df['smiles'])],
            'neg': [standardize_smiles(s) for s in list(neg_df['smiles'])],
        }

        graph_dit.prepare_data(pos_df, neg_df, med_df, batch_size=args.batch_size)
        print(f"Generating {args.num_samples} valid molecules...")
        time_start = time.time()

        while len(all_valid_results) < args.num_samples:
            graph_dit.prepare_data(pos_df, neg_df, med_df, batch_size=args.batch_size)
            context_target_map = graph_dit.generate(task_name=task, deterministic=args.deterministic)

            is_polymer_task = task.startswith("polymer_")
            process_func = partial(process_molecule,
                                   context_smiles_dict=context_smiles_standardized,
                                   is_polymer=is_polymer_task)

            num_cores = 5
            with multiprocessing.Pool(processes=num_cores) as pool:
                results = pool.map(process_func, context_target_map.items())
                new_results = [r for r in results if r is not None]
                if not new_results: continue

                filtered = []
                for result in new_results:
                    std = standardize_smiles(result['smiles'])
                    if std in train_smiles_list or std in unique_list: continue
                    unique_list.add(std)
                    filtered.append(result)

                all_valid_results.extend(filtered)
                print(f"Added {len(filtered)} unique molecules")

            elapsed = time.time() - time_start
            print(f"Generated {len(all_valid_results)}/{args.num_samples}, time: {elapsed:.1f}s")

            if len(all_valid_results) >= args.num_samples:
                all_valid_results = all_valid_results[:args.num_samples]
                break

        total_time = time.time() - time_start

        # Oracle evaluation
        all_valid_results = sorted(all_valid_results, key=lambda x: x['consistency_score'], reverse=True)[:args.top_k]
        if task.startswith("polymer_"):
            oracle = PolymerOracle(task)
        elif task.startswith("docking_"):
            oracle = DockingOracle(task)
        else:
            oracle = Oracle(name=task)

        for result in all_valid_results:
            result['score'] = oracle([result['smiles']])[0]

        results_df = pd.DataFrame(all_valid_results)
        results_df = results_df.sort_values(by='score', ascending=False)
        results_df = results_df[['smiles', 'score', 'consistency_score']]
        results_df.to_csv(f'{args.output_dir}/{task}.csv', index=False)
        print(f"Results saved to {args.output_dir}/{task}.csv")
        print(f"Total time for {task}: {total_time:.1f}s")

    print(f"\nAll tasks completed.")


if __name__ == '__main__':
    main()
