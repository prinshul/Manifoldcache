import re
import os
import glob
import argparse
from tqdm.auto import tqdm
from functools import partial
from multiprocessing import Pool
from omegaconf import OmegaConf
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
import rdkit.Chem as Chem
from rdkit.Chem import rdMolAlign
from rdkit.Geometry import Point3D

from dit_mc.data_loader.utils import load_pkl, save_pkl


def mol_from_xyz(atomic_numbers, coords):
    mol = Chem.RWMol()
    conf = Chem.Conformer(len(atomic_numbers))

    for i, (z, pos) in enumerate(zip(atomic_numbers, coords)):
        atom = Chem.Atom(int(z))
        mol_idx = mol.AddAtom(atom)
        conf.SetAtomPosition(mol_idx, Point3D(*pos))

    mol = mol.GetMol()
    mol.AddConformer(conf)
    return mol


def get_best_rmsd(ref_mol, gen_mol, use_alignmol=False):
    try:
        if use_alignmol:
            return rdMolAlign.AlignMol(gen_mol, ref_mol)
        else:
            rmsd = rdMolAlign.GetBestRMS(gen_mol, ref_mol)
    except:
        rmsd = np.nan

    return rmsd


def worker_fn(job, use_alignmol=False):
    smiles, ref_idx, gen_idx, ref_mol, gen_mol = job
    rmsd_val = get_best_rmsd(ref_mol, gen_mol, use_alignmol=use_alignmol)
    return smiles, ref_idx, gen_idx, rmsd_val


def calc_coverage_recall(rmsd_array, thresholds):
    min_rmsd_per_conf = np.nanmin(rmsd_array, axis=1, keepdims=True)
    hits_per_conf = min_rmsd_per_conf < thresholds
    return np.mean(hits_per_conf, axis=0)


def calc_coverage_precision(rmsd_array, thresholds):
    thresholds = np.expand_dims(thresholds, 1)
    min_rmsd_per_pred = np.nanmin(rmsd_array, axis=0, keepdims=True)
    hits_per_pred = min_rmsd_per_pred < thresholds
    return np.mean(hits_per_pred, axis=1)


def calc_amr_recall(rmsd_array):
    return np.mean(np.nanmin(rmsd_array, axis=1))


def calc_amr_precision(rmsd_array):
    return np.mean(np.nanmin(rmsd_array, axis=0))


def print_covmat_results(results, step, threshold):
    df = pd.DataFrame.from_dict(
        {
            "Threshold": results["thresholds"],
            "COV-R_mean": np.mean(results["CoverageR"], 0),
            "COV-R_median": np.median(results["CoverageR"], 0),
            "COV-P_mean": np.mean(results["CoverageP"], 0),
            "COV-P_median": np.median(results["CoverageP"], 0),
        }
    )

    mask = np.abs(results['thresholds'] - threshold) < 1e-6

    metrics = {
        "epoch": step,
        "COV-R_mean": df["COV-R_mean"][mask].values[0],
        "COV-R_median": df["COV-R_median"][mask].values[0],
        "COV-P_mean": df["COV-P_mean"][mask].values[0],
        "COV-P_median": df["COV-P_median"][mask].values[0],
        "MAT-R_mean": np.mean(results["MatchingR"]),
        "MAT-R_median": np.median(results["MatchingR"]),
        "MAT-P_mean": np.mean(results["MatchingP"]),
        "MAT-P_median": np.median(results["MatchingP"]),
    }

    return df, metrics


class CovMatEvaluator(object):
    def __init__(
        self,
        num_workers=8,
        use_alignmol=False,
        allow_permutations=False,
        thresholds=np.arange(0.05, 3.05, 0.05),
    ):
        self.num_workers = num_workers
        self.use_alignmol = use_alignmol
        self.allow_permutations = allow_permutations
        self.thresholds = np.array(thresholds).flatten()

    def __call__(self, mols_true, mols_pred):
        jobs = []
        rmsd_results = {}

        for smiles in mols_pred:
            num_true = len(mols_true[smiles])
            num_pred = len(mols_pred[smiles])
            rmsd_results[smiles] = np.nan * np.ones((num_true, num_pred))

            ref_mols = [Chem.rdmolops.RemoveHs(m) for m in mols_true[smiles]]
            gen_mols = [Chem.rdmolops.RemoveHs(m) for m in mols_pred[smiles]]

            for i, ref_mol in enumerate(ref_mols):
                for j, gen_mol in enumerate(gen_mols):
                    jobs.append((smiles, i, j, ref_mol, gen_mol))

        def populate(res):
            smiles, i, j, val = res
            rmsd_results[smiles][i, j] = val

        fn = partial(worker_fn, use_alignmol=self.use_alignmol)

        if self.num_workers > 1:
            with Pool(self.num_workers) as p:
                for res in tqdm(p.imap_unordered(fn, jobs), total=len(jobs)):
                    populate(res)
        else:
            for res in tqdm(map(fn, jobs), total=len(jobs)):
                populate(res)

        amr_r, amr_p, cov_r, cov_p = [], [], [], []

        for arr in rmsd_results.values():
            amr_r.append(calc_amr_recall(arr))
            amr_p.append(calc_amr_precision(arr))
            cov_r.append(calc_coverage_recall(arr, self.thresholds))
            cov_p.append(calc_coverage_precision(arr, self.thresholds))

        return {
            "thresholds": self.thresholds,
            "CoverageR": np.array(cov_r),
            "MatchingR": np.array(amr_r),
            "CoverageP": np.array(cov_p),
            "MatchingP": np.array(amr_p),
        }, rmsd_results


def evaluate_main(workdir, step, seed, num_workers=1):
    cfg = OmegaConf.load(os.path.join(workdir, ".hydra", "config.yaml"))

    if step == -1:
        files = glob.glob(os.path.join(workdir, "pred_mols_epoch*.pkl"))
        step = max(int(re.search(r"epoch(\d+)", f).group(1)) for f in files)
        print(f"Using latest step: {step}")

    file_key = f"{step}" if seed == -1 else f"{step}_seed{seed}"

    pred_path = os.path.join(workdir, f"pred_mols_epoch{file_key}.pkl")
    gt_path = os.path.join(
        cfg.data_loader.data_cfg["data_dir"],
        "geom",
        cfg.data_loader.data_cfg["dataset"],
        "test_mols.pkl"
    )

    print("Loading data...")
    pred = load_pkl(pred_path)
    gt = load_pkl(gt_path)

    print("Running evaluation...")
    evaluator = CovMatEvaluator(num_workers=num_workers)
    results, rmsd_results = evaluator(gt, pred)

    threshold = 0.5 if cfg.data_loader.data_cfg["dataset"] == "qm9" else 0.75
    cov_df, metrics = print_covmat_results(results, step, threshold)

    print("\n===== FINAL METRICS =====")
    for k, v in metrics.items():
        print(f"{k}: {v}")

    cov_df.to_csv(os.path.join(workdir, f"cov_results_epoch{file_key}.csv"), index=False)

    rmsd_vals = []
    for r in rmsd_results.values():
        rmsd_vals.extend(r[~np.isnan(r)].tolist())

    save_pkl(os.path.join(workdir, f"rmsd_values_epoch{file_key}.pkl"), rmsd_vals)

    plt.hist(rmsd_vals, bins=50)
    plt.xlabel("RMSD")
    plt.ylabel("Frequency")
    plt.title("RMSD Histogram")
    plt.savefig(os.path.join(workdir, f"rmsd_hist_epoch{file_key}.png"))
    plt.close()

    print("Done.")


if __name__ == "__main__":
    import multiprocessing as mp
    mp.set_start_method('spawn', force=True)

    parser = argparse.ArgumentParser()
    parser.add_argument("--workdir", type=str, required=True)
    parser.add_argument("--step", type=int, default=-1)
    parser.add_argument("--seed", type=int, default=-1)
    parser.add_argument("--num_workers", type=int, default=1)

    args = parser.parse_args()

    evaluate_main(
        workdir=args.workdir,
        step=args.step,
        seed=args.seed,
        num_workers=args.num_workers,
    )
