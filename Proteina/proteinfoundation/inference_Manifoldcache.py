"""
Usage:
  python proteinfoundation/inference_Manifoldcache.py --config_name inference_cond_autoguidance --shallow_stride 2 --deep_stride 4
  python proteinfoundation/inference_Manifoldcache.py --config_name inference_cond_autoguidance --warmup tstar
  python proteinfoundation/inference_Manifoldcache.py --config_name inference_cond_autoguidance --no_cache
"""

import os
import sys
import time
import shutil

root = os.path.abspath(".")
sys.path.append(root)

import argparse
import loralib as lora
import hydra
import lightning as L
import pandas as pd
import torch
from dotenv import load_dotenv
from loguru import logger
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from proteinfoundation.utils.lora_utils import replace_lora_layers
from proteinfoundation.metrics.designability import scRMSD
from proteinfoundation.metrics.metric_factory import GenerationMetricFactory, generation_metric_from_list
from proteinfoundation.proteinflow.proteina import Proteina
from proteinfoundation.utils.ff_utils.pdb_utils import write_prot_to_pdb
from proteinfoundation.inference import GenDataset, split_nlens, parse_nlens_cfg, parse_len_cath_code

from manifoldcache_proteina import patch_proteina_model, ForwardFlopMeter


if __name__ == "__main__":
    load_dotenv()

    parser = argparse.ArgumentParser()
    parser.add_argument("--config_name", type=str, default="inference_base")
    parser.add_argument("--config_number", type=int, default=-1)
    parser.add_argument("--config_subdir", type=str, default=None)
    parser.add_argument("--shallow_stride", type=int, default=2, help="Stride for the first transformer layer")
    parser.add_argument("--deep_stride", type=int, default=4, help="Stride for the last transformer layer")
    parser.add_argument("--warmup", type=str.lower, default="frac", choices=["frac", "tstar", "t*"],
                        help="'frac': no caching for the first --warmup_frac of steps (default). "
                             "'tstar': no caching before T*, where log-SNR(t) >= 0.")
    parser.add_argument("--warmup_frac", type=float, default=0.10)
    parser.add_argument("--cooldown_frac", type=float, default=0.05)
    parser.add_argument("--no_cache", action="store_true", help="Run the uncached baseline")
    args = parser.parse_args()
    warmup = "tstar" if args.warmup in ("tstar", "t*") else "frac"

    assert torch.cuda.is_available()
    logger.add(sys.stdout, format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {file}:{line} | {message}")

    config_path = ("../configs/experiment_config" if args.config_subdir is None
                   else f"../configs/experiment_config/{args.config_subdir}")
    with hydra.initialize(config_path, version_base=hydra.__version__):
        config_name = f"inf_{args.config_number}" if args.config_number != -1 else args.config_name
        cfg = hydra.compose(config_name=config_name)
        logger.info(f"Config: {cfg}")

    root_path = f"./inference/{config_name}{'_manifoldcache' if not args.no_cache else ''}"
    if os.path.exists(root_path):
        shutil.rmtree(root_path)
    os.makedirs(root_path, exist_ok=True)

    ckpt_file = os.path.join(cfg.ckpt_path, cfg.ckpt_name)
    if not cfg.lora.use:
        model = Proteina.load_from_checkpoint(ckpt_file)
    else:
        model = Proteina.load_from_checkpoint(ckpt_file, strict=False)
        replace_lora_layers(model, cfg["lora"]["r"], cfg["lora"]["lora_alpha"], cfg["lora"]["lora_dropout"])
        lora.mark_only_lora_as_trainable(model, bias=cfg["lora"]["train_bias"])
        ckpt = torch.load(ckpt_file, map_location="cpu")
        model.load_state_dict(ckpt["state_dict"])

    L.seed_everything(cfg.seed)

    nn_ag = None
    if cfg.get("autoguidance_ratio", 0.0) > 0 and cfg.get("guidance_weight", 1.0) != 1.0:
        model_ag = Proteina.load_from_checkpoint(cfg.autoguidance_ckpt_path)
        nn_ag = model_ag.nn

    model.configure_inference(cfg, nn_ag=nn_ag)

    caches, meter = None, None
    if args.no_cache:
        meter = ForwardFlopMeter()
        meter.register(model.nn, "main")
        if model.nn_ag is not None:
            meter.register(model.nn_ag, "autoguidance")
    else:
        caches = patch_proteina_model(
            model, dt=cfg.dt,
            shallow_stride=args.shallow_stride, deep_stride=args.deep_stride,
            warmup_frac=args.warmup_frac, cooldown_frac=args.cooldown_frac,
            warmup=warmup,
            schedule_mode=cfg.schedule.schedule_mode, schedule_p=cfg.schedule.schedule_p,
        )

    nlens_dict = parse_nlens_cfg(cfg)
    lens_sample, nsamples = split_nlens(nlens_dict, max_nsamples=cfg.max_nsamples, n_replica=1)
    len_cath_codes = parse_len_cath_code(cfg) if cfg.fold_cond else None
    dataset = GenDataset(nres=lens_sample, nsamples=nsamples, dt=cfg.dt, len_cath_codes=len_cath_codes)
    dataloader = DataLoader(dataset, batch_size=1)

    flat_cfg = OmegaConf.to_container(cfg, resolve=True, enum_to_str=True)
    flat_dict = pd.json_normalize(flat_cfg, sep="_").to_dict(orient="records")[0]
    flat_dict = {k: str(v) for k, v in flat_dict.items()}
    columns = list(flat_dict.keys())

    t0 = time.time()
    trainer = L.Trainer(accelerator="gpu", devices=1, inference_mode=False)
    predictions = trainer.predict(model, dataloader)
    sample_time = time.time() - t0
    logger.info(f"Sampling: {sample_time:.1f}s")

    if caches is not None:
        cache_main, cache_ag = caches
        summ = {"label": "manifoldcache", "config_name": config_name,
                "shallow_stride": args.shallow_stride, "deep_stride": args.deep_stride,
                "warmup": warmup, "tstar_step": cache_main.tstar_step,
                "warmup_frac": args.warmup_frac, "cooldown_frac": args.cooldown_frac,
                "sample_time_s": sample_time}
        total_flops, baseline_flops, num_forwards, flops_ok = 0.0, 0.0, 0.0, True
        for cache, name in [(cache_main, "main"), (cache_ag, "autoguidance")]:
            if cache is None:
                continue
            cache.print_stats()
            summ.update(cache.flops_summary(name))
            num_forwards += cache.num_forwards()
            if cache.total_flops() is None:
                flops_ok = False
            else:
                total_flops += cache.total_flops()
                baseline_flops += cache.baseline_flops()
        summ["num_network_forwards"] = num_forwards

        logger.info("===== FLOPs / NFE report [ManifoldCache] =====")
        logger.info(f"  network forwards (NFE, unchanged by caching): {num_forwards:.0f}")
        if flops_ok and baseline_flops > 0:
            summ["total_flops"] = total_flops
            summ["baseline_flops"] = baseline_flops
            summ["flops_ratio_vs_baseline"] = total_flops / baseline_flops
            summ["flops_saved_pct"] = 100.0 * (1.0 - total_flops / baseline_flops)
            logger.info(f"  total FLOPs (cached): {total_flops:.4e}")
            logger.info(f"  baseline FLOPs:       {baseline_flops:.4e}")
            logger.info(f"  --> FLOPs ratio: {total_flops / baseline_flops:.3f}x "
                        f"(saved {summ['flops_saved_pct']:.1f}%)")
        else:
            logger.warning("  FLOPs not measured (dispatch counter unavailable).")
        nfe_csv = os.path.join(root_path, "..", f"results_{config_name}_manifoldcache_nfe_flops.csv")
    else:
        meter.remove_hooks()
        summ = meter.summary_dict()
        summ["config_name"] = config_name
        summ["sample_time_s"] = sample_time
        logger.info("===== FLOPs / NFE report [baseline] =====")
        logger.info(f"  network forwards (NFE): {summ['num_network_forwards']}")
        logger.info(f"  total FLOPs: {summ['total_flops']:.4e}")
        nfe_csv = os.path.join(root_path, "..", f"results_{config_name}_baseline_nfe_flops.csv")
    logger.info(f"  wall-clock sampling: {sample_time:.1f}s")
    pd.DataFrame([summ]).to_csv(nfe_csv, index=False)
    logger.info(f"Wrote NFE/FLOPs summary to {nfe_csv}")

    if cfg.compute_designability:
        columns += ["id_gen", "pdb_path", "L", "_res_scRMSD", "_res_scRMSD_all"]
        results = []
        samples_per_length = {}
        for coors_atom37 in predictions:
            n = coors_atom37.shape[-3]
            samples_per_length.setdefault(n, 0)
            for i in range(coors_atom37.shape[0]):
                dir_name = f"n_{n}_id_{samples_per_length[n]}"
                samples_per_length[n] += 1
                sample_root_path = os.path.join(root_path, dir_name)
                os.makedirs(sample_root_path, exist_ok=False)
                pdb_path = os.path.join(sample_root_path, dir_name + ".pdb")
                write_prot_to_pdb(coors_atom37[i].numpy(), pdb_path, overwrite=True, no_indexing=True)
                res_row = list(flat_dict.values()) + [i, pdb_path, n]
                try:
                    res_designability = scRMSD(pdb_path, ret_min=False, tmp_path=sample_root_path)
                    res_row += [min(res_designability), res_designability]
                except Exception as e:
                    logger.warning(f"Designability failed for {pdb_path}: {e}")
                    res_row += [float("nan"), []]
                results.append(res_row)
        df = pd.DataFrame(results, columns=columns)

    if cfg.compute_fid:
        samples_dir_fid = os.path.join(root_path, "samples_fid")
        os.makedirs(samples_dir_fid, exist_ok=True)
        list_of_pdbs = []
        for coors_atom37 in predictions:
            for i in range(coors_atom37.shape[0]):
                pdb_path = os.path.join(samples_dir_fid, f"{len(list_of_pdbs)}_fid.pdb")
                write_prot_to_pdb(coors_atom37[i].numpy(), pdb_path, overwrite=True, no_indexing=True)
                list_of_pdbs.append(pdb_path)
        res_row = list(flat_dict.values())
        for cfg_mf in cfg.metric_factory:
            metric_factory = GenerationMetricFactory(**cfg_mf).cuda()
            metrics = generation_metric_from_list(list_of_pdbs, metric_factory)
            for k, v in metrics.items():
                columns += ["_res_" + k]
                res_row += [v.cpu().item()]
        df = pd.DataFrame([res_row], columns=columns)
        df = df.drop("metric_factory", axis=1)

    suffix = "fid" if cfg.compute_fid else ("manifoldcache" if not args.no_cache else "baseline")
    csv_file = os.path.join(root_path, "..", f"results_{config_name}_{suffix}.csv")
    df.to_csv(csv_file, index=False)
    logger.info(f"Results: {csv_file}")
    logger.info(f"Total: {sample_time:.1f}s")
