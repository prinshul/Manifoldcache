# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

from collections import defaultdict
import os
import sys
from typing import Dict, List, Optional

root = os.path.abspath(".")
sys.path.append(root)  # Adds project's root directory

import argparse
import random
import shutil

import hydra
import lightning as L
import numpy as np
import pandas as pd
import torch
from dotenv import load_dotenv
from loguru import logger
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, Dataset

from proteinfoundation.metrics.designability import scRMSD
from proteinfoundation.proteinflow.proteina import Proteina
from proteinfoundation.utils.ff_utils.pdb_utils import write_prot_to_pdb
from proteinfoundation.nn.motif_factory import parse_motif, save_motif_csv


class GenMotifDataset(Dataset):
    """
    This class provides length-centric and fold-centric sampling for unconditional
    and conditional protein structure generation. Each returned item is a dictionary
    with key information for generation, which contains the length of proteins, the
    number of proteins and cath codes if conditional sampling is used.

    If length distribution is specified, sample `nsamples` proteins for each length,
    cath codes are randomly sampled based on empirical distribution.
    Otherwise, if cath code set is specified, sample `nsamples` proteins for each cath code,
    lengths are randomly sampled based on empirical distribution.

    Each sample returned by this dataset is a 2-tuple (L, nsamples) or 3-tuple (L, nsamples, cath_code) where
      - nres (int) is the number of residues in the proteins to be samples
      - nsamples (int) is the number of proteins to generate (happens in parallel),
        so if nsamples=10 it means that it will produce 10 proteins of length L (all sampled in parallel)
      - cath_code (List[str]) is the cath code for the nsamples if conditional generation is used
    """

    def __init__(
        self,
        dt = 0.0025,
        nsamples: Optional[int] = 1,
        max_nsamples_per_batch: Optional[int] = 1,
        n_replicas: int = 1,
        contig_string: Optional[str] = None,
        motif_pdb_path: Optional[str] = None,
        motif_only = False,
        motif_min_length: Optional[int] = None,
        motif_max_length: Optional[int] = None,
        motif_task_name = None,
        segment_order = 'A',
    ):
        """
        Args:
            nlens_cfg (Optional[Dict]): Config dict for specifying length distribution. If not None, use length-centric sampling.
                Defaults to None.
            cath_codes (Optional[List[str]]): List of cath codes to sample.. If not None and nlens_cfg is None, use fold-centric sampling.
                Defaults to None.
            empirical_distribution_cfg (Optional[Dict]): Config dict for specifying (length, cath code) distribution.
                Defaults to None.
            nsamples (Optional[int]): Number of samples to generate for each length or each cath code.
                Defaults to 1.
            max_nsamples_per_batch (Optional[int]): Maximum number of samples for each batch.
                Defaults to 1.
                
                
        nlens_cfg:
    nres_lens: [50, 100, 150, 200, 250]
  nsamples: 100
        """
        super(GenMotifDataset, self).__init__()
        self.motif_task_name = motif_task_name
        self.dt = dt
        ##################################################################################
        ################### 1. Parse length and cath codes ###############################
        ##################################################################################
        logger.info("Use motif-conditioned sampling.")
        nsamples = [nsamples]

        ##################################################################################
        ################### 3. Generate data points ######################################
        ##################################################################################
        self.motif_masks, self.motif_structures = [None]*len(nsamples), None

        self.nsamples = nsamples
        self.cath_codes = [None] * len(nsamples)
        assert n_replicas == 1
        self.motif_masks, self.motif_structures = self.generate_motif_info(contig_string, 
                                                                           motif_pdb_path, 
                                                                           self.motif_task_name, 
                                                                           nsamples[0], 
                                                                           motif_only = motif_only, 
                                                                           min_length = motif_min_length, 
                                                                           max_length = motif_max_length, 
                                                                           segment_order=segment_order)

        ##################################################################################
        # 4. Make sure the nsamples for each data point is not greater than max_nsamples #
        ##################################################################################
        if max_nsamples_per_batch:
            self.nres, self.cath_codes, self.nsamples, self.general_masks, self.motif_masks, self.motif_structures = self.flatten_motif(max_nsamples_per_batch)
        ##################################################################################
        # 5. Make sure this won't cause an error during validation on multiple devices ###
        ##################################################################################

        assert all(
            [n <= max_nsamples_per_batch for n in self.nsamples]
        ), f"The nsamples for each len shouldn't be greater than {max_nsamples_per_batch}"
        assert (
            len(self.nsamples) % n_replicas == 0
        ), f"Should be evenly splitable over {n_replicas} devices"

        logger.info(
            f"Adding generation dataset to sample {self.nsamples} sequences of length {self.nres}."
        )
        
    def generate_motif_info(self, contig_string, motif_pdb_path, motif_task_name, nsamples, sort = True, motif_only = False, min_length = None, max_length = None, segment_order = 'A'):
        mask, x_motif, outstr = parse_motif(motif_pdb_path, contig_string, nsamples=nsamples, make_tensor=False, motif_only = motif_only, min_length = min_length, max_length = max_length)
        if sort:
            lengths = [x.shape[0] for x in mask]
            idx = np.argsort(lengths)
            mask = [mask[i] for i in idx]
            x_motif = [x_motif[i] for i in idx]
            outstr = [outstr[i] for i in idx]
        save_motif_csv(motif_pdb_path, motif_task_name, outstr, segment_order = segment_order)
        return mask, x_motif
    
    def flatten_motif(self, max_nsamples: int):
        """Flatten the list to make sure each data point have no more than max_nsamples"""
        nres, cath_codes, nsamples = [], [], []
        masks, motif_masks = [], []
        structures = []
        for i in range(len(self.nsamples)):
            for j in range(0, self.nsamples[i], max_nsamples):
                
                if self.cath_codes[i] is not None:
                    cath_codes.append(self.cath_codes[i][j : j + max_nsamples])
                else:
                    cath_codes.append(None)
                if j + max_nsamples <= self.nsamples[i]:
                    nsamples.append(max_nsamples)
                    new_mask = self.motif_masks[j:j+max_nsamples]
                    new_structure = self.motif_structures[j:j+max_nsamples]
                else:
                    nsamples.append(self.nsamples[i] - j)
                    new_mask = self.motif_masks[j:j+self.nsamples[i]]
                    new_structure = self.motif_structures[j:j+self.nsamples[i]]
                lengths = [torch.Tensor([True]*x.shape[0]) for x in new_mask]
                general_mask = torch.nn.utils.rnn.pad_sequence(lengths, batch_first=True, padding_value=False)
                masks.append(general_mask)
                padded_masks = torch.nn.utils.rnn.pad_sequence(new_mask, batch_first=True, padding_value=False)
                padded_structures = torch.nn.utils.rnn.pad_sequence(new_structure, batch_first=True, padding_value=0)
                motif_masks.append(padded_masks)
                structures.append(padded_structures)
                nres.append(padded_masks.shape[1])
        return nres, cath_codes, nsamples, masks, motif_masks, structures

    def pad_nlens(self, n_replicas: int):
        """Split nlens into data points (len, nsample) as val dataset and guarantee that
        1. len(val_dataset) should be a multiple of n_replica, to ensure that we don't introduce additional samples for multi-gpu validation
        2. nsample should be the same for all data points if n_replica > 1 (multi-gpu)
        """
        # Add samples to the small bins
        max_nsamples = max(self.nsamples)
        for i in range(len(self.nsamples)):
            while self.cath_codes[i] != None and len(self.cath_codes[i]) < max_nsamples:
                self.cath_codes[i] += self.cath_codes[i][
                    : (max_nsamples - len(self.cath_codes[i]))
                ]
            self.nsamples[i] += max_nsamples - self.nsamples[i]

        # Keep adding lengths in the dataset to make it a multiple of n_replica
        while len(self.nres) % n_replicas != 0:
            self.nres.append(self.nres[-1])
            self.nsamples.append(max_nsamples)
            self.cath_codes.append(self.cath_codes[-1])
    def __len__(self):
        return len(self.nres)

    def __getitem__(self, index: int):
        result = {
            "nres": self.nres[index],
            "dt": self.dt,
            "nsamples": self.nsamples[index],
        }
        if self.cath_codes[index] is not None:
            result["cath_code"] = self.cath_codes[index]

        if self.motif_masks and index < len(self.motif_masks) and self.motif_masks[index] is not None:
            result["motif_seq_mask"] = self.motif_masks[index]
            result["motif_structure"] = self.motif_structures[index] / 10
            result["mask"] = self.general_masks[index].bool()
        return result

def save_motif_predictions(
    root_path: str,
    predictions: List[torch.Tensor],
    job_id: int = 0,
    pdb_name: str = None
) -> None:
    samples_per_length = defaultdict(int)
    count = 0
    for j, pred in enumerate(predictions):
        coors_atom37 = pred  # [b, n, 37, 3], prediction_step returns atom37
        n = coors_atom37.shape[-3]

        # Save each generation as a pdb file
        for i in range(coors_atom37.shape[0]):
            # Create directory where everything related to this sample will be stored
            suffix = f"_{count}"
            count += 1
            dir_name = f"{pdb_name}{suffix}"
            samples_per_length[n] += 1
            fname = dir_name + ".pdb"
            pdb_path = os.path.join(root_path, fname)
            write_prot_to_pdb(
                coors_atom37[i].numpy(),
                pdb_path,
                overwrite=True,
                no_indexing=True,
            )


if __name__ == "__main__":
    load_dotenv()

    parser = argparse.ArgumentParser(description="Job info")
    parser.add_argument(
        "--config_name",
        type=str,
        default="inference_motif",
        help="Name of the config yaml file.",
    )
    parser.add_argument(
        "--config_number", type=int, default=-1, help="Number of the config yaml file."
    )
    parser.add_argument(
        "--config_subdir",
        type=str,
        help="(Optional) Name of directory with config files, if not included uses base inference config.\
            Likely only used when submitting to the cluster with script.",
    )
    parser.add_argument(
        "--split_id",
        type=int,
        default=0,
        help="Leave as 0.",
    )
    args = parser.parse_args()
    logger.info(" ".join(sys.argv))

    assert (
        torch.cuda.is_available()
    ), "CUDA not available"  # Needed for ESMfold and designability
    logger.add(
        sys.stdout,
        format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {file}:{line} | {message}",
    )  # Send to stdout
    
    # Inference config
    # If config_subdir is None then use base inference config
    # Otherwise use config_subdir/some_config
    if args.config_subdir is None:
        config_path = "../configs/experiment_config"
    else:
        config_path = f"../configs/experiment_config/{args.config_subdir}"

    with hydra.initialize(config_path, version_base=hydra.__version__):
        # If number provided use it, otherwise name
        if args.config_number != -1:
            config_name = f"inf_{args.config_number}"
        else:
            config_name = args.config_name
        cfg = hydra.compose(config_name=config_name)
        logger.info(f"Inference config {cfg}")
        run_name = cfg.run_name_

    assert (
        not cfg.compute_designability or not cfg.compute_fid
    ), "Designability cannot be computed together with FID"

    # Set root path for this inference run
    root_path = f"./inference/{config_name}"
    if os.path.exists(root_path):
        shutil.rmtree(root_path)
    os.makedirs(root_path, exist_ok=True)

    # Load model from checkpoint
    ckpt_path = cfg.ckpt_path
    ckpt_file = os.path.join(ckpt_path, cfg.ckpt_name)
    logger.info(f"Using checkpoint {ckpt_file}")
    assert os.path.exists(ckpt_file), f"Not a valid checkpoint {ckpt_file}"
    model = Proteina.load_from_checkpoint(ckpt_file)

    # Set seed
    logger.info(f"Seeding everything to seed {cfg.seed}")
    L.seed_everything(cfg.seed)

    # Set inference variables and potentially load autoguidance
    nn_ag = None
    if (
        cfg.get("autoguidance_ratio", 0.0) > 0
        and cfg.get("guidance_weight", 1.0) != 1.0
    ):
        assert cfg.autoguidance_ckpt_path is not None
        ckpt_ag_file = cfg.autoguidance_ckpt_path
        model_ag = Proteina.load_from_checkpoint(ckpt_ag_file)
        nn_ag = model_ag.nn

    model.configure_inference(cfg, nn_ag=nn_ag)


    dataset = GenMotifDataset(dt=cfg.dt,
                              nsamples=cfg.nsamples, 
                              max_nsamples_per_batch = cfg.max_nsamples_per_batch, 
                              contig_string =cfg.contig_string, 
                              motif_pdb_path= cfg.motif_pdb_path,
                              motif_only = cfg.motif_only, 
                              motif_min_length = cfg.motif_min_length, 
                              motif_max_length = cfg.motif_max_length, 
                              motif_task_name = cfg.motif_task_name)
    dataloader = DataLoader(dataset, batch_size=1, shuffle = False)
    # Note: Batch size should be left as 1, it is not the actual batch size.
    # Each sample returned by this loader is a 3-tuple (L, nsamples, dt) where
    #   - L (int) is the number of residues in the proteins to be samples
    #   - nsamples (int) is the number of proteins to generate (happens in parallel),
    #     so if nsamples=10 it means that it will produce 10 proteins of length L (all sampled in parallel)
    #   - dt (float) step-size used for the ODE integrator
    #   - cath_code (Optional[List[str]]) cath code for conditional generation

    # Flatten config and use it to initialize results dataframes columns
    flat_cfg = OmegaConf.to_container(cfg, resolve=True, enum_to_str=True)
    flat_dict = pd.json_normalize(flat_cfg, sep="_").to_dict(orient="records")[0]
    flat_dict = {k: str(v) for k, v in flat_dict.items()}
    columns = list(flat_dict.keys())

    # Sample the model
    trainer = L.Trainer(accelerator="gpu", devices=1)
    predictions = trainer.predict(model, dataloader)

    save_motif_predictions(
            root_path, predictions, job_id=args.split_id, pdb_name=cfg.motif_task_name.split('_')[0] #cfg_gen.dataset.motif_pdb_path.split('/')[-1][:4]
        )
    import shutil
    shutil.copy(f"./{cfg.motif_task_name.split('_')[0]}_motif_info.csv", root_path)
    
    # Code for designability and
    # Store samples generated as pdbs and also scRMSD
    if cfg.compute_designability:

        # Add some columns to store per-sample results
        columns += ["id_gen", "pdb_path", "L"]
        if cfg.compute_designability:
            columns += ["_res_scRMSD", "_res_scRMSD_all"]

        results = []
        samples_per_length = {}
        for pred in predictions:
            coors_atom37 = pred  # [b, n, 37, 3], prediction_step returns atom37
            n = coors_atom37.shape[-3]
            if n not in samples_per_length:
                samples_per_length[n] = 0

            # Save each generation as a pdb file
            for i in range(coors_atom37.shape[0]):
                # Create directory where everything related to this sample will be stored
                dir_name = f"n_{n}_id_{samples_per_length[n]}"
                samples_per_length[n] += 1
                sample_root_path = os.path.join(
                    root_path, dir_name
                )  # ./inference/conf_{}/n_{}_id_{}
                os.makedirs(sample_root_path, exist_ok=False)

                # Save generated structure as pdb
                fname = dir_name + ".pdb"
                pdb_path = os.path.join(sample_root_path, fname)
                write_prot_to_pdb(
                    coors_atom37[i].numpy(),
                    pdb_path,
                    overwrite=True,
                    no_indexing=True,
                )

                res_row = list(flat_dict.values()) + [i, pdb_path, n]

                # If needed run designability, storing all intermediate values generated in sample_root_path
                if cfg.compute_designability:
                    res_designability = scRMSD(
                        pdb_path, ret_min=False, tmp_path=sample_root_path
                    )
                    res_row += [min(res_designability), res_designability]
                    print(res_designability)

                results.append(res_row)

        # Create the dataframe with results
        df = pd.DataFrame(results, columns=columns)


    csv_file = os.path.join(root_path, "..", f"results_{config_name}.csv")
    df.to_csv(csv_file, index=False)

