# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

import os
import sys

root = os.path.abspath(".")
sys.path.append(root)  # Adds project's root directory

import os
import hydra
import pprint
import argparse
from dotenv import load_dotenv
load_dotenv()

import torch
from torch_geometric.data import Data
from torch_geometric import transforms as T
import lightning as L

from proteinfoundation.metrics.metric_factory import GenerationMetricFactory, generation_metric_from_list


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Job info")
    parser.add_argument("--data_dir", type=str, help="Path to the pdb directory.")
    parser.add_argument("--ca_only", action="store_true", help="Whether to use ca_only model.")
    parser.add_argument("--batch_size", type=int, default=12, help="Batch size for representation computation.")
    parser.add_argument("--num_workers", type=int, default=32, help="Number of workers for data loading.")
    args = parser.parse_args()

    pdb_list = [os.path.join(args.data_dir, f) for f in os.listdir(args.data_dir)]

    data_path = os.environ["DATA_PATH"]
    model_name = "gearnet_ca.pth" if args.ca_only else "gearnet.pth"
    ckpt_path = os.path.join(data_path, "metric_factory", "model_weights", model_name)
    feat_name = "%seval_ca_features.pth" if args.ca_only else "%seval_features.pth"
    feat_path = os.path.join(data_path, "metric_factory", "features", feat_name)

    ref_db = ["pdb_", "D_FS_"]
    results = {}
    for db in ref_db:
        metric_factory = GenerationMetricFactory(
            ckpt_path=ckpt_path, 
            ca_only=args.ca_only, 
            metrics=["FID", "fJSD_C", "fJSD_A", "fJSD_T"], 
            real_features_path=feat_path % db,
            reset_real_features=False,
            prefix=db.upper(),
        )
        metric_factory = metric_factory.cuda()
        metric = generation_metric_from_list(
            pdb_list, 
            metric_factory, 
            batch_size=args.batch_size, 
            num_workers=args.num_workers,
            verbose=True,
        )
        results.update(metric)

    metric_factory = GenerationMetricFactory(
        ckpt_path=ckpt_path, 
        ca_only=args.ca_only, 
        metrics=["fS_C", "fS_A", "fS_T"], 
        real_features_path=None,
        reset_real_features=False,
    )
    metric_factory = metric_factory.cuda()
    metric = generation_metric_from_list(
        pdb_list, 
        metric_factory, 
        batch_size=args.batch_size, 
        num_workers=args.num_workers,
        verbose=True,
    )
    results.update(metric)

    print(pprint.pformat(results))
