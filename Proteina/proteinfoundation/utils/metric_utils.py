# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.


import torch

from proteinfoundation.utils.ff_utils.pdb_utils import from_pdb_file


def transform_global_percentage_to_mask_dropout(fold_label_sample_ratio):
    assert (
        len(fold_label_sample_ratio) == 4
    ), "Length of fold_label_sample_ratio should be 4"
    assert (
        sum(fold_label_sample_ratio) == 1.0
    ), "Sum of fold_label_sample_ratio should be 1.0"
    mask_T_prob = sum(fold_label_sample_ratio[:3]) / sum(
        fold_label_sample_ratio
    )  # Among all samples, how many T-level labels are dropped?       (null + C + CA) / (null + C + CA + CAT)
    mask_A_prob = sum(fold_label_sample_ratio[:2]) / (
        sum(fold_label_sample_ratio[:3]) + 1e-10
    )  # Among samples with T labels dropped, how many A-level labels are dropped?    (null + C) / (null + C + CA)
    mask_C_prob = sum(fold_label_sample_ratio[:1]) / (
        sum(fold_label_sample_ratio[:2]) + 1e-10
    )  # Among samples with A and T labels dropped, how many C-level labels are dropped?     null / (null + C)
    return mask_T_prob, mask_A_prob, mask_C_prob


def load_alpha_carbon_coordinates(pdb_file):
    prot = from_pdb_file(pdb_file)
    mask = torch.Tensor(prot.atom_mask).long().bool()  # [n, 37]
    coors_atom37 = torch.Tensor(prot.atom_positions)  # [n, 37, 3]
    mask_ca = mask[:, 1]  # [n]
    return coors_atom37[mask_ca, 1, :]  # [n_unmasked, 3]


def compute_ca_metrics(pdb_path):
    try:
        coors = load_alpha_carbon_coordinates(pdb_path)  # [n, 3]
        consecutive_ca_ca_distances = torch.norm(
            coors[1:, :] - coors[:-1, :], dim=-1
        )  # [n-1]
        pairwise_ca_ca_distances = torch.norm(
            coors[None, :, :] - coors[:, None, :], dim=-1
        )  # [n, n]
        num_collisions = (
            torch.sum(
                (pairwise_ca_ca_distances > 0.01) & (pairwise_ca_ca_distances < 2.0)
            )
            / 2.0
        )
        # The greater than is to avoid diagonal elements which do not count as collisions
        return {
            "ca_ca_dist_avg": torch.mean(consecutive_ca_ca_distances),
            "ca_ca_dist_median": torch.median(consecutive_ca_ca_distances),
            "ca_ca_dist_std": torch.std(consecutive_ca_ca_distances),
            "ca_ca_dist_min": torch.min(consecutive_ca_ca_distances),
            "ca_ca_dist_max": torch.max(consecutive_ca_ca_distances),
            "ca_ca_collisions(2A)": num_collisions,
        }
    except Exception as e:
        print(f"Error in ca-ca metrics {e}")
        return {
            "ca_ca_dist_avg": 0.0,
            "ca_ca_dist_median": 0.0,
            "ca_ca_dist_std": 0.0,
            "ca_ca_dist_min": 0.0,
            "ca_ca_dist_max": 0.0,
            "ca_ca_collisions(2A)": 0.0,
        }

def compute_structural_metrics(pdb_path):
    """Computes a bunch of validation metrics, returns them as a dictionary."""
    metrics_ca_ca = compute_ca_metrics(pdb_path)
    return metrics_ca_ca
