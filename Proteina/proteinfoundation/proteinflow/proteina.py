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
from math import prod
from typing import Dict

import torch
from jaxtyping import Bool, Float
from lightning.pytorch.utilities.rank_zero import rank_zero_only
from scipy.spatial.transform import Rotation
from torch import Tensor

from proteinfoundation.flow_matching.r3n_fm import R3NFlowMatcher
from proteinfoundation.nn.protein_transformer import ProteinTransformerAF3
from proteinfoundation.proteinflow.model_trainer_base import ModelTrainerBase
from proteinfoundation.utils.align_utils.align_utils import kabsch_align
from proteinfoundation.utils.coors_utils import ang_to_nm, trans_nm_to_atom37
from proteinfoundation.nn.motif_factory import SingleMotifFactory


@rank_zero_only
def create_dir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir, exist_ok=True)


def sample_uniform_rotation(
    shape=tuple(), dtype=None, device=None
) -> Float[Tensor, "*batch 3 3"]:
    """
    Samples rotations distributed uniformly.

    Args:
        shape: tuple (if empty then samples single rotation)
        dtype: used for samples
        device: torch.device

    Returns:
        Uniformly samples rotation matrices [*shape, 3, 3]
    """
    return torch.tensor(
        Rotation.random(prod(shape)).as_matrix(),
        device=device,
        dtype=dtype,
    ).reshape(*shape, 3, 3)


class Proteina(ModelTrainerBase):
    def __init__(self, cfg_exp, store_dir=None):
        super(Proteina, self).__init__(cfg_exp=cfg_exp, store_dir=store_dir)
        self.save_hyperparameters()

        # Define flow matcher
       
        self.motif_conditioning = cfg_exp.training.get("motif_conditioning", False)
        self.fm = R3NFlowMatcher(zero_com= not self.motif_conditioning, scale_ref=1.0)  # Work in nm
        if self.motif_conditioning:
            self.motif_conditioning_sequence_rep = cfg_exp.training.get("motif_conditioning_sequence_rep", False)
            if self.motif_conditioning_sequence_rep:
                if "motif_sequence_mask" not in cfg_exp.model.nn.feats_init_seq:
                    cfg_exp.model.nn.feats_init_seq.append("motif_sequence_mask")
                if "motif_x1" not in cfg_exp.model.nn.feats_init_seq:
                    cfg_exp.model.nn.feats_init_seq.append("motif_x1")
                
            if "motif_structure_mask" not in cfg_exp.model.nn.feats_pair_repr:
                cfg_exp.model.nn.feats_pair_repr.append("motif_structure_mask")
            if "motif_x1_pair_dists" not in cfg_exp.model.nn.feats_pair_repr:
                cfg_exp.model.nn.feats_pair_repr.append("motif_x1_pair_dists")
            self.motif_factory = SingleMotifFactory(motif_prob=cfg_exp.training.get("motif_prob", 1.0))

        # Neural network
        self.nn = ProteinTransformerAF3(**cfg_exp.model.nn)

        self.nparams = sum(p.numel() for p in self.nn.parameters() if p.requires_grad)

        create_dir(self.val_path_tmp)

    def align_wrapper(self, x_0, x_1, mask):
        """Performs Kabsch on the translation component of x_0 and x_1."""
        return kabsch_align(mobile=x_0, target=x_1, mask=mask)

    def extract_clean_sample(self, batch):
        """
        Extracts clean sample, mask, batch size, protein length n, and dtype from the batch.
        Applies augmentations if those are required.

        Args:
            batch: batch from dataloader.

        Returns:
            Tuple (x_1, mask, batch_shape, n, dtype)
        """
        x_1 = batch["coords"][:,:,1,:]  # [b, n, 3]
        mask = batch["mask_dict"]["coords"][..., 0, 0]  # [b, n] boolean
        if self.cfg_exp.model.augmentation.global_rotation:
            # CAREFUL: If naug_rot is > 1 this increases "batch size"
            x_1, mask = self.apply_random_rotation(
                x_1, mask, naug=self.cfg_exp.model.augmentation.naug_rot
            )
        batch_shape = x_1.shape[:-2]
        n = x_1.shape[-2]
        return (
            ang_to_nm(x_1),
            mask,
            batch_shape,
            n,
            x_1.dtype,
        )  # Since we work in nm throughout

    def apply_random_rotation(self, x, mask, naug=1):
        """
        Applies random rotation augmentation. Each sample in the batch may receive more than one augmentation,
        specified by the parameters naug. If naug > 1 this is basically increaseing the batch size from b to
        naug * b. This should likely be implemented in the dataloaders.

        Args:
            - x: Data batch, shape [b, n, 3]
            - mask: Binary, shape [b, n]
            - naug: Number of augmentations to apply to each sample, effectively increasing batch size if >1.

        Returns:
            Augmented samples and mask, shapes [b * naug, n, 3] and [B * naug, n].
        """
        assert (
            x.ndim == 3
        ), f"Augmetations can only be used for simple (x_1) batches [b, n, 3], current shape is {x.shape}"
        assert (
            mask.ndim == 2
        ), f"Augmetations can only be used for simple (mask) batches [b, n], current shape is {mask.shape}"
        assert naug >= 1, f"Number of augmentations (int) should >= 1, currently {naug}"

        # Repeat for multiple augmentations per sample
        x = x.repeat([naug, 1, 1])  # [naug * b, n, 3]
        mask = mask.repeat([naug, 1])  # [naug * b, n]

        # Sample and apply rotations
        rots = sample_uniform_rotation(
            shape=x.shape[:-2], dtype=x.dtype, device=x.device
        )  # [naug * b, 3, 3]
        x_rot = torch.matmul(x, rots)
        return self.fm._mask_and_zero_com(x_rot, mask), mask

    def compute_loss_weight(
        self, t: Float[Tensor, "*"], eps: float = 1e-3
    ) -> Float[Tensor, "*"]:
        t = t.clamp(min=eps, max=1.0 - eps)  # For safety
        return t / (
            1.0 - t
        )

    def compute_fm_loss(
        self,
        x_1: Float[Tensor, "* n 3"],
        x_1_pred: Float[Tensor, "* n 3"],
        x_t: Float[Tensor, "* n 3"],
        t: Float[Tensor, "*"],
        mask: Bool[Tensor, "* nres"],
        log_prefix: str,
    ) -> Float[Tensor, "*"]:
        """
        Computes and logs flow matching loss.

        Args:
            x_1: True clean sample, shape [*, n, 3].
            x_1_pred: Predicted clean sample, shape [*, n, 3].
            x_t: Sample at interpolation time t (used as input to predict clean sample), shape [*, n, 3].
            t: Interpolation time, shape [*].
            mask: Boolean residue mask, shape [*, nres].

        Returns:
            Flow matching loss.
        """
        nres = torch.sum(mask, dim=-1) * 3  # [*]

        err = (x_1 - x_1_pred) * mask[..., None]  # [*, n, 3]
        loss = torch.sum(err**2, dim=(-1, -2)) / nres  # [*]

        total_loss_w = 1.0 / ((1.0 - t) ** 2 + 1e-5)

        loss = loss * total_loss_w  # [*]
        if log_prefix:
            self.log(
                f"{log_prefix}/trans_loss",
                torch.mean(loss),
                on_step=True,
                on_epoch=True,
                prog_bar=True,
                logger=True,
                batch_size=mask.shape[0],
                sync_dist=True,
                add_dataloader_idx=False,
            )
        return loss

    def compute_auxiliary_loss(
        self,
        x_1: Float[Tensor, "* n 3"],
        x_1_pred: Float[Tensor, "* n 3"],
        x_t: Float[Tensor, "* n 3"],
        t: Float[Tensor, "*"],
        mask: Bool[Tensor, "* n"],
        nn_out: Dict[str, Tensor],
        log_prefix: str,
        batch: Dict[str, Tensor] = None,
    ) -> Float[Tensor, ""]:
        """
        Computes and logs auxiliary losses.

        Args:
            x_1: True clean sample, shape [*, n, 3].
            x_1_pred: Predicted clean sample, shape [*, n, 3].
            x_t: Sample at interpolation time t (used as input to predict clean sample), shape [*, n, 3].
            t: Interpolation time, shape [*].
            mask: Boolean residue mask, shape [*, n].
            nn_out: Dictionary of output from neural network

        Returns:
            Auxiliary loss.
        """
        bs = x_1.shape[0]
        n = x_1.shape[1]
        nres = mask.sum(-1)  # [*]

        gt_ca_coors = x_1 * mask[..., None]  # [*, n, 3]
        pred_ca_coors = x_1_pred * mask[..., None]  # [*, n, 3]
        pair_mask = mask[..., None, :] * mask[..., None]  # [*, n, n]

        # Pairwise distances
        gt_pair_dists = torch.linalg.norm(
            gt_ca_coors[:, :, None, :] - gt_ca_coors[:, None, :, :], dim=-1
        )  # [*, n, n]
        pred_pair_dists = torch.linalg.norm(
            pred_ca_coors[:, :, None, :] - pred_ca_coors[:, None, :, :], dim=-1
        )  # [*, n, n]
        gt_pair_dists = gt_pair_dists * pair_mask  # [*, n, n]
        pred_pair_dists = pred_pair_dists * pair_mask  # [*, n, n]

        # Add mask to only account for pairs that are closer than thr in ground truth
        max_dist = self.cfg_exp.loss.thres_aux_2d_loss
        if max_dist is None:
            max_dist = 1e10
        pair_mask_thr = gt_pair_dists < max_dist  # [*, n, n]
        total_pair_mask = pair_mask * pair_mask_thr  # [*, n, n]

        # Compute loss
        den = torch.sum(total_pair_mask, dim=(-1, -2)) - nres
        dist_mat_loss = torch.sum(
            (gt_pair_dists - pred_pair_dists) ** 2 * total_pair_mask, dim=(-1, -2)
        )  # [*]
        dist_mat_loss = dist_mat_loss / den  # [*]

        # Distogram loss
        num_dist_buckets = self.cfg_exp.loss.get("num_dist_buckets", 64)
        pair_pred = nn_out.get("pair_pred", None)
        if num_dist_buckets and pair_pred is not None:
            assert (
                num_dist_buckets == pair_pred.shape[-1]
            ), "The number of distance buckets should be equal with the output dim of pair pred head"
            assert num_dist_buckets > 1, "Need more than one bucket for distogram loss"

            # Bucketize pair distance
            max_dist_boundary = self.cfg_exp.loss.get("max_dist_boundary", 1.0)
            boundaries = torch.linspace(
                0.0, max_dist_boundary, num_dist_buckets - 1, device=pair_pred.device
            )
            gt_pair_dist_bucket = torch.bucketize(
                gt_pair_dists, boundaries
            )  # [*, n, n], each value in [0, num_dist_buckets)

            # Distogram loss
            pair_pred = pair_pred.view(bs * n * n, num_dist_buckets)
            gt_pair_dist_bucket = gt_pair_dist_bucket.view(bs * n * n)
            distogram_loss = torch.nn.functional.cross_entropy(
                pair_pred, gt_pair_dist_bucket, reduction="none"
            )  # [bs * n * n]
            distogram_loss = distogram_loss.view(bs, n, n)
            distogram_loss = torch.sum(distogram_loss * pair_mask, dim=(-1, -2))  # [*]
            distogram_loss = distogram_loss / (
                pair_mask.sum(dim=(-1, -2)) + 1e-10
            )  # [*]
        else:
            distogram_loss = dist_mat_loss * 0

        auxiliary_loss = (
            distogram_loss
            * (t > self.cfg_exp.loss.aux_loss_t_lim)
            * self.cfg_exp.loss.aux_loss_weight
        )
        auxiliary_loss_no_w = distogram_loss * (t > self.cfg_exp.loss.aux_loss_t_lim)
        motif_aux_loss_weight = self.cfg_exp.loss.get("motif_aux_loss_weight", 0)
        scaffold_aux_loss_weight = self.cfg_exp.loss.get("scaffold_aux_loss_weight", 0)
        if scaffold_aux_loss_weight > 0:
            scaffold_loss = scaffold_aux_loss_weight * self.compute_fm_loss(
                        x_1=x_1,
                        x_1_pred=x_1_pred,
                        x_t=x_t,
                        mask=~batch["fixed_sequence_mask"]*batch["mask"],
                        t=t,
                        log_prefix=None
                    )
            auxiliary_loss += scaffold_loss
            self.log(
                f"{log_prefix}/scaffold_loss",
                torch.mean(scaffold_loss),
                on_step=True,
                on_epoch=True,
                prog_bar=False,
                logger=True,
                batch_size=mask.shape[0],
                sync_dist=True,
                add_dataloader_idx=False,
            )
        elif motif_aux_loss_weight:
            mask_to_use = batch["fixed_sequence_mask"] * batch["mask"]
            check_weight = 1.0
            if not batch["fixed_sequence_mask"].any():
                check_weight = 0
                mask_to_use = batch["mask"]
            motif_loss = motif_aux_loss_weight * self.compute_fm_loss(
                x_1=x_1,
                x_1_pred=x_1_pred,
                x_t=x_t,
                mask=mask_to_use,
                t=t,
                log_prefix=None,
            )
            auxiliary_loss += check_weight * motif_loss
            self.log(
                f"{log_prefix}/motif_loss",
                torch.mean(motif_loss * check_weight),
                on_step=True,
                on_epoch=True,
                prog_bar=False,
                logger=True,
                batch_size=mask.shape[0],
                sync_dist=True,
                add_dataloader_idx=False,
            )

        self.log(
            f"{log_prefix}/distogram_loss",
            torch.mean(distogram_loss),
            on_step=True,
            on_epoch=True,
            prog_bar=False,
            logger=True,
            batch_size=mask.shape[0],
            sync_dist=True,
            add_dataloader_idx=False,
        )
        self.log(
            f"{log_prefix}/dist_mat_loss",
            torch.mean(dist_mat_loss),
            on_step=True,
            on_epoch=True,
            prog_bar=False,
            logger=True,
            batch_size=mask.shape[0],
            sync_dist=True,
            add_dataloader_idx=False,
        )
        self.log(
            f"{log_prefix}/auxiliary_loss",
            torch.mean(auxiliary_loss),
            on_step=True,
            on_epoch=True,
            prog_bar=False,
            logger=True,
            batch_size=mask.shape[0],
            sync_dist=True,
            add_dataloader_idx=False,
        )
        self.log(
            f"{log_prefix}/auxiliary_loss_no_w",
            torch.mean(auxiliary_loss_no_w),
            on_step=True,
            on_epoch=True,
            prog_bar=False,
            logger=True,
            batch_size=mask.shape[0],
            sync_dist=True,
            add_dataloader_idx=False,
        )
        return auxiliary_loss

    def detach_gradients(self, x):
        """Detaches gradients from sample x"""
        return x.detach()

    def samples_to_atom37(self, samples):
        """
        Transforms samples to atom37 representation.

        Args:
            samples: Tensor of shape [b, n, 3]

        Returns:
            Samples in atom37 representation, shape [b, n, 37, 3].
        """
        return trans_nm_to_atom37(samples)  # [b, n, 37, 3]
