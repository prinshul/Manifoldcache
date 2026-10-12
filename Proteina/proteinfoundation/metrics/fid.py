# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

from copy import deepcopy
from typing import Any, Dict

import torch
from torch import Tensor
from torchmetrics.metric import Metric


def _compute_fid(mu1: Tensor, sigma1: Tensor, mu2: Tensor, sigma2: Tensor) -> Tensor:
    r"""Compute adjusted version of `Fid Score`_.

    The Frechet Inception Distance between two multivariate Gaussians X_x ~ N(mu_1, sigm_1)
    and X_y ~ N(mu_2, sigm_2) is d^2 = ||mu_1 - mu_2||^2 + Tr(sigm_1 + sigm_2 - 2*sqrt(sigm_1*sigm_2)).

    Args:
        mu1: mean of activations calculated on predicted (x) samples
        sigma1: covariance matrix over activations calculated on predicted (x) samples
        mu2: mean of activations calculated on target (y) samples
        sigma2: covariance matrix over activations calculated on target (y) samples

    Returns:
        Scalar value of the distance between sets.

    """
    a = (mu1 - mu2).square().sum(dim=-1)
    b = sigma1.trace() + sigma2.trace()
    c = torch.linalg.eigvals(sigma1 @ sigma2).sqrt().real.sum(dim=-1)

    return a + b - 2 * c


class ProteinFrechetInceptionDistance(Metric):

    higher_is_better: bool = False
    is_differentiable: bool = False
    full_state_update: bool = False
    plot_lower_bound: float = 0.0

    real_features_sum: Tensor
    real_features_cov_sum: Tensor
    real_features_num_samples: Tensor

    fake_features_sum: Tensor
    fake_features_cov_sum: Tensor
    fake_features_num_samples: Tensor

    def __init__(
        self,
        num_features: int,
        reset_real_features: bool = True,
        **kwargs: Any,
    ) -> None:
        """
        Calculate Protein Fréchet inception distance (FID) which is used to access the similarity between two protein structure distribution.

        Args:
            num_features (int): Feature dimensions.
            reset_real_features (Optional[bool]): Whether to reset features of the real dataset for FID and fJSD metrics.
                Defaults to False.

        """
        super().__init__(**kwargs)

        if not isinstance(reset_real_features, bool):
            raise ValueError("Argument `reset_real_features` expected to be a bool")
        self.reset_real_features = reset_real_features

        mx_num_feats = (num_features, num_features)
        self.add_state(
            "real_features_sum",
            torch.zeros(num_features).double(),
            dist_reduce_fx="sum",
        )
        self.add_state(
            "real_features_cov_sum",
            torch.zeros(mx_num_feats).double(),
            dist_reduce_fx="sum",
        )
        self.add_state(
            "real_features_num_samples", torch.tensor(0).long(), dist_reduce_fx="sum"
        )

        self.add_state(
            "fake_features_sum",
            torch.zeros(num_features).double(),
            dist_reduce_fx="sum",
        )
        self.add_state(
            "fake_features_cov_sum",
            torch.zeros(mx_num_feats).double(),
            dist_reduce_fx="sum",
        )
        self.add_state(
            "fake_features_num_samples", torch.tensor(0).long(), dist_reduce_fx="sum"
        )

    def load_real_features(self, real_features_dict: Dict[str, Tensor]):
        """Load metric states of the real dataset."""
        self.real_features_sum = real_features_dict["real_features_sum"]
        self.real_features_cov_sum = real_features_dict["real_features_cov_sum"]
        self.real_features_num_samples = real_features_dict["real_features_num_samples"]

    def dump_real_features(self) -> Dict[str, Tensor]:
        """Return metric states of the real dataset."""
        return {
            "real_features_sum": self.real_features_sum,
            "real_features_cov_sum": self.real_features_cov_sum,
            "real_features_num_samples": self.real_features_num_samples,
        }

    def update(self, features: Tensor, real: bool) -> None:
        """Update the state with extracted features.

        Args:
            featrues: Input features.
            real: Whether given feature is real or fake.

        """
        batch_size = features.shape[0]
        self.orig_dtype = features.dtype
        features = features.double()

        if features.dim() == 1:
            features = features.unsqueeze(0)
        if real:
            self.real_features_sum += features.sum(dim=0)
            self.real_features_cov_sum += features.t().mm(features)
            self.real_features_num_samples += batch_size
        else:
            self.fake_features_sum += features.sum(dim=0)
            self.fake_features_cov_sum += features.t().mm(features)
            self.fake_features_num_samples += batch_size

    def compute(self) -> Tensor:
        """Calculate FID score based on accumulated extracted features from the two distributions."""
        if self.real_features_num_samples < 2 or self.fake_features_num_samples < 2:
            raise RuntimeError(
                "More than one sample is required for both the real and fake distributed to compute FID"
            )
        mean_real = (self.real_features_sum / self.real_features_num_samples).unsqueeze(
            0
        )
        mean_fake = (self.fake_features_sum / self.fake_features_num_samples).unsqueeze(
            0
        )

        cov_real_num = (
            self.real_features_cov_sum
            - self.real_features_num_samples * mean_real.t().mm(mean_real)
        )
        cov_real = cov_real_num / (self.real_features_num_samples - 1)
        cov_fake_num = (
            self.fake_features_cov_sum
            - self.fake_features_num_samples * mean_fake.t().mm(mean_fake)
        )
        cov_fake = cov_fake_num / (self.fake_features_num_samples - 1)
        return _compute_fid(
            mean_real.squeeze(0), cov_real, mean_fake.squeeze(0), cov_fake
        ).to(self.orig_dtype)

    def reset(self) -> None:
        """Reset metric states."""
        if not self.reset_real_features:
            real_features_sum = deepcopy(self.real_features_sum)
            real_features_cov_sum = deepcopy(self.real_features_cov_sum)
            real_features_num_samples = deepcopy(self.real_features_num_samples)
            super().reset()
            self.real_features_sum = real_features_sum
            self.real_features_cov_sum = real_features_cov_sum
            self.real_features_num_samples = real_features_num_samples
        else:
            super().reset()


if __name__ == "__main__":
    num_features = 128
    batch_size = 64
    total_sample = 65536
    num_experiment = 10

    metric = ProteinFrechetInceptionDistance(num_features, reset_real_features=False)

    # Real dataset N(0, 9)
    real_dataset = torch.randn((total_sample, num_features)) * 3
    metric.update(real_dataset, real=True)

    # Fake dataset N(3, 4)
    for i in range(num_experiment):
        for _ in range(0, total_sample, batch_size):
            fake_data = torch.randn(batch_size, num_features) * 2 + 3
            metric.update(fake_data, real=False)
        fid = metric.compute()
        print(i, "FID between N(0, 9) and N(3, 4): %.5f" % fid)
        metric.reset()

    # Fake dataset N(0, 4)
    for i in range(num_experiment):
        for _ in range(0, total_sample, batch_size):
            fake_data = torch.randn(batch_size, num_features) * 2
            metric.update(fake_data, real=False)
        fid = metric.compute()
        print(i, "FID between N(0, 9) and N(0, 4): %.5f" % fid)
        metric.reset()

    # Fake dataset N(0, 9)
    for i in range(num_experiment):
        for _ in range(0, total_sample, batch_size):
            fake_data = torch.randn(batch_size, num_features) * 3
            metric.update(fake_data, real=False)
        fid = metric.compute()
        print(i, "FID between N(0, 9) and N(0, 9): %.5f" % fid)
        metric.reset()
