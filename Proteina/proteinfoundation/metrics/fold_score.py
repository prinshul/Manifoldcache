# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

from typing import Any, List, Tuple

import torch
from torch import Tensor
from torchmetrics.metric import Metric
from torchmetrics.utilities import rank_zero_warn
from torchmetrics.utilities.data import dim_zero_cat


class ProteinFoldScore(Metric):

    is_differentiable: bool = False
    higher_is_better: bool = True
    full_state_update: bool = False
    plot_lower_bound: float = 0.0

    features: List

    def __init__(
        self,
        splits: int = 10,
        **kwargs: Any,
    ) -> None:
        """
        Calculate Fold Score which is used to access the quality of protein structures.

        Args:
            splits (int): Integer determining how many splits the fold score calculation should be split among.
                Defaults to 10.

        """
        super().__init__(**kwargs)

        rank_zero_warn(
            "Metric `FoldScore` will save all extracted features in buffer."
            " For large datasets this may lead to large memory footprint.",
            UserWarning,
        )

        self.splits = splits
        self.add_state("features", [], dist_reduce_fx=None)

    def update(self, features: Tensor) -> None:
        """Update the state with extracted features."""
        self.features.append(features)

    def compute(self) -> Tuple[Tensor, Tensor]:
        """Compute metric."""
        features = dim_zero_cat(self.features)
        # random permute the features
        idx = torch.randperm(features.shape[0])
        features = features[idx]

        # calculate probs and logits
        prob = features.softmax(dim=1)
        log_prob = features.log_softmax(dim=1)

        # split into groups
        prob = prob.chunk(self.splits, dim=0)
        log_prob = log_prob.chunk(self.splits, dim=0)

        # calculate score per split
        mean_prob = [p.mean(dim=0, keepdim=True) for p in prob]
        kl_ = [
            p * (log_p - (m_p + 1e-10).log())
            for p, log_p, m_p in zip(prob, log_prob, mean_prob)
        ]
        kl_ = [k.sum(dim=1).mean().exp() for k in kl_]
        kl = torch.stack(kl_)

        return kl.mean()


if __name__ == "__main__":
    num_features = 128
    batch_size = 64
    total_sample = 65536
    num_experiment = 10

    metric = ProteinFoldScore(splits=10)

    # Balanced and high confidence dataset
    for i in range(num_experiment):
        for _ in range(0, total_sample, batch_size):
            data = torch.ones((batch_size, num_features)) * -1e6
            label = torch.randint(num_features, (batch_size,))
            data[torch.arange(batch_size), label] = 1e6
            metric.update(data)
        is_mean, is_std = metric.compute()
        print(
            i,
            "Fold Score for balanced and high confidence dataset: %.5f +- %.5f"
            % (is_mean, is_std),
        )
        metric.reset()

    # Balanced but low confidence dataset
    for i in range(num_experiment):
        for _ in range(0, total_sample, batch_size):
            data = torch.rand((batch_size, num_features))
            metric.update(data)
        is_mean, is_std = metric.compute()
        print(
            i,
            "Fold Score for balanced but low confidence dataset: %.5f +- %.5f"
            % (is_mean, is_std),
        )
        metric.reset()

    # Skewed but high confidence dataset
    for i in range(num_experiment):
        for _ in range(0, total_sample, batch_size):
            data = torch.ones((batch_size, num_features)) * -1e6
            label = torch.randint(num_features // 10, (batch_size,))
            data[torch.arange(batch_size), label] = 1e6
            metric.update(data)
        is_mean, is_std = metric.compute()
        print(
            i,
            "Fold Score for skewed but high confidence dataset: %.5f +- %.5f"
            % (is_mean, is_std),
        )
        metric.reset()

    # Skewed and low confidence dataset
    for i in range(num_experiment):
        for _ in range(0, total_sample, batch_size):
            data = torch.rand((batch_size, num_features))
            data[num_features // 10 :] = -1e6
            metric.update(data)
        is_mean, is_std = metric.compute()
        print(
            i,
            "Fold Score for skewed and low confidence dataset: %.5f +- %.5f"
            % (is_mean, is_std),
        )
        metric.reset()
