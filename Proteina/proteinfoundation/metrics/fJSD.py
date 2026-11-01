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
from torch.nn import functional as F
from torchmetrics.metric import Metric


class FoldJensenShannonDivergence(Metric):

    higher_is_better: bool = False
    is_differentiable: bool = False
    full_state_update: bool = False
    plot_lower_bound: float = 0.0

    real_features_sum: Tensor
    real_features_num_samples: Tensor

    fake_features_sum: Tensor
    fake_features_num_samples: Tensor

    def __init__(
        self,
        num_classes: int,
        reset_real_features: bool = True,
        **kwargs: Any,
    ) -> None:
        """
        Calculate Fold Jensen Shannon Divergence (fJSD) which is used to access the similarity between two protein structure label distribution.

        Args:
            num_classes (int): Number of classes for the label distribution.
            reset_real_features (Optional[bool]): Whether to reset features of the real dataset for FID and fJSD metrics.
                Defaults to False.

        """
        super().__init__(**kwargs)

        if not isinstance(reset_real_features, bool):
            raise ValueError("Argument `reset_real_features` expected to be a bool")
        self.reset_real_features = reset_real_features

        num_features = num_classes
        self.add_state(
            "real_features_sum",
            torch.zeros(num_features).double(),
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
            "fake_features_num_samples", torch.tensor(0).long(), dist_reduce_fx="sum"
        )

    def load_real_features(self, real_features_dict: Dict[str, Tensor]):
        """Load metric states of the real dataset."""
        self.real_features_sum = real_features_dict["real_features_sum"]
        self.real_features_num_samples = real_features_dict["real_features_num_samples"]

    def dump_real_features(self) -> Dict[str, Tensor]:
        """Return metric states of the real dataset."""
        return {
            "real_features_sum": self.real_features_sum,
            "real_features_num_samples": self.real_features_num_samples,
        }

    def update(self, features: Tensor, real: bool) -> None:
        """Update the state with extracted features.

        Args:
            features: Input features.
            real: Whether given feature is real or fake.

        """
        batch_size = features.shape[0]
        features = features.softmax(
            dim=-1
        )  # A better solution should be to use logits and log_sum_exp to reduce
        if features.dim() == 1:
            features = features.unsqueeze(0)
        if real:
            self.real_features_sum += features.sum(dim=0)
            self.real_features_num_samples += batch_size
        else:
            self.fake_features_sum += features.sum(dim=0)
            self.fake_features_num_samples += batch_size

    def compute(self) -> Tensor:
        """Compute metric."""
        if self.real_features_num_samples < 2 or self.fake_features_num_samples < 2:
            raise RuntimeError(
                "More than one sample is required for both the real and fake distributed to compute FID"
            )
        mean_real = (self.real_features_sum / self.real_features_num_samples)[None, :]
        mean_fake = (self.fake_features_sum / self.fake_features_num_samples)[None, :]

        m = 0.5 * (mean_real + mean_fake)
        jsd = F.kl_div((m + 1e-10).log(), mean_real, reduction="batchmean") + F.kl_div(
            (m + 1e-10).log(), mean_fake, reduction="batchmean"
        )
        jsd /= 2.0

        return jsd * 10

    def reset(self) -> None:
        """Reset metric states."""
        if not self.reset_real_features:
            real_features_sum = deepcopy(self.real_features_sum)
            real_features_num_samples = deepcopy(self.real_features_num_samples)
            super().reset()
            self.real_features_sum = real_features_sum
            self.real_features_num_samples = real_features_num_samples
        else:
            super().reset()


if __name__ == "__main__":
    num_features = 128
    batch_size = 64
    total_sample = 65536
    num_experiment = 10

    metric = FoldJensenShannonDivergence(num_features, reset_real_features=False)

    # Real dataset: uniform over the first half
    real_dataset = torch.cat(
        [
            torch.ones((total_sample, num_features // 2)) * 1e6,
            torch.ones((total_sample, num_features // 2)) * -1e6,
        ],
        dim=-1,
    )
    metric.update(real_dataset, real=True)

    # Fake dataset: uniform over all
    for i in range(num_experiment):
        for _ in range(0, total_sample, batch_size):
            fake_data = torch.ones(batch_size, num_features)
            metric.update(fake_data, real=False)
        fjsd = metric.compute()
        print(
            i,
            "fJSD between (uniform over first half) and (uniform over all)): %.5f"
            % fjsd,
        )
        metric.reset()

    # Fake dataset: uniform over the second half
    for i in range(num_experiment):
        for _ in range(0, total_sample, batch_size):
            fake_data = torch.cat(
                [
                    torch.ones((batch_size, num_features // 2)) * -1e6,
                    torch.ones((batch_size, num_features // 2)) * 1e6,
                ],
                dim=-1,
            )
            metric.update(fake_data, real=False)
        fjsd = metric.compute()
        print(
            i,
            "fJSD between (uniform over first half) and (uniform over second half)): %.5f"
            % fjsd,
        )
        metric.reset()

    # Fake dataset: uniform over the first half
    for i in range(num_experiment):
        for _ in range(0, total_sample, batch_size):
            fake_data = torch.cat(
                [
                    torch.ones((batch_size, num_features // 2)) * 1e6,
                    torch.ones((batch_size, num_features // 2)) * -1e6,
                ],
                dim=-1,
            )
            metric.update(fake_data, real=False)
        fjsd = metric.compute()
        print(
            i,
            "fJSD between (uniform over first half) and (uniform over first half)): %.5f"
            % fjsd,
        )
        metric.reset()
