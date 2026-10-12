# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

import lightning as L
from lightning.pytorch.callbacks import Callback


class SeedCallback(Callback):
    def on_save_checkpoint(self, trainer, pl_module, checkpoint):
        checkpoint["global_step"] = trainer.global_step

    def on_load_checkpoint(self, trainer, pl_module, checkpoint):
        global_step = checkpoint["global_step"]
        seed = global_step
        L.seed_everything(seed)
        print(f"Seeding rank {trainer.global_rank} with seed {seed}")
