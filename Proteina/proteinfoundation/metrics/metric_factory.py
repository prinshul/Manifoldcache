# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

from typing import Any, Dict, List, Optional, Union

import lightning as L  # noqa
import torch

import torch.distributed as dist
from lightning_utilities.core.rank_zero import rank_zero_only
from torch import Tensor
from torch.nn import Module, ModuleList
from torch_geometric.data import Batch, Data, Dataset
from torch_geometric.loader import DataLoader
from torchmetrics.metric import Metric
from tqdm import tqdm

from proteinfoundation.metrics.fid import ProteinFrechetInceptionDistance
from proteinfoundation.metrics.fJSD import FoldJensenShannonDivergence
from proteinfoundation.metrics.gearnet_utils import NoTrainBBGearNet, NoTrainCAGearNet
from proteinfoundation.metrics.fold_score import ProteinFoldScore
from proteinfoundation.utils.constants import PDB_TO_OPENFOLD_INDEX_TENSOR

from graphein_utils.graphein_utils import protein_to_pyg


class GenerationMetricFactory(ModuleList):

    structure_encoder: Module
    feature_network: str = "gearnet"

    def __init__(
        self,
        metrics: List[str],
        ckpt_path: str = "./data/metric_factory/model_weights/gearnet.pth",
        ca_only: bool = False,
        reset_real_features: bool = False,
        real_features_path: str = None,
        prefix: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        """
        Initialize the GeneationMetric Factory with specified structure encoder nd metrics.

        Args:
            metrics (List[str]): List of metric names to be included. Metric names should be in ["FID", "fS_C", "fS_A", "fS_T", "fJSD_C", "fJSD_A", "fJSD_T"]
            ckpt_path (str): Path to the checkpoint of the structure encoder.
            ca_only (Optional[bool]): Whether to use CA-only structure model or Backbone structure model.
                Defaults to False.
            reset_real_features (Optional[bool]): Whether to reset features of the real dataset for FID and fJSD metrics.
                Defaults to False.
            real_features_path (Optional[str]): Path to pre-saved features of the real dataset. Will be loaded in the rank zero process during initialization. If not None, reset_real_features will be turned off.
                Defaults to None.
            prefix (Optional[str]): Prefix prepended to metric names in the returned results.
                Defaults to None.

        Raises:
            ValueError: If metric is not supported.
        """
        super().__init__(**kwargs)

        if ca_only:
            self.structure_encoder = NoTrainCAGearNet(ckpt_path)
        else:
            self.structure_encoder = NoTrainBBGearNet(ckpt_path)

        num_features = self.structure_encoder.output_dim
        reset_real_features = reset_real_features and (real_features_path is None)

        self.prefix = prefix if prefix is not None else ""
        self.metrics = metrics
        self.metric_modules = ModuleList()
        for metric in metrics:
            if metric == "FID":
                _metric = ProteinFrechetInceptionDistance(
                    num_features, reset_real_features=reset_real_features
                )
            elif metric in ["fS_C", "fS_A", "fS_T"]:
                _metric = ProteinFoldScore(splits=1)
            elif metric in ["fJSD_C", "fJSD_A", "fJSD_T"]:
                for k, v in self.structure_encoder.num_classes:
                    if k == metric[-1]:
                        num_class = v
                _metric = FoldJensenShannonDivergence(
                    num_class, reset_real_features=reset_real_features
                )
            else:
                raise ValueError(
                    f"{metric} is not supported in GenerationMetricFactory"
                )
            self.metric_modules.append(_metric)

        if real_features_path:
            self.load_real_dataset_features(real_features_path)

        # Don't have structure_encoder in our state_dict
        def RemoveStructureEncoderKeys(module, incompatible_keys):
            incompatible_keys.missing_keys.clear()

        self.register_load_state_dict_post_hook(RemoveStructureEncoderKeys)

    @rank_zero_only
    def load_real_dataset_features(self, real_features_path: str):
        """Load features of the real dataset. Only loaded in the rank zero process to avoid duplicate on different devices.

        Args:
            real_features_path: Path to the real dataset features.

        """
        real_features = torch.load(real_features_path)
        for metric, metric_module in zip(self.metrics, self.metric_modules):
            if metric in ["FID", "fJSD_C", "fJSD_A", "fJSD_T"]:
                features_dict = {}
                for k, v in real_features.items():
                    if k.startswith(metric + "_"):
                        features_dict[k[len(metric + "_") :]] = v
                metric_module.load_real_features(features_dict)

    def dump_real_dataset_features(self, real_features_path: str):
        """Save features of the real dataset. Only support single device.

        Args:
            real_features_path: Path to the real dataset features.

        """
        if dist.is_initialized():
            total_processes = dist.get_world_size()
            assert (
                total_processes == 1
            ), f"Only support dumping real dataset features with 1 process, but got {total_processes}"
        real_features = {}
        for metric, metric_module in zip(self.metrics, self.metric_modules):
            if metric in ["FID", "fJSD_C", "fJSD_A", "fJSD_T"]:
                features_dict = metric_module.dump_real_features()
                for k, v in features_dict.items():
                    real_features[metric + "_" + k] = v.cpu()
        torch.save(real_features, real_features_path)

    def update(self, proteins: Batch, real: bool = False) -> None:
        """Update the state with extracted features.

        Args:
            proteins: Input proteins to evaluate. By default in pyg format.
            real: Whether given protein is real or fake.

        """
        with torch.no_grad():
            output = self.structure_encoder(proteins)
        for metric, metric_module in zip(self.metrics, self.metric_modules):
            if metric == "FID":
                metric_module.update(output["protein_feature"], real=real)
            elif metric in ["fS_C", "fS_A", "fS_T"] and not real:
                level = metric[-1]
                metric_module.update(output[f"pred_{level}"])
            elif metric in ["fJSD_C", "fJSD_A", "fJSD_T"]:
                level = metric[-1]
                metric_module.update(output[f"pred_{level}"], real=real)

    def compute(self) -> Tensor:
        """Compute metric."""
        output = {}
        for metric, metric_module in zip(self.metrics, self.metric_modules):
            output[self.prefix + metric] = metric_module.compute()
        return output

    def reset(self) -> None:
        """Reset metric states."""
        for metric_module in self.metric_modules:
            metric_module.reset()

    def set_dtype(self, dst_type: Union[str, torch.dtype]) -> "Metric":
        """Transfer all metric state to specific dtype. Special version of standard `type` method.

        Arguments:
            dst_type: the desired type as ``torch.dtype`` or string

        """
        out = super().set_dtype(dst_type)
        if isinstance(out.structure_encoder, NoTrainBBGearNet) or isinstance(
            out.structure_encoder, NoTrainCAGearNet
        ):
            out.structure_encoder._dtype = dst_type
        return out

    def state_dict(self, *args, **kwargs):
        return {}  # don't dump structure_encoder state_dict


class DatasetWrapper(Dataset):
    def __init__(self, pdb_list: List[str], **kwargs):
        super().__init__(**kwargs)
        self.pdb_list = pdb_list

    def __len__(
        self,
    ):
        return len(self.pdb_list)

    def __getitem__(self, index: int) -> Data:
        graph = protein_to_pyg(self.pdb_list[index], deprotonate=False)
        coord_mask = graph.coords != 1e-5
        graph.coord_mask = coord_mask[..., 0]

        graph.coords = graph.coords[:, PDB_TO_OPENFOLD_INDEX_TENSOR, :]
        graph.coord_mask = graph.coord_mask[:, PDB_TO_OPENFOLD_INDEX_TENSOR]
        # Need to have an attribute with 'node' so that PyG can infer the number of nodes
        #   See https://github.com/pyg-team/pytorch_geometric/blob/6eac972f9896e34c49118ac5d33afa9833c2ce8d/torch_geometric/data/storage.py#L407
        graph.node_id = torch.arange(graph.coords.shape[0]).unsqueeze(-1)
        # print(graph)
        # exit()
        return graph


def update_generation_metric(
    pdb_list: List[str],
    metric_factory: GenerationMetricFactory,
    batch_size: int = 12,
    num_workers: int = 32,
    real: bool = False,
    verbose: bool = False,
):
    dataset = DatasetWrapper(pdb_list)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    device = metric_factory.structure_encoder.atom_embedding.weight.device

    if verbose:
        dataloader = tqdm(dataloader)
    for batch in dataloader:
        batch = batch.to(device)
        metric_factory.update(batch, real=real)


def generation_metric_from_list(
    pdb_list: List[str],
    metric_factory: GenerationMetricFactory,
    batch_size: int = 12,
    num_workers: int = 32,
    verbose: bool = False,
) -> Dict[str, Tensor]:
    """
    Computes generation metrics in a GenerationMetricFactory.

    Args:
        pdb_list: List of paths to all PDBs we want to score.
        metric_factory: Database to compare against. So far only "pdb", we'll extend to a variant of "afdb".
        batch_size: Batch size for updating metrics.
            Default to 12, which should be fitted in 80G GPU memory.
        num_workers: Number of CPUs used for dataloading.
        verbase: Whether to turn on tqdm progress bar.

    Returns:
        Dictionary of metric names to metrics scores.
    """
    update_generation_metric(
        pdb_list, metric_factory, batch_size, num_workers, real=False, verbose=verbose
    )
    metrics = metric_factory.compute()
    metric_factory.reset()
    return metrics


if __name__ == "__main__":
    import os
    import pprint

    size = 1000

    metric_factory = GenerationMetricFactory(
        metrics=["FID", "fS_C", "fS_A", "fS_T", "fJSD_C", "fJSD_A", "fJSD_T"],
        ckpt_path="./model_weights/gearnet_ca.pth",
        ca_only=True,
        reset_real_features=False,
    )
    metric_factory = metric_factory.cuda()
    pdb_list = sorted(os.listdir("./data/scop/raw/scop_data/"))
    pdb_list = [os.path.join("./data/scop/raw/scop_data/", fname) for fname in pdb_list]
    pdb_listA = pdb_list[:size]

    update_generation_metric(pdb_listA, metric_factory, real=True, num_workers=0)
    for i in range(0, len(pdb_list), size):
        metric = generation_metric_from_list(
            pdb_list[i : i + size], metric_factory, num_workers=0
        )
        print("Dataset from [%d, %d)" % (i, i + size), pprint.pformat(metric))
