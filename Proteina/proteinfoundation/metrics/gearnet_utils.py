# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

import math
import os
from typing import List, Optional, Tuple

import torch
from torch import nn
from torch_geometric.data import Batch
from torch_geometric.nn import radius_graph
from torch_scatter import scatter_mean, scatter_sum


def rbf(d, rbf_dim, d_min=0.0, d_max=20.0):
    d_mu = torch.linspace(d_min, d_max, rbf_dim, device=d.device)
    d_mu = d_mu.view([1, -1])
    d_sigma = (d_max - d_min) / rbf_dim
    d_expand = torch.unsqueeze(d, -1)

    rbf = torch.exp(-(((d_expand - d_mu) / d_sigma) ** 2))
    return rbf


def clockwise_angle(p1, p2):
    assert p1.shape[-1] == 3
    assert p2.shape[-1] == 3
    x = (p1 * p2).sum(dim=-1)
    y = torch.cross(p1, p2, dim=-1)
    angle = torch.atan2(y.norm(dim=-1), x) * torch.sign(y[..., 2])
    return angle


def get_index_embedding(indices, embed_size, max_len=2056):
    """Creates sine / cosine positional embeddings from a prespecified indices.

    Args:
        indices: offsets of size [..., N_edges] of type integer
        max_len: maximum length.
        embed_size: dimension of the embeddings to create

    Returns:
        positional embedding of shape [N, embed_size]
    """
    K = torch.arange(embed_size // 2, device=indices.device)
    pos_embedding_sin = torch.sin(
        indices[..., None] * math.pi / (max_len ** (2 * K[None] / embed_size))
    ).to(indices.device)
    pos_embedding_cos = torch.cos(
        indices[..., None] * math.pi / (max_len ** (2 * K[None] / embed_size))
    ).to(indices.device)
    pos_embedding = torch.cat([pos_embedding_sin, pos_embedding_cos], axis=-1)
    return pos_embedding


class Linear(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        dropout: float = 0.0,
        bias: bool = False,
        leakyrelu_negative_slope: float = 0.1,
        momentum: float = 0.2,
    ):
        super(Linear, self).__init__()

        module = []
        module.append(nn.BatchNorm1d(in_channels, momentum=momentum))
        module.append(nn.LeakyReLU(leakyrelu_negative_slope))
        module.append(nn.Dropout(dropout))
        module.append(nn.Linear(in_channels, out_channels, bias=bias))
        self.module = nn.Sequential(*module)

    def forward(self, x):
        return self.module(x)


class MLP(nn.Module):
    def __init__(
        self,
        in_channels: int,
        mid_channels: int,
        out_channels: int,
        dropout: float = 0.0,
        bias: bool = True,
        leakyrelu_negative_slope: float = 0.2,
        momentum: float = 0.2,
    ):
        super(MLP, self).__init__()

        module = []
        module.append(nn.BatchNorm1d(in_channels, momentum=momentum))
        module.append(nn.LeakyReLU(leakyrelu_negative_slope))
        module.append(nn.Dropout(dropout))
        if mid_channels is None:
            module.append(nn.Linear(in_channels, out_channels, bias=bias))
        else:
            module.append(nn.Linear(in_channels, mid_channels, bias=bias))
        if mid_channels is None:
            module.append(nn.BatchNorm1d(out_channels, momentum=momentum))
        else:
            module.append(nn.BatchNorm1d(mid_channels, momentum=momentum))
        module.append(nn.LeakyReLU(leakyrelu_negative_slope))
        if mid_channels is None:
            module.append(nn.Dropout(dropout))
        else:
            module.append(nn.Linear(mid_channels, out_channels, bias=bias))

        self.module = nn.Sequential(*module)

    def forward(self, input):
        return self.module(input)


class GeometricRelationalGraphConv(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        edge_input_dim: Optional[int] = 0,
        num_relation: Optional[int] = 1,
        leakyrelu_negative_slope: Optional[float] = 0.1,
        dropout: Optional[float] = 0.2,
        bias: Optional[bool] = False,
    ):
        """
        Geometry-aware relational graph convolution operator from
        `Protein Representation Learning by Geometric Structure Pretraining`_.

        .. _Protein Representation Learning by Geometric Structure Pretraining:
            https://arxiv.org/abs/2203.06125

        Args:
            input_dim (int): Input dimension
            output_dim (int): Output dimension
            edge_input_dim (int): Input dimension of edge features
            num_relation (int): Number of relations.
            leakyrelu_negative_slope (Optional[float], optional): Controls the angle of the negative slope in LeakyReLU.
                Defaults to 0.1.
            dropout (Optional[float], optional): Probability in Dropout.
                Defaults to 0.2.
            bias (Optional[bool], optional): Whether to have bias in Linear layers.
                Defaults to False.
        """
        super(GeometricRelationalGraphConv, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.edge_input_dim = edge_input_dim
        self.num_relation = num_relation

        if input_dim != output_dim:
            self.identity = Linear(
                in_channels=input_dim,
                out_channels=output_dim,
                dropout=dropout,
                bias=bias,
                leakyrelu_negative_slope=leakyrelu_negative_slope,
            )
        else:
            self.identity = nn.Sequential()

        self.input = MLP(
            in_channels=input_dim,
            mid_channels=None,
            out_channels=input_dim,
            dropout=dropout,
            leakyrelu_negative_slope=leakyrelu_negative_slope,
        )

        if edge_input_dim > 0:
            self.edge_input = MLP(
                in_channels=edge_input_dim,
                mid_channels=None,
                out_channels=input_dim,
                dropout=dropout,
                leakyrelu_negative_slope=leakyrelu_negative_slope,
            )

        self.linear = Linear(
            in_channels=num_relation * input_dim,
            out_channels=output_dim,
            dropout=dropout,
            bias=bias,
            leakyrelu_negative_slope=leakyrelu_negative_slope,
        )

        self.output = Linear(
            in_channels=output_dim,
            out_channels=output_dim,
            dropout=dropout,
            bias=bias,
            leakyrelu_negative_slope=leakyrelu_negative_slope,
        )

    def forward(self, h_v, edge_index, h_e=None):
        identity = self.identity(h_v)
        h_v = self.input(h_v)

        node_in, node_out, relation_type = edge_index
        message = h_v[node_in]
        if self.edge_input_dim > 0:
            message = message + self.edge_input(h_e)

        assert relation_type.max() < self.num_relation
        node_out = node_out * self.num_relation + relation_type
        update = scatter_sum(
            message, node_out, dim=0, dim_size=h_v.shape[0] * self.num_relation
        )
        update = update.view(h_v.shape[0], self.num_relation * self.input_dim)

        output = self.linear(update)

        out = self.output(output) + identity
        return out


class GearNet(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_layers: int,
        edge_input_dim: int,
        num_relation: int,
        leakyrelu_negative_slope: Optional[float] = 0.1,
        dropout: Optional[float] = 0.2,
        radius: Optional[float] = 5.0,
        num_classes: Optional[List[Tuple[str, int]]] = None,
        max_len: Optional[int] = 3000,
        ca_only: Optional[bool] = False,
    ) -> None:
        """
        GearNet for protein structure reprentation learning and fold classification from
        `Protein Representation Learning by Geometric Structure Pretraining`_.

        .. _Protein Representation Learning by Geometric Structure Pretraining:
            https://arxiv.org/abs/2203.06125

        Args:
            input_dim (int): Input dimension
            hidden_dim (int): Hidden dimension
            num_layers (int): Number of layers
            edge_input_dim (int): Input dimension of edge features
            num_relation (int): Number of relations. One type for spatial edges, all the other types for sequential edges.
            leakyrelu_negative_slope (Optional[float], optional): Controls the angle of the negative slope in LeakyReLU.
                Defaults to 0.1.
            dropout (Optional[float], optional): Probability in Dropout.
                Defaults to 0.2.
            radius (Optional[float], optional): Spatial radius (\A) for constructing structure graph
                Defaults to 5.0.
            num_classes (Optional[List[Tuple[str, int]]], optional): List of tuples (level, num_class), indicating which fold level to predict and the number of classes at this level.
                Defaults to None.
            max_len (Optional[int], optional): Maximum length for calculating positional embeddings.
                Defaults to 3000.
            ca_only (Optional[bool], optional): Whether to use backbone structure model or CA-only structure model.
                Defaults to False.
        """
        super(GearNet, self).__init__()
        self.input_dim = input_dim
        self.output_dim = hidden_dim
        self.dims = [input_dim] + [hidden_dim] * num_layers
        self.edge_input_dim = edge_input_dim
        self.num_relation = num_relation
        self.radius = radius
        self.num_classes = num_classes
        self.max_len = max_len
        self.rbf_dim = edge_input_dim // 2
        self.ca_only = ca_only

        if self.ca_only:
            self.atom_embedding = nn.Embedding(
                num_embeddings=1, embedding_dim=input_dim // 2
            )  # CA embedding
        else:
            self.atom_embedding = nn.Embedding(
                num_embeddings=3, embedding_dim=input_dim // 2
            )  # N, CA, C embeddings
        self.layers = nn.ModuleList()
        for i in range(len(self.dims) - 1):
            self.layers.append(
                GeometricRelationalGraphConv(
                    self.dims[i],
                    self.dims[i + 1],
                    edge_input_dim=self.edge_input_dim,
                    num_relation=num_relation,
                    leakyrelu_negative_slope=leakyrelu_negative_slope,
                    dropout=dropout,
                )
            )

        self.mlp = MLP(
            in_channels=self.output_dim,
            mid_channels=None,  # 2 * self.output_dim,
            out_channels=self.output_dim,
            bias=True,
            dropout=dropout,
            leakyrelu_negative_slope=leakyrelu_negative_slope,
        )

        if num_classes:
            # num_classes should be given as a subset of [('H', num_class_H), ('T', num_class_T), ('A', num_class_A), ('C', num_class_C)]
            for k, num_class in num_classes:
                setattr(self, "pred_head_%s" % k, nn.Linear(self.output_dim, num_class))

    def construct_graph(self, atom_seq_pos, coords, atom2batch):
        # Sequential graph
        max_distance = (self.num_relation - 1) // 2
        node_in, node_out = radius_graph(
            atom_seq_pos.float(), max_distance + 0.1, batch=atom2batch
        )
        relation = atom_seq_pos[node_out] - atom_seq_pos[node_in] + max_distance
        relation = relation.clamp(0, self.num_relation - 2)
        seq_edge_list = torch.stack([node_in, node_out, relation], dim=0)

        # Spatial graph
        node_in, node_out = radius_graph(
            coords, self.radius, batch=atom2batch, max_num_neighbors=64
        )
        radius_edge_list = torch.stack(
            [node_in, node_out, torch.ones_like(node_in) * (self.num_relation - 1)],
            dim=0,
        )

        edge_list = torch.cat([seq_edge_list, radius_edge_list], dim=1)
        return edge_list

    def node_feature(self, atom_type, atom_seq_pos):
        # Atom type embedding
        if self.ca_only:
            # Reindex CA atom type for embedding
            atom_type_embedding = self.atom_embedding(atom_type - 1)
        else:
            atom_type_embedding = self.atom_embedding(atom_type)

        # Positional embedding
        position_embedding = get_index_embedding(
            atom_seq_pos, self.input_dim // 2, max_len=self.max_len
        )

        h_v = torch.cat(
            [
                atom_type_embedding,
                position_embedding,
            ],
            dim=-1,
        )

        return h_v

    def edge_feature(self, edge_list, atom_seq_pos, coords, atom2batch):
        node_in, node_out, _ = edge_list

        # RBF features
        pos_in, pos_out = coords[node_in], coords[node_out]
        rbf_feat = rbf((pos_out - pos_in).norm(dim=-1), self.rbf_dim)

        # Relative position embeddings
        rel_pos = atom_seq_pos[node_out] - atom_seq_pos[node_in]
        rel_pos_emebdding = get_index_embedding(
            rel_pos, self.rbf_dim - 2, max_len=self.max_len
        )  # Leave two dims for angle feat

        # Angle features to break reflection symmetry
        center_coord = scatter_mean(coords, atom2batch, dim=0)  # (batch_size, 3)
        diff_node_in = pos_in - center_coord[atom2batch[node_in]]  # (num_edge, 3)
        diff_node_out = pos_out - center_coord[atom2batch[node_out]]  # (num_edge, 3)
        angle = clockwise_angle(diff_node_in, diff_node_out)  # (num_edge, )
        angle_feat = torch.stack([angle.sin(), angle.cos()], dim=-1)

        h_e = torch.cat([rbf_feat, rel_pos_emebdding, angle_feat], dim=-1)

        return h_e

    def atom_info(self, batch, atom_mask):
        # Flatten residue info into atom info
        device = atom_mask.device
        coords = batch.coords[atom_mask]  # (num_atom, 3)
        atom2batch = batch.batch[:, None].expand_as(atom_mask)
        atom2batch = atom2batch[atom_mask]
        atom_type = torch.arange(atom_mask.shape[-1], device=device)[None, :].expand_as(
            atom_mask
        )
        atom_type = atom_type[atom_mask]

        num_residues = scatter_sum(
            torch.ones_like(batch.batch), batch.batch, dim=0
        )  # (batch_size, )
        num_cum_residues = num_residues.cumsum(dim=0)
        residue_id = torch.arange(batch.batch.shape[0], device=device)
        residue_id = (
            residue_id - (num_cum_residues - num_residues)[batch.batch]
        )  # Remove shift from batching
        residue_id = residue_id[:, None].expand_as(atom_mask)  # (num_residue, 37)
        atom_seq_pos = residue_id[atom_mask]  # (num_atom, )

        return atom_type, atom_type, coords, atom_seq_pos, atom2batch

    def forward(self, batch: Batch):
        atom_mask = batch.coord_mask.bool()  # (num_residue, 37)
        # Mask irrelevant atoms
        if self.ca_only:
            atom_mask[:, 2:] = 0
            atom_mask[:, 0] = 0
        else:
            atom_mask[:, 3:] = 0
            assert (
                atom_mask[:, 0].any() or atom_mask[:, 2].any()
            ), "Only find CA atoms if the structure, please set ca_only=True if you are using CA-only structures"

        atom_type, atom_type, coords, atom_seq_pos, atom2batch = self.atom_info(
            batch, atom_mask
        )
        h_v = self.node_feature(atom_type, atom_seq_pos)

        edge_list = self.construct_graph(atom_seq_pos, coords, atom2batch)
        h_e = self.edge_feature(edge_list, atom_seq_pos, coords, atom2batch)

        for i in range(len(self.layers)):
            h_v = self.layers[i](h_v, edge_list, h_e)

        protein_feature = scatter_sum(h_v, atom2batch, dim=0)
        protein_feature = self.mlp(protein_feature)

        output = {
            "protein_feature": protein_feature,
        }

        # Predict fold class based on different levels
        if self.num_classes:
            for k, num_class in self.num_classes:
                pred = getattr(self, "pred_head_%s" % k)(protein_feature)
                output["pred_%s" % k] = pred

        return output


class NoTrainBBGearNet(GearNet):
    """
    Pre-trained GearNet model on backbone structures
    """

    def __init__(self, ckpt_path: str):
        super().__init__(
            input_dim=512,
            hidden_dim=512,
            edge_input_dim=256,
            num_layers=8,
            num_relation=6,
            dropout=0.2,
            radius=10.0,
            ca_only=False,
            num_classes=[["T", 1336], ["A", 43], ["C", 5]],
        )

        if os.path.exists(ckpt_path):
            state_dict = torch.load(ckpt_path, map_location="cpu")
            self.load_state_dict(state_dict)
        else:
            raise ValueError(
                f"NoTrainBBGearNet checkpoint path {ckpt_path} does not exist."
            )
        # put into evaluation mode
        self.eval()
        for p in self.parameters():
            p.requires_grad = False  # Turn off gradient

    def train(self, mode: bool) -> "NoTrainBBGearNet":
        """Force network to always be in evaluation mode."""
        return super().train(False)


class NoTrainCAGearNet(GearNet):
    """
    Pre-trained GearNet model on CA-only structures
    """

    def __init__(self, ckpt_path: str):
        super().__init__(
            input_dim=512,
            hidden_dim=512,
            edge_input_dim=256,
            num_layers=8,
            num_relation=6,
            dropout=0.2,
            radius=10.0,
            ca_only=True,
            num_classes=[["T", 1336], ["A", 43], ["C", 5]],
        )

        if os.path.exists(ckpt_path):
            state_dict = torch.load(ckpt_path, map_location="cpu")
            self.load_state_dict(state_dict)
        else:
            raise ValueError(
                f"NoTrainAAGearNet checkpoint path {ckpt_path} does not exist."
            )
        # put into evaluation mode
        self.eval()
        for p in self.parameters():
            p.requires_grad = False

    def train(self, mode: bool) -> "NoTrainCAGearNet":
        """Force network to always be in evaluation mode."""
        return super().train(False)


if __name__ == "__main__":
    model = NoTrainBBGearNet(ckpt_path="./model_weights/gearnet.pth")
    ca_model = NoTrainCAGearNet(ckpt_path="./model_weights/gearnet_ca.pth")

    breakpoint()
