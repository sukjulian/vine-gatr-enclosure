# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear
import os
from typing import Dict, Literal, Tuple, Union

import torch
import torch_geometric as pyg
from gatr.interface import (
    embed_oriented_plane,
    embed_point,
    extract_oriented_plane,
    extract_scalar,
)
from omegaconf import DictConfig
from torch.nn.functional import l1_loss, mse_loss

from gatr_enclosure.datasets import ArterialSurfaceDataset
from gatr_enclosure.models import GATr, ProjectiveGeometricAlgebraInterface, ViNEGATr
from gatr_enclosure.nn.loss import AttractiveLoss
from gatr_enclosure.transforms import PointCloudSingularValueDecomposition, Subsampling
from gatr_enclosure.transforms.functional import surface_normal

from .base import BaseExperiment
from .utils import compute_approximation_error, get_generic_splits_idcs


class ArterialSurfaceExperiment(BaseExperiment):

    def get_model(self, config: DictConfig) -> torch.nn.Module:

        model = ViNEGATr if config.model.id == "vine_gatr" else GATr

        return model(Interface(), **config.model)

    def get_dataset(self, config: DictConfig) -> pyg.data.Dataset:
        dir_ = os.path.join("datasets", "arterial-surface")

        pre_transform = pyg.transforms.Compose(
            (
                surface_normal,
                PointCloudSingularValueDecomposition(
                    use_orientation=config.model.virtual_nodes_use_orientation
                ),
            )
        )

        transform = Subsampling(num_samples=config.num_samples) if "num_samples" in config else None

        return ArterialSurfaceDataset(dir_, pre_transform=pre_transform, transform=transform)

    def get_dataset_splits_idcs(
        self, config: DictConfig
    ) -> Dict[Literal["training", "validation", "test"], torch.Tensor]:

        return get_generic_splits_idcs(len(self._dataset), 0.8, 0.1)

    @staticmethod
    def loss_fn(y: torch.Tensor, data: pyg.data.Data, config: DictConfig) -> torch.Tensor:
        factors = config.training.loss_term_factors

        generic_term = mse_loss(y, data.y) if config.training.loss == "mse" else l1_loss(y, data.y)

        if factors.attractive >= 1e-6:
            attractive_term = AttractiveLoss()(
                data.pos,
                data.virtual_nodes_pos,
                data.batch,
                data.virtual_nodes_batch if "virtual_nodes_batch" in data else None,
            )
        else:
            attractive_term = torch.tensor(0.0)

        loss = factors.generic * generic_term + factors.attractive * attractive_term

        return loss

    @staticmethod
    def metric_fn(
        y: torch.Tensor, y_data: torch.Tensor, scatter_idcs: torch.Tensor, config: DictConfig
    ) -> torch.Tensor:

        metric = compute_approximation_error(y, y_data, scatter_idcs)

        return metric


class Interface(ProjectiveGeometricAlgebraInterface):
    in_mv_channels = 2
    out_mv_channels = 1
    in_s_channels = 1
    out_s_channels = None

    @staticmethod
    @torch.no_grad()
    def embed(data: pyg.data.Data) -> Tuple[torch.Tensor, torch.Tensor]:

        mv = torch.cat(
            (
                embed_point(data.pos).view(-1, 1, 16),
                embed_oriented_plane(data.surf_normal, data.pos).view(-1, 1, 16),
            ),
            dim=1,
        )

        s = data.geodesic_dist_inlet.view(-1, 1)

        return mv, s

    @staticmethod
    def extract(mv: torch.Tensor, s: Union[torch.Tensor, None]) -> torch.Tensor:
        return extract_scalar(mv).view(-1, 1) * extract_oriented_plane(mv).squeeze()
