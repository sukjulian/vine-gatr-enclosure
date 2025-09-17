# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear
import os
from typing import Dict, Literal, Tuple, Union, cast

import torch
import torch_geometric as pyg
from gatr.interface import embed_point, extract_oriented_plane, extract_scalar
from lab_gatr import PointCloudPoolingScales
from omegaconf import DictConfig
from torch.nn.functional import l1_loss, mse_loss

from gatr_enclosure.datasets import ArterialVolumeDataset
from gatr_enclosure.models import (
    GATr,
    LaBGATr,
    ProjectiveGeometricAlgebraInterface,
    ViNEGATr,
)
from gatr_enclosure.transforms import PointCloudSingularValueDecomposition, Subsampling
from gatr_enclosure.transforms.functional import idcs_euclidean_dist

from .base import BaseExperiment
from .utils import compute_approximation_error, get_generic_splits_idcs


class ArterialVolumeExperiment(BaseExperiment):

    def get_model(self, config: DictConfig) -> torch.nn.Module:

        match config.scaling.strategy:
            case "gatr_and_subsampling":
                model = cast(torch.nn.Module, GATr)
            case "lab_gatr":
                model = cast(torch.nn.Module, LaBGATr)
            case "vine_gatr":
                model = cast(torch.nn.Module, ViNEGATr)

        return model(Interface(), **config.model)

    def get_dataset(self, config: DictConfig) -> pyg.data.Dataset:
        dir_ = os.path.join("datasets", "arterial-volume")

        pre_transforms = [idcs_euclidean_dist]
        if config.scaling.strategy == "lab_gatr":
            pre_transforms.append(
                PointCloudPoolingScales(
                    rel_sampling_ratios=(config.scaling.compression,), interp_simplex="tetrahedron"
                )
            )
        elif config.scaling.strategy == "vine_gatr":
            pre_transforms.append(PointCloudSingularValueDecomposition())

        pre_transform = pyg.transforms.Compose(pre_transforms)

        transform = (
            Subsampling(num_samples=config.scaling.num_samples)  # also applied at test time
            if config.scaling.strategy == "gatr_and_subsampling"
            else None
        )

        return ArterialVolumeDataset(dir_, pre_transform=pre_transform, transform=transform)

    def get_dataset_splits_idcs(
        self, config: DictConfig
    ) -> Dict[Literal["training", "validation", "test"], torch.Tensor]:

        return get_generic_splits_idcs(len(self._dataset), 0.8, 0.1)

    @staticmethod
    def loss_fn(y: torch.Tensor, data: pyg.data.Data, config: DictConfig) -> torch.Tensor:

        loss = mse_loss(y, data.y) if config.training.loss == "mse" else l1_loss(y, data.y)

        return loss

    @staticmethod
    def metric_fn(
        y: torch.Tensor, y_data: torch.Tensor, scatter_idcs: torch.Tensor, config: DictConfig
    ) -> torch.Tensor:

        metric = compute_approximation_error(y, y_data, scatter_idcs)

        return metric


class Interface(ProjectiveGeometricAlgebraInterface):
    in_mv_channels = 1
    out_mv_channels = 1
    in_s_channels = 3
    out_s_channels = None

    @staticmethod
    @torch.no_grad()
    def embed(data: pyg.data.Data) -> Tuple[torch.Tensor, torch.Tensor]:

        mv = embed_point(data.pos).view(-1, 1, 16)

        s = torch.cat(
            (
                data.euclidean_dist_inlet.view(-1, 1),
                data.euclidean_dist_lumen_wall.view(-1, 1),
                data.euclidean_dist_outlets.view(-1, 1),
            ),
            dim=1,
        )

        return mv, s

    @staticmethod
    def extract(mv: torch.Tensor, s: Union[torch.Tensor, None]) -> torch.Tensor:
        return extract_scalar(mv).view(-1, 1) * extract_oriented_plane(mv).squeeze()
