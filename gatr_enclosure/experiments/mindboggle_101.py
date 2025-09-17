# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear
import os
from typing import Dict, Literal, Tuple, Union, cast

import torch
import torch_geometric as pyg
from gatr.interface import embed_oriented_plane, embed_point, extract_scalar
from lab_gatr import PointCloudPoolingScales
from omegaconf import DictConfig

from gatr_enclosure.datasets import Mindboggle101Dataset
from gatr_enclosure.models import (
    GATr,
    LaBGATr,
    ProjectiveGeometricAlgebraInterface,
    RNGGATr,
    ViNEGATr,
)
from gatr_enclosure.transforms import (
    Curvature,
    PointCloudSingularValueDecomposition,
    Subsampling,
)
from gatr_enclosure.transforms.functional import sulcal_depth, surface_normal

from .base import BaseExperiment
from .utils import (
    compute_classification_accuracy,
    compute_dice_coefficients,
    get_identified_splits_idcs,
)

VALIDATION_SPLIT = (
    "mmrr-21-2",
    "nki-rs-22-2",
    "nki-trt-20-2",
    "oasis-trt-20-2",
)
TEST_SPLIT = (
    "afterthought-1",
    "mmrr-21-1",
    "mmrr-21-21",
    "nki-rs-22-1",
    "nki-rs-22-22",
    "nki-trt-20-1",
    "nki-trt-20-20",
    "oasis-trt-20-1",
    "oasis-trt-20-20",
)


class Mindboggle101Experiment(BaseExperiment):

    def get_model(self, config: DictConfig) -> torch.nn.Module:

        match config.scaling.strategy:
            case "gatr_and_subsampling":
                model = cast(torch.nn.Module, GATr)
            case "lab_gatr":
                model = cast(torch.nn.Module, LaBGATr)
            case "vine_gatr":
                model = cast(torch.nn.Module, ViNEGATr)
            case "rng_gatr":
                model = cast(torch.nn.Module, RNGGATr)

        return model(Interface(), **config.model)

    def get_dataset(self, config: DictConfig) -> pyg.data.Dataset:
        dir_ = os.path.join("datasets", "mindboggle-101")

        pre_transforms = [surface_normal, sulcal_depth, Curvature(num_rings=3)]
        if config.scaling.strategy == "lab_gatr":
            pre_transforms.append(
                PointCloudPoolingScales(
                    rel_sampling_ratios=(config.scaling.compression,), interp_simplex="triangle"
                )
            )
        elif config.scaling.strategy == "vine_gatr":
            pre_transforms.append(PointCloudSingularValueDecomposition())

        pre_transform = pyg.transforms.Compose(pre_transforms)

        if config.scaling.strategy == "gatr_and_subsampling":
            transform = Subsampling(
                num_samples=config.scaling.num_samples
            )  # also applied at test time
        elif config.scaling.strategy == "rng_gatr":
            transform = Subsampling(num_samples=config.scaling.num_samples, in_place=False)
        else:
            transform = None

        return Mindboggle101Dataset(dir_, pre_transform=pre_transform, transform=transform)

    def get_dataset_splits_idcs(
        self, config: DictConfig
    ) -> Dict[Literal["training", "validation", "test"], torch.Tensor]:
        return get_identified_splits_idcs(self._dataset.id_list, VALIDATION_SPLIT, TEST_SPLIT)

    @staticmethod
    def loss_fn(y: torch.Tensor, data: pyg.data.Data, config: DictConfig) -> torch.Tensor:

        loss = torch.nn.functional.cross_entropy(y, data.y.max(dim=1).indices)

        return loss

    @staticmethod
    def metric_fn(
        y: torch.Tensor, y_data: torch.Tensor, scatter_idcs: torch.Tensor, config: DictConfig
    ) -> torch.Tensor:

        if config.metric == "dice_coefficients":
            metric = compute_dice_coefficients(y, y_data, scatter_idcs)

        else:
            metric = compute_classification_accuracy(y, y_data, scatter_idcs)

        return metric


class Interface(ProjectiveGeometricAlgebraInterface):
    in_mv_channels = 2
    out_mv_channels = 36
    in_s_channels = 3
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

        s = torch.cat(
            (
                data.hemisphere_id.view(-1, 1),
                data.sulcal_depth.view(-1, 1),
                data.curvature.view(-1, 1),
            ),
            dim=1,
        )

        return mv, s

    @staticmethod
    def extract(mv: torch.Tensor, s: Union[torch.Tensor, None]) -> torch.Tensor:
        return extract_scalar(mv).squeeze()
