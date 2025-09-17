# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear
from typing import Dict, Literal, Tuple, Union, cast

import torch
import torch_geometric as pyg
from gatr.interface import embed_oriented_plane, embed_point, extract_oriented_plane
from omegaconf import DictConfig

from gatr_enclosure.datasets import RandomDummyDataset
from gatr_enclosure.models import GATr, ProjectiveGeometricAlgebraInterface, ViNEGATr
from gatr_enclosure.transforms import PointCloudSingularValueDecomposition

from .base import BaseExperiment
from .utils import compute_approximation_error


class RandomDummyExperiment(BaseExperiment):

    def get_model(self, config: DictConfig) -> torch.nn.Module:

        match config.model.id:
            case "gatr":
                model = cast(torch.nn.Module, GATr)
            case "vine_gatr":
                model = cast(torch.nn.Module, ViNEGATr)

        return model(Interface(), **config.model)

    def get_dataset(self, config: DictConfig) -> pyg.data.Dataset:

        transform = (
            PointCloudSingularValueDecomposition() if config.model.id == "vine_gatr" else None
        )

        return RandomDummyDataset(num_pos=config.num_pos, transform=transform)

    def get_dataset_splits_idcs(
        self, config: DictConfig
    ) -> Dict[Literal["training", "validation", "test"], torch.Tensor]:
        idcs = torch.arange(len(self._dataset))  # whole dataset

        return {"training": idcs, "validation": idcs, "test": idcs}

    @staticmethod
    def loss_fn(y: torch.Tensor, data: pyg.data.Data, config: DictConfig) -> torch.Tensor:
        factor = config.training.loss_term_factors.generic

        loss = factor * torch.nn.functional.l1_loss(y, data.y)

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
                embed_oriented_plane(data.orientation, data.pos).view(-1, 1, 16),
            ),
            dim=1,
        )

        s = data.colour.view(-1, 1)

        return mv, s

    @staticmethod
    def extract(mv: torch.Tensor, s: Union[torch.Tensor, None]) -> torch.Tensor:
        return extract_oriented_plane(mv).squeeze()
