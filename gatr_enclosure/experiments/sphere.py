# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear
from typing import Dict, Literal, Tuple, Union

import torch
import torch_geometric as pyg
from gatr.interface import embed_point, extract_scalar
from omegaconf import DictConfig

from gatr_enclosure.datasets import SphereDataset
from gatr_enclosure.models import GATr, ProjectiveGeometricAlgebraInterface

from .base import BaseExperiment
from .utils import compute_mean_absolute_error


class SphereExperiment(BaseExperiment):

    def get_model(self, config: DictConfig) -> torch.nn.Module:
        return GATr(Interface(), **config.model)

    def get_dataset(self, config: DictConfig) -> pyg.data.Dataset:
        return SphereDataset(num_point_sources=3, variance_gaussians=config.variance_gaussians)

    def get_dataset_splits_idcs(
        self, config: DictConfig
    ) -> Dict[Literal["training", "validation", "test"], torch.Tensor]:
        idcs = torch.arange(len(self._dataset))  # whole dataset

        return {"training": idcs, "validation": idcs, "test": idcs}

    @staticmethod
    def loss_fn(y: torch.Tensor, data: pyg.data.Data, config: DictConfig) -> torch.Tensor:

        loss = torch.nn.functional.l1_loss(y, data.y)

        return loss

    @staticmethod
    def metric_fn(
        y: torch.Tensor, y_data: torch.Tensor, scatter_idcs: torch.Tensor, config: DictConfig
    ) -> torch.Tensor:

        metric = compute_mean_absolute_error(y, y_data, scatter_idcs)

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
        s = data.point_source_masks

        return mv, s

    @staticmethod
    def extract(mv: torch.Tensor, s: Union[torch.Tensor, None]) -> torch.Tensor:
        return extract_scalar(mv).squeeze()
