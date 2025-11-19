# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear
import os
from typing import Dict, Literal, Tuple, Union

import torch
import torch_geometric as pyg
from gatr.interface import embed_oriented_plane, embed_point
from omegaconf import DictConfig

from gatr_enclosure.datasets import Mindboggle101Dataset
from gatr_enclosure.models import ProjectiveGeometricAlgebraInterface, ViNEGATr
from gatr_enclosure.nn.loss import (  # OptimalTransportLoss,
    AttractiveLoss,
    ManifoldRepulsiveLoss,
    RepulsiveLoss,
)
from gatr_enclosure.transforms import (
    Curvature,
    LaplacianEigenvectors,
    PointCloudSingularValueDecomposition,
)
from gatr_enclosure.transforms.functional import sulcal_depth, surface_normal

from .base import BaseExperiment


class VirtualNodeEmbedMindboggle101Experiment(BaseExperiment):

    def get_model(self, config: DictConfig) -> torch.nn.Module:
        return ViNEGATr(Interface(), encoder_only=True, **config.model)

    def get_dataset(self, config: DictConfig) -> pyg.data.Dataset:
        dir_ = os.path.join("datasets", "mindboggle-101")

        pre_transform = pyg.transforms.Compose(
            (
                surface_normal,
                sulcal_depth,
                Curvature(num_rings=3),
                PointCloudSingularValueDecomposition(),
                LaplacianEigenvectors(num=64),
            )
        )

        return Mindboggle101Dataset(dir_, pre_transform=pre_transform)

    def get_dataset_splits_idcs(
        self, config: DictConfig
    ) -> Dict[Literal["training", "validation", "test"], torch.Tensor]:
        idcs = torch.arange(len(self._dataset))

        return {"training": idcs, "validation": idcs, "test": idcs}

    @staticmethod
    def loss_fn(
        y: torch.Tensor, data: pyg.data.Data, config: DictConfig
    ) -> torch.Tensor:
        factors = config.training.loss_term_factors

        pos = data.pos
        batch = data.batch

        virtual_nodes_pos = data.virtual_nodes_pos
        virtual_nodes_batch = (
            data.virtual_nodes_batch if "virtual_nodes_batch" in data else None
        )

        match config.training.loss:

            case "attractive_repulsive":
                repulsive_term = RepulsiveLoss(radius_repulsion=1.0)(
                    virtual_nodes_pos, batch=virtual_nodes_batch
                )

            case "attractive_manifold_repulsive":
                repulsive_term = ManifoldRepulsiveLoss()(
                    virtual_nodes_pos,
                    pos,
                    data.eigenvectors,
                    data.eigenvalues,
                    virtual_nodes_batch,
                    batch,
                )

            # case "optimal_transport":
            #     loss = OptimalTransportLoss()(
            #         virtual_nodes_pos, data.pos, virtual_nodes_batch, batch
            #     )

        if "attractive" in config.training.loss and "repulsive_term" in locals():
            attractive_term = AttractiveLoss()(
                pos, virtual_nodes_pos, batch, virtual_nodes_batch
            )

            loss = (
                factors.attractive * attractive_term
                + factors.repulsive * repulsive_term
            )

        return loss

    @staticmethod
    def metric_fn(
        y: torch.Tensor,
        y_data: torch.Tensor,
        scatter_idcs: torch.Tensor,
        config: DictConfig,
    ) -> torch.Tensor:

        dummy = torch.full((scatter_idcs.unique().numel(),), 4.20)

        return dummy

    @staticmethod
    def get_custom_pos_visualisation(data: pyg.data.Data) -> torch.Tensor:
        return data.virtual_nodes_pos


class Interface(ProjectiveGeometricAlgebraInterface):
    in_mv_channels = 2
    in_s_channels = 3

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

        dummy = torch.zeros(mv.size(0))

        return dummy
