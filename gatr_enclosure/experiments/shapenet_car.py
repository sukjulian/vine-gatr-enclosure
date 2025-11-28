# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear
import os
from typing import Dict, List, Literal, Tuple, Union, cast

import numpy as np
import torch
import torch_geometric as pyg
from gatr.interface import embed_oriented_plane, embed_point, extract_scalar
from lab_gatr import PointCloudPoolingScales
from lab_gatr.nn.positional_encoding import PositionalEncoding
from omegaconf import DictConfig
from torch.nn.functional import l1_loss, mse_loss

from gatr_enclosure.datasets import ShapenetCarDataset, ShapenetCarDatasetUPT
from gatr_enclosure.models import (
    LaBGATr,
    ProjectiveGeometricAlgebraInterface,
    RNGGATr,
    ViNEGATr,
    ViNEGATrWithRegularization,
)
from gatr_enclosure.transforms import (
    LaplacianEigenvectors,
    PointCloudSingularValueDecomposition,
    Subsampling,
)
from gatr_enclosure.transforms.functional import idcs_euclidean_dist, surface_normal

from .base import BaseExperiment
from .utils import compute_approximation_error, get_identified_splits_idcs

# Training split statistics
PRESSURE_AVERAGE = -36.3886
PRESSURE_STD = 48.7215


class ShapenetCarExperiment(BaseExperiment):

    def get_model(self, config: DictConfig) -> torch.nn.Module:

        match config.model.id:
            case "vine_gatr":
                model = cast(torch.nn.Module, ViNEGATr)
            case "vine_gatr_reg":
                model = cast(torch.nn.Module, ViNEGATrWithRegularization)
            case "lab_gatr":
                model = cast(torch.nn.Module, LaBGATr)
            case "rng_gatr":
                model = cast(torch.nn.Module, RNGGATr)

        return model(
            (
                InterfaceImproved()
                if hasattr(config.training, "improved") and config.training.improved
                else Interface()
            ),
            # decoder_id_query_idcs="dirichlet_boundary",
            **config.model,
        )

    def get_dataset(self, config: DictConfig) -> pyg.data.Dataset:
        upt = hasattr(config.training, "upt") and config.training.upt

        if upt:
            dir_ = os.path.join("datasets", "shapenet-car-upt")
        else:
            dir_ = os.path.join("datasets", "shapenet-car")

        # subset_index_key = 'dirichlet_boundary_index' if config.training.improved else None

        def identify_idcs(data: pyg.data.Data) -> pyg.data.Data:

            for coord, id_ in zip(data.pos.T, ("passenger_side", "roof", "bumper")):
                idcs = torch.where(coord <= coord.quantile(0.01))[0]
                data[f"{id_}_index"] = idcs.int()

            return data

        pre_transforms = (
            [] if upt else [surface_normal, identify_idcs, idcs_euclidean_dist]
        )
        if config.model.id == "vine_gatr":
            pre_transforms.append(
                PointCloudSingularValueDecomposition()
            )  # subset_index_key=subset_index_key))
        elif config.model.id == "vine_gatr_reg":
            # pre_transforms.append(PointCloudSingularValueDecomposition(subset_index_key=subset_index_key))
            pre_transforms.append(PointCloudSingularValueDecomposition())
            pre_transforms.append(
                LaplacianEigenvectors(
                    # num=32 if config.model.spectral_n_neigh <= 32 else 256,
                    num=min(
                        [
                            n
                            for n in [32, 64, 128, 256, 512, 1024]
                            if n >= config.model.spectral_n_neigh
                        ]
                    ),
                    sigma=config.model.laplacian_sigma,
                    # subset_index_key=subset_index_key
                )
            )
        elif config.model.id == "lab_gatr":
            pre_transforms.append(
                PointCloudPoolingScales(
                    rel_sampling_ratios=(config.model.compression,),
                    interp_simplex="triangle",
                )
            )

        pre_transform = pyg.transforms.Compose(pre_transforms)

        # print("Pre-transform name:")
        # print(repr(pre_transform))

        def standardise_pressure(data: pyg.data.Data) -> pyg.data.Data:
            data.y = (data.y - PRESSURE_AVERAGE) / PRESSURE_STD

            return data

        transforms = [standardise_pressure]

        if config.model.id == "rng_gatr":

            if (
                hasattr(config.model, "decoder_id_module")
                and config.model.decoder_id_module == "interpolation"
            ):
                num_nearest_neighbours_interpolation = 3
            else:
                num_nearest_neighbours_interpolation = None

            transforms.append(
                Subsampling(
                    num_samples=config.model.num_virtual_nodes,
                    in_place=False,
                    num_nearest_neighbours_interpolation=num_nearest_neighbours_interpolation,
                )
            )

        transform = pyg.transforms.Compose(transforms)

        if upt:
            return ShapenetCarDatasetUPT(
                dir_, pre_transform=pre_transform, transform=transform
            )
        else:
            return ShapenetCarDataset(
                dir_, pre_transform=pre_transform, transform=transform
            )

    def get_dataset_splits_idcs(
        self, config: DictConfig
    ) -> Dict[Literal["training", "validation", "test"], torch.Tensor]:

        # Folder "param0" (first 100 samples) according to Alkin et al. (2025)
        id_test_list: List[str] = np.arange(100).astype(str).tolist()

        id_validation_list: List[str] = np.arange(100, 170).astype(str).tolist()

        return get_identified_splits_idcs(
            self._dataset.id_list, id_validation_list, id_test_list
        )

    @staticmethod
    def loss_fn(
        y: torch.Tensor, data: pyg.data.Data, config: DictConfig
    ) -> torch.Tensor:

        loss = (
            mse_loss(y, data.y) if config.training.loss == "mse" else l1_loss(y, data.y)
        )

        return loss

    @staticmethod
    def metric_fn(
        y: torch.Tensor,
        y_data: torch.Tensor,
        scatter_idcs: torch.Tensor,
        config: DictConfig,
    ) -> torch.Tensor:

        y = y * PRESSURE_STD + PRESSURE_AVERAGE
        y_data = y_data * PRESSURE_STD + PRESSURE_AVERAGE

        assert (
            y.dim() == y_data.dim() == 1
        ), "Relative mean squared error not implemented."
        y = y.unsqueeze(1)
        y_data = y_data.unsqueeze(1)

        metric = compute_approximation_error(y, y_data, scatter_idcs)

        return metric

    # # TODO make sure this attr exists otherwise handle the exception
    @staticmethod
    def get_custom_pos_visualisation(data: pyg.data.Data) -> torch.Tensor:
        if not hasattr(data, "virtual_nodes_pos"):
            return []

        to_return = []

        to_return += [data.virtual_nodes_pos]

        if hasattr(data, "frame_id"):
            to_return += [data.frame_id]

        return to_return


class Interface(ProjectiveGeometricAlgebraInterface):
    in_mv_channels = 2
    out_mv_channels = 1
    in_s_channels = 96
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

        # s = data.euclidean_dist_dirichlet_boundary.view(-1, 1)
        s = torch.cat(
            (
                data.euclidean_dist_passenger_side.view(-1, 1),
                data.euclidean_dist_roof.view(-1, 1),
                data.euclidean_dist_bumper.view(-1, 1),
            ),
            dim=1,
        )

        # Positional embedding
        s = PositionalEncoding(num_channels=96)(s)

        return mv, s

    @staticmethod
    def extract(mv: torch.Tensor, s: Union[torch.Tensor, None]) -> torch.Tensor:
        return extract_scalar(mv).squeeze()


class InterfaceImproved(ProjectiveGeometricAlgebraInterface):
    in_mv_channels = 4
    out_mv_channels = 1
    in_s_channels = 1
    out_s_channels = None

    @staticmethod
    @torch.no_grad()
    def embed(data: pyg.data.Data) -> Tuple[torch.Tensor, torch.Tensor]:

        idx = data.dirichlet_boundary_index

        # a bit hacky, but it's easier if we do it here
        data.spectral_subset_index = idx

        N = data.pos.shape[0]

        mv = torch.cat(
            (
                embed_point(data.pos).view(-1, 1, 16),
                embed_oriented_plane(data.field, data.pos).view(-1, 1, 16),
            ),
            dim=1,
        )

        assert mv.shape == (N, 2, 16)

        onehot = torch.zeros((N, 2), dtype=mv.dtype, device=mv.device)
        # onehot[:, 0] = 0
        onehot[:, 1] = 1
        onehot[idx, 1] = 0
        onehot[idx, 0] = 1

        mv = (mv.view(N, 1, 2, 16) * onehot.view(N, 2, 1, 1)).reshape(N, 4, 16)

        if hasattr(data, "sdf"):
            s = data.sdf.view(-1, 1)
        else:
            s = data.euclidean_dist_dirichlet_boundary.view(-1, 1)

        return mv, s

    @staticmethod
    def extract(mv: torch.Tensor, s: Union[torch.Tensor, None]) -> torch.Tensor:
        return extract_scalar(mv).squeeze()
