# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear
import os
from typing import Dict, Literal, Tuple, Union, cast

import numpy as np
import torch
import torch_geometric as pyg
from gatr.interface import embed_oriented_plane, embed_point, extract_scalar
from lab_gatr import PointCloudPoolingScales
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
from .utils import compute_mean_squared_error, get_identified_splits_idcs

VALIDATION_SPLIT_IDCS = (
    57,
    414,
    787,
    569,
)
TEST_SPLIT_IDCS = (
    550,
    592,
    229,
    547,
    62,
    464,
    798,
    836,
    5,
    732,
    876,
    843,
    367,
    496,
    142,
    87,
    88,
    101,
    303,
    352,
    517,
    8,
    462,
    123,
    348,
    714,
    384,
    190,
    505,
    349,
    174,
    805,
    156,
    417,
    764,
    788,
    645,
    108,
    829,
    227,
    555,
    412,
    854,
    21,
    55,
    210,
    188,
    274,
    646,
    320,
    4,
    344,
    525,
    118,
    385,
    669,
    113,
    387,
    222,
    786,
    515,
    407,
    14,
    821,
    239,
    773,
    474,
    725,
    620,
    401,
    546,
    512,
    837,
    353,
    537,
    770,
    41,
    81,
    664,
    699,
    373,
    632,
    411,
    212,
    678,
    528,
    120,
    644,
    500,
    767,
    790,
    16,
    316,
    259,
    134,
    531,
    479,
    356,
    641,
    98,
    294,
    96,
    318,
    808,
    663,
    447,
    445,
    758,
    656,
    177,
    734,
    623,
    216,
    189,
    133,
    427,
    745,
    72,
    257,
    73,
    341,
    584,
    346,
    840,
    182,
    333,
    218,
    602,
    99,
    140,
    809,
    878,
    658,
    779,
    65,
    708,
    84,
    653,
    542,
    111,
    129,
    676,
    163,
    203,
    250,
    209,
    11,
    508,
    671,
    628,
    112,
    317,
    114,
    15,
    723,
    746,
    765,
    720,
    828,
    662,
    665,
    399,
    162,
    495,
    135,
    121,
    181,
    615,
    518,
    749,
    155,
    363,
    195,
    551,
    650,
    877,
    116,
    38,
    338,
    849,
    334,
    109,
    580,
    523,
    631,
    713,
    607,
    651,
    168,
)


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

        pre_transforms = [] if upt else [idcs_euclidean_dist]
        if config.model.id == "vine_gatr":
            pre_transforms.append(
                PointCloudSingularValueDecomposition()
            )  # subset_index_key=subset_index_key))
            pre_transforms.append(surface_normal)
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

        if config.model.id == "rng_gatr":
            transform = Subsampling(
                num_samples=config.model.num_virtual_nodes, in_place=False
            )
        # elif config.model.id == "vine_gatr_reg" and subset_index_key is not None:
        # transform = PointCloudSingularValueDecomposition(subset_index_key=subset_index_key)
        # elif config.model.id == "lab_gatr":
        #     transform = PointCloudPoolingScales(
        #             rel_sampling_ratios=(config.model.compression,), interp_simplex="triangle"
        #         )
        else:
            transform = None

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

        id_array = np.array(self._dataset.id_list)

        N = id_array.shape[0]
        N_test = len(TEST_SPLIT_IDCS)

        if not hasattr(config.training, "split_valid") or config.training.split_valid:
            N_val = int(0.12 * (N - N_test))
            rng = np.random.RandomState(42)
            idxs = np.arange(N)
            mask = (
                idxs.reshape(-1, 1) == np.array(TEST_SPLIT_IDCS).reshape(1, -1)
            ).any(axis=1)
            idxs = idxs[~mask]
            rng.shuffle(idxs)
            validation_split_idcs = idxs[:N_val]

            id_validation_list = id_array[np.array(validation_split_idcs)].tolist()

        else:
            N_val = 0
            id_validation_list = []

        # id_validation_list = id_array[np.array(VALIDATION_SPLIT_IDCS)].tolist()
        id_test_list = id_array[np.array(TEST_SPLIT_IDCS)].tolist()

        return get_identified_splits_idcs(
            id_array.tolist(), id_validation_list, id_test_list
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

        metric = compute_mean_squared_error(y, y_data, scatter_idcs)

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

        # s = data.euclidean_dist_dirichlet_boundary.view(-1, 1)
        s = torch.zeros((data.pos.size(0), 1), device=data.pos.device)  # dummy scalars

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
