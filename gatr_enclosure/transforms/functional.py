# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear
import torch_geometric as pyg
from torch_cluster import knn

from .utils import compute_diffusion_dist, compute_sulcal_depth, compute_surface_normal


def idcs_diffusion_dist(data: pyg.data.Data) -> pyg.data.Data:
    idcs = {key: value for key, value in data.items() if "_index" in key and key != "edge_index"}

    iterator = zip(idcs.keys(), compute_diffusion_dist(data.pos, idcs.values()))
    for key, value in iterator:

        data[f"diffusion_dist_{key.replace('_index', '')}"] = value

    return data


def idcs_euclidean_dist(data: pyg.data.Data) -> pyg.data.Data:
    idcs = {key: value for key, value in data.items() if "_index" in key and key != "edge_index"}

    for key, value in idcs.items():
        nearest = value[knn(data.pos[value], data.pos, k=1)[1]]

        data[f"euclidean_dist_{key.replace('_index', '')}"] = (data.pos[nearest] - data.pos).norm(
            dim=1
        )

    return data


def sulcal_depth(data: pyg.data.Data) -> pyg.data.Data:

    data.sulcal_depth = compute_sulcal_depth(data.pos, data.face)

    return data


def surface_normal(data: pyg.data.Data) -> pyg.data.Data:

    data.surf_normal = compute_surface_normal(data.pos, data.face)  # avoid sub-string "face"

    return data
