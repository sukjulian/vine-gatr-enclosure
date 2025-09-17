# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear
import torch
import torch_geometric as pyg

from .utils import disambiguated_svd, product


class PointCloudSingularValueDecomposition(torch.nn.Module):
    def __init__(self, use_orientation: bool = False, subset_index_key: str = None):
        
        super().__init__()

        self.use_orientation = use_orientation
        self.subset_index_key = subset_index_key

        ambiguities = torch.tensor(
            list(product((1, -1), repeat=7 if self.use_orientation else 3))
        )
        self.register_buffer('ambiguities', ambiguities)

    def __call__(self, data: pyg.data.Data) -> pyg.data.Data:

        point_cloud = data.pos

        if self.subset_index_key is not None:
            point_cloud = point_cloud[data[self.subset_index_key]]

        origin = point_cloud.mean(dim=0)

        if self.use_orientation:

            # subset_index = data[self.subset_index_key] if self.subset_index_key else None

            point_cloud = torch.cat(
                # (
                #     (point_cloud, data.surf_normal[subset_index], data.geodesic_dist_inlet.view(-1, 1)[subset_index]) 
                #     if self.subset_index_key else
                (point_cloud, data.surf_normal, data.geodesic_dist_inlet.view(-1, 1)),
                # ),
                dim=1
            )
            origin = torch.nn.functional.pad(origin, (0, 4))

        # Invariance under translation
        singular_values, right_singular_vectors = disambiguated_svd(point_cloud - origin, self.ambiguities)

        # Batch dimension
        data.origin = origin[None, ...]
        data.singular_values = singular_values[None, ...]
        data.right_singular_vectors = right_singular_vectors[None, ...]

        return data

    def __repr__(self) -> str:
        if self.subset_index_key is not None:
            return f"{self.__class__.__name__}(use_orientation={self.use_orientation})(subset_index_key={self.subset_index_key})"
        else:
            return f"{self.__class__.__name__}(use_orientation={self.use_orientation})"
