# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear
from typing import Tuple

import potpourri3d as pp3d
import pyvista
import torch
import torch_geometric as pyg


class SphereDataset(pyg.data.Dataset):
    def __init__(self, num_point_sources: int, variance_gaussians: float = 1.0):
        super().__init__()
        self.num_point_sources = num_point_sources
        self.variance_gaussians = variance_gaussians

        self._pos, self._face = self._get_sphere()
        self._solver = pp3d.MeshHeatMethodDistanceSolver(self._pos.numpy(), self._face.T.numpy())

    @staticmethod
    def _get_sphere() -> Tuple[torch.Tensor, torch.Tensor]:
        sphere = pyvista.Sphere()

        pos = torch.from_numpy(sphere.points)
        face = torch.from_numpy(sphere.regular_faces.astype("i4").T)

        return pos, face

    def len(self) -> int:
        return 1

    def get(self, idx: int) -> pyg.data.Data:

        point_source_masks, gaussian_mixture = self._sample_gaussian_mixture()

        data = pyg.data.Data(
            y=gaussian_mixture,
            pos=self._pos,
            face=self._face,
            point_source_masks=point_source_masks,
        )

        return data

    def _sample_gaussian_mixture(self) -> Tuple[torch.Tensor, torch.Tensor]:
        num_pos = self._pos.size(0)

        gaussian_mixture = torch.zeros(num_pos)

        idcs = torch.randint(num_pos, size=(self.num_point_sources,))
        for idx in idcs:

            geodesic_dist_point = torch.from_numpy(self._solver.compute_distance(idx).astype("f4"))
            gaussian_mixture += (-(geodesic_dist_point**2) / (2 * self.variance_gaussians)).exp()

        point_source_masks = torch.nn.functional.one_hot(idcs, num_classes=num_pos).T.float()

        return point_source_masks, gaussian_mixture
