# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear
import torch
import torch_geometric as pyg

from .utils import compute_gaussian_curvature


class Curvature:
    def __init__(self, num_rings: int):
        self.num_rings = num_rings

    def __call__(self, data: pyg.data.Data) -> pyg.data.Data:

        average_edge_len = self._compute_average_edge_len(data.pos, data.face)
        data.curvature = compute_gaussian_curvature(
            data.pos, data.face, radius=self.num_rings * average_edge_len
        )

        return data

    @staticmethod
    def _compute_average_edge_len(pos: torch.Tensor, face: torch.Tensor) -> float:

        edge = torch.stack(
            (pos[face[1]] - pos[face[0]], pos[face[2]] - pos[face[0]], pos[face[2]] - pos[face[1]])
        )

        return edge.norm(dim=2).mean().item()

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(num_rings={self.num_rings})"
