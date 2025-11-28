# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear
from typing import Optional

import lab_gatr
import torch
import torch_geometric as pyg
from torch_cluster import knn


class Subsampling:
    def __init__(
        self,
        num_samples: int,
        in_place: bool = True,
        num_nearest_neighbours_interpolation: Optional[int] = None,
    ):
        self.num_samples = num_samples
        self.in_place = in_place
        self.num_nearest_neighbours_interpolation = num_nearest_neighbours_interpolation

    def __call__(self, data: pyg.data.Data) -> pyg.data.Data:
        num_pos = data.pos.size(0)

        idcs = torch.randperm(num_pos)[: self.num_samples]

        if self.in_place is True:
            for key, value in data.items():

                if "_pool_" not in key and value.size(0) == num_pos:
                    data[key] = value[idcs]

                elif value.size(0) == 1:
                    pass

                else:
                    delattr(data, key)

        else:
            data.scale0_sampling_index = idcs.int()

            if self.num_nearest_neighbours_interpolation is not None:

                idcs_target, idcs_source = knn(
                    data.pos[idcs],
                    data.pos,
                    k=self.num_nearest_neighbours_interpolation,
                )

                data.scale0_interp_source = idcs_source.int()
                data.scale0_interp_target = idcs_target.int()

                data = lab_gatr.data.Data(**data)

        return data

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(num_samples={self.num_samples}, in_place={self.in_place}, num_nearest_neighbours_interpolation={self.num_nearest_neighbours_interpolation})"
