# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear
import torch
import torch_geometric as pyg


class Subsampling:
    def __init__(self, num_samples: int, in_place: bool = True):
        self.num_samples = num_samples
        self.in_place = in_place

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
            data.scale0_sampling_index = idcs

        return data

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(num_samples={self.num_samples}, in_place={self.in_place})"
        )
