# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear
from typing import Callable, Optional

import torch
import torch_geometric as pyg
from e3nn import o3


class RandomDummyDataset(pyg.data.Dataset):
    def __init__(self, num_pos: int, transform: Optional[Callable] = None):
        super().__init__(transform=transform)

        self._pos = torch.randn((num_pos, 3))
        self._orientation = torch.randn((num_pos, 3))
        self._colour = torch.randn(num_pos)

        self._y = self._colour.view(-1, 1) * self._orientation @ o3.rand_matrix()

    def len(self) -> int:
        return 1

    def get(self, idx: int) -> pyg.data.Data:

        data = pyg.data.Data(
            y=self._y, pos=self._pos, orientation=self._orientation, colour=self._colour
        )

        return data
