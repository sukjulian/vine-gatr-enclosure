# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear
from functools import partial
from math import sqrt
from typing import Literal, Optional, Tuple

import torch


class LearnedVirtualNodes(torch.nn.Module):
    def __init__(
        self,
        num_virtual_nodes: int,
        num_dim: int = 3,
        init_distribution: Literal["normal", "uniform", "truncated_normal"] = "normal",
        init_distribution_std: float = 1.0,
        broken: bool = False
    ):
        super().__init__()
        self.num_dim = num_dim

        self._num_frames = int(2**self.num_dim / 2) 
        self._per_frame = round(num_virtual_nodes / self._num_frames) if not broken else num_virtual_nodes

        self.linear_combination = torch.nn.Linear(self.num_dim, self._per_frame, bias=False)

        match init_distribution:
            case "normal":
                self._init_fun = partial(torch.nn.init.normal_, std=init_distribution_std)
            case "uniform":
                bound_param = 0.5 * sqrt(init_distribution_std**2 * 12.0)
                self._init_fun = partial(torch.nn.init.uniform_, a=-bound_param, b=bound_param)
            case "truncated_normal":
                bound_param = 0.5 * sqrt(init_distribution_std**2 * 12.0)
                self._init_fun = partial(
                    torch.nn.init.trunc_normal_,
                    std=init_distribution_std,
                    a=-bound_param,
                    b=bound_param,
                )

        self._init_fun(self.linear_combination.weight)

    def forward(
        self,
        singular_values: torch.Tensor,
        right_singular_vectors: torch.Tensor,
        num_pos: torch.Tensor,
        origin: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        This function is ultimately not used in ViNE-GATr, but might be helpful to understand the
        concept of learned virtual nodes.
        """
        
        assert not self.broken

        bases = self.get_bases(singular_values, right_singular_vectors, num_pos)

        bases = bases.transpose(2, 3)  # (batch_size, num_frames, num_dim, num_dim)
        virtual_nodes = self.linear_combination(
            bases
        )  # (batch_size, num_frames, num_dim, per_frame)
        virtual_nodes = virtual_nodes.transpose(
            2, 3
        )  # (batch_size, num_frames, per_frame, num_dim)

        batch_size, num_frames, num_dim = self._get_shape(right_singular_vectors)
        virtual_nodes = virtual_nodes.reshape(batch_size, num_frames * self._per_frame, num_dim)

        if origin is not None:
            virtual_nodes += origin.view(batch_size, 1, num_dim)

        return virtual_nodes

    def get_bases(
        self,
        singular_values: torch.Tensor,
        right_singular_vectors: torch.Tensor,
        num_pos: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, _, num_dim = self._get_shape(right_singular_vectors)

        bases = (
            singular_values.view(batch_size, 1, num_dim, 1)
            / (num_pos - 1).sqrt().view(batch_size, 1, 1, 1)
            * right_singular_vectors
        )

        return bases

    def _get_shape(self, right_singular_vectors: torch.Tensor) -> Tuple[int, int, int]:

        batch_size, num_frames, _, num_dim = right_singular_vectors.shape
        assert num_frames == self._num_frames and num_dim == self.num_dim, "Dimensions mismatch."

        return batch_size, num_frames, num_dim
