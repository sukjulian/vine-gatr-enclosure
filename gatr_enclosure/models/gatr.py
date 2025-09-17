# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear
from typing import Any, Optional

import gatr
import torch
import torch_geometric as pyg

from .utils import (
    ProjectiveGeometricAlgebraInterface,
    construct_join_reference,
    get_attention_mask,
)


class GATr(torch.nn.Module):

    def __init__(
        self,
        pga_interface: ProjectiveGeometricAlgebraInterface,
        hidden_mv_channels: int,
        num_heads: int,
        num_blocks: int,
        dropout_prob: Optional[float] = None,
        **kwargs: Any,
    ):
        super().__init__()

        self.pga_interface = pga_interface

        self.backend = gatr.GATr(
            pga_interface.in_mv_channels,
            pga_interface.out_mv_channels,
            hidden_mv_channels,
            pga_interface.in_s_channels,
            pga_interface.out_s_channels,
            hidden_s_channels=4 * hidden_mv_channels,
            attention=gatr.SelfAttentionConfig(num_heads=num_heads),
            mlp=gatr.MLPConfig(),
            num_blocks=num_blocks,
            dropout_prob=dropout_prob,
        )

        self.num_param = sum(param.numel() for param in self.parameters() if param.requires_grad)
        print(f"GATr ({self.num_param} parameters)")

    def forward(self, data: pyg.data.Data, batch: Optional[torch.Tensor] = None) -> torch.Tensor:
        mv, s = self.pga_interface.embed(data)

        attention_mask = get_attention_mask(batch)
        join_reference = construct_join_reference(mv, batch)

        mv, s = self.backend(mv, s, attention_mask, join_reference)

        return self.pga_interface.extract(mv, s)
