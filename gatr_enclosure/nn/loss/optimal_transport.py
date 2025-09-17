# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear
from typing import Any, Optional

import geomloss
import torch


class OptimalTransportLoss:
    def __init__(self, **kwargs: Any):
        self.geomloss = geomloss.SamplesLoss(**kwargs)

    def __call__(
        self,
        pos_source: torch.Tensor,
        pos_target: torch.Tensor,
        batch_source: Optional[torch.Tensor] = None,
        batch_target: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:

        if batch_source is not None and batch_source.unique().numel() > 1:
            raise NotImplementedError("Batch optimal transport not yet implemented.")

        return self.geomloss(pos_source, pos_target)
