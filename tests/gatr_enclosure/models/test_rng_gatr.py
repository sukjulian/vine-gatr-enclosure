# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear
from functools import partial
from typing import Callable

import torch

from gatr_enclosure.models import RNGGATr
from gatr_enclosure.transforms import Subsampling


def test_rng_gatr(assert_o3_equivariance: Callable) -> None:
    torch.manual_seed(0)

    model = partial(RNGGATr, hidden_mv_channels=4, num_heads=4, num_blocks=4)
    assert_o3_equivariance(model, Subsampling(num_samples=int(1e2), in_place=False))
