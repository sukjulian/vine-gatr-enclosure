# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear
from functools import partial
from typing import Callable

from gatr_enclosure.models import GATr


def test_gatr(assert_o3_equivariance: Callable) -> None:

    model = partial(GATr, hidden_mv_channels=4, num_heads=4, num_blocks=4)
    assert_o3_equivariance(model)
