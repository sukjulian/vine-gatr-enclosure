# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear
import torch
from xformers.ops.fmha import BlockDiagonalMask

from gatr_enclosure.models.utils import construct_join_reference, get_attention_mask


def test_construct_join_reference() -> None:

    num_pos = int(1e3)

    mv = torch.randn((num_pos, 1, 16))
    batch = torch.zeros(num_pos, dtype=torch.long)

    join_reference = construct_join_reference(mv, batch)

    assert join_reference.shape == (num_pos, 1, 16), "Join reference is not a multivector."


def test_get_attention_mask() -> None:

    num_pos = int(1e3)

    source_batch = target_batch = torch.zeros(num_pos, dtype=torch.long)

    attention_mask = get_attention_mask(target_batch, source_batch)

    assert isinstance(attention_mask, BlockDiagonalMask), "Improper attention mask."
