# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear
from typing import Any, Optional, Tuple, Union

import torch
from lab_gatr.models.lab_gatr import CrossAttentionHatchling

from .lab_gatr import LaBGATr
from .utils import (
    ProjectiveGeometricAlgebraInterface,
    get_attention_mask,
    get_decoder_query,
)


class RNGGATr(LaBGATr):

    def __init__(
        self,
        pga_interface: ProjectiveGeometricAlgebraInterface,
        hidden_mv_channels: int,
        num_heads: int,
        num_blocks: int,
        decoder_id_query_idcs: Optional[str] = None,
        dropout_prob: Optional[float] = None,
        **kwargs: Any,
    ):
        super().__init__(
            pga_interface,
            hidden_mv_channels,
            num_heads,
            num_blocks,
            decoder_id_query_idcs=None,
            dropout_prob=dropout_prob,
            **kwargs,
        )

        self.decoder = CrossAttentionHatchling(
            num_input_channels_source=hidden_mv_channels,
            num_input_channels_target=pga_interface.in_mv_channels,
            num_output_channels=pga_interface.out_mv_channels,
            num_input_scalars_source=pga_interface.out_s_channels,
            num_input_scalars_target=pga_interface.in_s_channels,
            num_output_scalars=1,  # soothe the 🐊
            num_attn_heads=num_heads,
            num_latent_channels=hidden_mv_channels,
            dropout_probability=dropout_prob,
        )
        self.decoder_id_query_idcs = decoder_id_query_idcs
        self.pga_interface = self.backend.tokeniser.geometric_algebra_interface

        self.backend.tokeniser.lift = self._tokeniser_lift
        delattr(self.backend.tokeniser, "mlp")

        self.num_param = sum(param.numel() for param in self.parameters() if param.requires_grad)
        print(f"...syke! It's actually RNG-GATr ({self.num_param} parameters)")

    def _tokeniser_lift(
        self, mv: torch.Tensor, s: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        mv_target, s_target, batch_source, batch_target, join_reference, decoder_query_idcs = (
            self._parse_tokeniser_cache()
        )

        if decoder_query_idcs is not None:
            mv_target, s_target, batch_target, join_reference = get_decoder_query(
                decoder_query_idcs,
                mv_target,
                s_target,
                batch_target,
                join_reference,
            )

        attention_mask = get_attention_mask(batch_target, batch_source)
        mv, s = self.decoder(mv, mv_target, s, s_target, attention_mask, join_reference)

        return self.pga_interface.extract(mv, s)

    def _parse_tokeniser_cache(
        self,
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        Union[None, torch.Tensor],
        Union[None, torch.Tensor],
        torch.Tensor,
        Union[None, torch.Tensor],
    ]:
        tokeniser_cache = self.backend.tokeniser.cache
        data = tokeniser_cache["data"]

        mv_target, s_target = tokeniser_cache["multivectors"], tokeniser_cache["scalars"]

        batch_source = None if data.batch is None else data.batch[data.scale0_sampling_index]
        batch_target = data.batch

        join_reference = tokeniser_cache["reference_multivector"]

        decoder_query_idcs = (
            None
            if self.decoder_id_query_idcs is None
            else data[f"{self.decoder_id_query_idcs}_index"]
        )

        return mv_target, s_target, batch_source, batch_target, join_reference, decoder_query_idcs
