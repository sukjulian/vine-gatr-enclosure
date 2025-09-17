# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear
from functools import partial
from typing import Optional

import torch
from gatr.interface import embed_point
from gatr.primitives.attention import _lin_square_normalizer, geometric_attention

from gatr_enclosure.models.utils import get_attention_mask


class ManifoldRepulsiveLoss:
    def __init__(self, diffusion_time: float = 1.0):
        self.diffusion_time = diffusion_time

    def __call__(
        self,
        pos: torch.Tensor,
        pos_manifold: torch.Tensor,
        laplacian_eigenvectors: torch.Tensor,
        laplacian_eigenvalues: torch.Tensor,
        batch: Optional[torch.Tensor] = None,
        batch_manifold: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """This does not yet incorporate the mass matrices in the truncated eigenvector bases."""

        query_mv = embed_point(pos).view(-1, 1, 16)
        key_mv = embed_point(pos_manifold).view(-1, 1, 16)
        value_s = laplacian_eigenvectors
        dummy_query_s = torch.zeros(query_mv.size(0), device=query_mv.device).view(-1, 1)
        dummy_key_s = torch.zeros(key_mv.size(0), device=key_mv.device).view(-1, 1)
        dummy_value_mv = torch.zeros_like(key_mv)

        fun_normalisation = partial(_lin_square_normalizer, epsilon=1e-3)
        attention_mask = get_attention_mask(batch, batch_manifold)

        _, eigenvectors_projection = geometric_attention(
            query_mv,
            key_mv,
            dummy_value_mv,
            dummy_query_s,
            dummy_key_s,
            value_s,
            fun_normalisation,
            attn_mask=attention_mask,
        )

        spectral_diffusion = (-(self.diffusion_time * laplacian_eigenvalues)).exp()

        if batch is not None:
            spectral_diffusion = spectral_diffusion[batch]

        heat_kernel = eigenvectors_projection * spectral_diffusion @ eigenvectors_projection.T
        loss = (-1.0) * heat_kernel.triu(diagonal=1).mean()

        return loss
