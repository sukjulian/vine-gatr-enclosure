# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear
from typing import Any, Optional

import lab_gatr
from lab_gatr import PointCloudPoolingScales
import torch

from .utils import ProjectiveGeometricAlgebraInterface, Stopwatch


def link_class_attr(object_: object, attr_name: str, attr_name_link: str) -> None:
    setattr(object_, attr_name_link, getattr(object_, attr_name))


class LaBGATr(torch.nn.Module):

    def __init__(
        self,
        pga_interface: ProjectiveGeometricAlgebraInterface,
        hidden_mv_channels: int,
        num_heads: int,
        num_blocks: int,
        decoder_id_query_idcs: Optional[str] = None,
        dropout_prob: Optional[float] = None,
        online_fps: bool = False,
        **kwargs: Any,
    ):
        super().__init__()

        pga_interface = self._convert_pga_interface(pga_interface)

        self.backend = lab_gatr.LaBGATr(
            geometric_algebra_interface=pga_interface,
            d_model=hidden_mv_channels,
            num_blocks=num_blocks,
            num_attn_heads=num_heads,
            dropout_probability=dropout_prob,
            pooling_mode="cross_attention",
        )

        self._decoder_id_query_idcs = decoder_id_query_idcs

        names_modules = ["encoding", "backend", "decoding"]
        if online_fps:
            self.fps = PointCloudPoolingScales(
                rel_sampling_ratios=(kwargs['compression'],), interp_simplex="triangle"
            )
            names_modules += ["preprocessing"]
        else:
            self.fps = None

        self.stopwatch = Stopwatch(names_splits=names_modules)

        # self.backend.tokeniser.register_forward_hook(self._build_outshapes_hook('tokenizer output'))
        
        self.backend.tokeniser.register_forward_hook(self._build_stopwatch_hook('encoding'))
        self.backend.gatr.register_forward_hook(self._build_stopwatch_hook('backend'))


    @staticmethod
    def _convert_pga_interface(
        pga_interface: ProjectiveGeometricAlgebraInterface,
    ) -> ProjectiveGeometricAlgebraInterface:

        link_class_attr(pga_interface, "in_mv_channels", "num_input_channels")
        link_class_attr(pga_interface, "out_mv_channels", "num_output_channels")
        link_class_attr(pga_interface, "in_s_channels", "num_input_scalars")
        link_class_attr(pga_interface, "out_s_channels", "num_output_scalars")

        link_class_attr(pga_interface, "extract", "dislodge")

        return pga_interface

    def forward(
        self, data: lab_gatr.data.Data, batch: Optional[torch.Tensor] = None
    ) -> torch.Tensor:

        if self.training is False:
            self.stopwatch.reset()
        
        if self.fps is not None:
            data = self.fps(data)
            if self.training is False:
                self.stopwatch.time_split("preprocessing")

        y = self.backend(data)

        if self._decoder_id_query_idcs is not None:
            y = y[data[f"{self._decoder_id_query_idcs}_index"]]  # workaround: mask output

        if self.training is False:
            self.stopwatch.time_split("decoding")

        return y

    def _build_stopwatch_hook(self, name:str):

        def hook(*args, **kwargs):
            if self.training is False:
                self.stopwatch.time_split(name)

        return hook
    
    def _build_outshapes_hook(self, name:str):
        def hook(model, _input, _output):
            if not isinstance(_output, tuple) and not isinstance(_output, list):
                _output = (_output,)
            s = f'{name}: '
            for o in _output:
                if isinstance(o, torch.Tensor):
                    s += f'{o.shape} '
            print(s)
        return hook