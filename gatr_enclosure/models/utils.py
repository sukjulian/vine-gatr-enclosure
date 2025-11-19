# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear
from abc import ABC, abstractmethod
from inspect import signature
from statistics import mean
from time import time
from typing import Dict, Iterable, Iterator, List, Optional, Tuple, Union, cast

import torch
import torch_geometric as pyg
from torch_scatter import scatter
from typing_extensions import Self
from xformers.ops.fmha import BlockDiagonalMask


def construct_join_reference(
    mv: torch.Tensor, batch: Optional[torch.Tensor] = None, expand_batch: bool = True
) -> torch.Tensor:

    if batch is None:
        join_reference = mv.mean(dim=(0, 1), keepdim=True)
        batch = torch.zeros(mv.size(0), dtype=torch.int, device=mv.device)

    else:
        join_reference = scatter(mv, batch, dim=0, reduce="mean").mean(
            dim=1, keepdim=True
        )

    return join_reference[batch] if expand_batch is True else join_reference


def get_attention_mask(
    batch_target: Union[None, torch.Tensor], batch_source: Optional[torch.Tensor] = None
) -> Union[None, BlockDiagonalMask]:

    if batch_target is None:
        attention_mask = None

    else:
        attention_mask = BlockDiagonalMask.from_seqlens(
            q_seqlen=torch.bincount(batch_target).tolist(),
            kv_seqlen=(
                torch.bincount(batch_source).tolist()
                if batch_source is not None
                else None
            ),
        )

    return attention_mask


def get_decoder_query(
    decoder_query_idcs: torch.Tensor,
    mv: torch.Tensor,
    s: torch.Tensor,
    batch: Union[None, torch.Tensor],
    join_reference: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, Union[None, torch.Tensor], torch.Tensor]:

    mv, s = mv[decoder_query_idcs], s[decoder_query_idcs]

    if batch is not None:
        batch = batch[decoder_query_idcs]

    join_reference = join_reference[decoder_query_idcs]

    return mv, s, batch, join_reference


class ProjectiveGeometricAlgebraInterface(ABC):
    in_mv_channels: int
    out_mv_channels: int
    in_s_channels: int
    out_s_channels: Optional[int] = None

    @staticmethod
    @abstractmethod
    def embed(data: pyg.data.data) -> Tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def extract(mv: torch.Tensor, s: Union[torch.Tensor, None]) -> torch.Tensor:
        raise NotImplementedError

    def __call__(self) -> Self:
        return self


class GATrSequential(torch.nn.Module):
    def __init__(self, *modules: torch.nn.Module):
        super().__init__()

        for i, module in enumerate(modules):
            self.add_module(str(i), module)

    def __iter__(self) -> Iterator[Union[torch.nn.Module, None]]:
        return iter(self._modules.values())

    def forward(
        self, mv: torch.Tensor, s: torch.Tensor, join_reference: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        for module in self:
            module = cast(torch.nn.Module, module)

            if "reference_mv" in signature(module.forward).parameters:
                mv, s = module(mv, scalars=s, reference_mv=join_reference)

            else:
                mv, s = module(mv, s)

        return mv, s


class Stopwatch:
    def __init__(self, names_splits: Iterable[str]):
        self.names_splits = tuple(names_splits)
        self.num_splits = len(tuple(names_splits))

        self._time_cache = time()
        self._durations_splits_cache: Dict[str, List[float]] = {
            name: [] for name in self.names_splits
        }

    def restart(self) -> None:
        self._durations_splits_cache = {name: [] for name in self.names_splits}

    def reset(self) -> None:
        self._time_cache = time()

    def time_split(self, name_split: str) -> None:
        self._durations_splits_cache[name_split].append(self._get_elapsed_time())

    def _get_elapsed_time(self) -> float:

        time_ = time()
        elapsed_time = time_ - self._time_cache

        self._time_cache = time_

        return elapsed_time

    @property
    def mean_duration_splits(self) -> Dict[str, float]:
        return {key: mean(value) for key, value in self._durations_splits_cache.items()}
