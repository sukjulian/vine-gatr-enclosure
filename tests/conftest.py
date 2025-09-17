# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear
"""Provides common fixtures used across all unit and integration tests."""

from typing import Callable, Optional, Tuple, Union

import pytest
import torch
from click.testing import CliRunner
from e3nn import o3
from gatr.interface import embed_oriented_plane, extract_oriented_plane
from torch_geometric.data import Data
from torch_geometric.transforms import Compose

from gatr_enclosure.models import ProjectiveGeometricAlgebraInterface


@pytest.fixture(scope="function")
def click_runner() -> CliRunner:
    """Provides a click Runner to test click-enhanced functions."""
    return CliRunner()


@pytest.fixture
def assert_o3_equivariance(
    pga_inputs: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, ProjectiveGeometricAlgebraInterface]
) -> Callable:
    pos, orientation, invariant, pga_interface = pga_inputs

    def fun(model: Callable, transform: Optional[Callable] = None) -> None:
        model = model(pga_interface)

        if transform is not None:
            model = Compose((transform, model))

        R = o3.rand_matrix()

        # O(3)-equivariance: f(Rx) = Rf(x)
        f_Rx = model(Data(pos=pos @ R.T, orientation=orientation @ R.T, invariant=invariant))
        Rf_x = model(Data(pos=pos, orientation=orientation, invariant=invariant)) @ R.T

        assert torch.allclose(f_Rx, Rf_x, atol=3e-4), "O(3)-equivariance seems to be broken."

    return fun


@pytest.fixture
def pga_inputs() -> (
    Tuple[torch.Tensor, torch.Tensor, torch.Tensor, ProjectiveGeometricAlgebraInterface]
):
    num_pos = int(1e3)

    pos = torch.randn((num_pos, 3))
    orientation = torch.randn((num_pos, 3))
    invariant = torch.randn(num_pos)

    class Interface(ProjectiveGeometricAlgebraInterface):
        in_mv_channels = out_mv_channels = in_s_channels = 1
        out_s_channels = None

        @staticmethod
        def embed(data: Data) -> Tuple[torch.Tensor, torch.Tensor]:

            mv = embed_oriented_plane(data.orientation, data.pos).view(-1, 1, 16)
            s = data.invariant.view(-1, 1)

            return mv, s

        @staticmethod
        def extract(mv: torch.Tensor, s: Union[torch.Tensor, None]) -> torch.Tensor:
            return extract_oriented_plane(mv).squeeze()

    return pos, orientation, invariant, Interface()
