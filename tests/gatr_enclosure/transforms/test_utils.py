# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear
import pyvista
import torch

from gatr_enclosure.transforms.utils import compute_laplacian_eigenvectors


def test_compute_laplacian_eigenvectors() -> None:

    sphere = pyvista.Sphere()
    pos = torch.from_numpy(sphere.points)
    face = torch.from_numpy(sphere.regular_faces.astype("i4").T)

    num_pos = pos.size(0)
    num_eigenvectors = int(num_pos * 1e-1)

    mass, eigenvalues, eigenvectors = compute_laplacian_eigenvectors(pos, face, num_eigenvectors)

    assert (
        mass.numel() == num_pos
        and eigenvalues.numel() == num_eigenvectors
        and eigenvectors.shape == (num_pos, num_eigenvectors)
    )
