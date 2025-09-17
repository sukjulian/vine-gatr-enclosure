# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear
import torch_geometric as pyg

from .utils import compute_laplacian_eigenvectors
from .utils import compute_pointcloud_laplacian_eigenvectors


class LaplacianEigenvectors:
    def __init__(self, num: int, sigma: float = None, subset_index_key: str = None):
        self.num = num
        self.sigma = sigma
        self.subset_index_key = subset_index_key

    def __call__(self, data: pyg.data.Data) -> pyg.data.Data:

        pos = data.pos
        if self.subset_index_key is not None:
            pos = pos[data[self.subset_index_key]]

        if hasattr(data, 'face') and data.face is not None:
            data.mass, eigenvalues, data.eigenvectors = compute_laplacian_eigenvectors(
                pos, data.face, self.num
            ) 
        else:
            data.mass, eigenvalues, data.eigenvectors = compute_pointcloud_laplacian_eigenvectors(
                pos, self.num, sigma=self.sigma,
                device='cpu'
            )

        data.eigenvalues = eigenvalues.view(1, -1)

        return data

    def __repr__(self) -> str:

        repr = f"{self.__class__.__name__}(num={self.num})"

        if self.sigma is not None and self.sigma != 0.06:
            repr += f"(sigma={self.sigma})"

        if self.subset_index_key is not None:
            repr += f"(subset_index_key={self.subset_index_key})"
        
        return repr

