# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear
import h5py
import torch
import torch_geometric as pyg

from .base import HierachicalDataFormatDatasetMemory


class ArterialSurfaceDataset(HierachicalDataFormatDatasetMemory):

    @staticmethod
    def hdf5_to_pyg(data_hdf5: h5py.Group) -> pyg.data.Data:

        data = pyg.data.Data(
            y=torch.from_numpy(data_hdf5["wss"][()]) * 1e-1,  # [dyn/cm^2] to [Pa]
            pos=torch.from_numpy(data_hdf5["pos"][()]),
            face=torch.from_numpy(data_hdf5["face"][()].T),
            geodesic_dist_inlet=torch.from_numpy(data_hdf5["geodesic_distances_to_inlet"][()]),
        )

        return data
