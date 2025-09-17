# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear
import h5py
import torch
import torch_geometric as pyg

from .base import HierachicalDataFormatDatasetDisk


class ArterialVolumeDataset(HierachicalDataFormatDatasetDisk):

    @staticmethod
    def hdf5_to_pyg(data_hdf5: h5py.Group) -> pyg.data.Data:

        data = pyg.data.Data(
            y=torch.from_numpy(data_hdf5["velocity"][()]),
            pos=torch.from_numpy(data_hdf5["pos_tets"][()]),
            inlet_index=torch.from_numpy(data_hdf5["inlet_idcs"][()]),
            lumen_wall_index=torch.from_numpy(data_hdf5["lumen_wall_idcs"][()]),
            outlets_index=torch.from_numpy(data_hdf5["outlets_idcs"][()]),
        )

        return data
