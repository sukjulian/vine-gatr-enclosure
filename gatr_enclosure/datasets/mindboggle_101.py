# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear
import h5py
import torch
import torch_geometric as pyg

from .base import HierachicalDataFormatDatasetMemory


class Mindboggle101Dataset(HierachicalDataFormatDatasetMemory):

    @staticmethod
    def hdf5_to_pyg(data_hdf5: h5py.Group) -> pyg.data.Data:

        parcellation_id = torch.from_numpy(data_hdf5["parcellation_id"][()])
        parcellation_id_one_hot = torch.nn.functional.one_hot(
            parcellation_id.long(), num_classes=36
        ).float()

        data = pyg.data.Data(
            y=parcellation_id_one_hot,
            pos=torch.from_numpy(data_hdf5["pos"][()]) * 1e-2,  # [mm] to [dm],
            face=torch.from_numpy(data_hdf5["face"][()].T),
            hemisphere_id=torch.from_numpy(data_hdf5["hemisphere_id"][()]).float(),
        )

        return data
