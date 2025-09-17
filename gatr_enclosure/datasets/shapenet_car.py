# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear
import h5py
import torch
import torch_geometric as pyg
import numpy as np

from .base import HierachicalDataFormatDatasetMemory

PRESSURE_AVERAGE = -36.3099
PRESSURE_STD = 48.5743


class ShapenetCarDataset(HierachicalDataFormatDatasetMemory):

    def __init__(self, root, transform = None, pre_transform = None):
        super().__init__(root, transform, pre_transform)


        # diff = (self.data.singular_values.view(-1, 3, 1) - self.data.singular_values.view(-1, 1, 3)).abs()
        # import numpy as np
        # np.set_printoptions(precision=5, suppress=True, linewidth=100000)
        # print(diff.mean(0).cpu().numpy())

        if hasattr(self.data, "eigenvalues"):
            print(self.data.eigenvalues.shape)
            me = self.data.eigenvalues.min(1)[0]
            Me = self.data.eigenvalues.max(1)[0]

            print(f'Min eig: {me.min().item()} - {me.mean().item()} - {me.max().item()}')
            print(f'Max eig: {Me.min().item()} - {Me.mean().item()} - {Me.max().item()}')
        
        # print(self.data)
        # num_mesh_points = self.data.y.shape[0]
        # num_tot_points = self.data.pos.shape[0]
        # num_batches = self.data.origin.shape[0]
        # print('Avg mesh points:', num_mesh_points / num_batches)
        # print('Avg tot points:', num_tot_points / num_batches)
        
        # data_list = self._get_data_list()
        # exit()


    @staticmethod
    def hdf5_to_pyg(data_hdf5: h5py.Group) -> pyg.data.Data:

        pressure = torch.from_numpy(data_hdf5["pressure"][()])
        pressure_normalised = (pressure - PRESSURE_AVERAGE) / PRESSURE_STD

        pos_domain = torch.from_numpy(data_hdf5["pos_velocity"][()])
        velocity = torch.from_numpy(data_hdf5["velocity"][()])
        average_velocity = velocity.mean(dim=0)
        direction_average_velocity = average_velocity / average_velocity.norm()

        pos_dirichlet_boundary = torch.from_numpy(data_hdf5["pos_pressure"][()])
        dirichlet_boundary_normal = torch.from_numpy(data_hdf5["car_surface_normal"][()])

        # print(pos_domain.shape, velocity.shape, average_velocity.shape, pos_dirichlet_boundary.shape, dirichlet_boundary_normal.shape)
        # print(direction_average_velocity)

        pos = torch.cat((pos_dirichlet_boundary, pos_domain))
        field = torch.cat(
            (dirichlet_boundary_normal, direction_average_velocity.expand(pos_domain.size(0), -1))
        )
        dirichlet_boundary_idcs = torch.arange(pos_dirichlet_boundary.size(0), dtype=torch.int)

        data = pyg.data.Data(
            y=pressure_normalised,
            pos=pos,
            field=field,
            dirichlet_boundary_index=dirichlet_boundary_idcs,
        )
        return data


class ShapenetCarDatasetUPT(HierachicalDataFormatDatasetMemory):

    def __init__(self, root, transform = None, pre_transform = None):
        super().__init__(root, transform, pre_transform)


        # diff = (self.data.singular_values.view(-1, 3, 1) - self.data.singular_values.view(-1, 1, 3)).abs()
        # import numpy as np
        # np.set_printoptions(precision=5, suppress=True, linewidth=100000)
        # print(diff.mean(0).cpu().numpy())

        if hasattr(self.data, "eigenvalues"):
            print(self.data.eigenvalues.shape)
            me = self.data.eigenvalues.min(1)[0]
            Me = self.data.eigenvalues.max(1)[0]

            print(f'Min eig: {me.min().item()} - {me.mean().item()} - {me.max().item()}')
            print(f'Max eig: {Me.min().item()} - {Me.mean().item()} - {Me.max().item()}')
        
        # print(self.data)
        # num_mesh_points = self.data.y.shape[0]
        # num_tot_points = self.data.pos.shape[0]
        # num_batches = self.data.origin.shape[0]
        # print('Avg mesh points:', num_mesh_points / num_batches)
        # print('Avg tot points:', num_tot_points / num_batches)
        
        # data_list = self._get_data_list()
        # exit()


    @staticmethod
    def hdf5_to_pyg(data_hdf5: h5py.Group) -> pyg.data.Data:

        pressure = torch.from_numpy(data_hdf5["pressure"][()])
        pressure_normalised = (pressure - PRESSURE_AVERAGE) / PRESSURE_STD

        # pos_domain = torch.from_numpy(data_hdf5["pos_velocity"][()])
        velocity = torch.from_numpy(data_hdf5["velocity"][()])
        average_velocity = velocity.mean(dim=0)
        direction_average_velocity = average_velocity / average_velocity.norm()

        pos_dirichlet_boundary = torch.from_numpy(data_hdf5["pos_pressure"][()])
        dirichlet_boundary_normal = torch.from_numpy(data_hdf5["car_surface_normal"][()])


        resolution = 32
        domain_min = torch.tensor([-2.0, -1.0, -4.5])
        domain_max = torch.tensor([2.0, 4.5, 6.0])
        tx = np.linspace(domain_min[0], domain_max[0], resolution)
        ty = np.linspace(domain_min[1], domain_max[1], resolution)
        tz = np.linspace(domain_min[2], domain_max[2], resolution)
        grid = np.stack(np.meshgrid(tx, ty, tz, indexing="ij"), axis=-1).astype(np.float32).reshape(resolution**3, 3)

        pos_domain = torch.from_numpy(grid)

        sdf = torch.from_numpy(data_hdf5[f"sdf_res{resolution}"][()])
        sdf = torch.cat(
            (torch.zeros(pos_dirichlet_boundary.size(0), dtype=sdf.dtype), sdf.reshape(-1))
        )

        # print(pos_domain.shape, velocity.shape, average_velocity.shape, pos_dirichlet_boundary.shape, dirichlet_boundary_normal.shape)
        # print(direction_average_velocity)

        pos = torch.cat((pos_dirichlet_boundary, pos_domain))
        field = torch.cat(
            (dirichlet_boundary_normal, direction_average_velocity.expand(pos_domain.size(0), -1))
        )
        dirichlet_boundary_idcs = torch.arange(pos_dirichlet_boundary.size(0), dtype=torch.int)

        data = pyg.data.Data(
            y=pressure_normalised,
            pos=pos,
            field=field,
            dirichlet_boundary_index=dirichlet_boundary_idcs,
            sdf=sdf
        )

        return data
