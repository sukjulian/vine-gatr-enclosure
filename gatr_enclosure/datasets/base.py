# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear
import os
from abc import abstractmethod
from functools import cached_property
from typing import Callable, List, Optional

import h5py
import torch
import torch_geometric as pyg
from tqdm import tqdm

from .utils import get_hash_id


class HierachicalDataFormatDatasetDisk(pyg.data.Dataset):

    def __init__(
        self,
        root: str,
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
    ):

        self._path_hdf5 = os.path.join(root, "raw", "dataset.hdf5")
        self._hash_id = get_hash_id(repr(pre_transform))

        with h5py.File(self._path_hdf5, "r") as file:
            self._id_tuple = tuple(file)

        super().__init__(root, transform, pre_transform)

    @property
    def processed_dir(self) -> str:
        return os.path.join(self.root, "processed", self._hash_id)

    @property
    def processed_file_names(self) -> List[str]:
        return [f"{id_}.pt" for id_ in self._id_tuple]

    def process(self) -> None:

        for id_ in tqdm(self._id_tuple, desc="Pre-processing dataset", leave=False):
            data = self._get_data(id_)

            if self.pre_transform is not None:
                data = self.pre_transform(data)

            torch.save(data, os.path.join(self.processed_dir, f"{id_}.pt"))

    def _get_data(self, id_: str) -> pyg.data.Data:

        with h5py.File(self._path_hdf5, "r") as file:
            data_hdf5 = file[id_]

            data = self.hdf5_to_pyg(data_hdf5)

        return data

    @staticmethod
    @abstractmethod
    def hdf5_to_pyg(data_hdf5: h5py.Group) -> pyg.data.Data:
        raise NotImplementedError

    def len(self) -> int:
        return len(self._id_tuple)

    def get(self, idx: int) -> pyg.data.Data:

        id_ = self._id_tuple[idx]
        data = torch.load(os.path.join(self.processed_dir, f"{id_}.pt"))

        return data


class HierachicalDataFormatDatasetMemory(pyg.data.InMemoryDataset):

    def __init__(
        self,
        root: str,
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
    ):

        self._path_hdf5 = os.path.join(root, "raw", "dataset.hdf5")
        self._hash_id = get_hash_id(repr(pre_transform))

        super().__init__(root, transform, pre_transform)

        self.load(self.processed_paths[0])

    @property
    def processed_dir(self) -> str:
        return os.path.join(self.root, "processed", self._hash_id)

    @property
    def processed_file_names(self) -> List[str]:
        return ["data.pt"]

    def process(self) -> None:
        data_list = self._get_data_list()

        if self.pre_transform is not None:
            iterator = tqdm(data_list, desc="Pre-processing dataset", leave=False)
            data_list = [self.pre_transform(data) for data in iterator]
            # data_list = preprocess_data_in_parallel(data_list, self.pre_transform)

        self.save(data_list, self.processed_paths[0])

    def _get_data_list(self) -> List[pyg.data.Data]:
        data_list = []

        with h5py.File(self._path_hdf5, "r") as file:

            for id_ in file:
                data_hdf5 = file[id_]

                data = self.hdf5_to_pyg(data_hdf5)

                data_list.append(data)

        return data_list

    @staticmethod
    @abstractmethod
    def hdf5_to_pyg(data_hdf5: h5py.Group) -> pyg.data.Data:
        raise NotImplementedError

    @cached_property
    def id_list(self) -> List[str]:

        with h5py.File(self._path_hdf5, "r") as file:
            id_list = list(file)

        return id_list





class TransformedDataset(torch.utils.data.Dataset):
    def __init__(self, datalist, transform):
        super(TransformedDataset).__init__()

        self.datalist = datalist
        self.transform = transform
    
    def __len__(self):
        return len(self.datalist)
    
    def __getitem__(self, idx):
        return self.transform(self.datalist[idx])

def worker_init_fn(worker_id):
    import os
    os.environ['OMP_NUM_THREADS'] = '8'  # Set the number of threads for each worker
    # os.environ['OMP_STACKSIZE'] = '2G'  # Adjust the size as needed
    print(f"Worker {worker_id} initialized with {os.environ['OMP_NUM_THREADS']} threads") 


def id_collate_fn(x):
    return x[0]

def preprocess_data_in_parallel(data_list, pre_transform):
    tmpdataset = TransformedDataset(data_list, pre_transform)
    dataloader = torch.utils.data.DataLoader(
        tmpdataset,
        batch_size=1,
        shuffle=False,
        num_workers=8,
        worker_init_fn=worker_init_fn,
        multiprocessing_context='spawn',
        prefetch_factor=20,
        collate_fn = id_collate_fn,
    )
    iterator = tqdm(dataloader, desc="Pre-processing dataset", leave=False)
    data_list = [data.clone() for data in iterator]
    # data_list = [data for data in iterator]
    return data_list
