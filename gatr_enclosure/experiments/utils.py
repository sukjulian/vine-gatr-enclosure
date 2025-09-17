# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear
from math import ceil, floor
from typing import Dict, Iterable, Literal, cast

import numpy as np
import torch
from torch_scatter import scatter


def get_generic_splits_idcs(
    len_dataset: int, proportion_training: float, proportion_validation: float
) -> Dict[Literal["training", "validation", "test"], torch.Tensor]:
    idcs = torch.arange(len_dataset)

    ptr_training = ceil(len_dataset * proportion_training)
    ptr_validation = ptr_training + floor(len_dataset * proportion_validation)

    return {
        "training": idcs[:ptr_training],
        "validation": idcs[ptr_training:ptr_validation],
        "test": idcs[ptr_validation:],
    }


def get_identified_splits_idcs(
    id_: Iterable[str], id_validation: Iterable[str], id_test: Iterable[str]
) -> Dict[Literal["training", "validation", "test"], torch.Tensor]:
    id_array = np.array(id_)[..., None]

    id_validation_array = np.array(id_validation)[None, ...]
    id_test_array = np.array(id_test)[None, ...]

    mask_test = (id_array == id_test_array).any(axis=1)

    if len(id_validation) == 0:
        mask_validation = np.zeros_like(mask_test)
    else:
        mask_validation = (id_array == id_validation_array).any(axis=1)

    mask_training = ~(mask_validation | mask_test)

    iterator = zip(("training", "validation", "test"), (mask_training, mask_validation, mask_test))
    idcs = {key: torch.from_numpy(value.nonzero()[0]) for key, value in iterator}

    return cast(Dict[Literal["training", "validation", "test"], torch.Tensor], idcs)


def compute_approximation_error(
    y: torch.Tensor, y_data: torch.Tensor, scatter_idcs: torch.Tensor
) -> torch.Tensor:

    sum_of_squared_l2_errors = scatter(
        (y_data - y).norm(dim=1) ** 2, scatter_idcs, dim=0, reduce="sum"
    )
    sum_of_squared_values = scatter(y_data.norm(dim=1) ** 2, scatter_idcs, dim=0, reduce="sum")

    return (sum_of_squared_l2_errors / sum_of_squared_values).sqrt()


def compute_classification_accuracy(
    y: torch.Tensor, y_data: torch.Tensor, scatter_idcs: torch.Tensor
) -> torch.Tensor:

    y_one_hot = torch.nn.functional.one_hot(y.max(dim=1)[1], num_classes=y.size(1))

    correct = scatter(torch.logical_and(y_one_hot, y_data).sum(dim=1), scatter_idcs, reduce="sum")
    total = scatter_idcs.unique(return_counts=True)[1]

    return correct / total


def compute_dice_coefficients(
    y: torch.Tensor, y_data: torch.Tensor, scatter_idcs: torch.Tensor
) -> torch.Tensor:

    y_one_hot = torch.nn.functional.one_hot(y.max(dim=1)[1], num_classes=y.size(1))

    true_positives = scatter(
        torch.logical_and(y_one_hot, y_data).int(), scatter_idcs, dim=0, reduce="sum"
    )
    false_positives_negatives = scatter(
        torch.logical_xor(y_one_hot, y).int(), scatter_idcs, dim=0, reduce="sum"
    )

    return 2.0 * true_positives / (2.0 * true_positives + false_positives_negatives + 1e-16)


def compute_mean_absolute_error(
    y: torch.Tensor, y_data: torch.Tensor, scatter_idcs: torch.Tensor
) -> torch.Tensor:

    mean_absolute_error = scatter((y_data - y).abs(), scatter_idcs, reduce="mean")

    return mean_absolute_error


def compute_mean_squared_error(
    y: torch.Tensor, y_data: torch.Tensor, scatter_idcs: torch.Tensor
) -> torch.Tensor:

    mean_squared_error = scatter((y_data - y) ** 2, scatter_idcs, reduce="mean")

    return mean_squared_error
