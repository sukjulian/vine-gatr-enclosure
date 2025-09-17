# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear
from typing import Optional

import torch
from torch_cluster import radius, knn

import torch_geometric as pyg


class RepulsiveLoss:
    def __init__(self, radius_repulsion: float, dense: bool = False):
        self.radius_repulsion = radius_repulsion
        self.dense = dense

    def __call__(self, pos: torch.Tensor, batch: Optional[torch.Tensor] = None) -> torch.Tensor:

        N = pos.shape[0]

        if batch is not None:
            n_nodes_per_batch = pyg.utils.degree(batch)[batch]
            min_nodes_per_batch = int(n_nodes_per_batch.min().item())
        else:
            n_nodes_per_batch = N
            min_nodes_per_batch = N


        if self.dense:
            idcs_target, idcs_source = knn(
                pos, pos,
                k=min_nodes_per_batch,
                batch=batch, loop=False
            )
            mask = idcs_target != idcs_source
            idcs_source = idcs_source[mask]
            idcs_target = idcs_target[mask]
        else:
            idcs_target, idcs_source = radius(
                pos, pos, 
                # r=4*self.radius_repulsion,
                r=20*self.radius_repulsion,
                batch_x=batch, batch_y=batch, ignore_same_index=True
            )

        squared_dist = ((pos[idcs_target] - pos[idcs_source]) ** 2).sum(dim=1)

        potential = torch.exp( - 0.5 * squared_dist / (self.radius_repulsion ** 2))

        if batch is not None:
            n_nodes_per_batch = pyg.utils.degree(batch)[batch]
        else:
            n_nodes_per_batch = N

        # first we sum across all the neighbors of each node
        potential_per_node = pyg.nn.pool.global_add_pool(potential, batch=idcs_target, size=N) / n_nodes_per_batch

        # # then, we average across all the nodes in a graph
        # avg_potential_per_graph = pyg.nn.pool.global_mean_pool(potential_per_node, batch=batch)
        # # batch = batch[idcs_source]
        # # avg_potential_per_graph = pyg.nn.pool.global_mean_pool(potential, batch=batch)

        # # finally, we average over all graphs
        # loss = avg_potential_per_graph.mean()

        # because all graphs have the same number of virtual nodes we can merge the last two steps
        loss = potential_per_node.mean()

        return loss
