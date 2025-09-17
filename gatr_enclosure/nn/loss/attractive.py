# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear
from typing import Optional

import torch
from torch_cluster import knn

from torch_cluster import radius

from torch_cluster import knn_graph

import torch_geometric as pyg

class AttractiveLossOld:
    def __init__(self, num_nearest_pos: int = 32):
        self.num_nearest_pos = num_nearest_pos

    def __call__(
        self,
        pos_source: torch.Tensor,
        pos_target: torch.Tensor,
        batch_source: Optional[torch.Tensor] = None,
        batch_target: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:

        N = pos_source.shape[0]
        M = pos_target.shape[0]

        idcs_target, idcs_source = knn(
            pos_source,
            pos_target,
            k=self.num_nearest_pos,
            batch_x=batch_source,
            batch_y=batch_target,
        )

        squared_dist = ((pos_target[idcs_target] - pos_source[idcs_source]) ** 2).sum(dim=1)

        # first we average across all the neighbors of each node
        avg_dist_per_node = pyg.nn.pool.global_mean_pool(squared_dist, batch=idcs_target, size=M)

        potential = torch.exp( - 0.5 * avg_dist_per_node / (self.radius_repulsion ** 2))

        # then, we average across all the nodes in a graph
        avg_dist_per_graph = pyg.nn.pool.global_mean_pool(avg_dist_per_node, batch=batch_target)

        # finally, we average over all graphs
        loss = avg_dist_per_graph.mean()

        return loss
        # self.just_log_gt(pos_target, batch_target)

    def just_log_gt(self, pos, batch):

        with torch.no_grad():
            idcs_target, idcs_source = knn_graph(
                pos,
                k=8,
                batch=batch,
                loop=False
            )
            squared_dist = ((pos[idcs_target] - pos[idcs_source]) ** 2).sum(dim=1).sqrt()

            avg_dist_per_node = pyg.nn.pool.global_mean_pool(squared_dist, batch=idcs_target)
            max_dist_per_node = pyg.nn.pool.global_max_pool(squared_dist, batch=idcs_target)

            idcs_target, idcs_source = knn_graph(
                pos,
                k=1,
                batch=batch,
                loop=False
            )
            min_dist_per_node = ((pos[idcs_target] - pos[idcs_source]) ** 2).sum(dim=1).sqrt()

            print(f'graph stats: 8-th neigh {max_dist_per_node.mean().item():.3f} | n.est neigh {min_dist_per_node.mean().item():.3f} | avg 8 neighs: {avg_dist_per_node.mean().item():.3f}')



class AttractiveLoss:
    def __init__(self, radius_attraction: int = 32, dense: bool = False):
        self.radius_attraction = radius_attraction
        self.dense = dense

    def __call__(
        self,
        pos_source: torch.Tensor,
        pos_target: torch.Tensor,
        batch_source: Optional[torch.Tensor] = None,
        batch_target: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:

        N = pos_source.shape[0]
        M = pos_target.shape[0]

        if batch_source is not None and batch_target is not None:
            n_nodes_per_batch = pyg.utils.degree(batch_source)[batch_target]
            min_nodes_per_batch = int(n_nodes_per_batch.min().item())
        else:
            n_nodes_per_batch = N
            min_nodes_per_batch = N


        if self.dense:
            idcs_target, idcs_source = knn(
                pos_source, pos_target,
                k=min_nodes_per_batch,
                batch_x=batch_source,
                batch_y=batch_target,
            )
        else:
            idcs_target, idcs_source = radius(
                pos_source, pos_target, 
                # r=4*self.radius_attraction,
                r=20*self.radius_attraction,
                batch_x=batch_source, batch_y=batch_target
            )

        squared_dist = ((pos_target[idcs_target] - pos_source[idcs_source]) ** 2).sum(dim=1)

        potential = torch.exp( - 0.5 * squared_dist / (self.radius_attraction ** 2))

        # first we sum across all the neighbors of each node
        potential_per_node = pyg.nn.pool.global_add_pool(potential, batch=idcs_target, size=M) / n_nodes_per_batch

        # # then, we average across all the nodes in a graph
        # avg_potential_per_graph = pyg.nn.pool.global_mean_pool(potential_per_node, batch=batch_target)
        # # finally, we average over all graphs
        # loss = avg_potential_per_graph.mean()
        
        # because all graphs have the same number of virtual nodes we can merge the last two steps
        loss = potential_per_node.mean()

        # Note that we want to maximize this potential energy
        return -1 * loss