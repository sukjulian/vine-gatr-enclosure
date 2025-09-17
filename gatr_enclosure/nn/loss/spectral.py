# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear
from typing import Optional

import torch
from torch_cluster import radius, knn

import torch_geometric as pyg


from gatr.primitives.attention import geometric_attention
from gatr.layers.attention.attention import GeometricAttention

import numpy as np



class SpectralLoss:
    def __init__(self, num_nearest_pos: int, diffusion_t: float = 1, n_eig: int = 32):
        self.num_nearest_pos = num_nearest_pos
        self.diffusion_t = diffusion_t
        self.n_eig = n_eig

    def __call__(
            self,
            data,
            layer: GeometricAttention,
            batch_source: Optional[torch.Tensor] = None,
            batch_target: Optional[torch.Tensor] = None,
            subset_index: Optional[torch.Tensor] = None,
        ) -> torch.Tensor:

        cached_cross_attention_input = data.cross_attention_pos['encoder'][1]
        q_mv, k_mv, _, q_s, k_s, _ = cached_cross_attention_input[:6]

        if len(cached_cross_attention_input) > 6:
            attention_mask = cached_cross_attention_input[7]
        else:
            attention_mask = None
        
        # print(q_mv.shape, k_mv.shape, q_s.shape, k_s.shape)

        # N virtual nodes
        N = q_mv.shape[1]

        # num input nodes
        M = k_mv.shape[1]

        n_heads = q_mv.shape[0]


        # k_mv has shape     1 x   input_tokens x hidden_channels x 16
        # q_mv has shape heads x virtual_tokens x hidden_channels x 16
        # hidden_channels = 4
        # layer.log_weights.shape = (heads, 1, hidden_channels)

        # need to give it a non-zero size otherwise attention crashes, ortherwise we would've used (1, M, 0, 16)
        v_mv = torch.zeros((1, M, 1, 16), device=k_mv.device)

        assert (data.eigenvectors.shape[-1] >= self.n_eig), (data.eigenvectors.shape[-1], self.n_eig)

        eigenvectors = data.eigenvectors[:, :self.n_eig]

        if subset_index is not None:
            assert eigenvectors.shape[0] == subset_index.shape[0], (eigenvectors.shape, subset_index.shape)

            # set to 0 the values over the input nodes not in the subset index
            v_s = torch.zeros((1, M, self.n_eig), dtype=eigenvectors.dtype, device=eigenvectors.device)
            v_s[:, subset_index, :] = data.eigenvectors.view(1, -1, self.n_eig)
        else:
            v_s = eigenvectors.view(1, M, self.n_eig)

        # h_s has shape heads x virtual_tokens x n_eig
        _, h_s = compute_spectral_scores(layer, q_mv, k_mv, v_mv, q_s, k_s, v_s, attention_mask=attention_mask)


        if subset_index is not None:
            source_embedding = eigenvectors.view(-1, self.n_eig)
            batch_source = batch_source[subset_index]
        else:
            source_embedding = v_s.view(M, self.n_eig)
        target_embedding = h_s.reshape(n_heads * N, self.n_eig)

        if batch_target is not None:
            batch_target = batch_target.view(1, N).expand(n_heads, N).reshape(-1)
            # assert batch_target.shape == (N*n_heads,), (batch_target.shape, N, n_heads, n*n_heads)
        
        N = N * n_heads

        # data.mass
        # eigenvalues has shape num_batches x n_eig
        eigenvalues = data.eigenvalues[:, :self.n_eig]

        # np.set_printoptions(precision=3, suppress=True, linewidth=100000)
        # print()
        # # print(eigenvalues.detach().cpu().numpy())
        # # print(f"Eigenvalues range:{eigenvalues.min().item()} - {torch.quantile(eigenvalues, q=0.25).item()} - {torch.quantile(eigenvalues, q=0.5).item()} - {torch.quantile(eigenvalues, q=0.75).item()} - {eigenvalues.max().item()}")
        # # print(torch.quantile(eigenvalues.cpu(), q=torch.tensor([0., 0.25, 0.5, 0.75, 1.]), dim=1))
        # print(eigenvalues.shape)
        # for i in range(eigenvalues.shape[0]):
        #     print(torch.quantile(eigenvalues[i].cpu(), q=torch.tensor([0., 0.25, 0.5, 0.75, 1.])).reshape(-1))
        # # print(f"Eigenvalues range:{eigenvalues.min().item()} - {torch.quantile(eigenvalues, q=0.25).item()} - {torch.quantile(eigenvalues, q=0.5).item()} - {torch.quantile(eigenvalues, q=0.75).item()} - {eigenvalues.max().item()}")
        # print()

        with torch.no_grad():
            eigenvalues = eigenvalues.view(-1, self.n_eig)

            eigenvalues[eigenvalues <= 0.] = 0.

            # spectral heat kernel diffusion
            eigenvalues = torch.exp(- self.diffusion_t * eigenvalues)

            # if batch_source is not None:
            #     n_nodes_per_batch_source = pyg.utils.degree(batch_source).reshape(-1, 1)
            # else:
            #     n_nodes_per_batch_source = M
            # eigenvalues = eigenvalues / eigenvalues.sum(-1).view(-1, 1) * n_nodes_per_batch_source
        
        # return self.compute_losses_dense(source_embedding, target_embedding, batch_source, batch_target, eigenvalues)
        return self.compute_losses_dense_L2(source_embedding, target_embedding, batch_source, batch_target, eigenvalues)
    
    def compute_losses_dense(self, source_embedding, target_embedding, batch_source, batch_target, eigenvalues):

        # N virtual nodes
        N = target_embedding.shape[0]

        # num input nodes
        M = source_embedding.shape[0]

        if batch_source is not None:
            n_nodes_per_batch_source = pyg.utils.degree(batch_source).reshape(-1, 1)
        else:
            n_nodes_per_batch_source = M
        total_source_embedding = pyg.nn.pool.global_add_pool(source_embedding, batch=batch_source) 
        avg_source_embedding = total_source_embedding / n_nodes_per_batch_source

        
        if batch_target is not None:
            n_nodes_per_batch_target = pyg.utils.degree(batch_target).reshape(-1, 1)
        else:
            n_nodes_per_batch_target = N
        total_target_embedding = pyg.nn.pool.global_add_pool(target_embedding, batch=batch_target) 
        avg_target_embedding = total_target_embedding / n_nodes_per_batch_target


        attractive_potential = (avg_target_embedding * eigenvalues * avg_source_embedding).sum(-1)  

        per_node_eigenvalues = eigenvalues
        if batch_target is not None:
            per_node_eigenvalues = per_node_eigenvalues[batch_target]

        norms_target = (target_embedding * per_node_eigenvalues * target_embedding).sum(-1)  #/ n_nodes_per_batch_target
        
        # E[x]^T\lambda E[x] - 1/N E[x^T \lambda x]
        repulsive_potential = (avg_target_embedding * eigenvalues * avg_target_embedding).sum(-1) - pyg.nn.pool.global_add_pool(norms_target, batch=batch_target) / n_nodes_per_batch_target**2

        loss_repulsive = repulsive_potential.mean() #* self.diffusion_t
        loss_attractive = attractive_potential.mean() #* self.diffusion_t

        return loss_repulsive, loss_attractive
    
    def compute_losses_dense_L2(self, source_embedding, target_embedding, batch_source, batch_target, eigenvalues):

        # N virtual nodes
        N = target_embedding.shape[0]

        # num input nodes
        M = source_embedding.shape[0]

        if batch_source is not None:
            n_nodes_per_batch_source = pyg.utils.degree(batch_source).reshape(-1, 1)
        else:
            n_nodes_per_batch_source = M

        # if batch_target is not None:
        #     n_nodes_per_batch_target = pyg.utils.degree(batch_target).reshape(-1, 1)
        # else:
        #     n_nodes_per_batch_target = N

        with torch.no_grad():
            # normalize s.t. the diagonal of the diffusion operator has ~ value one everywhere
            # alpha = eigenvalues.sum(-1).view(-1, 1) / n_nodes_per_batch_source
            # normalize s./t we use as distance function 1/M sum_j |x_i - x_j|^2
            # eigenvalues = eigenvalues / n_nodes_per_batch_source.sqrt()

            # mhmm actually should be this one... This way we don't need to divide my sqrt(M) above either probably
            alpha = torch.linalg.norm(eigenvalues, dim=-1).view(-1, 1) / n_nodes_per_batch_source.sqrt()

            eigenvalues = eigenvalues / alpha

            # this term should essentially disappear in the loss below due to the normalization above
            alpha2 = (eigenvalues**2).sum(-1).view(-1, 1) 


        per_node_eigenvalues = eigenvalues
        if batch_target is not None:
            per_node_eigenvalues = per_node_eigenvalues[batch_target]

        avg_target_embedding = pyg.nn.pool.global_mean_pool(target_embedding, batch=batch_target) 
        avg_source_embedding = pyg.nn.pool.global_mean_pool(source_embedding, batch=batch_source) 

        # norm squared of the average of the target embeddings E[x]^2
        avg_targets_norm = (avg_target_embedding**2 * eigenvalues**2).sum(-1)

        # average of the norm squared of the target_embedding
        norms_target = (target_embedding **2 * per_node_eigenvalues ** 2).sum(-1)
        avg_norms_target = pyg.nn.pool.global_mean_pool(norms_target, batch=batch_target)


        # norm squared of the average diffused target_embedding
        attractive_potential = 0.5 * avg_norms_target + 0.5 * alpha2  / n_nodes_per_batch_source - (avg_target_embedding * eigenvalues**2 * avg_source_embedding).sum(-1)

        
        # average L^2 distance between all pairs of virtual nodes. This is equal to the L^2 norm of the averaged embedding minus the average L^2 norm of each embedding
        # 0.5 E_ij[|\lambda x_i - \lambda x_j|^2] = E[x^T \lambda^2 x] - E[x]^T\lambda^2 E[x]
        # this is essentially the standard deviation of the target embeddings
        repulsive_potential = avg_norms_target - avg_targets_norm

        # the standard deviation should be non-negative but sometimes it has some small errors
        # assert (repulsive_potential >= -1e-3).all(), (repulsive_potential.min().item())

        loss_repulsive = repulsive_potential.mean() 
        loss_attractive = attractive_potential.mean() 

        return -1 * loss_repulsive, -1 * loss_attractive




def compute_spectral_scores(layer: GeometricAttention, q_mv, k_mv, v_mv, q_s, k_s, v_s, attention_mask=None):
    weights = layer.log_weights.exp()
    h_mv, h_s = geometric_attention(
        q_mv,
        k_mv,
        v_mv,
        q_s,
        k_s,
        v_s,
        normalizer=layer.normalizer,
        weights=weights,
        attn_mask=attention_mask,
    )
    return h_mv, h_s

