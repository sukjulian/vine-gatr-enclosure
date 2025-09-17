# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear
from itertools import product
from typing import Iterable, List, Tuple

import potpourri3d as pp3d
import scipy.sparse
import torch
from torch.linalg import cross, det, svd
from torch.nn.functional import normalize
from torch_cluster import knn
from torch_scatter import scatter
from trimesh import Trimesh
from trimesh.convex import convex_hull
from trimesh.curvature import discrete_gaussian_curvature_measure
from trimesh.sample import sample_surface_even


def compute_diffusion_dist(pos: torch.Tensor, idcs: Iterable[torch.Tensor]) -> List[torch.Tensor]:

    solver = pp3d.PointCloudHeatSolver(pos)
    compute_fun = solver.compute_distance_multisource

    diffusion_dist = [torch.from_numpy(compute_fun(idcs_).astype("f4")) for idcs_ in idcs]

    return diffusion_dist


def compute_gaussian_curvature(
    pos: torch.Tensor, face: torch.Tensor, radius: float
) -> torch.Tensor:

    curvature = discrete_gaussian_curvature_measure(Trimesh(pos, face.T), pos, radius).astype("f4")
    curvature = torch.from_numpy(curvature)

    return curvature


from torch_cluster import radius, knn
import torch_geometric as pyg


def compute_pointcloud_laplacian_eigenvectors(
    pos: torch.Tensor, num_eigenvectors: int, sigma: float,
    device='cpu'
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

    orig_device = pos.device

    pos = pos.to(device)

    edge_index = radius(
        pos, pos,
        r=2.5 * sigma,
        batch_x=None, 
        batch_y=None, 
        max_num_neighbors=48,
        num_workers=-1
    )    
    edge_index = torch.cat([
        edge_index,
        knn(
            pos, pos,
            batch_x=None, 
            batch_y=None, 
            k=6,
            num_workers=-1
        )
    ], dim=1)

    edge_index = edge_index.sort(0)[0]
    edge_index = torch.unique(edge_index, dim=1)
    i, j = edge_index
    edge_index = edge_index[:, i != j]
    i, j = edge_index

    N = pos.shape[0]

    E = edge_index.shape[1]


    dist = ((pos[i] - pos[j])**2).sum(-1)

    kernel = torch.exp(- 0.5 * dist / sigma**2)

    N = pos.shape[0]

    diagonal = pyg.nn.pool.global_add_pool(kernel, batch=i, size=N) + pyg.nn.pool.global_add_pool(kernel, batch=j, size=N)

    edge_index = torch.cat([
        torch.arange(N, device=edge_index.device).view(1, N).expand(2, N),
        edge_index,
        edge_index.flip(0), # is it necessary? lobpcg should anyways assume symmetric martix
    ], dim=1)
    laplacian = torch.cat([
        diagonal,
        -1 * kernel,
        -1 * kernel,
    ], dim=0)

    degree = pyg.utils.degree(edge_index[0], num_nodes=N) + pyg.utils.degree(edge_index[1], num_nodes=N)
    print(degree.min().item(), torch.quantile(degree, 0.01).item(), degree.mean().item(), degree.max().item(), diagonal.min().item(), torch.quantile(diagonal, 0.01).item(), diagonal.mean().item())

    if(degree.min() < 1 or diagonal.min().item() < 0.005):
        print('Error, too low of a radius')
        print(degree.min().item(), torch.quantile(degree, 0.01).item(), degree.mean().item(), degree.max().item(), diagonal.min().item(), torch.quantile(diagonal, 0.01).item(), diagonal.mean().item())

    if pos.is_cuda:
        L = torch.sparse_coo_tensor(
            edge_index, laplacian, size=(N, N)
        )
        eigenvalues, eigenvectors = torch.lobpcg(
            L,
            k=num_eigenvectors,
            largest=False,
            niter=-1,
            method='ortho'
        )
    else:
        L = scipy.sparse.coo_matrix((laplacian.numpy(), (edge_index[0].numpy(), edge_index[1].numpy())), shape=(N, N))

        eigenvalues, eigenvectors = scipy.sparse.linalg.eigsh(
            L, k=num_eigenvectors, M=None, sigma=1e-6, 
        )
        eigenvalues, eigenvectors = torch.from_numpy(eigenvalues), torch.from_numpy(eigenvectors)

    mass = None

    assert eigenvalues.min() > -1e-6, (eigenvalues.min().item(), eigenvalues.max().item(), eigenvalues.mean().item())

    return mass, eigenvalues.to(orig_device), eigenvectors.to(orig_device)


def compute_laplacian_eigenvectors(
    pos: torch.Tensor, face: torch.Tensor, num_eigenvectors: int
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

    mass = scipy.sparse.diags(pp3d.vertex_areas(pos, face.T).astype("f4"))
    laplacian = pp3d.cotan_laplacian(pos.numpy(), face.T)

    eigenvalues, eigenvectors = scipy.sparse.linalg.eigsh(
        laplacian, k=num_eigenvectors, M=mass, sigma=1e-6
    )

    mass = torch.from_numpy(mass.data.squeeze())
    eigenvalues, eigenvectors = torch.from_numpy(eigenvalues), torch.from_numpy(eigenvectors)

    return mass, eigenvalues, eigenvectors


def compute_sulcal_depth(pos: torch.Tensor, face: torch.Tensor) -> torch.Tensor:

    convex_hull_ = convex_hull(Trimesh(pos, face.T))

    pos_convex_hull = sample_surface_even(convex_hull_, pos.size(0))[0].astype("f4")
    pos_convex_hull = torch.from_numpy(pos_convex_hull)

    nearest = knn(pos_convex_hull, pos, k=1)[1]
    sulcal_depth = (pos_convex_hull[nearest] - pos).norm(dim=1)

    return sulcal_depth


def compute_surface_normal(pos: torch.Tensor, face: torch.Tensor) -> torch.Tensor:

    face_normal = normalize(cross(pos[face[1]] - pos[face[0]], pos[face[2]] - pos[face[0]]))
    scatter_idcs = torch.cat([face[0], face[1], face[2]])

    surface_normal = normalize(
        scatter(face_normal.repeat(3, 1), scatter_idcs.long(), dim=0, reduce="add")
    )

    return surface_normal


def disambiguated_svd(tensor: torch.Tensor, ambiguities=None) -> Tuple[torch.Tensor, torch.Tensor]:

    _, singular_values, right_singular_vectors = svd(tensor, full_matrices=False)

    # Sign ambiguity
    if ambiguities is None:
        num_dim = tensor.size(1)
        ambiguities = torch.tensor(
            list(product((1, -1), repeat=num_dim)), device=right_singular_vectors.device
        )

    # O(n) to SO(n)
    o_elements = right_singular_vectors[None, ...] * ambiguities[..., None]
    so_elements = o_elements[det(o_elements).isclose(torch.tensor(1.0))]

    return singular_values, so_elements
