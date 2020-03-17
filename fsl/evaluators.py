from __future__ import print_function, division

import numpy as np
import torch

from pykeops.torch import LazyTensor

__all__ = ['evaluate_on_test']

# https://www.kernel-operations.io/keops/_auto_tutorials/kmeans/plot_kmeans_torch.html
def KMeans(x, K=10, Niter=10):
    N, D = x.shape  # Number of samples, dimension of the ambient space

    # K-means loop:
    # - x  is the point cloud,
    # - cl is the vector of class labels
    # - c  is the cloud of cluster centroids
    c = x[:K, :].clone()  # Simplistic random initialization
    x_i = LazyTensor(x[:, None, :])  # (Npoints, 1, D)

    for i in range(Niter):

        c_j = LazyTensor(c[None, :, :])  # (1, Nclusters, D)
        D_ij = ((x_i - c_j) ** 2).sum(-1)  # (Npoints, Nclusters) symbolic matrix of squared distances
        cl = D_ij.argmin(dim=1).long().view(-1)  # Points -> Nearest cluster

        # Ncl = torch.bincount(cl).type(torchtype[dtype])  # Class weights
        Ncl = torch.bincount(cl)  # Class weights
        for d in range(D):  # Compute the cluster centroids with torch.bincount:
            c[:, d] = torch.bincount(cl, weights=x[:, d]) / Ncl

    return cl, c

def cluster_similarity(assigned_labels, ground_labels, num_classes):
    cluster_counts = np.zeros(shape=(num_classes, num_classes), dtype=int)
    # Within each cluster, count the ground truth labels with maximum occurrence
    # Let this be the ground truth for that cluster.
    # Calculate cluster accuracy based on this.
    for (assigned, actual) in zip(assigned_labels, ground_labels):
        cluster_count[assigned, actual] += 1

    # total accuracy is correct / incorrect
    # sum of max over each row / total sum
    return cluster_count.max(axis=1).sum() / cluster_count.sum()

def evaluate_on_test(model, test_loader, options):
    model.eval()
    num_classes = len(test_loader.dataset.classes)
    for batch, labels in test_loader:
        feat = model(batch.to(device=options.cuda_device))
        assigned_labels, cluster_centers = KMeans(
                                            x=feat,
                                            K=num_classes,
                                            Niter=options.cluster_iters
                                        )
        return cluster_similarity(assigned_labels, labels, num_classes)
        break
