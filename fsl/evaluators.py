from __future__ import print_function, division

import torch
import numpy as np
from sklearn.cluster import MiniBatchKMeans

__all__ = ['kmeans_on_data']


def cluster_similarity(assigned_labels, ground_labels, num_classes):
    cluster_count = np.zeros(shape=(num_classes, num_classes), dtype=int)
    # Within each cluster, count the ground truth labels with maximum occurrence
    # Let this be the ground truth for that cluster.
    # Calculate cluster accuracy based on this.

    for (assigned, actual) in zip(assigned_labels, ground_labels):
        assigned = int(assigned)
        actual = int(actual)
        cluster_count[assigned, actual] += 1

    # total accuracy is correct / incorrect
    # sum of max over each row / total sum
    return cluster_count.max(axis=1).sum() / cluster_count.sum()


def kmeans_on_data(model, data_loader, options):
    num_examples = len(data_loader.dataset)
    num_classes = len(data_loader.dataset.classes)

    features = torch.empty(num_examples, options.backbone_output_size)
    targets = data_loader.dataset.targets

    model.eval()
    idx = 0
    for batch, _ in data_loader:
        feat = model.module.backbone(batch.to(device=options.cuda_device))
        num_batch = feat.shape[0]
        features[idx:idx+num_batch] = feat
        idx += num_batch

    # assigned_labels, _ = kmeans(X=features, num_clusters=num_classes, distance='cosine')
    assigned_labels = MiniBatchKMeans(n_clusters=num_classes, batch_size=512).fit_predict(X=features.detach().cpu().numpy())
    return cluster_similarity(assigned_labels, targets, num_classes)
