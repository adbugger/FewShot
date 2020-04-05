from __future__ import print_function, division

import torch
import numpy as np
from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import adjusted_mutual_info_score, adjusted_rand_score

__all__ = ['kmeans_on_data']


def kmeans_on_data(model, data_loader, options):
    num_examples = len(data_loader.dataset)
    num_classes = len(data_loader.dataset.classes)

    features = np.empty(shape=(num_examples, options.backbone_output_size))
    targets = np.empty(shape=num_examples)

    model.eval()
    idx = 0
    for data, labels in data_loader:
        feat = model.module.backbone(data.to(device=options.cuda_device)).detach().cpu().numpy()
        num_batch = feat.shape[0]
        features[idx:idx+num_batch] = feat
        targets[idx:idx+num_batch] = labels.numpy()
        idx += num_batch

    assigned_labels = MiniBatchKMeans(
                        n_clusters=num_classes,
                        batch_size=512
                    ).fit_predict(
                        StandardScaler(copy=False).fit_transform(features)
                    )
    model.train()
    # return adjusted_mutual_info_score(targets, assigned_labels)
    return adjusted_rand_score(targets, assigned_labels)
