from __future__ import print_function, division

import torch
import numpy as np
import time
from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import adjusted_mutual_info_score, adjusted_rand_score

from utils import get_printer
__all__ = ['kmeans_on_data']

# This seems to stall in a distributed setting, do not call with --distributed
def kmeans_on_data(model, data_loader, options):
    scaler = StandardScaler(copy=False)

    num_examples = len(data_loader.dataset)
    num_classes = len(data_loader.dataset.classes)

    # features = np.empty(shape=(num_examples, options.backbone_output_size))
    features = np.empty(shape=(num_examples, options.projection_dim))
    targets = np.empty(shape=num_examples)

    Print = get_printer(options)
    idx = 0
    for data, labels in data_loader:
        # feat = model.module.backbone(data.to(device=options.cuda_device)).detach().cpu().numpy()
        feat = model(data.to(device=options.cuda_device)).detach().cpu().numpy()
        num_batch = feat.shape[0]
        features[idx:idx+num_batch] = feat
        targets[idx:idx+num_batch] = labels.numpy()
        idx += num_batch

        t = time.time()
        scaler.partial_fit(feat)
        t = time.time() - t
        Print(f"Partial feat took {t}s, cycle {idx} of {num_examples}")

    feat_scaled = scaler.transform(features)

    assigned_labels = MiniBatchKMeans(
                        n_clusters=num_classes,
                        batch_size=256
                    ).fit_predict(feat_scaled)
    # return adjusted_mutual_info_score(targets, assigned_labels)
    return adjusted_rand_score(targets, assigned_labels)
