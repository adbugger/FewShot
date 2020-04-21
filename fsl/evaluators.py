from __future__ import print_function, division

import torch
import numpy as np
import time
from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import adjusted_mutual_info_score, adjusted_rand_score

from utils import get_printer, do_nothing
__all__ = ['kmeans_on_data']


def kmeans_on_data(model, data_loader, options):
    # Print = get_printer(options)
    Print = do_nothing
    Print("doing kmeans on data")
    scaler = StandardScaler(copy=False)

    num_examples = len(data_loader.dataset)
    num_classes = len(data_loader.dataset.classes)

    # features = np.empty(shape=(num_examples, options.backbone_output_size))
    features = np.empty(shape=(num_examples, options.projection_dim))
    targets = np.empty(shape=num_examples)

    idx = 0
    s = time.time()
    for data, labels in data_loader:
        Print(f"data cycling at idx {idx} of {num_examples}")
        # feat = model.module.backbone(data.to(device=options.cuda_device)).detach().cpu().numpy()
        feat = model(data.to(device=options.cuda_device)).detach().cpu().numpy()
        num_batch = feat.shape[0]
        features[idx:idx+num_batch] = feat
        targets[idx:idx+num_batch] = labels.numpy()
        idx += num_batch

        scaler.partial_fit(feat)
    Print(f"data cycle time: {time.time() - s}")

    s = time.time()
    feat_scaled = scaler.transform(features)
    Print(f"transform time: {time.time() - s}")

    s = time.time()
    assigned_labels = MiniBatchKMeans(
                        n_clusters=num_classes,
                        batch_size=512
                    ).fit_predict(feat_scaled)
    Print(f"KMeans time: {time.time() - s}")
    # return adjusted_mutual_info_score(targets, assigned_labels)
    return adjusted_rand_score(targets, assigned_labels)
