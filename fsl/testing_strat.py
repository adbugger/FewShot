from __future__ import print_function, division

import torch

import numpy
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

__all__ = ['Classify1NN', 'SoftCosAttn']

def Classify1NN(options, full_data, full_labels):
    classifier = KNeighborsClassifier(n_neighbors=1)
    full_data = full_data.detach().cpu().numpy()
    full_labels = full_labels.cpu().numpy()

    # SimpleShot center and unit norm
    if options.scaler is not None:
        full_data = options.scaler.transform(full_data)
    if options.normalizer is not None:
        full_data = options.normalizer.transform(full_data)

    # The train data is at the start of the mini-batch
    num_train = options.n_way * options.k_shot
    train_data, test_data = full_data[:num_train], full_data[num_train:]
    train_labels, test_labels = full_labels[:num_train], full_labels[num_train:]

    if options.k_shot == 1:
        classifier.fit(train_data, train_labels)
    elif options.k_shot == 5:
        # For 5 shots we need to calculate the centroids for every k_shot group
        fit_data = numpy.empty((options.n_way, train_data.shape[1]))
        fit_labels = numpy.empty(options.n_way)
        idx = 0
        while idx < options.n_way:
            fit_data[idx] = numpy.mean(train_data[idx*options.k_shot : (idx+1)*options.k_shot], axis=0)
            fit_labels[idx] = train_labels[idx*options.k_shot]
            idx += 1
        classifier.fit(fit_data, fit_labels)
    return classifier.score(test_data, test_labels)

def SoftCosAttn(options, full_data, full_labels):
    full_data = full_data.detach()

    # The train data is at the start of the mini-batch
    num_train = options.n_way * options.k_shot
    num_test = options.n_way * options.num_query
    train_data, test_data = full_data[:num_train], full_data[num_train:]
    train_labels, test_labels = full_labels[:num_train], full_labels[num_train:]

    # https://arxiv.org/pdf/1606.04080.pdf
    # The simplest attention kernel from Matching Networks for One-Shot Learning
    # softmax over cosine similarity

    # The cosine sim matrix is just a matrix multiply after unit-normalization
    train_data = torch.nn.functional.normalize(train_data, p=2, dim=1)
    test_data = torch.nn.functional.normalize(test_data, p=2, dim=1)

    # cosine similarity
    cos_sim = torch.mm(test_data, train_data.t())
    attn_kernel = torch.nn.functional.softmax(cos_sim, dim=1)
    assert attn_kernel.shape == (num_test, num_train)

    # predict labels based on argmax
    predict_labels = train_labels[attn_kernel.argmax(dim=1)]

    return accuracy_score(
        y_true=test_labels.cpu().numpy(),
        y_pred=predict_labels.cpu().numpy())

