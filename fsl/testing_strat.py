from __future__ import print_function, division

import torch

import numpy
from scipy.special import softmax
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import Normalizer
from sklearn.metrics import accuracy_score

__all__ = ['Classify1NN', 'SoftCosAttn']

def partition_data(options, full_data, full_labels):
    # Inputs and outputs are all numpy arrays

    # The train data is at the start of the mini-batch
    num_train = options.n_way * options.k_shot
    train_data, test_data = full_data[:num_train], full_data[num_train:]
    train_labels, test_labels = full_labels[:num_train], full_labels[num_train:]

    if options.k_shot == 5 and options.centroid:
        # For 5 shot, we need to calculate centroids for every k_shot group
        fit_data = numpy.empty((options.n_way, train_data.shape[1]))
        fit_labels = numpy.empty(options.n_way)
        
        idx = 0
        while idx < options.n_way:
            fit_data[idx] = numpy.mean(train_data[idx*options.k_shot : (idx+1)*options.k_shot], axis=0)
            fit_labels[idx] = train_labels[idx*options.k_shot]
            idx += 1
        
        return (fit_data, test_data, fit_labels, test_labels)

    return (train_data, test_data, train_labels, test_labels)

def Classify1NN(options, full_data, full_labels):
    classifier = KNeighborsClassifier(n_neighbors=1)
    full_data = options.pre_classifier_pipeline.transform(full_data.detach().cpu().numpy())
    full_labels = full_labels.cpu().numpy()

    train_data, test_data, train_labels, test_labels = partition_data(options, full_data, full_labels)

    classifier.fit(train_data, train_labels)
    return classifier.score(test_data, test_labels)

def SoftCosAttn(options, full_data, full_labels):
    full_data = options.pre_classifier_pipeline.transform(full_data.detach().cpu().numpy())
    full_labels = full_labels.cpu().numpy()

    # We normalize here before the cosine similarity matrix
    norm = Normalizer(copy=False)
    full_data = norm.transform(full_data)

    train_data, test_data, train_labels, test_labels = partition_data(options, full_data, full_labels)

    # https://arxiv.org/pdf/1606.04080.pdf
    # The simplest attention kernel from Matching Networks for One-Shot Learning
    # softmax over cosine similarity

    # cosine similarity
    cos_sim = numpy.matmul(test_data, train_data.T)
    attn_kernel = softmax(cos_sim, axis=1)

    # predict labels based on argmax instead of the mean
    predict_labels = train_labels[attn_kernel.argmax(axis=1)]

    return accuracy_score(y_true=test_labels, y_pred=predict_labels)
