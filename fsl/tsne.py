import torch
import numpy as np

import datasets
from arguments import parse_args
from utils import ( get_loader, get_loader, get_gpu_ids )
from models import get_model
from episode_strat import SimpleShotEpisodes

from tsnecuda import TSNE

import matplotlib.pyplot as plt

@torch.no_grad()
def do_tsne(options):
    options.shuffle = False
    options.batch_size = 2048
    options.cuda_device = f"cuda:{get_gpu_ids()[0]}"

    # Load dataset into memory
    dataset = getattr(datasets, options.dataset)(options).test_set
    num = len(dataset)

    # Get the model
    model = get_model(options)
    proj_dim = options.projection_dim

    X = np.empty((num, proj_dim))
    y = np.empty(num)

    # Get the dataloader
    loader = get_loader(dataset, options)

    idx = 0
    for (batch, labels) in loader:
        num_batch = batch.shape[0]

        X[idx:idx+num_batch] = model(batch).cpu().numpy()
        y[idx:idx+num_batch] = labels.cpu().numpy()

        idx += num_batch

    X_new = TSNE().fit_transform(X)
    return (X_new, y)

@torch.no_grad()
def do_tsne_on_episode(options):
    options.cuda_device = f"cuda:{get_gpu_ids()[0]}"
    episode_loader = SimpleShotEpisodes(options).episode_loader(options)
    model = get_model(options)

    idx = 0
    for (batch, labels) in episode_loader:
        X = model(batch).cpu().numpy()
        y = labels.cpu().numpy()
        X_new = TSNE(perplexity=10).fit_transform(X)
        visualize(X_new, y, f"figs/fig{idx:03d}.png")
        idx += 1

def visualize(X, y, save_path):
    unique_labels = np.unique(y)

    plt.figure()
    for i in unique_labels:
        x1 = X[y == i, 0]
        x2 = X[y == i, 1]
        # plt.scatter(x=x1, y=x2, label=int(i), s=1.5, c=cmap[i])
        plt.scatter(x=x1, y=x2, label=int(i))
    # plt.legend()
    # plt.show()
    plt.savefig(save_path)
    plt.close()

if __name__ == "__main__":
    options = parse_args()
    # X, y = do_tsne(options)
    do_tsne_on_episode(options)
    # visualize(X, y)