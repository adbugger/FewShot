from __future__ import print_function, division
import datasets
import random

import torch

class SimpleShotEpisodes:
    def __init__(self, old_options):
        self.dataset = getattr(datasets, old_options.dataset)(old_options).test_set

        # map indices and classes for later sampling
        self.class_indices = dict()
        for idx, (_, label) in enumerate(self.dataset):
            if label in self.class_indices:
                self.class_indices[label].append(idx)
            else:
                self.class_indices[label] = [idx]

    def episode_loader(self, options):
        return torch.utils.data.DataLoader(self.dataset, batch_sampler=self.episode_sampler(options))

    def episode_sampler(self, options):
        for _ in range(options.num_test_tasks):
            classes = random.sample(self.class_indices.keys(), options.n_way)

            train_indices = list()
            test_indices = list()
            for class_id in classes:
                items = random.sample(self.class_indices[class_id], options.k_shot + options.num_query)
                train_indices.extend(items[:options.k_shot])
                test_indices.extend(items[-options.num_query:])


            assert (len(train_indices) == options.k_shot * options.n_way), (
                    f"Needed {options.k_shot * options.n_way} train indices, got {len(train_indices)}.")
            assert (len(test_indices) == options.num_query * options.n_way), (
                    f"Needed {options.num_query * options.n_way} test indices, got {len(test_indices)}.")

            # last n_way * num_query samples are test indices
            yield train_indices + test_indices
