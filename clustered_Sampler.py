from utils import get_md5sum
import numpy as np
import torch
import random
import os
import torchvision.transforms.functional as F


class ClusteredSampler(torch.utils.data.Sampler):
    def __init__(self, data_source, start_clustering, end_clustering):
        self.start_clustering = start_clustering
        self.end_clustering = end_clustering
        self.ds = data_source
        self.cluster_dict = None
        self.hiererchy = []
        self.do_dist = False
        self.center = 0
        self.n_cluster = int(os.environ['n_cluster'])

    def create_distribiouns(self, cluster_dict, eval_loss_dict, step):
        losses = np.zeros(self.n_cluster)
        amounts = np.zeros(self.n_cluster)
        self.step = step
        self.do_dist = self.start_clustering <= step <= self.end_clustering
        if self.do_dist:
            for k, v in eval_loss_dict.items():
                losses[cluster_dict[k]] += v
                amounts[cluster_dict[k]] += 1
            losses_mean = np.mean(losses)
            for idx in range(len(amounts)):
                if amounts[idx] == 0:
                    assert losses[idx] == 0
                    losses[idx] = losses_mean
                    amounts[idx] += 1
            while len(self.hiererchy) < self.n_cluster:
                max_idx = np.argmax(losses)
                self.hiererchy.append(max_idx)
                losses[max_idx] = -1
            self.cluster_dict = cluster_dict

    def __iter__(self):
        indexes = list(range(len(self.ds)))
        random.shuffle(indexes)
        if self.do_dist:

            curr_hiererchy = {}
            for i in range(self.n_cluster):
                curr_hiererchy[self.hiererchy[i]] = np.exp(-0.2 * abs(self.center-i))
            self.center += 1
            self.center = min(self.center, self.n_cluster)
        for idx in indexes:
            img, label = self.ds[idx]
            if self.do_dist and self.cluster_dict:
                hashed = get_md5sum(img.cpu().numpy().tobytes())
                cluster = self.cluster_dict[str(hashed)]
                assert cluster in curr_hiererchy
                if random.random() < curr_hiererchy[cluster]:
                    yield idx
            else:
                yield idx
        # raise StopIteration
