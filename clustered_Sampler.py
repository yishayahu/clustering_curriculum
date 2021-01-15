
import numpy as np
import torch
import random
import os
import torchvision.transforms.functional as F


class ClusteredSampler(torch.utils.data.Sampler):
    def __init__(self, data_source, tb,decrease_center = 1):
        self.ds = data_source
        self.cluster_dict = None
        self.hiererchy = []
        self.do_dist = False
        self.decrease_center = decrease_center
        self.n_cluster = int(os.environ['n_cluster'])
        self.center = self.n_cluster
        self.tb = tb



    def create_distribiouns(self, cluster_dict, eval_loss_dict):
        losses = np.zeros(self.n_cluster)
        amounts = np.zeros(self.n_cluster)
        assert self.cluster_dict is None
        assert self.center > 0
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

    def is_in_train(self, labels):
        to_return = [True] * len(labels)
        if self.do_dist and self.cluster_dict:
            for idx, label in enumerate(labels):
                if self.center > self.hiererchy.index(label):
                    to_return[idx] = False
        return to_return

    def __iter__(self):
        indexes = list(range(self.ds.batch_len))
        random.shuffle(indexes)
        assert self.cluster_dict
        print(f"self.center is {self.center}")
        curr_hiererchy = {}
        self.center -= self.decrease_center
        for i in range(self.n_cluster):
            curr_hiererchy[self.hiererchy[i]] = np.exp(-0.2 * abs(self.center - i)) if i < self.center else 1
        diffs = {}
        for i in range(self.n_cluster):
            diffs[i] = []
        for idx in indexes:
            (img,image_index), label = self.ds[idx]
            if self.center >= 0:
                cluster = self.cluster_dict[image_index]
                assert cluster in curr_hiererchy
                try:
                    if len(diffs[self.hiererchy.index(cluster)]) < 20:
                        diffs[self.hiererchy.index(cluster)].append(torch.Tensor(img))
                except:
                    print("excpetino 1")
                    print(type(img))
                    pass
                randi = random.random()
                if randi < curr_hiererchy[cluster]:
                    yield idx
            else:
                yield idx
        self.ds.restart()

        for k, v in diffs.items():
            try:
                if v:
                    self.tb.add_images(0, images=v, title=str(k), step=self.n_cluster - self.center)
            except:
                print("excpetino 2")
                print(type(v))
                pass


class RegularSampler(torch.utils.data.Sampler):
    def __init__(self, data_source):
        self.ds = data_source

    def __iter__(self):
        indexes = list(range(self.ds.batch_len))
        random.shuffle(indexes)
        for idx in indexes:
            yield idx
        self.ds.restart()
