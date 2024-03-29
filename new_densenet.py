import gc
import os

from torch import nn
import torch.nn.functional as F
from torch import Tensor
import torch
import numpy as np


class DenseNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, n_layers):
        super(DenseNetBlock, self).__init__()
        self._layers = nn.ModuleList()
        self.n_layers = n_layers
        for i in range(n_layers):
            if i == 0:
                conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(3, 3), padding=1)
            else:
                conv = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=(3, 3), padding=1)
            bn = nn.BatchNorm2d(out_channels)
            relu = nn.ReLU()
            self._layers += [conv, bn, relu]

    def forward(self, x: Tensor) -> Tensor:
        out = x
        for layer in self._layers:
            out = layer(out)
        return out


class DenseNet(nn.Module):
    def __init__(self, n_classes, clustering_algorithm=None):
        super(DenseNet, self).__init__()
        self.blocks = nn.ModuleList()
        self.bns = nn.ModuleList()
        self.blocks.append(DenseNetBlock(3, 32, 1))
        self.bns.append(None)
        self.blocks.append(DenseNetBlock(32, 128, 4))
        self.bns.append(nn.BatchNorm2d(160))
        self.blocks.append(DenseNetBlock(160, 256, 4))
        self.bns.append(nn.BatchNorm2d(416))
        self.blocks.append(DenseNetBlock(416, 512, 4))
        self.bns.append(nn.BatchNorm2d(928))

        self.last_conv = nn.Conv2d(928, n_classes, kernel_size=(1, 1))
        self.relu = nn.ReLU()
        self.maxpool2d = nn.MaxPool2d(kernel_size=2)
        self.softmax  = nn.LogSoftmax(-1)
        # clustering area
        self.cluster_dict = {}
        self.arrays_to_fit = []
        self.keys_to_fit = []
        self.clustering_algorithm = clustering_algorithm  #
        self.clustering_done = False
        self.test_time = False

    def do_clustering(self):
        return self.clustering_algorithm is not None

    def test_time_activate(self):
        self.test_time = True

    def done_test(self):
        self.test_time = False

    def get_clusters(self):
        if not self.clustering_algorithm:
            raise Exception("get clusters need clustering algo that is not None")
        self.clustering_done = True
        if self.arrays_to_fit:
            labels = self.clustering_algorithm.predict(self.arrays_to_fit)
            assert len(self.keys_to_fit) == len(labels)
            for k, l in zip(self.keys_to_fit, labels):
                self.cluster_dict[k] = int(l)
        return self.cluster_dict

    def forward(self, x: Tensor, image_indexes) -> Tensor:
        out = x
        for block, bn in zip(self.blocks, self.bns):
            skip = out
            out = block(out)
            if block.n_layers == 1:
                continue
            out = torch.cat([skip, out], dim=1)
            skip = None
            out = bn(out)
            out = self.relu(out)
            out = self.maxpool2d(out)
        if self.clustering_algorithm is not None and not self.clustering_done:
            keys = []
            arrays = []
            for i, image_index in enumerate(image_indexes):
                keys.append(int(image_index))
                arrays.append(out[i].cpu().detach().flatten().numpy().astype(np.int16))
            if self.test_time:
                labels = self.clustering_algorithm.predict(arrays)
                for k, l in zip(keys, labels):
                    self.cluster_dict[k] = int(l)
            else:
                self.keys_to_fit += keys
                self.arrays_to_fit += arrays
                if len(self.arrays_to_fit) >= int(os.environ["n_cluster"]):
                    self.clustering_algorithm.partial_fit(self.arrays_to_fit)
                    labels = self.clustering_algorithm.predict(self.arrays_to_fit)
                    assert len(self.keys_to_fit) == len(labels)
                    for k, l in zip(self.keys_to_fit, labels):
                        self.cluster_dict[k] = int(l)

                    self.arrays_to_fit = []
                    self.keys_to_fit = []
            labels = None # do not erase
            keys = []
            arrays = []
            gc.collect()
        out = self.last_conv(out)
        out = F.adaptive_avg_pool2d(out, (1, 1)).squeeze()
        out = self.softmax(out)
        return out
