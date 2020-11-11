import numpy as np
from sklearn.cluster import KMeans
from kmeanstf import KMeansTF
import tensorflow as tf


class AbstractClusteringAlgorithm:
    def __init__(self, **kwargs):
        raise NotImplemented()

    def fit(self, x: [np.ndarray]):
        raise NotImplemented()

    @property
    def labels(self):
        raise NotImplemented()


class KmeanSklearn(AbstractClusteringAlgorithm):
    def __init__(self, **kwargs):
        self.model = KMeans(**kwargs)

    def fit(self, x: [np.ndarray]):
        self.model.fit(x)

    @property
    def labels(self):
        return self.model.labels_


class KMeansTF(AbstractClusteringAlgorithm):
    def __init__(self, **kwargs):
        self.model = KMeansTF(**kwargs)

    def fit(self, x: [np.ndarray]):
        self.model.fit(tf.convert_to_tensor(x, dtype=tf.float32))

    @property
    def labels(self):
        return self.model.labels_
