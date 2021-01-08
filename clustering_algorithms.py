import numpy as np



class AbstractClusteringAlgorithm:
    def __init__(self, **kwargs):
        raise NotImplemented()

    def fit(self, x: [np.ndarray]):
        raise NotImplemented()

    @property
    def labels_(self):
        raise NotImplemented()
    def predict(self,X,cluster_dict):
        raise NotImplemented()


class KmeanSklearnByBatch(AbstractClusteringAlgorithm):
    def __init__(self, **kwargs):
        from sklearn.cluster import MiniBatchKMeans
        self.model = MiniBatchKMeans(**kwargs)
    def fit(self, x: [np.ndarray]):
        self.model.fit(x)

    @property
    def labels_(self):
        return self.model.labels_
    def partial_fit(self,x: [np.ndarray]):
        self.model.partial_fit(x)
    def predict(self,X,cluster_dict=None,from_image = False):

        to_pred = X
        return self.model.predict(np.array(to_pred,dtype=np.double()))

class KmeanSklearn(AbstractClusteringAlgorithm):
    def __init__(self, **kwargs):
        from sklearn.cluster import KMeans
        self.model = KMeans(**kwargs)

    def fit(self, x: [np.ndarray]):
        self.model.fit(x)

    @property
    def labels_(self):
        return self.model.labels_
    def predict(self,X,cluster_dict=None,from_image = False):

        to_pred = X
        return self.model.predict(np.array(to_pred,dtype=np.double()))



class BirchSklearn(AbstractClusteringAlgorithm):
    def __init__(self, **kwargs):
        from sklearn.cluster import Birch
        self.model = Birch(**kwargs)

    def fit(self, x: [np.ndarray]):
        self.model.fit(x)

    @property
    def labels_(self):
        return self.model.labels_


class KMeansTF(AbstractClusteringAlgorithm):
    def __init__(self, **kwargs):
        import tensorflow as tf
        from kmeanstf import KMeansTF
        self.model = KMeansTF(**kwargs)

    def fit(self, x: [np.ndarray]):
        self.model.fit(tf.convert_to_tensor(x, dtype=tf.float32))

    @property
    def labels_(self):
        return self.model.labels_
