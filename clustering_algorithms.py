import numpy as np




from utils import get_md5sum


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


class KmeanSklearn(AbstractClusteringAlgorithm):
    def __init__(self, **kwargs):
        from sklearn.cluster import KMeans
        self.model = KMeans(**kwargs)

    def fit(self, x: [np.ndarray]):
        self.model.fit(x)

    @property
    def labels_(self):
        return self.model.labels_
    def predict(self,X,cluster_dict):
        to_pred = []
        for x in X:
            hashed = get_md5sum(x.cpu().numpy().tobytes())
            to_pred.append(cluster_dict[str(hashed)].cpu().detach().numpy())

        return self.model.predict(np.array(to_pred))



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
