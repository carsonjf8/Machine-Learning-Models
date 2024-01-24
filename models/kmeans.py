import numpy as np
from numpy.random import default_rng
from models import metrics

class KMeans:
    '''
    num_clusters: int
        how many clusters the data should be put into
    '''
    def __init__(self, num_clusters):
        self.num_clusters = num_clusters
        self.centroids = None
        self.distance_fn = None

    '''
    computes centroids using data
    x_train: NumPy Array
        data to be clustered
    max_iters: int, default=5
        number of times to update the centroids
    distance_fn: function, default=metrics.euclidean_distance
        function used to calculate distance between points and centroids
    '''
    def train(self, x_train, max_iters=20, distance_fn=metrics.euclidean_distance):
        self.distance_fn = distance_fn
        # initialize centroids
        rng = default_rng()
        centroid_indices = rng.choice(self.num_clusters, self.num_clusters, replace=False)
        self.centroids = x_train[centroid_indices]

        for _ in range(max_iters):
            # calculate distance between data and current centroids
            distances = self.distance_fn(np.expand_dims(x_train, axis=1), np.expand_dims(self.centroids, axis=0))
            closest_centroids = np.argmin(distances, axis=1)

            # update centroids
            for centroid_id in range(self.num_clusters):
                centroid_data_points = x_train[np.where(closest_centroids == centroid_id)[0]]
                if centroid_data_points.shape[0] == 0:
                    continue
                updated_centroid = np.mean(centroid_data_points, axis=0)
                self.centroids[centroid_id] = updated_centroid

    '''
    predicts classes using closest centroid
    returns centroid predictions
    x_test: NumPy Array
        data for which to predict cluster
    '''
    def predict(self, x_test):
        distances = self.distance_fn(np.expand_dims(x_test, axis=1), np.expand_dims(self.centroids, axis=0))
        return np.argmin(distances, axis=1)

    '''
    returns a dictionary with class as key and centroid as value
    '''
    def get_class_centroid_dict(self):
        class_dict = {}
        for i in range(self.num_clusters):
            class_dict[i] = self.centroids[i]
        return class_dict
