import numpy as np
from scipy import stats

from models import metrics

class KNN:
    '''
    num_neighbors: int, default=5
        how many data points contribute to the class vote
    distance_metric: {'euclidean', 'manhattan', 'minkowski'}, default='euclidean'
        metric used to calculate distance between points
    p: int, default=2
        exponent value for minkowski distance
    '''
    def __init__(self, num_neighbors=5, distance_metric='euclidean', p=2):
        self.num_neighbors = num_neighbors
        if distance_metric == 'euclidean':
            self.distance_metric = metrics.euclidean_distance
        elif distance_metric == 'manhattan':
            self.distance_metric = metrics.manhattan_distance
        elif distance_metric == 'minkowski':
            self.distance_metric = lambda x1, x2: metrics.minkowski_distance(x1, x2, p)

    def train(self, x_train, y_train):
        self.x_train = x_train
        self.y_train = y_train

    def predict(self, x_test):
        distances = self.distance_metric(np.expand_dims(self.x_train, axis=0), np.expand_dims(x_test, axis=1))
        sorted_indices = np.argsort(distances, axis=1)
        closest_neighbors = sorted_indices[:, :self.num_neighbors]
        neighbor_classes = self.y_train[closest_neighbors]
        predicted_classes = stats.mode(neighbor_classes, axis=1).mode
        return predicted_classes
