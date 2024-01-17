import numpy as np
from scipy.stats import norm

class Naive_Bayes:
    def __init__(self):
        pass

    def train(self, x_train, y_train):
        self.x_train = x_train # training data (N, F)
        self.y_train = y_train # training labels (N)
        self.num_features = self.x_train.shape[1] # number of features for each data point (1)
        self.classes, self.class_counts = np.unique(self.y_train, return_counts=True) # unique class labels (C), number of data points of each class (C)
        self.num_classes = self.classes.shape[0] # number of classes (1)
        self.class_probabilities = np.divide(self.class_counts, np.sum(self.class_counts)) # probability of each class (C)

        self.id_to_class = {} # map from class id to label
        self.class_to_id = {} # map from class label to id
        for index, val in enumerate(self.classes):
            self.id_to_class[index] = val
            self.class_to_id[val] = index

        # split data into groups based on classes and extract information
        self.data_by_class = [] # list of lists where each the index of each inner list corresponds to a class id
                                # the elements of those inner list are the data points that belong to that class (C, N, F)
        self.class_means = np.zeros((self.num_classes, self.num_features)) # mean value for each feature in each class (C, F)
        self.class_stds = np.zeros((self.num_classes, self.num_features)) # standard deviation value for each feature in each class (C, F)
        for index, val in enumerate(self.classes):
            class_indices = np.argwhere(self.y_train == val)
            class_data = self.x_train[class_indices]
            self.data_by_class.append(class_data)
            # calculate means for each feature
            self.class_means[index] = np.mean(class_data, axis=0)
            # calculate standard deviations for each feature
            self.class_stds[index] = np.std(class_data, axis=0)

    def predict(self, x_test):
        # duplicate data by the number of classes
        x_test = np.expand_dims(x_test, axis=1)
        x_test = np.repeat(x_test, self.num_classes, axis=1)

        # calculate class probabilities for each data point
        data_class_probabilities = self.pdf(x_test)
        data_class_probabilities = np.prod(data_class_probabilities, axis=2)
        data_class_probabilities = np.multiply(data_class_probabilities, self.class_probabilities)

        # aggregate class predictions
        class_id_predictions = np.argmax(data_class_probabilities, axis=1)
        class_predictions = np.array([self.id_to_class[cip] for cip in class_id_predictions])
        return class_predictions

    # calculate the probability density function (PDF) for each feature in each data point
    def pdf(self, x_test):
        base = 1 / (np.multiply(self.class_stds, np.power(2 * np.pi, 0.5)))
        exponent = -np.divide(
            np.power(np.subtract(x_test, self.class_means), 2),
            np.multiply(2, np.power(self.class_stds, 2))
        )
        return np.multiply(base, np.power(np.e, exponent))
