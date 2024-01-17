import numpy as np

class SVM:
    def __init__(self):
        pass

    def train(self, x_train, y_train, epochs, learning_rate, regularization_constant):
        self.x_train = x_train # training data (N, F)
        self.y_train = y_train # training labels (N)
        self.num_features = self.x_train.shape[1] # number of features in each data point (1)
        self.classes = np.unique(self.y_train) # unique class labels (C)
        self.num_classes = self.classes.shape[0] # number of classes (1)
        self.weights = np.zeros((self.num_classes, self.num_features + 1)) # weights for OVR classifier (C, F + 1)

        # add bias term
        x_train = np.hstack((x_train, np.ones((x_train.shape[0], 1)))) # (N, F + 1)
        # training loop
        for _ in range(epochs):
            for (x_t, y_t) in zip(x_train, y_train): # ((1, F + 1), (1))
                mat_product = np.matmul(x_t, self.weights.T) # (C)

                # update all weights
                self.weights = np.subtract(self.weights, np.multiply(2 * learning_rate * regularization_constant, self.weights))

                # extra update for misclassifications
                for class_id in range(self.num_classes):
                    if class_id == y_t:
                        if mat_product[class_id] < 1:
                            self.weights[class_id] = np.add(self.weights[class_id], np.multiply(learning_rate, x_t))
                    else:
                        if mat_product[class_id] > -1:
                            self.weights[class_id] = np.add(self.weights[class_id], np.multiply(-learning_rate, x_t))

    def predict(self, x_test): # (N, F)
        x_test = np.hstack((x_test, np.ones((x_test.shape[0], 1)))) # (N, F + 1)
        predictions = np.matmul(x_test, self.weights.T) # (N, C)
        predictions = np.argmax(predictions, axis=1) # (N)
        return predictions
