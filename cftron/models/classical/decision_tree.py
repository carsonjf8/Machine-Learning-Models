import numpy as np
from scipy import stats
from ...utils.metrics import information_gain

class Decision_Tree:
    def __init__(self, n_levels: int) -> None:
        self.n_levels = n_levels
        self.splits = [None] * ((2 ** self.n_levels) - 1)
        self.classification_nodes = [None] * (2 ** self.n_levels)

    def train(self, x_train: np.array, y_train: np.array) -> None:
        self.train_helper(x_train, y_train, 1, 0)
    
    def train_helper(self, x_train: np.array, y_train: np.array, level_num: int, split_index: int) -> None:
        print('train_helper', level_num, split_index)
        # get best feature to split on
        feat_split_index, feat_split_val = self.get_best_split_feature(x_train, y_train)
        # store split feature and value
        self.splits[split_index] = (feat_split_index, feat_split_val)
        # split data base on the best split feature
        x_train_split_left, x_train_split_right, y_train_split_left, y_train_split_right = self.split_data(x_train, y_train, feat_split_index, feat_split_val)

        # add classification node if at the last level
        if level_num == self.n_levels:
            # get left and right path classes
            left_path_class = stats.mode(y_train_split_left)
            right_path_class = stats.mode(y_train_split_right)
            # create classification nodes
            self.classification_nodes[((split_index * 2) + 1) - ((2 ** self.n_levels) - 1)] = left_path_class
            self.classification_nodes[((split_index * 2) + 2) - ((2 ** self.n_levels) - 1)] = right_path_class
            return
        
        # recursively call sub tree splitting
        self.train_helper(x_train_split_left, y_train_split_left, level_num + 1, split_index * 2 + 1)
        self.train_helper(x_train_split_right, y_train_split_right, level_num + 1, split_index * 2 + 2)

    def split_data(self, x_train: np.array, y_train: np.array, feat_index: int, feat_split_val: float) -> tuple[np.array, np.array, np.array, np.array]:
        data_without_feat = np.hstack((x_train[:, 0 : feat_index], x_train[:, feat_index + 1 :]))

        feat_vals = x_train[:, feat_index]
        sorted_feat_vals_indices = feat_vals.argsort()

        split_index = self.get_split_index(sorted_feat_vals_indices, feat_split_val)
        left_split_indices = sorted_feat_vals_indices[: split_index]
        right_split_indices = sorted_feat_vals_indices[split_index :]

        return (data_without_feat[left_split_indices], data_without_feat[right_split_indices], y_train[left_split_indices], y_train[right_split_indices])

    def get_best_split_feature(self, x_train: np.array, y_train: np.array) -> tuple[int, float]:
        best_feat_index = 0
        best_feat_split_val = None
        best_feat_info_gain = 0
        # iterate through each feature
        for i in range(x_train.shape[1]):
            # get feature values
            feat_vals = x_train[:, i]
            # sort feature vals and classes base on feature vals
            sorted_indices = feat_vals.argsort()
            sorted_feat_vals = feat_vals[sorted_indices]
            sorted_y_train = y_train[sorted_indices]

            # get possible split values
            possible_split_vals = np.array([])
            for i in range(sorted_feat_vals.shape[0] - 1):
                # only consider possible splits if it divides different classes
                if sorted_y_train[i] == sorted_y_train[i + 1]:
                    continue
                possible_split_vals = np.append(possible_split_vals, (sorted_feat_vals[i] + sorted_feat_vals[i + 1]) / 2)
            
            for split_val in possible_split_vals:
                # split data based on feature value
                split_index = self.get_split_index(sorted_feat_vals, split_val)
                y_train_left_split = sorted_y_train[: split_index]
                y_train_right_split = sorted_y_train[split_index :]

                # calculate information gain for each split and track the best one
                info_gain = information_gain(y_train_left_split, y_train_right_split)
                if info_gain > best_feat_info_gain:
                    best_feat_index = i
                    best_feat_split_val = split_val
                    best_feat_info_gain = info_gain
        
        return (best_feat_index, best_feat_split_val)
    
    def get_split_index(self, vals: np.array, split_val: float) -> int:
        left = 0
        right = vals.shape[0] - 1
        while True:
            middle = (left + right) // 2
            if vals[middle] < split_val and vals[middle + 1] > split_val:
                return middle + 1
            elif vals[middle] < split_val:
                left = middle
            elif vals[middle] > split_val:
                right = middle

    def predict(self, x_test: np.array) -> np.array:
        predictions = np.zeros(x_test.shape[0])

        pred_index = 0
        for test in x_test:
            index = 0
            while index < len(self.splits):
                split_feat_index, split_feat_val = self.splits[0]
                if test[split_feat_index] < split_feat_val:
                    index = (index * 2) + 1
                else:
                    index = (index * 2) + 2
                
                test = np.hstack((test[: split_feat_index], test[split_feat_index :]))
            index -= ((2 ** self.n_levels) - 1)
            predictions[pred_index] = self.classification_nodes[index]

        return predictions