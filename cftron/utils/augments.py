import numpy as np

def one_hot_encoder(labels):
    labels = np.array(labels)
    n_classes = np.max(labels) + 1
    ohe_labels = np.zeros((labels.size, n_classes))
    ohe_labels[np.arange(labels.size), labels] = 1
    return ohe_labels

def normalize(data, epsilon=1e-12):
    return (data - np.min(data)) / np.max((np.max(data) - np.min(data), epsilon))

def shuffle(*arrs):
    output_arrs = []
    shuffle_order = np.random.randint(arrs[0].shape[0], size=arrs[0].shape[0])
    for arr in arrs:
        output_arrs.append(arr[shuffle_order])
    return output_arrs
