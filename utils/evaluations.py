import numpy as np


def confusion_mat(true, pred, C):
    if len(true) != len(pred):
        raise IndexError("predictions don't match ground truths")

    if max(true) > C - 1 or max(pred) > C - 1:
        raise ValueError("redundant category in Params true, pred!")

    cm = np.zeros((C, C)).astype(np.int)

    indexes = list(zip(np.array(true).astype(np.int), np.array(pred).astype(np.int)))
    for index in indexes:
        cm[index[0], index[1]] += 1

    return cm
