import numpy as np


def to_one_hot(hot_vals, max_val):
    '''Convert an int list of data into one-hot vectors.'''
    return np.eye(max_val)[np.array(hot_vals)]
