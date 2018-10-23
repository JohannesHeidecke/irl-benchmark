"""Utils module containing general helper functions."""

from typing import Callable, List, Union

import numpy as np
import torch


def to_one_hot(hot_vals: Union[int, List[int], np.ndarray],
               max_val: int,
               zeros_function: Callable = np.zeros
               ) -> Union[np.ndarray, torch.tensor]:
    """ Convert an integer or a list of integers to a one-hot array.

    Parameters
    ----------
    hot_vals: Union[int, List[int], np.ndarray]
        A single integer, or a list / vector of integers, corresponding to the
        hot values which will equal one in the returned array.
    max_val: int
        The maximum possible value in hot_values. All elements in hot_vals have
        to be smaller than max_val (since we start counting at 0).
    zeros_function: Callable
        Controls which function is used to create the array. It should
        be either `numpy.zeros` or `torch.zeros`.

    Returns
    -------
    Union[np.ndarray, torch.tensor]
        Either a numpy array or torch tensor with the one-hot encoded values.
        Type of returned data structure depends on the passed zeros_function.
        The default is numpy array.
        The returned data structure will be of shape (1, max_value) if hot_vals
        is a single integer, and (len(hot_vals), max_value) otherwise.
    """
    assert np.max(hot_vals) < max_val
    assert np.min(hot_vals) >= 0
    try:
        n_rows = len(hot_vals)
        res = zeros_function((n_rows, max_val))
        res[np.arange(n_rows), hot_vals] = 1.
    except TypeError:
        res = zeros_function((max_val, ))
        res[hot_vals] = 1.
    return res
