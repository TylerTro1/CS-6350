''' This file contains a utility functions.
    You don't need to change this file.
'''

from typing import Tuple

import numpy as np


def shuffle_data(x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    '''
    Helper function to shuffle two np.ndarrays s.t. if x[i] <- x[j] after shuffling,
    y[i] <- y[j] after shuffling for all i, j.

    Args:
        x (np.ndarray): the first array
        y (np.ndarray): the second array

    Returns
        (np.ndarray, np.ndarray): tuple of shuffled x and y
    '''

    assert len(x) == len(y), f'{len(x)=} and {len(y)=} must have the same length in dimension 0'
    p = np.random.permutation(len(x))
    return x[p], y[p]


def clip(x: np.ndarray, max_abs_value: float = 10000) -> np.ndarray:
    '''
    Helper function for clipping very large (or very small values)

    Args:
        x (np.ndarray): the value to be clipped. Can be an np.ndarray or a single float.
        max_abs_value (float): the maximum value that |x| will have after clipping s.t. -max_abs_value <= x <= max_abs_value

    Returns:
        np.ndarray: an np.ndarray containing the clipped values. Will be a float if x is a float.
    '''

    return np.minimum(np.maximum(x, -abs(max_abs_value)), abs(max_abs_value))
