#!/usr/bin/env python3

"""
This module defines a function for calculating the shape of a numpy.ndarray
"""


import numpy as np

def np_shape(matrix):
    """
    Calculate the shape of a numpy.ndarray.

    Args:
    matrix (numpy.ndarray): The input NumPy array.

    Returns:
    tuple: A tuple of integers representing the shape of the array.
    """
    return matrix.shape


if __name__ == "__main__":
    np_shape = __import__('10-ill_use_my_scale').np_shape
