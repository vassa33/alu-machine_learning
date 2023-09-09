#!/usr/bin/env python3
"""Slice Like A Ninja"""
import numpy as np


def np_slice(matrix, axes={}):
    """
    Slice a NumPy matrix along specific axes.

    Args:
    matrix (numpy.ndarray): The input matrix.
    axes (dict): A dictionary where the key is an axis to slice along,
                 and the value is a tuple representing the slice to make
                 along that axis.

    Returns:
    numpy.ndarray: A new NumPy matrix resulting from the specified slices.
    """
    slices = [slice(None)] * matrix.ndim

    for axis, axis_slice in axes.items():
        slices[axis] = slice(*axis_slice)

    return matrix[tuple(slices)]


if __name__ == "__main__":
    np_slice = __import__('100-slice_like_a_ninja').np_slice
