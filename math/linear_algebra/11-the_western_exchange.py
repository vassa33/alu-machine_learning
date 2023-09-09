#!/usr/bin/env python3

"""
This module defines a function for transposing a numpy.ndarray
"""


def np_transpose(matrix):
    """
    Transpose a numpy.ndarray without using loops, conditional statements, or imports.

    Args:
    matrix (numpy.ndarray): The input matrix to be transposed.

    Returns:
    numpy.ndarray: A new transposed numpy.ndarray.
    """
    # Determine the shape of the input matrix
    shape = matrix.shape

    # Create a new matrix with transposed shape
    transposed = np.empty(shape[::-1], dtype=matrix.dtype)

    # Calculate the index mapping for transposition
    index_map = np.arange(shape[0])[:, np.newaxis], np.arange(shape[1])

    # Rearrange the elements based on the index mapping
    transposed[index_map] = matrix

    return transposed
