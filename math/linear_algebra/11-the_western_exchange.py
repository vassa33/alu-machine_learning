#!/usr/bin/env python3

"""
This module defines a function for transposing a numpy.ndarray
"""


import numpy as np

def np_transpose(matrix):
    """
    Transpose a numpy.ndarray without using loops or conditional statements.

    Args:
    matrix (numpy.ndarray): The input matrix to be transposed.

    Returns:
    numpy.ndarray: A new transposed numpy.ndarray.
    """
    return np.transpose(matrix)
