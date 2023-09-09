#!/usr/bin/env python3

"""
Module for performing matrix multiplication using NumPy.
"""


import numpy as np

def mat_mul(mat1, mat2):
    """
    Perform matrix multiplication.

    Args:
        mat1 (list): The first 2D matrix.
        mat2 (list): The second 2D matrix.

    Returns:
        list: A new matrix resulting from the multiplication of mat1 and mat2.
        None: If the two matrices cannot be multiplied.
    """
    # Convert the input matrices to NumPy arrays
    np_mat1 = np.array(mat1)
    np_mat2 = np.array(mat2)

    # Check if mat1 and mat2 can be multiplied
    if np_mat1.shape[1] != np_mat2.shape[0]:
        return None

    # Perform matrix multiplication using NumPy's dot function
    result = np.dot(np_mat1, np_mat2)

    # Convert the result back to a Python list
    return result.tolist()


if __name__ == "__main__":
    mat_mul = __import__('8-ridin_bareback').mat_mul
