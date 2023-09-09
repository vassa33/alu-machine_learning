#!/usr/bin/env python3

"""
Module for performing matrix multiplication using NumPy.
"""


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
    # Check if mat1 and mat2 can be multiplied
    if len(mat1[0]) != len(mat2):
        return None

    # Initialize the result matrix with zeros
    result = [[0 for _ in range(len(mat2[0]))] for _ in range(len(mat1))]

    # Perform matrix multiplication
    for i in range(len(mat1)):
        for j in range(len(mat2[0])):
            for k in range(len(mat2)):
                result[i][j] += mat1[i][k] * mat2[k][j]

    return result


if __name__ == "__main__":
    mat_mul = __import__('8-ridin_bareback').mat_mul
