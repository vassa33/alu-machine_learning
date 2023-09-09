#!/usr/bin/env python3

"""
This module defines a function for calculating the transpose of a 2D matrix
"""


def matrix_transpose(matrix):
    """
    Calculate the transpose of a 2D matrix.

    Args:
    matrix (list): The input 2D matrix.

    Returns:
    list: The transpose of the input matrix.
    """
    # Use zip to transpose rows and columns
    transposed_matrix = [list(row) for row in zip(*matrix)]
    return transposed_matrix


if __name__ == "__main__":
    matrix_transpose = __import__('3-flip_me_over').matrix_transpose
