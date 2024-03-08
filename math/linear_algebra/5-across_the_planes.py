#!/usr/bin/env python3

"""
This module defines a function for adding 2D matrices element-wise
"""


def add_matrices2D(mat1, mat2):
    """
    Add two 2D matrices element-wise.

    Args:
    mat1 (list): The first 2D matrix.
    mat2 (list): The second 2D matrix.

    Returns:
    list: A new 2D matrix containing the element-wise sum of mat1 and mat2.
    None: If mat1 and mat2 are not the same shape.
    """
    # Check if mat1 and mat2 have the same shape
    if len(mat1) != len(mat2) or any(len(row1) != len(row2)
                                     for row1, row2 in zip(mat1, mat2)):
        return None

    # Perform element-wise addition
    result = [[a + b for a, b in zip(row1, row2)]
              for row1, row2 in zip(mat1, mat2)]
    return result


if __name__ == "__main__":
    add_matrices2D = __import__('5-across_the_planes').add_matrices2D
