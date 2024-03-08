#!/usr/bin/env python3

"""
This module defines a function for concatenating
two 2D matrices along a specific axis.
"""


def cat_matrices2D(mat1, mat2, axis=0):
    """
    Concatenate two matrices along a specific axis.

    Args:
    mat1 (list): The first 2D matrix.
    mat2 (list): The second 2D matrix.
    axis (int): The axis along which to
    concatenate (0 for rows, 1 for columns).

    Returns:
    list: A new matrix containing the concatenated matrices.
    None: If the two matrices cannot be concatenated.
    """
    # Check if the axis is valid
    if axis not in (0, 1):
        return None

    if axis == 0:
        # Concatenate along rows (vertical concatenation)
        if len(mat1[0]) != len(mat2[0]):
            return None
        return mat1 + mat2
    else:
        # Concatenate along columns (horizontal concatenation)
        if len(mat1) != len(mat2):
            return None
        return [row1 + row2 for row1, row2 in zip(mat1, mat2)]


if __name__ == "__main__":
    cat_matrices2D = __import__('7-gettin_cozy').cat_matrices2D
