#!/usr/bin/env python3
"""The Whole Barn"""


def add_matrices(mat1, mat2):
    """
    Add two matrices.

    Args:
    mat1 (list): The first matrix.
    mat2 (list): The second matrix.

    Returns:
    list: A new matrix resulting from the element-wise addition of mat1 and mat2.
    None: If mat1 and mat2 are not the same shape.
    """
    # Check if mat1 and mat2 have the same shape
    shape1 = len(mat1)
    shape2 = len(mat2)

    if shape1 != shape2:
        return None

    # Check if elements in each row have the same shape
    result = []

    for row1, row2 in zip(mat1, mat2):
        if len(row1) != len(row2):
            return None

        result.append([a + b for a, b in zip(row1, row2)])

    return result
