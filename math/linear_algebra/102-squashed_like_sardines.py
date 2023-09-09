#!/usr/bin/env python3
"""Squashed Like Sardines"""


def cat_matrices(mat1, mat2, axis=0):
    """
    Concatenates two matrices along a specific axis.

    Args:
    mat1 (list): The first matrix.
    mat2 (list): The second matrix.
    axis (int): The axis along which to concatenate. Default is 0.

    Returns:
    list: A new matrix resulting from concatenation.
    None: If concatenation is not possible.
    """

    def shape_of_matrix(matrix):
        # Recursively calculate the shape of a nested matrix
        if isinstance(matrix, list):
            return [len(matrix)] + shape_of_matrix(matrix[0])
        else:
            return []

    shape1 = shape_of_matrix(mat1)
    shape2 = shape_of_matrix(mat2)

    # Check if the shapes of both matrices match along the specified axis
    if shape1[:axis] != shape2[:axis] or shape1[axis + 1:] != shape2[axis + 1:]:
        return None

    def concatenate_matrices(matrix1, matrix2):
        # Recursively concatenate matrices along the specified axis
        if isinstance(matrix1, list):
            return [concatenate_matrices(m1, m2) for m1, m2 in zip(matrix1, matrix2)]
        else:
            return matrix1 + matrix2

    if axis == 0:
        return concatenate_matrices(mat1, mat2)
    elif axis == 1:
        return [row1 + row2 for row1, row2 in zip(mat1, mat2)]
    else:
        return None


# Test cases
if __name__ == "__main__":
    mat1 = [[11, 22, 33], [44, 55, 66]]
    mat2 = [[1, 2, 3], [4, 5, 6]]
    print(cat_matrices(mat1, mat2))
    print(cat_matrices(mat1, mat2, axis=1))
    print(cat_matrices(mat1, mat2, axis=2))

    mat3 = [[[[1, 2, 3, 4], [5, 6, 7, 8]],
             [[9, 10, 11, 12], [13, 14, 15, 16]],
             [[17, 18, 19, 20], [21, 22, 23, 24]]]]
    mat4 = [[[[11, 12, 13, 14], [15, 16, 17, 18]],
             [[19, 110, 111, 112], [113, 114, 115, 116]],
             [[117, 118, 119, 120], [121, 122, 123, 124]]]]
    
    print(cat_matrices(mat3, mat4, axis=3))
