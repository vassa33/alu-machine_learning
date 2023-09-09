#!/usr/bin/env python3
"""Squashed Like Sardines"""


def cat_matrices(mat1, mat2, axis=0):
    """
    Concatenate two matrices along a specific axis.

    Args:
    mat1 (list): The first matrix.
    mat2 (list): The second matrix.
    axis (int): The axis along which to concatenate. Default is 0.

    Returns:
    list: A new matrix resulting from concatenation along the specified axis.
    None: If matrices cannot be concatenated due to shape mismatch.
    """

    def shape_of_matrix(matrix):
        # Recursively calculate the shape of a nested matrix
        if isinstance(matrix, list):
            return [len(matrix)] + shape_of_matrix(matrix[0])
        else:
            return []

    shape1 = shape_of_matrix(mat1)
    shape2 = shape_of_matrix(mat2)

    if axis >= len(shape1) or axis >= len(shape2):
        return None

    # Check if dimensions along the specified axis match
    if shape1[axis] != shape2[axis]:
        return None

    # Concatenate matrices along the specified axis
    if axis == 0:
        return mat1 + mat2
    else:
        return [cat_matrices(mat1[i], mat2[i], axis - 1) for i in range(len(mat1))]

# Test cases
if __name__ == "__main__":
    mat1 = [1, 2, 3]
    mat2 = [4, 5, 6]
    print(cat_matrices(mat1, mat2))

    mat1 = [[1, 2], [3, 4]]
    mat2 = [[5, 6], [7, 8]]
    print(cat_matrices(mat1, mat2))

    mat3 = [[[[1, 2, 3, 4], [5, 6, 7, 8]],
             [[9, 10, 11, 12], [13, 14 ,15, 16]],
             [[17, 18, 19, 20], [21, 22, 23, 24]]]]
    mat4 = [[[[11, 12, 13, 14], [15, 16, 17, 18]],
             [[19, 110, 111, 112], [113, 114 ,115, 116]],
             [[117, 118, 119, 120], [121, 122, 123, 124]]]]
    mat5 = [[[[11, 12, 13, 14], [15, 16, 17, 18]],
             [[117, 118, 119, 120], [121, 122, 123, 124]]]]
    
    print(cat_matrices(mat3, mat4, axis=3))
    print(cat_matrices(mat3, mat5, axis=1))

    # Invalid concatenation due to shape mismatch
    mat6 = [9, 10]
    print(cat_matrices(mat1, mat6))
