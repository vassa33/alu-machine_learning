#!/usr/bin/env python3
"""The Whole Barn"""


def add_matrices(mat1, mat2):
    """
    Adds two matrices element-wise.

    Args:
    mat1 (list): The first matrix.
    mat2 (list): The second matrix.

    Returns:
    list: A new matrix resulting from element-wise addition.
    None: If matrices are not the same shape.
    """

    def shape_of_matrix(matrix):
        # Recursively calculate the shape of a nested matrix
        if isinstance(matrix, list):
            return [len(matrix)] + shape_of_matrix(matrix[0])
        else:
            return []

    shape1 = shape_of_matrix(mat1)
    shape2 = shape_of_matrix(mat2)

    # Check if the shapes of both matrices match
    if shape1 != shape2:
        return None

    def add_elementwise(matrix1, matrix2):
        # Recursively add elements of matrices
        if isinstance(matrix1, list):
            return [add_elementwise(m1, m2) for m1, m2 in zip(matrix1, matrix2)]
        else:
            return matrix1 + matrix2

    return add_elementwise(mat1, mat2)


# Test cases
if __name__ == "__main__":
    mat1 = [1, 2, 3]
    mat2 = [4, 5, 6]
    print(add_matrices(mat1, mat2))

    mat3 = [[1, 2], [3, 4]]
    mat4 = [[5, 6], [7, 8]]
    print(add_matrices(mat3, mat4))

    mat5 = [[[[1, 2, 3, 4], [5, 6, 7, 8]],
             [[9, 10, 11, 12], [13, 14 ,15, 16]],
             [[17, 18, 19, 20], [21, 22, 23, 24]]]]
    mat6 = [[[[11, 12, 13, 14], [15, 16, 17, 18]],
             [[19, 110, 111, 112], [113, 114 ,115, 116]],
             [[117, 118, 119, 120], [121, 122, 123, 124]]]]
    mat7 = [[[[11, 12, 13, 14], [15, 16, 17, 18]],
             [[117, 118, 119, 120], [121, 122, 123, 124]]]]
    
    print(add_matrices(mat5, mat6))
    print(add_matrices(mat5, mat7))

    # Invalid addition due to shape mismatch
    mat8 = [9, 10]
    print(add_matrices(mat1, mat8))
