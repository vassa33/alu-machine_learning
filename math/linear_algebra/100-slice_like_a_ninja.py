#!/usr/bin/env python3
"""Slice Like A Ninja"""


def np_slice(matrix, axes={}):
    """
    Slices a matrix along specific axes.

    Args:
    matrix (list): The input matrix.
    axes (dict): A dictionary where the key is an axis to slice along,
                 and the value is a tuple representing the slice to make
                 along that axis.

    Returns:
    list: A new matrix resulting from the specified slices.
    """

    def slice_matrix(matrix, axis, indices):
        # Recursively slice the matrix along the specified axis
        if axis == 0:
            return matrix[indices[0]:indices[1]]
        elif axis == 1:
            return [row[indices[0]:indices[1]] for row in matrix]
        else:
            return None

    result = matrix
    for axis, indices in axes.items():
        result = slice_matrix(result, axis, indices)

    return result


# Test cases
if __name__ == "__main__":
    # Test case 1
    mat1 = [[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]]
    print(np_slice(mat1, axes={1: (1, 3)}))  # Should return [[2, 3], [7, 8]]

    # Test case 2
    mat2 = [[[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]],
            [[11, 12, 13, 14, 15], [16, 17, 18, 19, 20]],
            [[21, 22, 23, 24, 25], [26, 27, 28, 29, 30]]]
    print(np_slice(mat2, axes={0: (2,), 2: (None, None, -2)}))
