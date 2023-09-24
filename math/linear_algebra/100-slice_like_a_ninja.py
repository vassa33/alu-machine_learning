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
        if axis < len(matrix):
            if len(indices) == 1:
                return matrix[indices[0]]
            elif len(indices) == 2:
                return matrix[indices[0]:indices[1]]
        return None

    result = matrix
    for axis, indices in axes.items():
        result = slice_matrix(result, axis, indices)

    return result


# Test cases
if __name__ == "__main__":
    # Test case 1 - Slice along axis=0
    mat1 = [[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]]
    print(np_slice(mat1, axes={0: (1,)}))  # Should return [6, 7, 8, 9, 10]

    # Test case 2 - Slice along axis=0, axis=1
    mat2 = [[[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]],
            [[11, 12, 13, 14, 15], [16, 17, 18, 19, 20]],
            [[21, 22, 23, 24, 25], [26, 27, 28, 29, 30]]]
    print(np_slice(mat2, axes={0: (2,), 1: (1,)}))
    # Should return [[17, 18, 19, 20]]

    # Test case 3 - Slice along axis=0, axis=2
    mat3 = [[[1, 2, 3], [4, 5, 6]],
            [[7, 8, 9], [10, 11, 12]],
            [[13, 14, 15], [16, 17, 18]]]
    print(np_slice(mat3, axes={0: (1,), 2: (1, None)}))
    # Should return [[[8, 9], [11, 12]], [[14, 15], [17, 18]]]

    # Test case 4 - Slice along axis=0, axis=3, axis=5
    mat4 = [[[[1, 2], [3, 4]], [[5, 6], [7, 8]]],
            [[[9, 10], [11, 12]], [[13, 14], [15, 16]]]]
    print(np_slice(mat4, axes={0: (1,), 3: (None,), 5: (None,)}))
    # Should return [[[[10], [12]], [[14], [16]]]]

    # Test case 5 - Slice along axis=1
    mat5 = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    print(np_slice(mat5, axes={1: (0, 2)}))
    # Should return [[1, 2], [4, 5], [7, 8]]

    # Test case 6 - Slice along axis=3 (Invalid axis, should return None)
    mat6 = [[[1, 2], [3, 4]], [[5, 6], [7, 8]]]
    print(np_slice(mat6, axes={3: (0, 2)}))  # Should return None

    # Test case 7 - Slice along axis=1, axis=3
    mat7 = [[[[1, 2], [3, 4]], [[5, 6], [7, 8]]],
            [[[9, 10], [11, 12]], [[13, 14], [15, 16]]]]
    print(np_slice(mat7, axes={1: (1,), 3: (None, 1)}))
    # Should return [[[4], [8]], [[12], [16]]]

    # Test case 8 - Slice using one parameter
    mat8 = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    print(np_slice(mat8, axes={0: (1,)}))  # Should return [[4, 5, 6]]

    # Test case 9 - Slice using two parameters
    mat9 = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    print(np_slice(mat9, axes={0: (1, 3)}))  # Should return [[4, 5, 6], [7, 8, 9]]
