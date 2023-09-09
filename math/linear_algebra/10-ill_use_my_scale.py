#!/usr/bin/env python3

"""
This module defines a function for calculating the shape of a numpy.ndarray
"""


def np_shape(matrix):
    """
    Calculate the shape of a numpy.ndarray without using loops or conditional statements.

    Args:
    matrix (list): The input nested list representing a matrix.

    Returns:
    tuple: A tuple of integers representing the shape of the matrix.
    """
    if not isinstance(matrix, list):
        return ()
    
    def shape_helper(matrix, shape):
        shape.append(len(matrix))
        shape.extend(shape_helper(matrix[0], [])) if matrix else None
        return shape
    
    return tuple(shape_helper(matrix, []))


if __name__ == "__main__":
    np_shape = __import__('10-ill_use_my_scale').np_shape