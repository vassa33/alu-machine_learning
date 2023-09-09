#!/usr/bin/env python3

matrix_shape = __import__('2-size_me_please').matrix_shape

def matrix_shape(matrix):
    """
    Calculate the shape (dimensions) of a matrix.

    Args:
    matrix (list): A nested list representing the matrix.

    Returns:
    list: A list of integers representing the dimensions of the matrix.
    """
    shape = []
    while isinstance(matrix, list):
        shape.append(len(matrix))
        matrix = matrix[0]
    return shape
