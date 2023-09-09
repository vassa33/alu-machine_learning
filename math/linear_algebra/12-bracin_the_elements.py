#!/usr/bin/env python3

"""
This module defines a function for performing element-wise calculations
"""


import numpy as np

def np_elementwise(mat1, mat2):
    """
    Perform element-wise addition, subtraction, multiplication, and division of two numpy.ndarrays.

    Args:
    mat1 (numpy.ndarray): The first input numpy.ndarray.
    mat2 (numpy.ndarray): The second input numpy.ndarray.

    Returns:
    tuple: A tuple containing the element-wise sum, difference, product, and quotient, respectively.
    """
    # Calculate element-wise operations
    add_result = mat1 + mat2
    sub_result = mat1 - mat2
    mul_result = mat1 * mat2
    div_result = mat1 / mat2

    return add_result, sub_result, mul_result, div_result
