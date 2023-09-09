#!/usr/bin/env python3

"""
This module defines a function for adding two arrays element-wise
"""


def add_arrays(arr1, arr2):
    """
    Add two arrays element-wise.

    Args:
    arr1 (list): The first array.
    arr2 (list): The second array.

    Returns:
    list: A new list containing the element-wise sum of arr1 and arr2.
    None: If arr1 and arr2 are not the same shape.
    """
    if len(arr1) != len(arr2):
        return None

    result = [a + b for a, b in zip(arr1, arr2)]
    return result


if __name__ == "__main__":
    add_arrays = __import__('4-line_up').add_arrays
