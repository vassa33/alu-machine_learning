#!/usr/bin/env python3

"""
This module defines a function for concatenating two arrays.
"""


def cat_arrays(arr1, arr2):
    """
    Concatenate two arrays.

    Args:
    arr1 (list): The first array.
    arr2 (list): The second array.

    Returns:
    list: A new list containing the elements of arr1 followed by the elements of arr2.
    """
    return arr1 + arr2


if __name__ == "__main__":
    cat_arrays = __import__('6-howdy_partner').cat_arrays
