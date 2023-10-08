#!/usr/bin/env python3
"""This module defines a function for performing
strided convolution on grayscale images."""

import numpy as np


def convolve_grayscale(images, kernel, padding='same', stride=(1, 1)):
    """
    Perform a convolution on grayscale images with striding.

    Args:
        images (numpy.ndarray): Input grayscale images with shape (m, h, w).
        kernel (numpy.ndarray): The convolution kernel with shape (kh, kw).
        padding (str or tuple): Padding type or a tuple of (ph, pw).
        stride (tuple): Stride options: (sh, sw).

    Returns:
        numpy.ndarray: Convolved images with shape (m, h', w').
    """
    m, h, w = images.shape
    kh, kw = kernel.shape
    sh, sw = stride

    if padding == 'same':
        ph = int(((h - 1) * sh + kh - h) / 2) + 1
        pw = int(((w - 1) * sw + kw - w) / 2) + 1
    elif padding == 'valid':
        ph, pw = 0, 0
    elif isinstance(padding, tuple) and len(padding) == 2:
        ph, pw = padding
    else:
        raise ValueError("Invalid padding option. Use 'same',
        'valid', or a (ph, pw) tuple.")

    # Pad the input images
    images_padded = np.pad(images, ((0, 0), (ph, ph), (pw, pw)),
                           mode='constant')

    # Calculate output dimensions
    output_h = (h + 2 * ph - kh) // sh + 1
    output_w = (w + 2 * pw - kw) // sw + 1

    # Initialize the output array
    output = np.zeros((m, output_h, output_w))

    # Perform the convolution using two for loops
    for i in range(output_h):
        for j in range(output_w):
            output[:, i, j] = np.sum(
                images_padded[:, i * sh:i * sh + kh,
                j * sw:j * sw + kw] * kernel,
                axis=(1, 2)
            )

    return output
