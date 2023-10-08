#!/usr/bin/env python3
"""This module defines a function for performing
convolution on images with multiple channels and various options."""


import numpy as np


def convolve_channels(images, kernel, padding='same', stride=(1, 1)):
    """
    Perform convolution on images with multiple channels.

    Args:
        images (numpy.ndarray): Input images with shape
        (m, h, w, c).
        kernel (numpy.ndarray): Convolution kernel with shape
        (kh, kw, c).
        padding (tuple or str): Padding options: 'same', 'valid',
        or tuple (ph, pw).
        stride (tuple): Stride options: (sh, sw).

    Returns:
        numpy.ndarray: Convolved images with shape (m, h', w'), where
        h' and w' are the height and width of the output images.
    """
    m, h, w, c = images.shape
    kh, kw, _ = kernel.shape
    sh, sw = stride

    # Calculate padding if it's not 'same' or 'valid'
    if padding == 'same':
        ph = ((h - 1) * sh + kh - h) // 2
        pw = ((w - 1) * sw + kw - w) // 2
    elif padding == 'valid':
        ph, pw = 0, 0
    else:
        ph, pw = padding

    # Pad the images with zeros
    padded_images = np.pad(images, ((0, 0), (ph, ph), (pw, pw),
                                    (0, 0)), mode='constant')

    # Calculate output dimensions
    output_h = (h + 2 * ph - kh) // sh + 1
    output_w = (w + 2 * pw - kw) // sw + 1

    # Initialize the output array
    output = np.zeros((m, output_h, output_w))

    for i in range(output_h):
        for j in range(output_w):
            output[:, i, j] = np.sum(
                padded_images[:, i * sh:i * sh + kh,
              j * sw:j * sw + kw, :] * kernel,
                axis=(1, 2, 3)
            )

    return output
