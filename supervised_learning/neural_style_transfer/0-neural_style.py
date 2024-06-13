#!/usr/bin/env python3
"""
Defines the NST class for performing Neural Style Transfer
"""

import numpy as np
import tensorflow as tf


class NST:
    """
    NST class for Neural Style Transfer

    Public class attributes:
        style_layers: list of layers for style extraction
        content_layer: layer for content extraction
    """

    style_layers = ['block1_conv1', 'block2_conv1', 'block3_conv1',
                    'block4_conv1', 'block5_conv1']
    content_layer = 'block5_conv2'

    def __init__(self, style_image, content_image, alpha=1e4, beta=1):
        """
        Initializes NST instance

        Args:
            style_image (np.ndarray): Style reference image
            content_image (np.ndarray): Content reference image
            alpha (float): Weight for content cost
            beta (float): Weight for style cost

        Raises:
            TypeError: If the inputs are not valid
        """
        if not isinstance(style_image, np.ndarray) or style_image.shape[-1] != 3:
            raise TypeError("style_image must be a numpy.ndarray with shape (h, w, 3)")
        if not isinstance(content_image, np.ndarray) or content_image.shape[-1] != 3:
            raise TypeError("content_image must be a numpy.ndarray with shape (h, w, 3)")
        if not isinstance(alpha, (int, float)) or alpha < 0:
            raise TypeError("alpha must be a non-negative number")
        if not isinstance(beta, (int, float)) or beta < 0:
            raise TypeError("beta must be a non-negative number")

        tf.compat.v1.enable_eager_execution()

        self.style_image = self.scale_image(style_image)
        self.content_image = self.scale_image(content_image)
        self.alpha = alpha
        self.beta = beta

    @staticmethod
    def scale_image(image):
        """
        Rescales an image such that its pixel values are between 0 and 1
        and its largest side is 512 pixels.

        Args:
            image (np.ndarray): Image to rescale

        Returns:
            tf.Tensor: Scaled image

        Raises:
            TypeError: If the input is not a valid image
        """
        if not isinstance(image, np.ndarray) or image.shape[-1] != 3:
            raise TypeError("image must be a numpy.ndarray with shape (h, w, 3)")

        max_dim = 512
        long_dim = max(image.shape[:-1])
        scale = max_dim / long_dim
        new_shape = (int(image.shape[0] * scale), int(image.shape[1] * scale))

        image = tf.image.resize(image[tf.newaxis, :], new_shape, method='bicubic')
        image = image / 255.0
        image = tf.clip_by_value(image, 0, 1)

        return image
