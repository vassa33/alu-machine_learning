#!/usr/bin/env python3
"""
Defines the NST class for neural style transfer
"""

import numpy as np
import tensorflow as tf


class NST:
    """
    A class to perform neural style transfer.
    
    Attributes:
        style_layers (list): List of layer names for style extraction.
        content_layer (str): Layer name for content extraction.
    """

    style_layers = ['block1_conv1', 'block2_conv1', 'block3_conv1',
                    'block4_conv1', 'block5_conv1']
    content_layer = 'block5_conv2'

    def __init__(self, style_image, content_image, alpha=1e4, beta=1):
        """
        Initialize the NST class with style and content images, and the weights.
        
        Args:
            style_image (numpy.ndarray): The image used as style reference.
            content_image (numpy.ndarray): The image used as content reference.
            alpha (float): The weight for content cost.
            beta (float): The weight for style cost.
            
        Raises:
            TypeError: If style_image or content_image is not a numpy.ndarray 
                       with shape (h, w, 3).
            TypeError: If alpha or beta is not a non-negative number.
        """
        
        # Enable eager execution in TensorFlow
        tf.enable_eager_execution()
        
        # Validate style_image
        if not isinstance(style_image, np.ndarray):
            raise TypeError("style_image must be a numpy.ndarray with shape (h, w, 3)")
        if style_image.ndim != 3 or style_image.shape[-1] != 3:
            raise TypeError("style_image must be a numpy.ndarray with shape (h, w, 3)")
        
        # Validate content_image
        if not isinstance(content_image, np.ndarray):
            raise TypeError("content_image must be a numpy.ndarray with shape (h, w, 3)")
        if content_image.ndim != 3 or content_image.shape[-1] != 3:
            raise TypeError("content_image must be a numpy.ndarray with shape (h, w, 3)")
        
        # Validate alpha
        if not isinstance(alpha, (int, float)) or alpha < 0:
            raise TypeError("alpha must be a non-negative number")
        
        # Validate beta
        if not isinstance(beta, (int, float)) or beta < 0:
            raise TypeError("beta must be a non-negative number")

        # Set instance attributes
        self.style_image = self.scale_image(style_image)  # Preprocessed style image
        self.content_image = self.scale_image(content_image)  # Preprocessed content image
        self.alpha = alpha  # Weight for content cost
        self.beta = beta  # Weight for style cost

    @staticmethod
    def scale_image(image):
        """
        Rescales an image so that its pixels values are between 0 and 1 
        and its largest side is 512 pixels.
        
        Args:
            image (numpy.ndarray): Image to be rescaled.
            
        Raises:
            TypeError: If image is not a numpy.ndarray with shape (h, w, 3).
        
        Returns:
            tensorflow.Tensor: The scaled image with shape (1, h_new, w_new, 3).
        """
        
        if not isinstance(image, np.ndarray):
            raise TypeError("image must be a numpy.ndarray with shape (h, w, 3)")
        if image.ndim != 3 or image.shape[-1] != 3:
            raise TypeError("image must be a numpy.ndarray with shape (h, w, 3)")
        
        # Determine new shape while maintaining aspect ratio
        max_dim = 512
        long_dim = max(image.shape[:-1])
        scale = max_dim / long_dim
        new_shape = tuple(map(lambda x: int(scale * x), image.shape[:-1]))

        # Convert image to tensor and add batch dimension
        image = tf.expand_dims(image, axis=0)
        
        # Resize image using bicubic interpolation
        image = tf.image.resize(image, new_shape, method='bicubic')
        
        # Normalize pixel values to range [0, 1]
        image = image / 255.0
        
        # Clip pixel values to ensure they are within [0, 1]
        image = tf.clip_by_value(image, clip_value_min=0, clip_value_max=1)
        
        return image
