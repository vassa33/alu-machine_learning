#!/usr/bin/env python3
"""
Defines class NST for performing Neural Style Transfer tasks
"""

import numpy as np
import tensorflow as tf


class NST:
    """
    Implements Neural Style Transfer

    Class Attributes:
        style_layers: list of layers for style extraction
        content_layer: layer for content extraction

    Instance Attributes:
        style_image: preprocessed style image
        content_image: preprocessed content image
        alpha: weight for content loss
        beta: weight for style loss
        model: Keras model for computing loss

    Methods:
        __init__(self, style_image, content_image, alpha=1e4, beta=1)
        scale_image(image)
        load_model()
    """
    style_layers = ['block1_conv1', 'block2_conv1', 'block3_conv1',
                    'block4_conv1', 'block5_conv1']
    content_layer = 'block5_conv2'

    def __init__(self, style_image, content_image, alpha=1e4, beta=1):
        """
        Initializes the NST class

        Args:
            style_image (np.ndarray): Style reference image
            content_image (np.ndarray): Content reference image
            alpha (float): Weight for content loss
            beta (float): Weight for style loss

        Raises:
            TypeError: If the images or weights are not valid
        """
        if not isinstance(style_image, np.ndarray) or style_image.shape[2] != 3:
            raise TypeError("style_image must be a numpy.ndarray with shape (h, w, 3)")
        if not isinstance(content_image, np.ndarray) or content_image.shape[2] != 3:
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
        self.load_model()

    @staticmethod
    def scale_image(image):
        """
        Rescales an image so that its largest dimension is 512 pixels
        and its values are between 0 and 1

        Args:
            image (np.ndarray): Image to rescale

        Returns:
            tf.Tensor: Scaled image tensor
        """
        if not isinstance(image, np.ndarray) or image.shape[2] != 3:
            raise TypeError("image must be a numpy.ndarray with shape (h, w, 3)")

        h, w, _ = image.shape
        max_dim = 512
        scale = max_dim / max(h, w)
        new_shape = (int(h * scale), int(w * scale))
        image = np.expand_dims(image, axis=0)
        image = tf.image.resize(image, new_shape, method='bicubic')
        image = image / 255.0
        image = tf.clip_by_value(image, 0, 1)
        return image

    def load_model(self):
        """
        Loads a VGG19 model, modifies it for style and content extraction,
        and stores it in the model attribute
        """
        vgg = tf.keras.applications.VGG19(include_top=False, weights='imagenet')
        vgg.trainable = False

        style_outputs = [vgg.get_layer(name).output for name in self.style_layers]
        content_output = vgg.get_layer(self.content_layer).output

        model_outputs = style_outputs + [content_output]
        self.model = tf.keras.models.Model(vgg.input, model_outputs)
