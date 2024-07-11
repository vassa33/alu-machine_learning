#!/usr/bin/env python3
"""
    Calculates the gradients
"""
import numpy as np
Q_affinities = __import__('5-Q_affinities').Q_affinities


def grads(Y, P):
    """
        Calculates the gradients of Y
        :param Y: numpy.ndarray of shape (n, ndim) containing low
        dimensional transformation of X
        :param P: numpy.ndarray of shape (n, n) containing P affinities of X
        :return: (dY, Q)
            dY is numpy.ndarray of shape (n, ndim) containing gradients of Y
            Q is numpy.ndarray of shape (n, n) containing Q affinities of Y
    """
    (n, ndim) = Y.shape
    dY = np.zeros((n, ndim))
    Q, num = Q_affinities(Y)
    PQ = P - Q

    for i in range(n):
        dY[i, :] = np.sum(
            np.tile(PQ[:, i] * num[:, i], (ndim, 1)).T * (Y[i, :] - Y), 0)

    return dY, Q
