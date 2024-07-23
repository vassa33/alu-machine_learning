#!/usr/bin/env python3
"""
    Computes to policy with a weight of a matrix.
"""
import numpy as np


def policy(matrix, weights):
    """
        The policy we will use
    """
    # For each col of weights we sum wi*si
    z = matrix.dot(weights)
    # For same results
    exp = np.exp(z)
    return exp / np.sum(exp)


def policy_gradient(state, weight):
    """
        The policy gradient
    """
    p = policy(state, weight)
    action = np.random.choice(len(p[0]), p=p[0])

    s = p.reshape(-1, 1)
    softmax = np.diagflat(s) - np.dot(s, s.T)

    dsoftmax = softmax[action, :]

    dlog = dsoftmax / p[0, action]
    grad = state.T.dot(dlog[None, :])

    return action, grad
