#!/usr/bin/env python3
"""
Defines function that performs forward propagation for bidirectional RNN
"""


import numpy as np

def bi_rnn(bi_cell, X, h_0, h_t):
    """
    Performs forward propagation for a bidirectional RNN.
    
    Args:
        bi_cell: an instance of BidirectionalCell that will be used for the forward propagation
        X: the data to be used, given as a numpy.ndarray of shape (t, m, i)
           - t is the maximum number of time steps
           - m is the batch size
           - i is the dimensionality of the data
        h_0: the initial hidden state in the forward direction, given as a numpy.ndarray of shape (m, h)
           - h is the dimensionality of the hidden state
        h_t: the initial hidden state in the backward direction, given as a numpy.ndarray of shape (m, h)

    Returns:
        H: numpy.ndarray containing all of the concatenated hidden states
        Y: numpy.ndarray containing all of the outputs
    """
    t, m, i = X.shape
    _, h = h_0.shape

    # Initialize hidden states for both directions
    H_f = np.zeros((t, m, h))
    H_b = np.zeros((t, m, h))

    # Forward direction
    h_prev = h_0
    for step in range(t):
        h_prev = bi_cell.forward(h_prev, X[step])
        H_f[step] = h_prev

    # Backward direction
    h_next = h_t
    for step in reversed(range(t)):
        h_next = bi_cell.backward(h_next, X[step])
        H_b[step] = h_next

    # Concatenate the hidden states from both directions
    H = np.concatenate((H_f, H_b), axis=-1)

    # Compute outputs
    Y = np.array([bi_cell.output(H[step]) for step in range(t)])

    return H, Y
