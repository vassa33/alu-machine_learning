#!/usr/bin/env python3
"""
Defines function that performs forward propagation for bidirectional RNN
"""


import numpy as np


def bi_rnn(bi_cell, X, h_0, h_t):
    """
    Performs forward propagation for a bidirectional RNN

    Args:
        bi_cell (BidirectionalCell): instance of BidirectionalCell
        X (np.ndarray): input data of shape (t, m, i)
        h_0 (np.ndarray): initial hidden state in the forward direction of shape (m, h)
        h_t (np.ndarray): initial hidden state in the backward direction of shape (m, h)

    Returns:
        H (np.ndarray): concatenated hidden states from both directions for each time step
        Y (np.ndarray): outputs for each time step
    """
    t, m, _ = X.shape
    H_f = np.zeros((t, m, h_0.shape[1]))
    H_b = np.zeros((t, m, h_t.shape[1]))

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

    # Compute the output for each time step using the concatenated hidden states
    Y = np.array([bi_cell.output(H[step])[0] for step in range(t)])

    return H, Y
