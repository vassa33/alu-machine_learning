#!/usr/bin/env python3
"""
Bayesian probability: Likelihood
"""


import numpy as np


def likelihood(x, n, P):
    """
    Calculates the likelihood of obtaining a given number of severe side effects
    in a cancer drug trial, assuming that the side effects follow
    a binomial distribution.

    Args:
        x (int): The number of patients who develop severe side effects.
        n (int): The total number of patients observed.
        P (np.ndarray): A 1D numpy.ndarray containing the various hypothetical
        probabilities of developing severe side effects.

    Returns:
        np.ndarray: A 1D numpy.ndarray containing the likelihood of obtaining
        the data, x and n, for each probability in P, respectively.
    """

    # Check if n is a positive integer.
    if not isinstance(n, int) or n <= 0:
        raise ValueError("n must be a positive integer")

    # Check if x is an integer that is greater than or equal to 0.
    if not isinstance(x, int) or x < 0:
        raise ValueError("x must be an integer that is greater than or equal to 0")

    # Check if x is not greater than n.
    if x > n:
        raise ValueError("x cannot be greater than n")

    # Check if P is a 1D numpy.ndarray.
    if not isinstance(P, np.ndarray) or P.ndim != 1:
        raise TypeError("P must be a 1D numpy.ndarray")

    # Check if all values in P are in the range [0, 1].
    if not np.all(P >= 0) or not np.all(P <= 1):
        raise ValueError("All values in P must be in the range [0, 1]")

    # Calculate the likelihood for each probability in P.
    likelihoods = []
    for p in P:
        likelihoods.append(np.binom.pmf(x, n, p))

    return np.array(likelihoods)
