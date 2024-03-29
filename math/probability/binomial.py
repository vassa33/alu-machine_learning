#!/usr/bin/env python3
"""A class Binomial representing a Binomial Probability Distribution.
 This distribution is commonly used to model the number of successes (usually denoted as "k")
 in a fixed number of independent Bernoulli trials, where each trial has two possible outcomes:
 success with probability "p" and failure with probability "1-p.""""


def factorial(n):
    """aux func"""
    if n == 0:
        return 1
    else:
        return n * factorial(n - 1)


class Binomial():
    """represents a binomial distribution"""

    def __init__(self, data=None, n=1, p=0.5):

        if data is None:
            if n <= 0:
                raise ValueError("n must be a positive value")
            self.n = int(n)
            if 0 >= p or p >= 1:
                raise ValueError("p must be greater than 0 and less than 1")
            self.p = float(p)
        else:
            if type(data) is not list:
                raise TypeError("data must be a list")
            if len(data) <= 1:
                raise ValueError("data must contain multiple values")

            # find avg
            mean = sum(data) / len(data)
            # find measure of dispersion
            variance = sum((x - mean)**2 for x in data) / (len(data))

            # calc n first
            self.n = round(mean / (1 - (variance / mean)))
            self.p = float(mean / self.n)

    def pmf(self, k):
        """Calculates the value of the PMF for a given num of successes"""

        if type(k) is not int:
            k = int(k)
        if k < 0:
            return 0
        combination = factorial(self.n) / (factorial(k) *
                                           factorial(self.n - k))
        return combination * (self.p ** k) * ((1 - self.p) ** (self.n - k))

    def cdf(self, k):
        """Calculates the value of the CDF for a given num of successes"""
        k = int(k)
        if k < 0:
            return 0
        return sum(self.pmf(i) for i in range(0, k + 1))
