#!/usr/bin/env python3
"""A class Normal representing a Normal Distribution.
The normal distribution, also known as the Gaussian distribution,
is characterized by a bell-shaped curve and is symmetrical around the mean.
Many real-world phenomena, such as the heights of individuals in a population
or measurement errors, can be approximated by the normal distribution."""


class Normal():
    """represents a normal distribution"""
    pi = 3.1415926536

    def __init__(self, data=None, mean=0., stddev=1.):

        if data is None:
            if stddev <= 0:
                raise ValueError("stddev must be a positive value")
            else:
                self.stddev = float(stddev)
            self.mean = float(mean)
        else:
            if type(data) is not list:
                raise TypeError("data must be a list")
            if len(data) <= 1:
                raise ValueError("data must contain multiple values")
            self.mean = float(sum(data) / len(data))
            s = sum(map(lambda x: (x - self.mean) ** 2, data))
            self.stddev = (s / len(data)) ** 0.5

    def z_score(self, x):
        """z-score of a given x-value"""
        # z = (x - μ) / σ
        return (x - self.mean) / self.stddev

    def x_value(self, z):
        """x-value of a given z-score"""
        # x = μ + (z * σ)
        return self.mean + (z * self.stddev)

    def pdf(self, x):
        """Calculates the value of the PDF for a given x-value"""
        # f(x) = (1 / (σ * √(2π))) * e^((z^2) / -2)
        pi = 3.1415926536
        e = 2.7182818285
        return (1 / (self.stddev * ((2 * pi) ** 0.5))) *\
               (e ** ((self.z_score(x) ** 2) / -2))

    def err(self, x):
        """Error function"""
        s = x - ((x ** 3)/3) + ((x ** 5)/10) - ((x ** 7)/42) + ((x ** 9)/216)
        return (2 * s) / self.pi ** 0.5

    def cdf(self, x):
        """Calculates the value of the CDF for a given x-value"""
        err_x = (x - self.mean) / (self.stddev * 2 ** 0.5)
        return 0.5 * (1 + self.err(err_x))
