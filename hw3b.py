"""
ChatGPT and Wikipedia were consulted for proper use of code,
clarity, and formatting.
"""

import math


def gamma_func(a):
    '''
    Wrapper for Python's gamma function.
    '''
    return math.gamma(a)


def Km(m):
    '''
    Computes the constant K_m used in the t-distribution PDF.
    '''
    num = gamma_func(0.5 * m + 0.5)
    den = math.sqrt(m * math.pi) * gamma_func(0.5 * m)
    return num / den


def t_integrand(u, m):
    '''
    Integrand used in the t-distribution integral.
    '''
    return (1 + (u * u) / m) ** (-(m + 1) / 2)


def simpson_integral(m, a, n=1000):
    '''
    Simpson's 1/3 rule integration from 0 to a.
    '''
    if n % 2 == 1:
        n += 1

    h = a / n
    s = t_integrand(0, m) + t_integrand(a, m)

    for i in range(1, n):
        x = i * h
        if i % 2 == 0:
            s += 2 * t_integrand(x, m)
        else:
            s += 4 * t_integrand(x, m)

    return (h / 3) * s


def t_cdf(z, m):
    '''
    Computes the cumulative distribution function
    for the Student t distribution.

    z : t value
    m : degrees of freedom
    '''
    k = Km(m)
    a = abs(z)

    area = simpson_integral(m, a)

    if z >= 0:
        return 0.5 + k * area
    else:
        return 0.5 - k * area