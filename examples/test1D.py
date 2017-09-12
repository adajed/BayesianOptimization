from bayes import optimize

import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt

def f(x):
    _x = x['x']
    return 1.0 / (1.0 + np.exp(-(_x ** 3) + 2 * (_x ** 2) + _x - 1))

def kernel(a, b):
    param = 0.1
    sqdist = a**2 + b**2 - 2*a*b
    return np.exp(-.5 * (1/param) * sqdist)

def ucb(u, stdv, k=0.1):
    return u + k * stdv

def ei(u, stdv, best):
    if stdv == 0.:
        return 0.
    else:
        Z = (u - best) / stdv
        return (u - best) * norm.cdf(Z) + stdv * norm.pdf(Z)

X = {'x' : np.linspace(-2, 3, 100)}
print(optimize(f, X, kernel, ucb, 15))
