from bayes import optimize

import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt

def f(x):
    _x = x['x']
    _y = x['y']
    return _y / (1.0 + np.exp(-(_x ** 3) + 2 * (_x ** 2) + _x - 1)) + np.sin(2*_y) - 0.1 * _y

def kernel(a, b):
    param = 0.1
    sqdist = np.sum((a - b) ** 2)
    return np.exp(-.5 * (1/param) * sqdist)

def ucb(u, stdv, k=0.1):
    return u + k * stdv

def ei(u, stdv, best):
    if stdv == 0.:
        return 0.
    else:
        Z = (u - best) / stdv
        return (u - best) * norm.cdf(Z) + stdv * norm.pdf(Z)

X = {
    'x' : np.linspace(-3, 3, 40),
    'y' : np.linspace(-3, 3, 40)
    }
print(optimize(f, X, kernel, ei, 100))
