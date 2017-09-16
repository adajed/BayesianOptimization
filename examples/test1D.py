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

X = {'x' : np.linspace(-2, 3, 100)}
x_best, y = optimize(f, X, kernel, 'ucb', 10)

print("x_best = {}".format(x_best))
print("f(x_best) = {}".format(y))
