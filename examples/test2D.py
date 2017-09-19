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

X = {
    'x' : np.linspace(-3, 3, 40),
    'y' : np.linspace(-3, 3, 40)
    }

x_best, y = optimize(f, X, 'se', 'ei', 10)

print("x_best = {}".format(x_best))
print("f(x_best) = {}".format(y))
