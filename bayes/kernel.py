import numpy as np

def r_squared(theta):
    t = np.array(theta, dtype=float)
    def f(x, y):
        return np.sum(((x - y) / t) ** 2)
    return f

'''
Implements squared exponential kernel
'''
def Kse(theta):
    r2 = r_squared(theta[1:])
    def f(x, y):
        return theta[0] * np.exp(-0.5 * r2(x, y))
    return f

'''
Implements Matern 5/2 kernel
'''
def Km52(theta):
    r2 = r_squared(theta[1:])
    def f(x, y):
        d = r2(x, y)

        res = 1. + np.sqrt(5. * d) + (5. + d) / 3.
        res *= np.exp(-np.sqrt(5. * d))
        return float(theta[0]) * res
    return f
