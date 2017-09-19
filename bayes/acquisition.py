from scipy.stats import norm

'''
Implements Probabilty of improvment (POI)
'''
def poi(**kwargs):
    mean = kwargs['mean']
    std = kwargs['std']
    best_y = kwargs['best_y']
    return norm.cdf((best_y - mean) / std)

'''
Implements Upper Confidence Bound (UCB)
'''
def ucb(k):
    def f(**kwargs):
        mean = kwargs['mean']
        std = kwargs['std']
        return mean - k * std
    return f

'''
Implements Expected Improvment (EI)
'''
def ei(**kwargs):
    mean = kwargs['mean']
    std = kwargs['std']
    best_y = kwargs['best_y']

    if std == 0.:
        return 0.
    Z = (mean - best_y) / std
    return (mean - best_y) * norm.cdf(Z) + std * norm.pdf(Z)

'''
Implements dummy random acquisition function
'''
def random(**kwargs):
    return np.random.rand()
