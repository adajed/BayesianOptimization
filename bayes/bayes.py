import numpy as np
import matplotlib.pyplot as plt
import itertools

from bayes.acquisition import poi, ucb, ei, random
from bayes.kernel import Kse, Km52

class BayesOptimalization:
    def __init__(self, input_space, kernel, acquisition):
        if type(acquisition) == str:
            assert acquisition in ['poi', 'ei', 'ucb', 'rand']
            if acquisition == 'poi':
                acquisition = poi
            elif acquisition == 'ucb':
                acquisition = ucb(1.)
            elif acquisition == 'ei':
                acquisition = ei
            elif acquisition == 'rand':
                acquisition = random

        if type(kernel) == str:
            assert kernel in ['se', 'm52']
            if kernel == 'se':
                kernel = Kse([1.] * (len(input_space) + 1))
            elif kernel == 'm52':
                kernel = Km52([1.] * (len(input_space) + 1))

        self._init = False
        self._kernel = self._kernel_matrix(kernel)
        self._acquisition = acquisition

        self._X_names = [name for name in input_space]
        self._X = np.array(list(itertools.product(*[input_space[name] for name in self._X_names])))
        self._K_ss = self._kernel(self._X, self._X)

        self._K = np.array([])
        self._K_s = np.array([])

        self._Xsamples = []
        self._ysamples = []

    def _kernel_matrix(self, kernel):
        def k(X, Y):
            matrix = []
            for x in X:
                arr = [kernel(np.array(x), np.array(y)) for y in Y]
                matrix.append(arr)
            return np.array(matrix)
        return k


    def _make_dict_from_xsample(self, xsample):
        return {
                name : x for name, x in zip(self._X_names, xsample)
                }

    def get_next(self):
        if not self._init:
            i = np.random.randint(len(self._X))
            return self._make_dict_from_xsample(self._X[i])

        Xtrain = self._Xsamples
        ytrain = self._ysamples
        bestY = np.max(ytrain)

        K_inv = np.linalg.inv(self._K)

        u = np.matmul(np.matmul(K_s, K_inv), ytrain)
        s2 = np.diag(K_ss) - np.diag(np.matmul(np.matmul(self._K_s, K_inv), self._K_s.T))
        stdv = np.sqrt(s2)

        '''
        plt.subplot(2, 1, 1)
        plt.plot(Xtrain, ytrain, 'bs', ms=8)
        plt.plot(X, u)
        plt.gca().fill_between(X.flat, u-stdv, u+stdv, color="#dddddd")
        plt.plot(X, [f(x) for x in X], 'r--', lw=2)
        '''

        besti = np.argmax([self._acquisition(ui, stdvi, bestY) for ui, stdvi in zip(u, stdv)])

        '''
        plt.subplot(2, 1, 2)
        plt.plot(X, ex)
        plt.plot([X[besti]], [ex[besti]], 'bs', ms=8)
        plt.show()
        '''

        return self._make_dict_from_xsample(self._X[besti])

    def set_value(self, x, y):
        self.init = True

        x_vec = [x[name] for name in self._X_names]

        if self._K.shape != (0,):
            self._K = np.append(self._K, self._kernel([x_vec], self._Xsamples), axis=0)

        self._Xsamples.append(x_vec)
        self._ysamples.append(y)

        if self._K.shape == (0,):
            self._K = self._kernel(self._Xsamples, [x_vec])
        else:
            self._K = np.append(self._K, self._kernel(self._Xsamples, [x_vec]), axis=1)
        self._K[-1, -1] += 1e-6

        if self._K_s.shape == (0,):
            self._K_s = self._kernel([x_vec], self._X)
        else:
            self._K_s = np.append(self._K_s, self._kernel([x_vec], self._X), axis=0)

    def get_best(self):
        Xtrain = self._Xsamples
        ytrain = self._ysamples

        besti = np.argmax(ytrain)
        return self._make_dict_from_xsample(Xtrain[besti]), ytrain[besti]

def optimize(f, input_space, kernel, acquisition, n_trials):
    bayes = BayesOptimalization(input_space, kernel, acquisition)

    for _ in range(n_trials):
        x = bayes.get_next()
        y = f(x)
        bayes.set_value(x, y)

    return bayes.get_best()
