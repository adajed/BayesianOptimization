import numpy as np
import matplotlib.pyplot as plt
import itertools

class BayesOptimalization:
    def __init__(self, input_space, kernel, acquisition):
        self._init = False
        self._kernel = self._kernel_matrix(kernel)
        self._acquisition = acquisition

        self._X_names = [name for name in input_space]
        self._X = np.array(list(itertools.product(*[input_space[name] for name in self._X_names])))
        self._K_ss = self._kernel(self._X, self._X)

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

        K = self._kernel(Xtrain, Xtrain)
        K += 1e-6 * np.eye(len(Xtrain))
        K_inv = np.linalg.inv(K)

        K_s = self._kernel(self._X, Xtrain)

        u = np.matmul(np.matmul(K_s, K_inv), ytrain)
        s2 = np.diag(K_ss) - np.diag(np.matmul(np.matmul(K_s, K_inv), K_s.T))
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
        self._Xsamples.append([x[name] for name in self._X_names])
        self._ysamples.append(y)

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
