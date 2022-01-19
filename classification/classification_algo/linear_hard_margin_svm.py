import numpy as np
import cvxpy as cp

'''
the inspiration is to make the hyperplane as farm as the vectors(points) in input space
which is equivalent to a quadratic programming problem.
'''

# a simple hard-margin svm demo，input label is -1 and 1 only
# when is non-linear separable, this svm would fail
class linear_hard_margin_svm:

    def __init__(self):
        self._dim = 0
        self._result = None
        self._w = None
    def __sign(self, x):
        return 1 if x >= 0 else -1
    def fit(self, X, y):
        self._dim = len(X[0]) + 1
        N = len(X)

        Q = np.identity(self._dim)
        Q[0][0] = 0
        p = np.zeros(self._dim)

        A = np.zeros((N, self._dim))
        for i in range(N):
            X_tmp = np.zeros(self._dim)
            X_tmp[0] = 1
            X_tmp[1:] = X[i]
            A[i] = y[i] * X_tmp
        l = np.array([1 for i in range(N)])
        x = cp.Variable(self._dim)
        prob = cp.Problem(cp.Minimize((1 / 2) * cp.quad_form(x, Q) + p.T @ x), [A @ x >= l])
        prob.solve(solver='OSQP', max_iter=2000)
        self._w = x.value
        return

    def predict(self, X):
        X_arg = np.zeros((len(X), self._dim))
        for i in range(len(X)):
            X_arg[i, 0] = 1
            X_arg[i, 1:] = X[i]
        y_hat = np.dot(X_arg, self._w)
        ret = np.array([self.__sign(a) for a in y_hat])
        return ret
    def err_rate(self, X, y):
        pass

