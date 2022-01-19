import numpy as np
import cvxpy as cp

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

class kernel_soft_margin_svm:
    def __init__(self, kernel='gaussian'):
        self._y = []
        self._X = []
        self._N = 0
        self._a = []
        self._b = []
        self._C = 0
        self._label_list = []
        if kernel == 'gaussian':
            self._kernel_func = self.__gaussian_kernel
        else:
            self._kernel_func = self.__poly_kernel

    def __clear(self):
        self._y = []
        self._X = []
        self._N = []
        self._a = []
        self._b = []
        self._label_list = []

    def __gaussian_kernel(self, x_m, x_n, k=10):
        x_diff = x_m - x_n
        return np.exp(-k * np.dot(x_diff, x_diff))

    def __poly_kernel(self, x_m, x_n, gamma=1, k=1, Q=2):
        return (k + gamma * np.dot(x_m, x_n)) ** Q

    def __fit_bin(self, X, y):
        # store basic info
        self._y.append(y)
        self._X.append(X)
        N = len(X)

        # generate qp problem matrix
        Q = np.zeros((N, N))
        for i in range(N):
            for j in range(N):
                Q[i][j] = y[i] * y[j] * self._kernel_func(X[i], X[j])
        p = -1 * np.ones(N)
        A = np.zeros((N + 1, N))
        for i in range(N):
            A[i][i] = 1
        A[N] = y
        l = np.zeros(N + 1)
        u = self._C * np.ones(N + 1)  # different from hard-margin svm
        u[N] = 0
        # solve qp problem
        x = cp.Variable(N)
        prob = cp.Problem(cp.Minimize((1 / 2) * cp.quad_form(x, Q) + p.T @ x),
                          [A @ x <= u, A @ x >= l])
        prob.solve(solver='OSQP', max_iter=2000)
        a = x.value

        # find free support vectors which means on the margin boundary
        free_sv_index = -1
        for i in range(N):
            if (a[i] > 0 and a[i] < self._C):  # support vector which is free a[i] < C
                free_sv_index = i
                break
        tmp = 0.0
        for i in range(N):
            tmp += a[i] * y[i] * self._kernel_func(X[i], X[free_sv_index])
        b = y[free_sv_index] - tmp

        # store necessary results
        self._a.append(a)
        self._b.append(b)
        return

    def __sigmoid(self, x):
        return 1 / (1 + np.exp(-x));

    def __sign(self, x):
        return 1 if x >= 0 else -1

    def __get_ova_data(self, x_train, y_train, chosen_label):
        assert (len(x_train) == len(y_train))
        ry_train = []
        for i in range(len(x_train)):
            if y_train[i] == chosen_label:
                ry_train.append(1)
            else:
                ry_train.append(-1)
        ry_train = np.array(ry_train)

        return (x_train, ry_train)

    '''
    uses ova(one versus all) to implement multiple classification using svm
    C is the weight of noise tolerance. when C is large, we would have vectors near the
    margin or even vectors in the error hyperplane side, so the noise tolerance ability
    increases.
    C is set to 1 as a default
    '''

    def fit(self, X, y, C=1, kernel='gaussian'):
        if kernel == 'gaussian':
            self._kernel_func = self.__gaussian_kernel
        else:
            self._kernel_func = self.__poly_kernel

        self.__clear()
        self._C = C
        self._N = len(X)

        for i in y:
            if i not in self._label_list:
                self._label_list.append(i)
        # print(self._label_list)

        for i, label in enumerate(self._label_list):
            x_ova_train, y_ova_train = self.__get_ova_data(X, y, label)
            self.__fit_bin(x_ova_train, y_ova_train)
        return

    def predict(self, X_test):
        N_test = len(X_test)
        y_hat = np.zeros((len(self._label_list), N_test))

        for i, label in enumerate(self._label_list):
            # labels'prob for X_test
            for j in range(N_test):
                # calcule for a single vector X_test[j] in label[i]
                tmp = 0.0
                for k in range(self._N):
                    tmp += self._a[i][k] * self._y[i][k] * self._kernel_func(self._X[i][k], X_test[j])
                y_hat[i][j] = self.__sigmoid(tmp + self._b[i])
        # get each X_test vectors' max prob label index
        ans = [0 for i in range(N_test)]
        for i in range(1, len(self._label_list)):
            for j in range(N_test):
                # if j's test prob of label label_list[i] is bigger than current
                if (y_hat[ans[j]][j] <= y_hat[i][j]):
                    ans[j] = i
        # transform index to label
        ans_label = np.array([self._label_list[i] for i in ans])
        return ans_label

    def err_rate(self, X, y):
        y_hat = self.predict(X)
        res = np.sum(y != y_hat)
        # print("error num is {}, totoal is {}".format(res, len(X)))
        return res / len(X)

    def score(self, X, y):
        return 1 - self.err_rate(X, y)

def get_data():
    iris = load_iris()
    x_train = iris.data
    y_train = iris.target
    x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.2, stratify=y_train)
    return x_train, y_train, x_test, y_test


max_iter = 20

sum_accuracy = 0.


for i in range(max_iter):
    x_train, y_train, x_test, y_test = get_data()
    model = kernel_soft_margin_svm()
    model.fit(x_train, y_train)
    acc = model.score(x_test, y_test)
    sum_accuracy += acc
    print("the {}'s prediction accuracy is {}".format(i, acc))

print("average accuracy of {} test is {}".format(max_iter, sum_accuracy/max_iter))