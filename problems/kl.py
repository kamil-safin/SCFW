import sys
import numpy as np
from numpy import matlib
from sklearn.datasets import load_svmlight_file

from problems.base_problem import BaseProblem


class KLProblem(BaseProblem):
    
    def __init__(self, path, Mf=1, nu=4, lam=0.005, M=None, R=30):
        data = load_svmlight_file(path)
        y = data[1].reshape(-1, 1)
        W = data[0].toarray()
        # multiplicative multiplication of the structure of the recognizable structure
        W = matlib.repmat(y, 1, W.shape[1]) * W
        # summation of feature descriptions
        sA = np.sum(W, 1)
        # if the sum of the string is negative, invert it
        W[sA < 0, :] = -W[sA < 0, :]
        # zero row deletion
        self.W = W[sA != 0, :]
        Bias = 1
        b = (Bias * y).squeeze()
        b = np.abs(b)
        if (b.any == 0):
            print(' Input parameter y error')
        self.y = b
        self.Mf =  Mf
        self.nu = nu
        self.N, self.n = self.W.shape
        self.n = self.n + 1
        self.lam = lam
        self.R = R
        
    def val(self, x, param=None):
        t = x[-1]
        self.t = t
        x = x[:-1]
        if param is None:
            self.param = self.W.dot(x)
            param = self.param
        first_term = param * np.log(param / self.y)
        return np.sum(first_term - param) + self.lam * t
    
    def grad(self, x, param=None):
        if param is None:
            param = self.param
        if min(x) < 0:
            sys.exit('x is not nonnegative')
        t = x[-1]
        x = x[:-1]
        first_part = self.W.T.dot(np.log(param / self.y + 1e-10))
        return np.hstack((first_part, self.lam))
    
    def hess_mult(self, s, param=None):
        if param is None:
            param = self.param
        s = s[:-1]
        num = self.W.dot(s) ** 2
        return np.sum(num / param)

    def linear_oracle(self, grad):
        s = np.zeros(self.n) + 1e-10
        i_max = np.argmax(-grad[:-1])
        if grad[i_max] < 0:
            s[i_max] = self.t # 1 x n
        return s
    
    def param_func(self, x):
        x = x[:-1]
        return self.W @ x

    def proj_simplex(self, y):
        ind = np.argsort(y)
        sum_y = sum(y)
        origin_y = sum_y
        n = len(y)
        Py = y.copy()
        for i in range(n):
            t = (sum_y - 1) / (n - i)
            if (origin_y > 1 and t < 0): #for numerical errors
                sum_y = sum(y[ind[i : n - 1]])
                t = (sum_y - 1) / (n - i)
            if i > 0:
                if t <= y[ind[i]] and t >= y[ind[i - 1]]:
                    break
            elif t <= y[ind[i]]:
                break
            sum_y -= y[ind[i]]
            Py[ind[i]] = 0
        Py = np.maximum(y - t, np.zeros(n))
        return Py

    def projection(self, x):
        r = x[-1]
        x = x[:-1]
        P_y = self.proj_simplex(x)
        P_y = P_y * self.t
        return np.hstack((P_y, np.max((r, 1e-5))))
    
    def generate_start_point(self):
        d = self.n - 1
        return np.hstack((np.ones(d) / d, self.R))