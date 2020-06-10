import sys
import numpy as np
from numpy import matlib
from sklearn.datasets import load_svmlight_file

from problems.base_problem import BaseProblem


class PoissonProblem(BaseProblem):
    
    def __init__(self, path, Mf=None, nu=3, lam=None, M=None):
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
        self.Mf =  max(2 / np.sqrt(self.y))
        self.nu = nu
        self.N, self.n = self.W.shape
        self.lam =  np.sqrt(self.N) / 2
        x0 = np.ones(self.n) / self.n
        self.M = self.val(x0) / self.lam
        
    def val(self, x, param=None):
        if param is None:
            self.param = self.W @ x
            param = self.param
        fst_term = np.sum(param)
        snd_term = self.y.dot(np.log(param))
        return fst_term - snd_term + self.lam * sum(x)
    
    def grad(self, x, param=None):
        if param is None:
            param = self.param
        if min(x) < 0:
            sys.exit('x is not nonnegative')
        e = np.ones(self.N)
        mult = (e - (self.y / param))
        x_term = (self.W.T @ mult) # n
        return x_term.T + self.lam * np.ones(self.n)
    
    def hess(self, x):
        if param is None:
            param = self.param
        denom = 1 / param
        snd_einsum = np.multiply(self.W, denom.reshape(-1, 1))
        fst_einsum = self.y.reshape(-1, 1) * snd_einsum
        return np.einsum('ij,ik->jk', fst_einsum, snd_einsum)
    
    def hess_mult(self, s, param=None):
        if param is None:
            param = self.param
        num = self.y.dot(((self.W @ s) / param) ** 2)
        return num
    
    def hess_mult_vec(self, s, param=None):
        if param is None:
            param = self.param
        return (((self.W @ s) * self.y)/((param) ** 2)).dot(self.W)
    
    def param_func(self, x):
        return self.W @ x
    
    def generate_start_point(self):
        return np.ones(self.n) / self.n
    
    def linear_oracle(self, grad):
        s = np.zeros(self.n)
        i_max = np.argmax(-grad)
        if grad[i_max] < 0:
            s[i_max] = self.M # 1 x n
        return s
    
    def projection(self, x):
        return np.maximum(x, 0)