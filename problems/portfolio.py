import scipy.io
import numpy as np

from problems.base_problem import BaseProblem


class PortfolioProblem(BaseProblem):
    
    def __init__(self, path, Mf=2, nu=3, rho=None):
       self.Mf = Mf
       self.nu = nu
       self.R = scipy.io.loadmat(path)['W']
       self.N, self.n = self.R.shape
       self.sigma_f = min(
           np.linalg.eigvalsh(
               self.hess(
                   self.generate_start_point()
                   )
               )
           )
       if rho is None:
           self.rho = np.sqrt(self.n)
       
    def val(self, x, param=None):
        if param is None:
            self.param = self.R @ x
            param = self.param
        return -np.sum(np.log(param))
    
    def grad(self, x, param=None):
        if param is None:
            param = self.param
        return -self.R.T.dot(1 / param)
    
    def hess(self, x, param=None):
        if param is None:
            param = self.R @ x
        Z = self.R / param.reshape(-1, 1)
        return np.einsum('ij,ik->jk', Z, Z, dtype=self.R.dtype)
    
    def hess_mult(self, s, param=None):
        if param is None:
            param = self.param
        Rs = self.R @ s
        Z = (Rs / param) ** 2
        return np.sum(Z)
    
    def hess_mult_vec(self, s, param=None):
        if param is None:
            param = self.param
        Rs = self.R @ s
        return self.R.T.dot(Rs / param ** 2)
    
    def param_func(self, x):
        return self.R @ x
    
    def generate_start_point(self):
        return np.ones(self.n) / self.n
    
    def linear_oracle(self, grad):
        grad_min = np.min(grad)
        s = np.array([el == grad_min for el in grad])
        s = s / sum(s)
        return s
    
    def lloo_oracle(self, x, grad, r):
        d = self.rho * r
        sum_threshold = min(d/2, 1)
        min_index = np.argmin(grad)
        p_pos = np.zeros(self.n)
        p_pos[min_index] = sum_threshold
        p_neg = np.zeros(self.n)
        sorted_indexes = (-grad).argsort() #this is ascending order which corresponds with descending order when you take grad
        k = 0
        tmp_sum = 0
        for k in range(len(sorted_indexes)):
            tmp_sum += x[sorted_indexes[k]]
            if tmp_sum >= sum_threshold:
                break
        for j in range(k):
            index = sorted_indexes[j]
            p_neg[index] = x[index]
        p_neg[sorted_indexes[k]] = sum_threshold - (tmp_sum - x[sorted_indexes[k]])
        return x + p_pos - p_neg

    def projection(self, y):
        ind = np.argsort(y)
        sum_y = sum(y)
        origin_y = sum_y
        n = self.n
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

