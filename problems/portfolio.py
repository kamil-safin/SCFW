import scipy.io
import numpy as np

from problems.base_problem import BaseProblem


class PortfolioProblem(BaseProblem):
    
    def __init__(self, path, Mf=2, nu=3, rho=None, llo_oracle=None):
        super().__init__(llo_oracle=llo_oracle)
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
        if self.sigma_f < 0:
            self.sigma_f = 1e-10
        if rho is None:
            self.rho = np.sqrt(self.n)
        self.name = 'portfolio'

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

