import numpy as np
from sklearn.datasets import load_svmlight_file

from problems.base_problem import BaseProblem


class DwdProblem(BaseProblem):
    
    def __init__(self, path, q=2, R=10, R_w=10, u=5, Mf=None, nu=None):
        data = load_svmlight_file(path)
        self.A = data[0]
        self.y = data[1]
        if max(self.y) == 2:
            self.y = 2 * self.y - 3
        # normalize
        for i, row in enumerate(self.A):
            if np.sum(row.multiply(row)) != 0:
                self.A[i] = row.multiply(1 / np.sqrt(np.sum(row.multiply(row))))
        self.p, self.d = self.A.shape
        self.n = self.p + self.d + 1
        self.q = q
        self.u = u
        self.R = R
        self.R_w = R_w
        self.c = np.array([1] * self.p)
        E = np.eye(self.p)
        max_norm = max([np.linalg.norm(np.hstack((self.A[i].toarray().flatten(), self.y[i], E[i])))**(self.q / (self.q + 2)) for i in range(self.p)])
        if Mf is None:
            self.Mf = (self.q + 2) / (self.q * (self.q + 1))**(1 / (self.q + 2)) * self.n**(1 / (self.q + 2)) * max_norm
        else:
            self.Mf = Mf
        if nu is None:
            self.nu = 2 * (self.q + 3) / (self.q + 2)
        else:
            self.nu = nu
        self.name = 'dwd'
    
    def val(self, x, param=None):
        # x is flatten vector of (w, mu, xi)
        w = x[:self.d]
        mu = x[self.d]
        xi = x[(self.d + 1):]
        if param is None:
            self.param = self.A @ w + mu * self.y + xi
            param = self.param
        val_1 = 1 / param**self.q
        val_1 = np.sum(val_1) * (1 / self.p)
        val_2 = self.c.dot(xi)
        return val_1 + val_2
    
    def grad(self, x, param=None):
        if param is None:
            param = self.param
        param = 1 / param**(self.q + 1)
        w_grad = (-self.q / self.p) * (self.A.T @ param)
        mu_grad = (-self.q / self.p) * (self.y.dot(param))
        xi_grad = (-self.q / self.p) * param + self.c
        return np.hstack((w_grad, mu_grad, xi_grad))
    
    def hess_mult(self, s, param=None):
        if param is None:
            param = self.param
        param = 1 / (param**(self.q + 2) + 1e-10)
        s_w = s[:self.d]
        s_mu = s[self.d]
        s_xi = s[(self.d + 1):]
        numer = (self.A @ s_w + s_mu * self.y + s_xi)**2
        val = self.q * (self.q + 1) / self.p * np.sum(numer * param)
        return val
    
    def hess_mult_vec(self, s, param=None):
        if param is None:
            param = self.param
        param = 1 / param**(self.q + 2)
        s_w = s[:self.d]
        new_vec = self.A @ s_w + self.y * s[self.d] + s[(self.d + 1):self.n]
        w_hess = (self.q * (self.q + 1) / self.p) * (self.A.T @ (new_vec * param))
        mu_hess = (self.q * (self.q + 1) / self.p) * ((self.y * new_vec).dot(param))
        xi_hess = (self.q * (self.q + 1) / self.p) * new_vec * param
        return np.hstack((w_hess, mu_hess, xi_hess))
    
    def param_func(self, x):
        w = x[:self.d]
        mu = x[self.d]
        xi = x[(self.d + 1):]
        return self.A @ w + mu * self.y + xi
    
    def generate_start_point(self):
        return np.array([0]*(self.d + 1) + [1 / self.p] * self.p)
    
    def l2_oracle(self, x, R=1):
        s = -1 * x
        s = s * R / np.linalg.norm(s)
        return s
    
    def linear_oracle(self, grad):
        w_grad = grad[:self.d]
        mu_grad = grad[self.d]
        xi_grad = grad[(self.d + 1):]
        w_s = self.l2_oracle(w_grad, R=self.R_w)
        xi_s = self.l2_oracle(xi_grad, R=self.R)
        mu_s = -1 * self.u * np.sign(mu_grad)
        return np.hstack((w_s, mu_s, xi_s))
    
    def l2_projection(self, x, R=1):
        x = x * R / np.linalg.norm(x)
        return x

    def projection(self, x):
        w = x[:self.d]
        mu = x[self.d]
        xi = x[(self.d + 1):]
        w_pr = self.l2_projection(w, R=self.R_w)
        xi_pr = self.l2_projection(xi, R=self.R)
        mu_pr = self.u * np.sign(mu)
        return np.hstack((w_pr, mu_pr, xi_pr))