import numpy as np

from problems.base_problem import BaseProblem


class CovEstProblem(BaseProblem):
    
    def __init__(self, path, Mf=2, nu=3, r=1):
        self.A = np.load(path)
        self.n = self.A.shape[0]
        self.r = r
        self.Mf = Mf
        self.nu = nu
        self.name = 'cov_estimation'
        np.random.seed(42)

    def val(self, x, param=None):
        if param is None:
            self.param = np.linalg.inv(x)
            param = self.param
        val = -np.sum(np.log(np.linalg.eigvalsh(x))) + np.trace(self.A.dot(x))
        return val

    def grad(self, x, param=None):
        if param is None:
            param = self.param
        else:
            # for beta combination in line_search policy
            param = np.linalg.inv(x)
        return self.A - param

    def hess_mult(self, s, param=None):
        if param is None:
            param = self.param
        temp = param.dot(s)
        return np.trace(temp.dot(temp))

    def hess_mult_vec(self, s, param=None):
        if param is None:
            param = self.param
        return param.dot(s.dot(param))

    def param_func(self, x):
        return x

    def linear_oracle(self, grad):
        i_max, j_max = np.unravel_index(np.argmax(np.abs(grad)), grad.shape)
        s = np.zeros(grad.shape)
        if i_max == j_max:
            sign = np.sign(grad[i_max, j_max])
            s[j_max, i_max] = -self.r * sign
            if sign > 0:
                diag = np.diag(grad)
                i_min = np.argmin(diag)
                s[i_min, i_min] = 2 * self.r
        else:
            s[j_max, i_max] = -self.r/2 * np.sign(grad[i_max, j_max])
            s[i_max, j_max] = s[j_max, i_max]
            diag = np.diag(grad)
            i_max = np.argmax(diag)
            s[i_max, i_max] = self.r
        return s

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
        diag = np.diag(x)
        if np.linalg.norm(diag, 1) <= self.r:
            return x
        else:
            diag_abs = np.abs(diag / self.r)
            P_y_abs = self.proj_simplex(diag_abs)
            P_y = P_y_abs * np.sign(diag) * self.r
            np.fill_diagonal(x, P_y)
        return x

    def generate_start_point(self):
        vec = np.random.rand(self.n,)
        vec /= np.linalg.norm(vec, ord=1)
        return np.diag(vec * self.r)