import numpy as np
import scipy.linalg as sc
from abc import ABCMeta, abstractmethod

from domain_functions.products import norm, dot_product


class BasePolicy():
    __metaclass__ = ABCMeta
    
    def _init_(self):
        pass
    
    @abstractmethod
    def get_alpha(self):
        pass
    
class StandardPolicy(BasePolicy):
        
    def get_alpha(self, fw_state, problem):
        if problem.name == 'cov_estimation':
            alpha = 2 / (fw_state['k'] + 3)
        else:
            alpha = 2 / (fw_state['k'] + 2)
        return alpha

class SelfConcordantPolicy(BasePolicy):
    
    def _compute_t_nu(self, fw_state, problem):
        e = problem.hess_mult(fw_state['delta_x'])**0.5
        beta = norm(fw_state['delta_x'])
        Mf = problem.Mf
        nu = problem.nu
        Gap = fw_state['Gap']
        if nu == 2:
            delta_v = Mf * beta
            t = 1 / delta_v * np.log(1 + (Gap*delta_v) / ( e ** 2))
        elif nu == 3:
            delta_v =  Mf * e / 2
            t = Gap / (Gap * delta_v + e ** 2)
        else:
            delta_v = (nu - 2) / 2 * Mf * (beta ** (3 - nu)) * e ** (nu - 2)
            if nu == 4:
                t = 1 / delta_v * (1 - np.exp(-delta_v * Gap / (e ** 2)))
            elif nu < 4 and nu > 2:
                const = (4 - nu) / (nu - 2)
                t = 1 / delta_v * (1 - (1 + (-delta_v * Gap * const / (e ** 2)))) ** (-1 / const)
        return t, delta_v
    
    def get_alpha(self, fw_state, problem):
        t, _ = self._compute_t_nu(fw_state, problem)
        return min(1, t)
    
class LLOOPolicy(BasePolicy):
    
    def get_alpha(self, fw_state, problem):
        if fw_state['k'] == 1:
            self.h = fw_state['Gap']
            self.r = np.sqrt(6 * self.h / problem.sigma_f)
        fw_state['r'] = self.r
        s = problem.llo_oracle(fw_state, problem)
        delta_x = fw_state['x'] - s
        fw_state['s'] = s
        fw_state['delta_x'] = delta_x
        fw_state['Gap'] = dot_product(fw_state['grad'], fw_state['delta_x'])
        e = np.sqrt(problem.hess_mult(delta_x)) * problem.Mf/2
        alpha = min(self.h * problem.Mf**2 / (4 * e**2), 1) * (1 / (1 + e))
        self.h = self.h * np.exp(-alpha / 2)
        self.r = self.r * np.sqrt(np.exp(-alpha / 2))
        return alpha


class BaseBetaPolicy(BasePolicy):
    
    def adjust_beta(self, problem, fw_state, extra_param, extra_param_s, case='general'):
        if problem.name == 'cov_estimation':
            x = fw_state['x']
            s = fw_state['s']
            L = np.linalg.cholesky(x)
            invL = np.linalg.inv(L)
            temp = invL @ (s - x)
            min_eig, _ = sc.eigh(temp@(invL.transpose()))
            min_eig = min(min_eig)
            if min_eig < 0:
                #(1-beta)*1+beta min_eig>0 => beta<=1/(1-min_eig)
                beta_max = min(1, 1 / abs(min_eig) - 1e-5)
            else:
                beta_max = 1
        else:
            if case == 'line_search':
                if min(extra_param_s) == 0: #if 0 it is not defines and beta is adjusted
                    beta = 0.5
                else:
                    beta = 1
                return beta
            else:
                if min(extra_param_s) < 0: #if 0 it is not defines and beta is adjusted
                    indexes = np.where(extra_param_s <= 0)
                    beta_max = min(extra_param[indexes] / (extra_param[indexes] - extra_param_s[indexes]    ))
                else:
                    beta_max = 1
        return beta_max
    
class LineSearchPolicy(BaseBetaPolicy):
    
    def __init__(self, accuracy=1e-10):
        self.accuracy = accuracy
        
    def get_alpha(self, fw_state, problem):
        extra_param_s = problem.param_func(fw_state['s'])
        extra_param = problem.param
        beta_max = self.adjust_beta(problem, fw_state, extra_param, extra_param_s, case='line_search')
        grad_beta = lambda beta: problem.grad((1 - beta) * fw_state['x'] + beta * fw_state['s'],
                                              (1 - beta) * extra_param + beta * extra_param_s)
        t_lb = 0
        delta_x = -fw_state['delta_x']
        ub = dot_product(grad_beta(beta_max), delta_x)
        t_ub = beta_max
        t = t_ub
        while (t_ub < 1) and (ub < 0):
            t_ub = 1 - (1 - t_ub) / 2
            ub = dot_product(grad_beta(t_ub), delta_x)
        while (t_ub - t_lb > self.accuracy):
            t = (t_lb + t_ub) / 2
            val = dot_product(grad_beta(t), delta_x)
            if val > 0:
                t_ub = t
            else:
                t_lb = t
        return t

class BacktrackingPolicy(BaseBetaPolicy):
    
    def __init__(self, tau=2, nu=0.25):
        self.tau = tau
        self.nu = nu
    
    def get_alpha(self, fw_state, problem):
        extra_param_s = problem.param_func(fw_state['s'])
        extra_param = problem.param
        beta_max = self.adjust_beta(problem, fw_state, extra_param, extra_param_s)
        func_beta = lambda beta: problem.val((1 - beta) * fw_state['x'] + beta * fw_state['s'],
                                             (1 - beta) * extra_param + beta * extra_param_s)
        delta_x = -fw_state['delta_x']
        fx = fw_state['f']
        L = self.nu * fw_state['L']
        qx = dot_product(fw_state['grad'], delta_x)
        qqx = L / 2 * norm(delta_x)**2
        t = min(-1 * qx / (L * norm(delta_x)**2), beta_max)
        while func_beta(t) > fx + t * qx + t**2 * qqx:
            L = self.tau * L
            qqx = qqx * self.tau
            t = min(-1 * qx / (2 * qqx), beta_max)
        fw_state['L'] = L
        return t