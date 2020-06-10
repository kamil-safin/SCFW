from abc import ABCMeta, abstractmethod


class BaseProblem():
    __metaclass__ = ABCMeta
    @abstractmethod
    def __init__(self):
        pass
    
    @abstractmethod
    def val(self):
        pass
    
    @abstractmethod
    def grad(self):
        pass
    
    @abstractmethod
    def hess_mult(self):
        pass
    
    @abstractmethod
    def hess_mult_vec(self):
        pass
    
    @abstractmethod
    def param_func(self):
        pass
    
    @abstractmethod
    def linear_oracle(self):
        pass
    
    @abstractmethod
    def projection(self):
        pass
    
    def lloo_oracle(self):
        pass
