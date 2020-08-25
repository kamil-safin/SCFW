from abc import ABCMeta, abstractmethod

from domain_functions.llo_oracles import LLO_ORACLES

class BaseProblem():
    __metaclass__ = ABCMeta
    @abstractmethod
    def __init__(self, llo_oracle):
        if llo_oracle is not None:
            self.llo_oracle = LLO_ORACLES[llo_oracle]
    
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

