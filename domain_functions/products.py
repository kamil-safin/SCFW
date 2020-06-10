import numpy as np


def dot_product(x, y):
    '''
    dot product for vector or matrix
    '''
    if x.ndim == 1:
        return np.dot(x, y)
    if x.ndim == 2:
        # for positive semi-definite matrices
        return np.trace(np.conjugate(x).T.dot(y)).real
    else:
        print('Invalid dimension')
        return None    

def norm(x):
    '''
    norm for vector or matrix
    '''
    if x.ndim == 1:
        return np.linalg.norm(x)
    if x.ndim == 2:
        return np.sqrt(dot_product(x, x))
    else:
        print('Invalid dimension')
        return None