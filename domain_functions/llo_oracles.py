import numpy as np


def lloo_simplex(fw_state, problem):
    x = fw_state['x']
    r = fw_state['r']
    n = problem.n
    rho = problem.rho
    grad = fw_state['grad']
    d = rho * r
    sum_threshold = min(d/2, 1)
    min_index = np.argmin(grad)
    p_pos = np.zeros(n)
    p_pos[min_index] = sum_threshold
    p_neg = np.zeros(n)
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

LLO_ORACLES = {'lloo_simplex': lloo_simplex}