import time
import sys
import numpy as np

from domain_functions.products import norm, dot_product
from scfw.policies import StandardPolicy, SelfConcordantPolicy, LineSearchPolicy, BacktrackingPolicy, LLOOPolicy


POLICY_DICT = {'standard': StandardPolicy(),
               'sc': SelfConcordantPolicy(),
               'line_search': LineSearchPolicy(),
               'backtracking': BacktrackingPolicy(),
               'lloo': LLOOPolicy()}

def run_frank_wolfe(problem, x_0=None, alpha_policy='standard', max_iter=1000, eps=1e-10, print_every=10):
    policy = POLICY_DICT[alpha_policy]
    fw_state = {}
    fw_state['L'] = 1
    lower_bound = float("-inf")
    upper_bound = float("inf")
    real_Gap = upper_bound - lower_bound
    criterion = 1e10 * eps
    
    if x_0 is None:
        x = problem.generate_start_point()
    else:
        x = x_0

    alpha_hist = []
    Gap_hist = []
    f_hist = []
    time_hist = [0]
    int_start = time.time()

    for k in range(1, max_iter + 1):
        fw_state['k'] = k
        start_time = time.time()
        f = problem.val(x)
        
        #find optimal
        grad = problem.grad(x)
        fw_state['f'] = f
        fw_state['grad'] = grad
        fw_state['x'] = x
        fw_state['s'] = problem.linear_oracle(grad)
        fw_state['delta_x'] = x - fw_state['s']
        fw_state['Gap'] = dot_product(grad, fw_state['delta_x'])
            
        alpha = policy.get_alpha(fw_state, problem)

        x_nxt = x + alpha * (fw_state['s'] - x)
        time_hist.append(time.time() - start_time)
        x_last = x.copy()
        alpha_hist.append(alpha)
        Gap_hist.append(fw_state['Gap'])
        f_hist.append(f)
        x = x_nxt
        if f < upper_bound:
            upper_bound = f
            x_best = x.copy()
        lower_bound = max(lower_bound, f - fw_state['Gap'])
        if (lower_bound - upper_bound) / abs(lower_bound) > 1e-3:
            print(f'upper_bound={upper_bound:.2e}, lower_bound={lower_bound:.2e}')
            sys.exit("Lower bound bigger than upper bound")
        real_Gap = upper_bound - lower_bound
        criterion = min(criterion, norm(x - x_last) / max(1, norm(x_last)))
        
        if k % print_every == 0 or k == 1:
            print(f'iter={k}, stepsize={alpha:.2e}, criterion={criterion:.2e},'
                  f' upper_bound={upper_bound:.2e}, lower_bound={lower_bound:.2e},'
                  f' real_Gap={real_Gap:.2e}, f_val={f}')

        if (criterion <= eps) and (upper_bound - lower_bound) / np.abs(lower_bound) <= eps:
            f_hist.append(f)
            f = problem.val(x_best)
            print('Convergence achieved!')
            print(f'iter = {k}, stepsize = {alpha}, crit = {criterion}, upper_bound={upper_bound}, lower_bound={lower_bound}, real_Gap={real_Gap}')
            return x_best, alpha_hist, Gap_hist, f_hist, time_hist

    #x_hist.append(x)
    f_hist.append(f)
    int_end = time.time()
    print(int_end - int_start)
    return x_best, alpha_hist, Gap_hist, f_hist, time_hist
