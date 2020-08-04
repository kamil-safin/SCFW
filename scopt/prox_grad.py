import sys
import time
import numpy as np

from domain_functions.products import norm, dot_product


def estimate_lipschitz_bb(x, x_old, grad, grad_old, bb_type=2):
    s = x - x_old
    y = grad - grad_old
    if bb_type == 2:
        est = norm(y) / norm(s)
    elif bb_type == 3:
        est = abs(dot_product(y, s)) / norm(s)
    else:
        est = np.sqrt(norm(y)) / norm(s)
    return est


def run_prox_grad(
    problem,
    x_0=None,
    max_iter=1000,
    eps=1e-10,
    bb_type=3,
    backtracking=True,
    btk_iters=100,
    print_every=10,
    ):

    if x_0 is None:
        x = problem.generate_start_point()
    else:
        x = x_0
    x_old = 0
    grad_old = 0
    alpha_hist = []
    f_hist = []
    time_hist = []
    err_hist = []
    int_start = time.time()
    time_hist.append(0)
    Mf = problem.Mf
    nu = problem.nu
    for k in range(1, max_iter + 1):
        start = time.time()
        f = problem.val(x)
        grad = problem.grad(x)
        Lips_cur = estimate_lipschitz_bb(x, x_old, grad, grad_old, bb_type=bb_type)
        x_nxt = problem.projection(x - 1/Lips_cur * grad)
        diffx = x_nxt - x
        nrm_dx = norm(diffx)
        lam_k = np.sqrt(Lips_cur * diffx.dot(diffx))
        beta_k = Mf * norm(diffx)
        if backtracking:
            for _ in range(btk_iters):
                if Lips_cur <= ((lam_k * lam_k) / (nrm_dx * nrm_dx)):
                    break
                else:
                    Lips_cur = Lips_cur / 2
                    x_nxt = problem.projection(x - 1/Lips_cur * grad)
        
        if backtracking:
            diffx = x_nxt - x
            nrm_dx = norm(diffx)
            lam_k = np.sqrt(Lips_cur * diffx.dot(diffx))
            beta_k = Mf * norm(diffx)
        alpha = min(beta_k / (lam_k * (lam_k + beta_k)), 1.)
        alpha_hist.append(alpha)
        x_old = x
        grad_old = grad
        x  = x + alpha * diffx
        end = time.time()
        alpha_hist.append(alpha)
        f_hist.append(f)
        rdiff = nrm_dx / max(1.0, norm(x))
        err_hist.append(rdiff)
        time_hist.append(end - start)

        if (rdiff <= eps) and (k > 1):
            print('Convergence achieved!')
            print('iter = %4d, stepsize = %3.3e, rdiff = %3.3e,value=%g' % (k, alpha, rdiff, f))
            break

        if (k % print_every == 0) or (k == 1):
            print('iter = %4d, stepsize = %3.3e, rdiff = %3.3e , f = %g' % (k, alpha, rdiff, f))
    int_end = time.time()
    if k >= max_iter:
        f_hist.append(f)
        print('Exceed the maximum number of iterations')
    print(int_end - int_start)
    return x, alpha_hist, f_hist, time_hist
