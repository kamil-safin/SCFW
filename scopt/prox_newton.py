import sys
import time
import numpy as np

from domain_functions.products import norm, dot_product


def estimate_lipschitz(problem, ndim):
    Lest = 1
    if ndim == 1:
        dirr = np.ones(problem.n)
    elif ndim == 2:
        dirr = np.eye(problem.n)
    if Lest == 1:
        # Estimate Lipschitz Constant
        for _ in range(1, 16):
            Dir = problem.hess_mult_vec(dirr)
            dirr = Dir / norm(Dir)
        Hd = problem.hess_mult_vec(dirr)
        dHd = dot_product(dirr, Hd)
        L = dHd / (dot_product(dirr, dirr))
    return L

def fista(
        problem, scopt_state, max_iter=1000, 
        tol=1e-5, fista_type='mfista', Lest='backtracking',
        print_fista=False
        ):
    x = scopt_state['x']
    def func(x_2):
        return 0.5 * problem.hess_mult(x_2 - x) + dot_product(scopt_state['grad'], x_2 - x)
    def grad_func(x_2):
        return problem.hess_mult_vec(x_2 - x) + scopt_state['grad']
    y = x.copy()
    if Lest == 'estimate':
        L = estimate_lipschitz(problem, ndim=x.ndim)
    elif Lest == 'backtracking':
        L = 1
    x_cur = y.copy()
    f_cur = func(x_cur)
    t = 1
    beta = 2
    for k in range(1, max_iter + 1):
        grad_y = grad_func(y)
        f_y = func(y)
        if Lest == 'estimate':
            x_tmp = y - 1 / L * grad_y
            z = problem.projection(x_tmp)
            f_z = func(z)
            diff_yz = z - y
        elif Lest == 'backtracking':
            z = y
            L = L / beta
            diff_yz = z - y
            f_z = f_y + 1
        while (f_z > f_y + dot_product(grad_y, diff_yz) + (L / 2) * norm(diff_yz)**2) or (f_z > f_y):
            L = L * beta
            x_tmp = y - 1 / L * grad_y
            z = problem.projection(x_tmp)
            f_z =  func(z)
            diff_yz = z - y
            if L > 1e+20:
                z = problem.projection(y)
                f_z = func(z)
                diff_yz = z - y
                L = L / beta
                break
        f_nxt = f_z
        if (f_nxt > f_cur) and (fista_type == 'mfista'):
            x_nxt = x_cur
            f_nxt = f_cur
        else:
            x_nxt = z
        zdiff = z - x_cur
        ndiff = norm(zdiff)
        if (ndiff < tol) and (k > 1) and print_fista:
            print('Fista err = %3.3e; Subiter = %3d; subproblem converged!' % (ndiff, k))
            break
        xdiff = x_nxt - x_cur
        t_nxt = 0.5 * (1 + np.sqrt(1 + 4 * (t ** 2)))
        y = x_nxt + (t - 1) / t_nxt * xdiff + t / t_nxt * (z-x_nxt)
        t = t_nxt
        x_cur = x_nxt
        f_cur = f_nxt
    return x_nxt


def run_prox_newton(
        problem, x_0=None, max_iter=1000, eps=1e-10, 
        use_two_phase=False, print_every=10, Lest='backtracking', 
        fista_iter=1000, fista_tol=1e-5, fista_type='mfista', print_fista=False
        ):
    if x_0 is None:
        x = problem.generate_start_point()
    else:
        x = x_0
    alpha_hist = []
    f_hist = []
    time_hist = []
    err_hist = []
    int_start = time.time()
    time_hist.append(0)
    bPhase2 = False
    Mf = problem.Mf
    nu = problem.nu
    scopt_state = {}
    for i in range(1, max_iter + 1):

        start = time.time()

        f = problem.val(x)
        grad = problem.grad(x)
        scopt_state['grad'] = grad
        scopt_state['x'] = x
        # compute local Lipschitz constant
        x_nxt = fista(
            problem, scopt_state, max_iter=fista_iter, 
            tol=fista_tol, fista_type=fista_type, Lest=Lest,
            print_fista=print_fista
            )
        diffx = x_nxt - x
        lam_k = np.sqrt(problem.hess_mult(diffx))
        beta_k = Mf * norm(diffx)
        # solution value stop-criterion
        nrm_dx = norm(diffx)
        rdiff = nrm_dx / max(1.0, norm(x))
        if use_two_phase and not bPhase2:
            if nu == 2:  # conditions to go to phase 2
                sys.exit('still under implementation')
            elif nu < 3:
                sys.exit('still under implementation')
            elif nu == 3:
                if lam_k * 2 * Mf < 1:
                    bPhase2 = True
        if not bPhase2:  # if we are not in phase 2
            if beta_k == 0:
                tau_k = 0
            else:
                if nu == 2:
                    tau_k = 1 / beta_k * np.log(1 + beta_k)
                elif nu == 3:
                    d_k = 0.5 * Mf * lam_k
                    tau_k = 1 / (1 + d_k)
                elif nu < 3:
                    d_k = (nu / 2 - 1) * (Mf * lam_k)**(nu - 2) * beta_k**(3 - nu)
                    nu_param = (nu - 2) / (4 - nu)
                    tau_k = (1 - (1 + d_k / nu_param)**(-nu_param)) / d_k
                else:
                    print('The value of nu is not valid')
                    return None
        else:  # if we are in phase 2
            tau_k = 1

        end = time.time()

        alpha_hist.append(tau_k)
        f_hist.append(f)
        err_hist.append(rdiff)
        time_hist.append(end - start)

        x = x + tau_k * diffx

        # Check the stopping criterion.
        if (rdiff <= eps) and (i > 1):
            print('Convergence achieved!')
            print('iter = %4d, stepsize = %3.3e, rdiff = %3.3e,value=%g' % (i, tau_k, rdiff, f))
            f_hist.append(f)
            break

        if (i % print_every == 0) or (i == 1):
            print('iter = %4d, stepsize = %3.3e, rdiff = %3.3e , f = %g' % (i, tau_k, rdiff, f))

    int_end = time.time()
    if i >= max_iter:
        f_hist.append(f)
        print('Exceed the maximum number of iterations')
    print(int_end - int_start)
    return x, alpha_hist, f_hist, time_hist