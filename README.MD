# SCFW

Implementation of the algorithms from paper [Self-concordant analysis of Frank-Wolfe algorithms](https://arxiv.org/pdf/2002.04320.pdf) (Proceedings of the 37 th International Conference on Machine Learning, Vienna, Austria, PMLR 119, 2020. Copyright 2020 by the author(s)).

# Requirements

All methods are implemented on Python 3.7 with mathematical packages:
 - numpy 1.18.1
 - scipy 1.4.1

# Running the code
You need to define the problem for optimization task:
```python
from problems.portfolio import PortfolioProblem
portfolio_problem = PortfolioProblem(path='./data/syn_1000_800_10_50.mat')
```
And run the method with parameters:
```python
from scfw.frank_wolfe import run_frank_wolfe
result = run_frank_wolfe(portfolio_problem, alpha_policy='backtracking', max_iter=100, print_every=10)
```
# Available implementations
__Problems__:
 - Poisson Inverse Problem
 - Portfolio Optimization Problem

__Methods__:
 - Frank-Wolfe with policies:
   - standard
   - line search
   - sc
   - backtracking
 - SCOPT (Proximal Newtom from the [SCOPT](https://www.epfl.ch/labs/lions/technology/scopt/) package)

# Problem statements

__Poisson Inverse Problem__

The maximum likelihood formulation of the Poisson inverse problem under sparsity constraints leads to the objective function:

<a href="https://www.codecogs.com/eqnedit.php?latex=\min_{x}\left\{f(x)=\sum\limits_{i=1}^m\omega_i^\top&space;x_i-\sum\limits_{i=1}^m&space;y_i\ln(\omega_i^\top&space;x_i):\&space;x\in\mathbb{R}_&plus;^n,&space;||x||_1\leq&space;M\right\}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\min_{x}\left\{f(x)=\sum\limits_{i=1}^m\omega_i^\top&space;x_i-\sum\limits_{i=1}^m&space;y_i\ln(\omega_i^\top&space;x_i):\&space;x\in\mathbb{R}_&plus;^n,&space;||x||_1\leq&space;M\right\}" title="\min_{x}\left\{f(x)=\sum\limits_{i=1}^m\omega_i^\top x_i-\sum\limits_{i=1}^m y_i\ln(\omega_i^\top x_i):\ x\in\mathbb{R}_+^n, ||x||_1\leq M\right\}" /></a>

__Portfolio Optimization Problem__

The goal of this problem is to minimize the utility function of the investor by choosing the weights of the assets in the portfolio. The task is to design a
portfolio solving the problem:

<a href="https://www.codecogs.com/eqnedit.php?latex=\min_{x}\left\{f(x)=-\sum\limits_{t=1}^T\log(r_t^\top&space;x):\&space;x_i\geq&space;0,&space;\sum\limits_{i=1}^n&space;x_i=1\right\}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\min_{x}\left\{f(x)=-\sum\limits_{t=1}^T\log(r_t^\top&space;x):\&space;x_i\geq&space;0,&space;\sum\limits_{i=1}^n&space;x_i=1\right\}" title="\min_{x}\left\{f(x)=-\sum\limits_{t=1}^T\log(r_t^\top x):\ x_i\geq 0, \sum\limits_{i=1}^n x_i=1\right\}" /></a>
