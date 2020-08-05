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
 - Proximal Newton
 - Proximal Gradient
 
 Proximal Newton and Proximal Gradient methids are from the [SCOPT](https://www.epfl.ch/labs/lions/technology/scopt/) package

# Problem statements

__Poisson Inverse Problem__

The maximum likelihood formulation of the Poisson inverse problem under sparsity constraints leads to the objective function:

<a href="https://www.codecogs.com/eqnedit.php?latex=\min_{x}\left\{f(x)=\sum\limits_{i=1}^m\omega_i^\top&space;x_i-\sum\limits_{i=1}^m&space;y_i\ln(\omega_i^\top&space;x_i):\&space;x\in\mathbb{R}_&plus;^n,&space;||x||_1\leq&space;M\right\}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\min_{x}\left\{f(x)=\sum\limits_{i=1}^m\omega_i^\top&space;x_i-\sum\limits_{i=1}^m&space;y_i\ln(\omega_i^\top&space;x_i):\&space;x\in\mathbb{R}_&plus;^n,&space;||x||_1\leq&space;M\right\}" title="\min_{x}\left\{f(x)=\sum\limits_{i=1}^m\omega_i^\top x_i-\sum\limits_{i=1}^m y_i\ln(\omega_i^\top x_i):\ x\in\mathbb{R}_+^n, ||x||_1\leq M\right\}" /></a>

__Portfolio Optimization Problem__

The goal of this problem is to minimize the utility function of the investor by choosing the weights of the assets in the portfolio. The task is to design a
portfolio solving the problem:

<a href="https://www.codecogs.com/eqnedit.php?latex=\min_{x}\left\{f(x)=-\sum\limits_{t=1}^T\log(r_t^\top&space;x):\&space;x_i\geq&space;0,&space;\sum\limits_{i=1}^n&space;x_i=1\right\}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\min_{x}\left\{f(x)=-\sum\limits_{t=1}^T\log(r_t^\top&space;x):\&space;x_i\geq&space;0,&space;\sum\limits_{i=1}^n&space;x_i=1\right\}" title="\min_{x}\left\{f(x)=-\sum\limits_{t=1}^T\log(r_t^\top x):\ x_i\geq 0, \sum\limits_{i=1}^n x_i=1\right\}" /></a>

__Distance Weighted Discrimination Problem__

In Distance Weighted Discrimination problem, the classification loss attains the form:

<a href="https://www.codecogs.com/eqnedit.php?latex=f(x)=\frac{1}{n}\sum_{i=1}^{p}(a_{i}^{\top}w&plus;\mu&space;y_{i}&plus;\xi_{i})^{-q}&plus;c^{\top}\xi" target="_blank"><img src="https://latex.codecogs.com/gif.latex?f(x)=\frac{1}{n}\sum_{i=1}^{p}(a_{i}^{\top}w&plus;\mu&space;y_{i}&plus;\xi_{i})^{-q}&plus;c^{\top}\xi" title="f(x)=\frac{1}{n}\sum_{i=1}^{p}(a_{i}^{\top}w+\mu y_{i}+\xi_{i})^{-q}+c^{\top}\xi" /></a>

over the compact set:

<a href="https://www.codecogs.com/eqnedit.php?latex=\chi&space;=\{(w,\mu,\xi)|\;||w||^{2}\leq&space;1,\mu\in[-u,u],||\xi||^{2}\leq&space;R,\xi\in\mathbb{R}_{&plus;}^p\}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\chi&space;=\{(w,\mu,\xi)|\;||w||^{2}\leq&space;1,\mu\in[-u,u],||\xi||^{2}\leq&space;R,\xi\in\mathbb{R}_{&plus;}^p\}" title="\chi =\{(w,\mu,\xi)|\;||w||^{2}\leq 1,\mu\in[-u,u],||\xi||^{2}\leq R,\xi\in\mathbb{R}_{+}^p\}" /></a>

__Nonnegative Linear Systems (general KL) problem__

Solving Nonnegative linear systems is a GSC optimization problem of the same form as in the Poisson linear inverse problem:

<a href="https://www.codecogs.com/eqnedit.php?latex=\min_{\theta\in\mathbb{R}^{d}_{&plus;},||\theta||_{1}\leq&space;t,t\in[0,R]}g(x)&plus;\lambda&space;t" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\min_{\theta\in\mathbb{R}^{d}_{&plus;},||\theta||_{1}\leq&space;t,t\in[0,R]}g(x)&plus;\lambda&space;t" title="\min_{\theta\in\mathbb{R}^{d}_{+},||\theta||_{1}\leq t,t\in[0,R]}g(x)+\lambda t" /></a>

where: 

<a href="https://www.codecogs.com/eqnedit.php?latex=g(\theta)=\sum_{i=1}^{N}&space;\left&space;\{\left&space;\langle&space;w_{i},\theta\right&space;\rangle\log\left(\frac{\left&space;\langle&space;w_{i},\theta\right&space;\rangle}{b_{i}}\right)-\left&space;\langle&space;w_{i},\theta\right&space;\rangle&space;\right\}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?g(\theta)=\sum_{i=1}^{N}&space;\left&space;\{\left&space;\langle&space;w_{i},\theta\right&space;\rangle\log\left(\frac{\left&space;\langle&space;w_{i},\theta\right&space;\rangle}{b_{i}}\right)-\left&space;\langle&space;w_{i},\theta\right&space;\rangle&space;\right\}" title="g(\theta)=\sum_{i=1}^{N} \left \{\left \langle w_{i},\theta\right \rangle\log\left(\frac{\left \langle w_{i},\theta\right \rangle}{b_{i}}\right)-\left \langle w_{i},\theta\right \rangle \right\}" /></a>
