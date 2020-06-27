# SCFW

Implementation of the algorithms from paper [Self-concordant analysis of Frank-Wolfe algorithms](https://arxiv.org/pdf/2002.04320.pdf).

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

<img src="http://www.sciweavers.org/tex2img.php?eq=%5Cmin_%7Bx%7D%5Cleft%5C%7Bf%28x%29%3D%5Csum%5Climits_%7Bi%3D1%7D%5Em%5Comega_i%5E%5Ctop%20x_i-%5Csum%5Climits_%7Bi%3D1%7D%5Em%20y_i%5Cln%28%5Comega_i%5E%5Ctop%20x_i%29%3A%5C%20x%5Cin%5Cmathbb%7BR%7D_%2B%5En%2C%20%7C%7Cx%7C%7C_1%5Cleq%20M%5Cright%5C%7D&bc=White&fc=Black&im=jpg&fs=12&ff=arev&edit=0" align="center" border="0" alt="\min_{x}\left\{f(x)=\sum\limits_{i=1}^m\omega_i^\top x_i-\sum\limits_{i=1}^m y_i\ln(\omega_i^\top x_i):\ x\in\mathbb{R}_+^n, ||x||_1\leq M\right\}" width="471" height="53" />

__Portfolio Optimization Problem__

The goal of this problem is to minimize the utility function of the investor by choosing the weights of the assets in the portfolio. The task is to design a
portfolio solving the problem:

<img src="http://www.sciweavers.org/tex2img.php?eq=%5Cmin_%7Bx%7D%5Cleft%5C%7Bf%28x%29%3D-%5Csum%5Climits_%7Bt%3D1%7D%5ET%5Clog%28r_t%5E%5Ctop%20x%29%3A%5C%20x_i%5Cgeq%200%2C%20%5Csum%5Climits_%7Bi%3D1%7D%5En%20x_i%3D1%5Cright%5C%7D&bc=White&fc=Black&im=jpg&fs=12&ff=arev&edit=0" align="center" border="0" alt="\min_{x}\left\{f(x)=-\sum\limits_{t=1}^T\log(r_t^\top x):\ x_i\geq 0, \sum\limits_{i=1}^n x_i=1\right\}" width="378" height="53" />
