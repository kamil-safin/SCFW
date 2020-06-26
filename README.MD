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
