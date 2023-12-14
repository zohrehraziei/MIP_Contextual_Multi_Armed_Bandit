# ALN with Contextual Multi-Armed Bandit for Solving Mixed Integer Programming

[![PyPI version](https://badge.fury.io/py/alns.svg)](https://badge.fury.io/py/alns)
[![ALNS](https://github.com/N-Wouda/ALNS/actions/workflows/alns.yaml/badge.svg)](https://github.com/N-Wouda/ALNS/actions/workflows/alns.yaml)

``alns`` is a general, well-documented and tested implementation of the adaptive
large neighbourhood search [ALNS](https://github.com/N-Wouda/ALNS) metaheuristic in Python. ALNS is an algorithm
that can be used to solve difficult combinatorial optimisation problems. 

`alns` depends only on `numpy` and `matplotlib`. It may be installed in the
usual way as

```
pip install alns
```

### How to cite `alns`

If you use `alns` in your research, please consider citing the following paper:

> Wouda, N.A., and L. Lan (2023). 
> ALNS: a Python implementation of the adaptive large neighbourhood search metaheuristic. 
> _Journal of Open Source Software_, 8(81): 5028. 
> https://doi.org/10.21105/joss.05028

Or, using the following BibTeX entry:

```bibtex
@article{Wouda_Lan_ALNS_2023, 
  doi = {10.21105/joss.05028}, 
  url = {https://doi.org/10.21105/joss.05028}, 
  year = {2023}, 
  publisher = {The Open Journal}, 
  volume = {8}, 
  number = {81}, 
  pages = {5028}, 
  author = {Niels A. Wouda and Leon Lan}, 
  title = {{ALNS}: a {P}ython implementation of the adaptive large neighbourhood search metaheuristic}, 
  journal = {Journal of Open Source Software} 
}
