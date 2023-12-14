# ALN with Contextual Multi-Armed Bandit for Solving Mixed Integer Programming

[![PyPI version](https://badge.fury.io/py/alns.svg)](https://badge.fury.io/py/alns)
[![ALNS](https://github.com/N-Wouda/ALNS/actions/workflows/alns.yaml/badge.svg)](https://github.com/N-Wouda/ALNS/actions/workflows/alns.yaml)


# ALNS with Contextual Multi-Armed Bandit for MIP using SCIP

This repository contains an implementation that combines Adaptive Large Neighbourhood Search (ALNS) with a contextual multi-armed bandit (MAB) approach for solving Mixed Integer Programming (MIP) problems using SCIP. The ALNS component of this implementation is based on the work available in Niels Wouda's ALNS repository: [ALNS](https://github.com/N-Wouda/ALNS)


## Overview

The implementation employs ALNS, an iterative method for solving difficult optimization problems, and enhances it with a contextual multi-armed bandit (MAB) strategy for dynamic operator selection. The MAB approach helps in adapting the selection of destroy and repair operators based on contextual information extracted from the MIP instances.

## Features

- **[ALNS](https://github.com/N-Wouda/ALNS) Algorithm**: Adaptive search algorithm for optimizing MIP solutions.
- **Contextual MAB**: Dynamic operator selection based on learned performance and problem context implemented on the top of ALNS (see [MAB](https://github.com/P-bibs/ALNS)).
- **SCIP Integration**: Utilizes SCIP for solving MIP instances effectively.
- **Customizable Operators**: Supports the addition of custom destroy and repair operators for ALNS.

## Requirements

- Python 3.x
- `numpy`
- `pandas`
- `pyscipopt` (Python interface for SCIP Optimization Suite)
- `mabwiser` (for the multi-armed bandit algorithm)

## Usage

1. **Initialization**: Create a new ALNS instance and configure it with SCIP-based problem states.
2. **Operator Setup**: Define and add custom destroy and repair operators to the ALNS instance.
3. **Context Extraction**: Utilize the `ContextExtractor` class to extract features from MIP instances.
4. **ALNS Iteration**: Run the ALNS algorithm with the configured operators and MAB selector.
5. **Solution Retrieval**: Access the best solution found by the ALNS process.

## Example

```python
instance_path = "path/to/mip_instance.mps"
run_banditalns(instance_path)
