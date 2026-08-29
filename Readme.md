# SamAdams

This repository provides the PyTorch reference implementation for the **SamAdams** adaptive-stepsize Langevin dynamics algorithm introduced in 

> **Benedict Leimkuhler, René Lohmann, and Peter A. Whalley.** (2026). "A Langevin sampling algorithm inspired by the Adam optimizer." *ACM Transactions on Probabilistic Machine Learning*. DOI: [10.1145/3806203](https://doi.org/10.1145/3806203)

**Note on the Codebase & Low-Dimensional Experiments**
> This repository is strictly scoped to the high-dimensional Neural Network implementation of the algorithm. The core C++ engine used for the two-dimensional sampling experiments, extensive numerical testing, and configuration-space observables is actively maintained in our primary framework: [SamplingSuite2D](https://github.com/SchroedingersLion/SamplingSuite2D).

## Repository Contents
* `Samplers.py`: Contains the PyTorch `nn.Module` implementations of the fixed-stepsize `BAOAB` integrator and the adaptive-stepsize `ZBAOABZ` (SamAdams) integrator. 
* `CNN_example.py`: A Minimal Working Example (MWE) demonstrating how to wrap a standard PyTorch CNN with the `ZBAOABZ` sampler and train it on a subset of CIFAR10.

## Dependencies
The code relies on standard deep learning and scientific computing libraries. The following package versions (or higher) are recommended to ensure compatibility:

* `torch` (>= 2.0.0)
* `torchvision` (>= 0.15.0)
* `numpy` (>= 1.21.0)
* `matplotlib` (>= 3.4.0)
