# SamAdams PyTorch Implementation

This repository provides the PyTorch reference implementation for the **SamAdams** adaptive-stepsize Langevin dynamics algorithm introduced in 

**Benedict Leimkuhler, René Lohmann, and Peter A. Whalley.** (2026). "A Langevin sampling algorithm inspired by the Adam optimizer." *ACM Transactions on Probabilistic Machine Learning*. DOI: [10.1145/3806203](https://doi.org/10.1145/3806203)

**Note on the Codebase & Low-Dimensional Experiments**  
This repository is strictly scoped to the Neural Network implementation of the algorithm. For the reference implementation for the two-dimensional models (e.g., the Beale or the Star potential), we refer to the repository [SamplingSuite2D](https://github.com/SchroedingersLion/SamplingSuite2D) (Zenodo DOI: [10.5281/zenodo.22072352](https://doi.org/10.5281/zenodo.22072352)).

## Repository Contents
* `src/Samplers.py`: Contains the PyTorch `nn.Module` implementations of the fixed-stepsize `BAOAB` integrator and the adaptive-stepsize `ZBAOABZ` (SamAdams) integrator. 
* `src/CNN_example.py`: A Minimal Working Example (MWE) demonstrating how to wrap a standard PyTorch CNN with the `ZBAOABZ` sampler and train it on a subset of CIFAR10.

## Dependencies
The MWE was tested using the following specific package versions:

* `torch` (== 2.6.0)
* `torchvision` (== 0.21.0)
* `numpy` (== 2.2.4)
* `matplotlib` (== 3.10.0)
