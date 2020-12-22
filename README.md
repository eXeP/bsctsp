# Optimizing preprocessing and local search heuristics of the traveling salesman problem with parallel programming

My Bachelor's thesis on a parallelizing a TSP solver. Contains CUDA implementations of 2-opt and subgradient optimization for alpha-nearness.

## Solver
Needs cuda (nvcc) and g++ installed to compile:
```bash
make solver
```

The executable has some parameters:
+ Problem instance (Only EUC_2D supported)
+ Algorithm (RAND, ALPHA1, ALPHA2, ALPHA3)
+ Device (CPU, GPU)
+ Candidate set size (Applies only to ALPHA1-3)
```bash
./solver pr1002 ALPHA1 GPU 10
```