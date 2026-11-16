# pyEBM2D: Python Translations of the 2D Energy Balance Model

This repository contains Python translations of the 2D seasonal Energy Balance Model (EBM) originally written in Fortran by Zhuang, North, and Stevens (2017). 

The primary purpose of this repository is to explore the speed, value, and feasibility of translating legacy Fortran climate models into the modern Python ecosystem.

## Implementations Included

1. **Original Fortran (`EBM_Zuang_North_Stevens/`)**
   - The original highly-optimized Fortran codebase utilizing an implicit Full Multigrid (FMG) solver.
   - Reference: [Zhuang, North, and Stevens (2017)](https://www.sciencedirect.com/science/article/pii/S2352711017300262)

2. **Native Python (`ebm_python_native.py`)**
   - **Method:** Replicates the Fortran implicit Crank-Nicolson scheme using exact sparse matrix construction and solves it using `scipy.sparse.linalg` (SuperLU pre-factorization).
   - **Performance:** Extremely fast. By factoring the constant grid properties once, it completes a 40-year simulation in roughly 3 seconds on a modern CPU, outperforming or matching the legacy FMG solver while remaining highly readable.
   - **Best For:** Standard scientific workflows, drop-in replacement for the Fortran model.

3. **JAX Python (`ebm_python_jax.py`)**
   - **Method:** A JAX-accelerated version using `jax.lax.scan` to compile the entire 40-year time loop into machine code via XLA.
   - **Differences & Performance:** JAX currently lacks a native sparse LU direct solver, so this implementation relies on a dense matrix LU factorization. This gives it a larger memory footprint ($O(N^3)$) and a noticeable upfront Just-In-Time (JIT) compilation overhead. 
   - **Best For:** Machine Learning integration. Because the loop is fully compiled in JAX, it supports Automatic Differentiation (AutoDiff). This allows researchers to compute exact gradients of the simulation for parameter tuning (e.g., automatically tuning diffusion coefficients to historical climate data).

## Running the Models

A wrapper script is provided to run all versions and generate final temperature contour plots:

```bash
./run_all.sh
```
Plots will be saved to the `figures/` directory.
