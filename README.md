# KraNS — Krastenov NullSpace

KraNS is a matrix-free C++17 library for computing null-space bases of large sparse matrices.

## Features

- **Abstract interfaces** for matrix and vector operations (Matrix-Free paradigm)
- **Backend Adapters**:
  - Eigen-backed vectors and matrices
  - Universal callback-based adapters (`StdMatrixAdapter`, `StdVectorAdapter`) for seamless integration with any external library via C++ lambdas.
- **Utilities**:
  - Modified Gram–Schmidt orthogonalization
  - Orthogonal projection onto nullspace
  - Small dense SVD (Eigen backend)
  - **RREF Subspace Comparison**: Numerically stable Reduced Row Echelon Form validation to rigorously prove identical subspaces.
  - Timer for benchmarking
- **I/O**: Matrix Market (.mtx) and simple IJA triplets (0-based indices)
- **Solvers**:
  - **Direct (dense/sparse-factorization) reference solvers**
    - `svd` — Jacobi/BDC SVD with FULL V; take trailing V columns
    - `qrcp` — Column-pivoted QR; rank via qr.rank(); explicit kernel via triangular solve + unpermutation
    - `cod` — Complete orthogonal decomposition; trailing Z with best of {Z_k, P*Z_k, P^T*Z_k} 
    - `fullpivlu` — Full-pivot LU; rank via lu.rank(); kernel from lu.kernel()
    - `suitesparse` *(Optional: USED FOR BENCHMARKING)* — Direct multifrontal sparse QR factorization (SPQR) defining the "Memory Wall"
  - **Iterative (matrix-free)**:
    - `cyclic` (augmented operator / harmonic interior on H(A), projector polishing, optional left nullspace)
    - `lobpcg` (Locally Optimal Block Preconditioned Conjugate Gradient method for finding the right nullspace)
    - `spectra` *(Optional: USED FOR BENCHMARKING)* — Implicitly Restarted Lanczos Method operating on normal equations (A^T A)
    - `primme` *(Optional: USED FOR BENCHMARKING)* — High-performance hybrid solver
    - `slepc` *(Optional: USED FOR BENCHMARKING)* — Distributed HPC standard utilizing SVDCYCLIC

- Compact test suite and clean CMake targets (`krans_core`, `krans_eigen`)

## Build Instructions

### Requirements

- C++17
- [Eigen 3](https://eigen.tuxfamily.org) (e.g., `brew install eigen`, `sudo apt install -y libeigen3-dev`). If not found locally, CMake will automatically fetch it.
- **Optional Benchmarking Dependencies**:
  - `make` (required to build PRIMME)
  - `MPI` and `PETSc` / `SLEPc` (required for SLEPc integration)
  - `SuiteSparse` (e.g., `brew install suitesparse`, required for SPQR)

### Build

**Recommended (Release):**

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

**Build with full Benchmarking Suite (Release):**

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
  -DENABLE_SPECTRA=ON \
  -DENABLE_PRIMME=ON \
  -DENABLE_SLEPC=ON \
  -DENABLE_SUITESPARSE=ON
cmake --build build -j
```

**Tests (Debug):**

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Debug
cmake --build build -j
```

### Build Tests

```bash
cmake --build build --target run_core_adapter_tests
# or run individual ./build/test_* binaries
```

Or you can run cross-validation of all available solvers (RREF verification)
```bash
./build/test_all_solvers
```

## CLI

- Builds by default. Accepts `.mtx` and `.ija` inputs, runs a selected solver, and writes:
  - `N.csv` : right-nullspace basis (columns are basis vectors)
  - `report.json` : sizes, timings, residuals, orthogonality
  - optionally `N_left.csv` : left-nullspace basis (for `cyclic` with `--left`)

### Usage:
```bash
cd build/
./krans_nullspace --solver [svd|qrcp|cod|fullpivlu|cyclic|lobpcg|spectra|primme|slepc|suitesparse] \
  --input A.(mtx|ija) \
  [--out N.csv] [--report report.json]
  [--tol <rel_tol>]           # direct solvers only (default 1e-12)
```
*(Note: `spectra`, `primme`, `slepc`, and `suitesparse` are only available if compiled with their respective CMake flags).*

#### Cyclic options (optional):

```bash
  --k <int>              # number of right null vectors (default 1)
  --ncv <int>            # subspace size (default 48)
  --max-restarts <int>   # restarts (default 80)
  --tol-abs <float>      # accept if ||A v|| <= max(tol_abs, tol_rel * ||A||_F), default 1e-12
  --tol-rel <float>      # relative part in acceptance, default 1e-10
  --proj-passes <int>    # projector polish passes (1 or 2)
  --cg-maxit <int>       # CG iters for normal equations
  --cg-rel <float>       # CG relative tol
  --verbose              # progress output
  --left                 # also write N_left.csv (left nullspace)
```

#### LOBPCG options (optional):

```bash
  --k <int>                  # number of right null vectors (default 1)
  --tol-abs <float>          # accept if ||A v|| <= max(tol_abs, tol_rel * ||A||_F), default 1e-12
  --tol-rel <float>          # relative part in acceptance, default 1e-10
  --lobpcg-block <int>       # block size (default min(16,k))
  --lobpcg-max-it <int>      # max outer iterations (default 1000)
  --lobpcg-restart <float>   # restart fraction in (0.1..1], default 0.9
  --lobpcg-soft-lock <0|1>   # soft locking on/off (default 1)
  --verbose                  # progress output
```

#### Example usage:

1) Direct SVD with tight tolerance:
```bash
./build/krans_nullspace --solver svd --input data/A.ija --tol 1e-12 \
  --out N.csv --report report.json
```

2) Cyclic: request 3 right-null vectors, export left space, verbose:
```bash
./build/krans_nullspace --solver cyclic --input data/A.ija \
  --k 3 --ncv 96 --max-restarts 200 \
  --tol-abs 1e-12 --tol-rel 1e-10 \
  --proj-passes 2 --cg-maxit 200 --cg-rel 1e-12 \
  --left --verbose \
  --out N.csv --report report.json
  ```

3) Custom output paths:
  ```bash
  ./build/krans_nullspace --solver qrcp --input data/A.mtx \
  --out results/nullspace.csv --report results/summary.json
  ```
### Notes
- `N.csv` is `n x k`; each column is a basis vector of `ker(A)`. Any solution of `A v = 0`can be written as `v = N * c`.
- For direct solvers, `--tol` is a relative rank threshold; loosen to detect larger near-null subspaces.
- For iterative solvers (`cyclic`, `lobpcg`, `primme`, `spectra`), use `--k` to request the number of right-null vectors you want to extract. For `cyclic`, `--left` writes `N_left.csv`.
- The framework includes a post-processing filter that validates every computed vector against `||Av||_2 < tau` to guarantee physical correctness regardless of the underlying algorithm's algebraic convergence.

## Origin / Credits

KraNS originated in the master’s thesis of Alexandr Krastenov (CTU in Prague, Faculty of Information Technology), originally developed at [gitlab.fit.cvut.cz/krastale/dp-sparse-nullspace](https://gitlab.fit.cvut.cz/krastale/dp-sparse-nullspace).
