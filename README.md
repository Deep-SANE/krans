# KraNS — Krastenov NullSpace

KraNS is a matrix-free C++17 library for computing null-space bases of large sparse matrices.

## Quick start

The smallest complete use of KraNS: compute the null space of a matrix that is given **only** by two
functions, `y = A x` (SpMV) and `y = Aᵀ x` (SpMᵀV). No matrix storage is required.

The matrix here is `A` of size `(n−N) × n` with `(A x)_i = x_{i+N} − c·x_i` (`n = 1000`, `N = 3`,
`c = ½`). Its null space is known exactly: it is `N`-dimensional, spanned by the vectors `v_j`
(`j = 0, …, N−1`) that have entries `c^k` at positions `j + kN` (`k = 0, 1, …`) and zeros elsewhere.
`A` is also well-conditioned: its nonzero singular values lie in `[1−c, 1+c]` for any `n`, so both
iterative solvers converge with default options. The example runs both, checks each computed basis
against the exact one, and prints a comparison.

```cpp
// KraNS quick start: null space of A, (n-N) x n, (A x)_i = x_{i+N} - c x_i.
// ker(A) has dimension N; the basis vector v_j has entries c^k at positions
// j + kN (k = 0, 1, ...). The nonzero singular values lie in [1-c, 1+c].
//
// Usage: krans_quickstart [cyclic|lobpcg|both]   (default: both)
#include "krans/Solvers/CyclicLanczosSolver.hpp"
#include "krans/Solvers/LOBPCGSolver.hpp"
#include "krans/adapters/universal/StdMatrixAdapter.hpp"
#include "krans/adapters/universal/StdVectorFactory.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

int main(int argc, char **argv) {
  const std::string which = argc > 1 ? argv[1] : "both";
  if (which != "cyclic" && which != "lobpcg" && which != "both") {
    std::fprintf(stderr, "usage: %s [cyclic|lobpcg|both]\n", argv[0]);
    return 2;
  }

  const std::size_t n = 1000, N = 3;
  const double c = 0.5;

  // y = A x  (SpMV)
  auto Ax = [=](const std::vector<double> &x, std::vector<double> &y) {
    for (std::size_t i = 0; i + N < n; ++i) y[i] = x[i + N] - c * x[i];
  };
  // y = A^T x  (SpMTV)
  auto ATx = [=](const std::vector<double> &x, std::vector<double> &y) {
    for (std::size_t j = 0; j < n; ++j)
      y[j] = (j >= N ? x[j - N] : 0.0) - (j + N < n ? c * x[j] : 0.0);
  };

  krans::universal::StdMatrixAdapter A(n - N, n, Ax, ATx);
  krans::universal::StdVectorFactory vf(n, /*seed=*/42);

  // Check a computed basis U: ||A u|| ~ 0 and u lies in span{v_0, ..., v_{N-1}}.
  int failures = 0;
  auto report = [&](const char *name, const auto &U, double seconds) {
    double max_res = 0.0, max_dist = 0.0;
    krans::universal::StdVectorAdapter r(n - N);
    for (const auto &u : U) {
      A.multiply(*u, r);
      max_res = std::max(max_res, r.norm());
      std::vector<double> d(u->data(), u->data() + n);
      for (std::size_t j = 0; j < N; ++j) { // subtract projection onto v_j
        double dot = 0.0, vv = 0.0;
        for (std::size_t i = j, k = 0; i < n; i += N, ++k) {
          dot += d[i] * std::pow(c, k);
          vv += std::pow(c, 2 * k);
        }
        for (std::size_t i = j, k = 0; i < n; i += N, ++k)
          d[i] -= dot / vv * std::pow(c, k);
      }
      double dist = 0.0;
      for (double e : d) dist += e * e;
      max_dist = std::max(max_dist, std::sqrt(dist));
    }
    std::printf("%-8s %-9zu %-13.1e %-20.1e %.3f s\n", name, U.size(),
                max_res, max_dist, seconds);
    if (U.size() != N || max_dist > 1e-8) ++failures;
  };
  auto since = [](auto t0) {
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0)
        .count();
  };

  std::printf("solver   vectors   max ||A u||   max dist(u, ker A)   time\n");
  if (which != "lobpcg") {
    krans::solvers::CyclicLanczosOptions opt;
    opt.k = N; // number of null vectors to compute
    auto t0 = std::chrono::steady_clock::now();
    auto U = krans::solvers::find_nullspace_cyclic_lanczos(A, vf, opt);
    report("cyclic", U, since(t0));
  }
  if (which != "cyclic") {
    krans::solvers::LOBPCGOptions opt;
    opt.nev = N; // number of null vectors to compute
    auto t0 = std::chrono::steady_clock::now();
    auto U = krans::solvers::find_nullspace_lobpcg(A, vf, opt);
    report("lobpcg", U, since(t0));
  }
  return failures == 0 ? 0 : 1;
}
```

This file is [`examples/quickstart.cpp`](examples/quickstart.cpp). Build and run it:

```bash
cmake -S . -B build
cmake --build build --target krans_quickstart -j
./build/krans_quickstart          # both solvers; or: ./build/krans_quickstart lobpcg
```

Expected output (timings vary by machine):

```
solver   vectors   max ||A u||   max dist(u, ker A)   time
cyclic   3         5.4e-13       6.1e-13              0.012 s
lobpcg   3         3.4e-11       6.3e-11              0.006 s
```

Notes:
- `StdMatrixAdapter` wraps any callables with the signature
  `void(const std::vector<double>& x, std::vector<double>& y)`; `y` is preallocated to the right size.
- `StdVectorFactory` creates the work vectors the solvers need (seeded for reproducibility).
- The result is a vector of `std::unique_ptr<krans::VectorInterface>`, an orthonormal basis of the
  computed null space; `->data()` gives the raw values.
- Ill-conditioned matrices may need tuned options; see
  [Convergence of the iterative solvers](#convergence-of-the-iterative-solvers).

## Header-only or compiled?

KraNS is **not header-only** — it has to be built and linked.

| Part | Where | Form |
|---|---|---|
| Interfaces (`MatrixInterface`, `VectorInterface`, `VectorFactory`) | `include/krans/` | header-only |
| Adapters (universal `Std*`, Eigen `Eigen*`) | `include/krans/adapters/` | header-only |
| Utilities (Gram–Schmidt, projection) | `src/Utils/` | static library `krans_core` |
| Solvers (`cyclic`, `lobpcg`, dense reference solvers, optional ones), matrix I/O | `src/Solvers/`, `src/IO/` | static library `krans_eigen` (depends on `krans_core`) |

- Link against **`krans_eigen`**; it pulls in `krans_core`.
- **Eigen** is required to *build* KraNS (it is used internally for the small dense projected problems;
  CMake fetches Eigen 3.4 if it is not installed). Your own code does not need to include Eigen when it uses
  only the universal `Std*` adapters, as in the quick start.

Using KraNS from your own code today:
1. **Inside the KraNS build** — add an executable to `CMakeLists.txt` the same way `krans_quickstart` is added
   (`target_link_libraries(<your_target> PRIVATE krans_eigen)`).
2. **Plain compiler command** — after building KraNS into `build/`:
   ```bash
   g++ -std=c++17 -O2 -I path/to/krans/include my_app.cpp \
       path/to/krans/build/libkrans_eigen.a path/to/krans/build/libkrans_core.a -o my_app
   ```
   The static libraries are built with link-time optimization, so use the same compiler that built KraNS.

CMake package support (`find_package(KraNS)` / `add_subdirectory`) is planned but not available yet.

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

## Convergence of the iterative solvers

The matrix-free solvers converge at a rate governed by the conditioning of the operators they iterate
with: `AᵀA` for `lobpcg`, and `AAᵀ` for the inner CG that `cyclic` uses to project candidates onto
`ker(A)`. For ill-conditioned `A` — e.g. discretized differential operators, where the condition number
grows like `n²` — the default iteration limits can be too small. The solvers then return **fewer vectors
than requested** instead of failing loudly.

**Detecting it.** Compare the number of returned vectors with the number requested, or use the `_ex`
variants (`find_nullspace_lobpcg_ex`, `find_nullspace_cyclic_lanczos_ex`) and check
`converged_to_target`. With `verbose = true`, both solvers print progress and a final diagnosis.

**`lobpcg` options** (`LOBPCGOptions`):
- `max_iters` (default 1000) — unpreconditioned LOBPCG may need many more. `LOBPCGResult` reports
  `iters`, `accept_threshold` and `best_unaccepted_residual` (how close a non-converged run got).
- `block_size` (default 0 = `min(16, nev)`) — values above `nev` are honored; the extra vectors cut the
  iteration count sharply but make each iteration more expensive.
- `M_inv` — a preconditioner for `AᵀA`, the standard way to speed up convergence when one is available.

**`cyclic` options** (`CyclicLanczosOptions`):
- `cg_maxit` (default 80) — the projection step works only if its CG on `AAᵀ` converges.
  `CyclicLanczosResult::cg_maxit_hits` counts the CG solves that stopped at this limit.
- `precond_AAt` — a preconditioner for `AAᵀ`.

Measured on the 1-D difference operator `D`, `(D x)_i = x_{i+1} − x_i` of size `(n−1) × n` with
`ker(D) = span{(1, …, 1)}` (`n = 1000`, one null vector, three random starts; see
[`tests/test_difference_operator.cpp`](tests/test_difference_operator.cpp)):

| Solver and settings | Iterations | Time | Result |
|---|---|---|---|
| `lobpcg`, defaults (`block_size` 1, `max_iters` 1000) | 1 000 | 0.02 s | no vector found |
| `lobpcg`, `block_size` 1, unlimited iterations | 5 940 – 11 505 | 0.14 – 0.27 s | found |
| `lobpcg`, `block_size` 16, unlimited iterations | 384 – 439 | 0.9 – 1.0 s | found |
| `cyclic`, defaults (`cg_maxit` 80) | all 245 CG solves hit `cg_maxit` | 0.17 s | no vector found |
| `cyclic`, `cg_maxit` 1000 | 1 CG solve | 0.007 s | found |

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
