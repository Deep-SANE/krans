// Regression tests on the 1-D difference operator D ((n-1) x n,
// (D x)_i = x_{i+1} - x_i, ker(D) = span{1}). D^T D and D D^T are
// ill-conditioned (cond ~ n^2), which stresses iteration limits.
//
// Checks are explicit (not assert) because the build defines NDEBUG.
#include "krans/Solvers/CyclicLanczosSolver.hpp"
#include "krans/Solvers/LOBPCGSolver.hpp"
#include "krans/adapters/universal/StdMatrixAdapter.hpp"
#include "krans/adapters/universal/StdVectorFactory.hpp"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <vector>

using krans::universal::StdMatrixAdapter;
using krans::universal::StdVectorAdapter;
using krans::universal::StdVectorFactory;

static int failures = 0;

static void check(bool ok, const char *what) {
  std::cout << (ok ? "  [ok]   " : "  [FAIL] ") << what << "\n";
  if (!ok)
    ++failures;
}

static StdMatrixAdapter difference_operator(std::size_t n) {
  auto Ax = [n](const std::vector<double> &x, std::vector<double> &y) {
    for (std::size_t i = 0; i + 1 < n; ++i) y[i] = x[i + 1] - x[i];
  };
  auto ATx = [n](const std::vector<double> &x, std::vector<double> &y) {
    for (std::size_t j = 0; j < n; ++j)
      y[j] = (j > 0 ? x[j - 1] : 0.0) - (j + 1 < n ? x[j] : 0.0);
  };
  return StdMatrixAdapter(n - 1, n, Ax, ATx);
}

// v = +-(1,...,1)/sqrt(n) up to tol
static bool is_constant_unit(const krans::VectorInterface &v, double tol) {
  const std::size_t n = v.size();
  const double *d = v.data();
  for (std::size_t i = 0; i < n; ++i)
    if (std::abs(std::abs(d[i]) - 1.0 / std::sqrt(double(n))) > tol)
      return false;
  return true;
}

int main() {
  const std::size_t n = 100;
  StdMatrixAdapter A = difference_operator(n);
  StdVectorFactory vf(n, /*seed=*/42);

  std::cout << "LOBPCG, block_size = 0 (auto), default max_iters\n";
  {
    krans::solvers::LOBPCGOptions opt;
    auto R = krans::solvers::find_nullspace_lobpcg_ex(A, vf, opt);
    check(R.block_size == 1, "auto block size is min(16, nev) = 1");
    check(R.converged_to_target == (R.right_null.size() == 1),
          "converged_to_target matches the number of vectors returned");
    if (!R.converged_to_target)
      check(R.iters == opt.max_iters &&
                R.best_unaccepted_residual > R.accept_threshold,
            "non-converged run reports iters == max_iters and the residual "
            "gap");
  }

  std::cout << "LOBPCG, block_size = 16 > nev = 1\n";
  {
    krans::solvers::LOBPCGOptions opt;
    opt.block_size = 16;
    auto R = krans::solvers::find_nullspace_lobpcg_ex(A, vf, opt);
    check(R.block_size == 16, "explicit block size above nev is honored");
    check(R.converged_to_target && R.right_null.size() == 1, "converged");
    if (!R.right_null.empty())
      check(is_constant_unit(*R.right_null[0], 1e-8),
            "vector is +-(1,...,1)/sqrt(n)");
  }

  std::cout << "LOBPCG, block_size larger than cols(A)\n";
  {
    StdMatrixAdapter A4 = difference_operator(4);
    StdVectorFactory vf4(4, /*seed=*/42);
    krans::solvers::LOBPCGOptions opt;
    opt.block_size = 16;
    auto R = krans::solvers::find_nullspace_lobpcg_ex(A4, vf4, opt);
    check(R.block_size == 4, "block size is capped at cols(A)");
    check(R.converged_to_target, "converged");
  }

  const std::size_t nc = 1000;
  StdMatrixAdapter Ac = difference_operator(nc);
  StdVectorFactory vfc(nc, /*seed=*/42);

  std::cout << "Cyclic, default cg_maxit, n = " << nc << "\n";
  {
    krans::solvers::CyclicLanczosOptions opt;
    auto R = krans::solvers::find_nullspace_cyclic_lanczos_ex(Ac, vfc, opt);
    check(R.cg_solves > 0, "polishing CG solves are counted");
    if (!R.converged_to_target)
      check(R.cg_maxit_hits > 0,
            "non-converged run reports polishing CG solves hitting cg_maxit");
  }

  std::cout << "Cyclic, cg_maxit = 1000, n = " << nc << "\n";
  {
    krans::solvers::CyclicLanczosOptions opt;
    opt.cg_maxit = 1000;
    auto R = krans::solvers::find_nullspace_cyclic_lanczos_ex(Ac, vfc, opt);
    check(R.converged_to_target && R.right_null.size() == 1, "converged");
    if (!R.right_null.empty())
      check(is_constant_unit(*R.right_null[0], 1e-8),
            "vector is +-(1,...,1)/sqrt(n)");
  }

  std::cout << (failures ? "FAILED: " : "All checks passed.")
            << (failures ? std::to_string(failures) + " check(s)\n" : "\n");
  return failures ? 1 : 0;
}
