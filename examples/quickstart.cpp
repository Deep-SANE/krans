// KraNS quick start: null space of the 1-D difference operator D, where
// (D x)_i = x_{i+1} - x_i, D is (n-1) x n, and ker(D) = span{(1,...,1)}.
#include "krans/Solvers/CyclicLanczosSolver.hpp"
#include "krans/adapters/universal/StdMatrixAdapter.hpp"
#include "krans/adapters/universal/StdVectorFactory.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <vector>

int main() {
  const std::size_t n = 1000;

  // y = D x  (SpMV)
  auto Ax = [n](const std::vector<double> &x, std::vector<double> &y) {
    for (std::size_t i = 0; i + 1 < n; ++i) y[i] = x[i + 1] - x[i];
  };
  // y = D^T x  (SpMTV)
  auto ATx = [n](const std::vector<double> &x, std::vector<double> &y) {
    for (std::size_t j = 0; j < n; ++j)
      y[j] = (j > 0 ? x[j - 1] : 0.0) - (j + 1 < n ? x[j] : 0.0);
  };

  krans::universal::StdMatrixAdapter A(n - 1, n, Ax, ATx);
  krans::universal::StdVectorFactory vf(n, /*seed=*/42);

  krans::solvers::CyclicLanczosOptions opt;
  opt.k = 1;           // number of null vectors to compute
  opt.cg_maxit = 1000; // inner CG on D D^T (ill-conditioned for large n)
  auto N = krans::solvers::find_nullspace_cyclic_lanczos(A, vf, opt);
  std::printf("null vectors found    : %zu\n", N.size());
  if (N.size() != 1) return 1;

  // Check: ||D v|| ~ 0 and v = +-(1,...,1)/sqrt(n).
  krans::universal::StdVectorAdapter r(n - 1);
  A.multiply(*N[0], r);
  const double *v = N[0]->data();
  double err = 0.0;
  for (std::size_t i = 0; i < n; ++i)
    err = std::max(err, std::abs(std::abs(v[i]) - 1.0 / std::sqrt(double(n))));

  std::printf("||D v||               : %.3e\n", r.norm());
  std::printf("max ||v_i|-1/sqrt(n)| : %.3e\n", err);
  return err < 1e-6 ? 0 : 1;
}
