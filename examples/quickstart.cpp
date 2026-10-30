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
