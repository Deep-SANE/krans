#include "krans/Solvers/CyclicLanczosSolver.hpp"

#include <Eigen/Dense>
#include <algorithm>
#include <cassert>
#include <cmath>
#include <iostream>
#include <limits>
#include <numeric>
#include <utility>

#include "krans/Utils/GramSchmidt.hpp"
#include "krans/Utils/Projection.hpp"

namespace krans::solvers {

// ---------------- Golub–Kahan helpers ---------------------------------------
// Lanczos on H(A) = [0 A; A^T 0] started from [0; y_0] is Golub–Kahan
// bidiagonalization: the Lanczos vectors alternate [0; y_0], [x_1; 0],
// [0; y_1], ..., every diagonal entry of T is 0, and each step needs a single
// product with A or A^T. Only the nonzero halves are stored (Y of size n,
// X of size m), so reorthogonalization runs on half-length vectors and only
// against vectors of the same kind.

// w <- w - (s . w) s for each s in S, in order (one Gram–Schmidt pass).
static inline void
orth_pass(VectorInterface &w,
          const std::vector<std::unique_ptr<VectorInterface>> &S) {
  for (const auto &s : S) {
    const double c = s->dot(w);
    w.add_scaled(*s, -c);
  }
}

static inline double residual_Av(const MatrixInterface &A,
                                 const VectorInterface &v,
                                 VectorInterface &tmp_m) {
  return krans::utils::residual_norm_Av(A, v, tmp_m);
}

static inline void apply_AAt(const MatrixInterface &A,
                             const VectorInterface &x_m, VectorInterface &y_m,
                             VectorInterface &tmp_n) {
  A.multiply_transpose(x_m, tmp_n); // tmp_n = A^T x
  A.multiply(tmp_n, y_m);           // y_m   = A tmp_n
}

static std::pair<std::size_t, double>
cg_normal_eq_prec(const MatrixInterface &A, const VectorFactory &vf,
                  const VectorInterface &rhs_m, VectorInterface &w_m,
                  const AAtPreconditioner &M_inv, std::size_t maxit,
                  double rel_tol, double breakdown_tol) {
  const std::size_t m = A.rows(), n = A.cols();
  auto r = vf.create_zero_of_size(m);
  auto z = vf.create_zero_of_size(m);
  auto p = vf.create_zero_of_size(m);
  auto Ap = vf.create_zero_of_size(m);
  auto tmp_n = vf.create_zero_of_size(n);

  w_m.set_zero();
  r->copy_from(rhs_m);

  const double rhsn = std::sqrt(std::max(0.0, r->dot(*r)));
  if (!(rhsn > 0.0) || !std::isfinite(rhsn))
    return {0, rhsn}; // zero (or non-finite) rhs: w = 0

  if (M_inv)
    M_inv(*r, *z);
  else
    z->copy_from(*r);
  p->copy_from(*z);

  double rz = r->dot(*z);
  if (!(rz > 0.0) || !std::isfinite(rz))
    return {0, rhsn}; // preconditioner not SPD on r
  double rnorm = std::sqrt(rz);
  const double tol = std::max(rel_tol * rnorm, 1e-30);
  if (rnorm <= tol)
    return {0, rnorm};

  std::size_t k = 0;
  for (; k < maxit; ++k) {
    apply_AAt(A, *p, *Ap, *tmp_n);
    const double pAp = p->dot(*Ap);
    if (!(std::isfinite(pAp)) || std::fabs(pAp) <= breakdown_tol)
      break; // breakdown or semidefinite roundoff

    const double alpha = rz / pAp;
    w_m.add_scaled(*p, alpha);
    r->add_scaled(*Ap, -alpha);

    if (M_inv)
      M_inv(*r, *z);
    else
      z->copy_from(*r);

    const double rz_new = r->dot(*z);
    if (!std::isfinite(rz_new))
      break;
    rnorm = std::sqrt(std::max(0.0, rz_new));
    if (rnorm <= tol) {
      ++k;
      break;
    }

    const double beta = rz_new / std::max(1e-300, rz);
    rz = rz_new;

    p->scale(beta);
    p->add_scaled(*z, 1.0);
  }
  return {k, rnorm};
}

static double estimate_fro_norm(const MatrixInterface &A,
                                const VectorFactory &vf, std::size_t samples) {
  samples = std::max<std::size_t>(1, samples);
  double sum2 = 0.0;
  for (std::size_t s = 0; s < samples; ++s) {
    auto r = vf.create_random_of_size(A.cols());
    auto Ar = vf.create_zero_of_size(A.rows());
    A.multiply(*r, *Ar);
    const double n = Ar->norm();
    sum2 += n * n;
  }
  return std::sqrt(sum2 / static_cast<double>(samples));
}

static inline void gradient_nullspace_step(const MatrixInterface &A,
                                           VectorInterface &v,
                                           VectorInterface &tmp_m,
                                           VectorInterface &tmp_n, double tau) {
  krans::utils::gradient_nullspace_step(A, v, tmp_m, tmp_n, tau);
}

// -------- Randomized fallback: project + gradient ----------------------------
// Attempts to produce one right-null vector by: GS vs current basis,
// projector pass v <- v - A^T(AA^T)^{-1}Av via CG, then gradient steps on
// f(v)=1/2||Av||^2 with safe steps via ||A||_F estimate.
static bool try_project_random_to_nullspace(
    const MatrixInterface &A, const VectorFactory &vf, double tol_accept,
    std::vector<std::unique_ptr<VectorInterface>> &basis, int grad_max_iters,
    int frob_samples) {
  const std::size_t m = A.rows(), n = A.cols();

  auto v = vf.create_random_of_size(n);
  // GS vs current basis
  for (auto &q : basis) {
    const double c = v->dot(*q);
    v->add_scaled(*q, -c);
  }
  double nv = v->norm();
  if (nv < 1e-14)
    return false;
  v->scale(1.0 / nv);

  // Work buffers
  auto tmp_m = vf.create_zero_of_size(m);
  auto w_m = vf.create_zero_of_size(m);
  auto tmp_n = vf.create_zero_of_size(n);

  // One projector pass (unpreconditioned here)
  A.multiply(*v, *tmp_m);
  auto [its, rnorm] = cg_normal_eq_prec(A, vf, *tmp_m, *w_m, /*M_inv*/ nullptr,
                                        /*maxit*/ 500, /*rel_tol*/ 1e-12,
                                        /*breakdown_tol*/ 1e-30);
  if (its > 0 || rnorm < tmp_m->norm()) {
    A.multiply_transpose(*w_m, *tmp_n);
    v->add_scaled(*tmp_n, -1.0);
  }

  // Re-GS + renorm
  for (auto &q : basis) {
    const double c = v->dot(*q);
    v->add_scaled(*q, -c);
  }
  nv = v->norm();
  if (nv >= 1e-14)
    v->scale(1.0 / nv);

  // Check acceptance
  double res = residual_Av(A, *v, *tmp_m);
  if (res <= tol_accept) {
    basis.push_back(std::move(v));
    return true;
  }

  // Gradient fallback
  const double fn = estimate_fro_norm(A, vf, std::max(1, frob_samples));
  const double L2 = std::max(1e-30, fn * fn);
  const double tau = 0.9 / L2;

  for (int it = 0; it < grad_max_iters && res > tol_accept; ++it) {
    gradient_nullspace_step(A, *v, *tmp_m, *tmp_n, tau);

    // GS + renorm
    for (auto &q : basis) {
      const double c = v->dot(*q);
      v->add_scaled(*q, -c);
    }
    nv = v->norm();
    if (nv < 1e-14)
      return false;
    v->scale(1.0 / nv);

    res = residual_Av(A, *v, *tmp_m);
  }

  if (res <= tol_accept) {
    basis.push_back(std::move(v));
    return true;
  }
  return false;
}

// ---------------- Tiny problems (plain & refined) ----------------------------
struct TinyEV {
  Eigen::VectorXd evals;
  Eigen::MatrixXd evecs;
};

// plain Ritz on tridiagonal T
static TinyEV solve_tiny_tridiag(const std::vector<double> &alpha,
                                 const std::vector<double> &beta) {
  const int l = static_cast<int>(alpha.size());
  Eigen::MatrixXd T = Eigen::MatrixXd::Zero(l, l);
  for (int i = 0; i < l; ++i) {
    T(i, i) = alpha[(std::size_t)i];
    if (i + 1 < l) {
      const double b = beta[(std::size_t)i];
      T(i, i + 1) = b;
      T(i + 1, i) = b;
    }
  }
  Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(T);
  return {es.eigenvalues(), es.eigenvectors()};
}

// Refined interior extraction at tau: with H V = V T + beta_last v_{l+1}
// e_l^T, ||(H - tau I) V c|| = ||Tbar c|| where Tbar = [T - tau I;
// beta_last e_l^T] is (l+1) x l. Its right singular vectors minimize the
// residual over the Krylov space; evals holds the singular values (the
// residual norms). Unlike inverting T - tau I, this stays well
// defined when tau is an exact Ritz value (e.g. an exactly captured null
// vector), which made the old LU-based harmonic solve return NaN.
static TinyEV solve_tiny_refined(const std::vector<double> &alpha,
                                 const std::vector<double> &beta,
                                 double beta_last, double tau) {
  const int l = static_cast<int>(alpha.size());
  Eigen::MatrixXd Tbar = Eigen::MatrixXd::Zero(l + 1, l);
  for (int i = 0; i < l; ++i) {
    Tbar(i, i) = alpha[(std::size_t)i] - tau;
    if (i + 1 < l) {
      const double b = beta[(std::size_t)i];
      Tbar(i, i + 1) = b;
      Tbar(i + 1, i) = b;
    }
  }
  Tbar(l, l - 1) = beta_last;
  Eigen::JacobiSVD<Eigen::MatrixXd> svd(Tbar, Eigen::ComputeFullV);
  return {svd.singularValues(), svd.matrixV()};
}

// Left null vectors of A are right null vectors of A^T.
namespace {
class TransposedMatrix final : public MatrixInterface {
public:
  explicit TransposedMatrix(const MatrixInterface &A) : A_(A) {}
  void multiply(const VectorInterface &x, VectorInterface &y) const override {
    A_.multiply_transpose(x, y);
  }
  void multiply_transpose(const VectorInterface &x,
                          VectorInterface &y) const override {
    A_.multiply(x, y);
  }
  std::size_t rows() const override { return A_.cols(); }
  std::size_t cols() const override { return A_.rows(); }

private:
  const MatrixInterface &A_;
};
} // namespace

// ----------------------------- Main solver -----------------------------------
// Right null space only. The Krylov space is started from z0 = [0; y], i.e.
// Lanczos on H(A) runs as Golub–Kahan bidiagonalization (see above) and never
// picks up components of ker(A^T): for tall A (m > n) that space is large and
// would otherwise flood the cluster of H(A) at 0, so that most interior
// candidates have a (numerically) zero right block. Accepted vectors v are
// locked as [0; v], so only the y-vectors are orthogonalized against them.
static CyclicLanczosResult
find_right_nullspace(const MatrixInterface &A, const VectorFactory &vf,
                     const CyclicLanczosOptions &opt) {
  const std::size_t m = A.rows(), n = A.cols();
  CyclicLanczosResult R;

  if (n == 0 || m == 0)
    return R;

  const std::size_t k_target = static_cast<std::size_t>(
      (opt.nullity_known >= 0) ? opt.nullity_known : static_cast<int>(opt.k));
  if (k_target == 0) {
    R.converged_to_target = true;
    return R;
  }
  if (opt.rank_known >= 0 && static_cast<int>(n) <= opt.rank_known) {
    // full column rank => no right nulls
    return R;
  }
  const std::size_t ncv =
      std::max<std::size_t>(std::max<std::size_t>(2, opt.ncv), k_target + 2);

  // Pre-compute ||A||_F estimate for scale-invariant thresholds
  R.frob_norm_est = estimate_fro_norm(A, vf, std::max(1, opt.frob_samples));
  const double accept_thresh =
      std::max(opt.tol_abs, opt.tol_rel * std::max(1.0, R.frob_norm_est));
  const double prescreen_thresh =
      std::max(0.0, opt.prescreen_factor) * std::max(1.0, R.frob_norm_est);

  // Accepted right-nullspace basis
  std::vector<std::unique_ptr<VectorInterface>> basis;
  basis.reserve(k_target);

  auto gs_against_basis = [&](VectorInterface &v) {
    for (int pass = 0; pass < 2; ++pass)
      for (auto &q : basis) {
        const double c = v.dot(*q);
        v.add_scaled(*q, -c);
      }
  };

  // Restart vector: right block only (top block stays zero).
  auto random_right = [&]() {
    auto y = vf.create_random_of_size(n);
    gs_against_basis(*y);
    return y;
  };
  std::unique_ptr<VectorInterface> y0;
  if (!opt.initial_right_space.empty() && opt.initial_right_space.front())
    y0 = opt.initial_right_space.front()->clone();
  else
    y0 = vf.create_random_of_size(n);

  // Golub–Kahan vectors of the current cycle. Lanczos vector i of H(A) is
  // [0; Y[i/2]] for even i and [X[(i-1)/2]; 0] for odd i.
  std::vector<std::unique_ptr<VectorInterface>> Y, X;
  Y.reserve(ncv / 2 + 1);
  X.reserve(ncv / 2 + 1);

  // Tridiagonal T of H(A) in that basis: alpha (all 0) and beta
  std::vector<double> alpha;
  alpha.reserve(ncv);
  std::vector<double> beta;
  beta.reserve(ncv);

  auto tmp_m = vf.create_zero_of_size(m); // reused for ||A v||
  auto w_m = vf.create_zero_of_size(m);
  auto tmp_n = vf.create_zero_of_size(n);
  std::size_t restarts_since_accept = 0;

  for (std::size_t restart = 0; restart < opt.max_restarts; ++restart) {
    R.restarts = restart;
    alpha.clear();
    beta.clear();

    // ---- start vector z0 = [0; y0], orthogonal to the locked vectors
    gs_against_basis(*y0);
    double ny = y0->norm();
    if (!std::isfinite(ny) || ny < 1e-14) {
      y0 = random_right();
      ny = y0->norm();
      if (!(ny >= 1e-14))
        break; // basis already spans R^n
    }
    y0->scale(1.0 / ny);
    Y.clear();
    X.clear();
    Y.push_back(std::move(y0));

    // ---- Golub–Kahan build up to l <= ncv Lanczos vectors of H(A):
    //   from y_j:  w = A y_j   - beta x_j,   reorth vs X,         -> x_{j+1}
    //   from x_j:  w = A^T x_j - beta y_{j-1}, reorth vs basis, Y -> y_j
    auto w_x = vf.create_zero_of_size(m);
    auto w_y = vf.create_zero_of_size(n);
    std::size_t local_steps = 0;
    double beta_last = 0.0;
    for (std::size_t j = 0; j < ncv; ++j) {
      const bool from_y = (Y.size() > X.size()); // last vector is [0; y]
      VectorInterface &w = from_y ? *w_x : *w_y;
      if (from_y) {
        A.multiply(*Y.back(), w);
        if (!beta.empty())
          w.add_scaled(*X.back(), -beta.back());
      } else {
        A.multiply_transpose(*X.back(), w);
        w.add_scaled(*Y.back(), -beta.back());
      }

      const int reorth_passes = static_cast<int>(opt.reorth);
      for (int pass = 0; pass < reorth_passes; ++pass) {
        if (from_y) {
          orth_pass(w, X);
        } else {
          orth_pass(w, basis); // locked [0; v]
          orth_pass(w, Y);
        }
      }

      const double bj = std::sqrt(w.dot(w));
      alpha.push_back(0.0);
      beta_last = bj;

      if (!std::isfinite(bj) || bj < 1e-14 || Y.size() + X.size() >= ncv)
        break;

      beta.push_back(bj);
      auto vnew = w.clone();
      vnew->scale(1.0 / bj);
      (from_y ? X : Y).push_back(std::move(vnew));
      ++local_steps;
    }
    R.krylov_steps_total += local_steps;

    const std::size_t l = alpha.size();
    if (l == 0 || !std::isfinite(beta_last))
      break;

    // ---- tiny problem & interior selection
    TinyEV tev;
    std::vector<int> ord;
    if (opt.use_harmonic) {
      tev = solve_tiny_refined(alpha, beta, beta_last, opt.target_tau);
      ord.resize(static_cast<std::size_t>(tev.evals.size()));
      std::iota(ord.begin(), ord.end(), 0);
      // singular values: smallest residual first
      std::stable_sort(ord.begin(), ord.end(), [&](int a, int b) {
        return tev.evals[a] < tev.evals[b];
      });
    } else {
      tev = solve_tiny_tridiag(alpha, beta);
      ord.resize(static_cast<std::size_t>(tev.evals.size()));
      std::iota(ord.begin(), ord.end(), 0);
      std::stable_sort(ord.begin(), ord.end(), [&](int a, int b) {
        return std::abs(tev.evals[a] - opt.target_tau) <
               std::abs(tev.evals[b] - opt.target_tau);
      });
    }
    if (!tev.evals.allFinite() || !tev.evecs.allFinite()) {
      if (opt.verbose)
        std::cout << "[Cyclic] non-finite projected problem, random restart\n";
      y0 = random_right();
      continue;
    }

    const std::size_t test_cap =
        std::min<std::size_t>(l, std::max<std::size_t>(k_target, 8));
    double best_res = std::numeric_limits<double>::infinity();
    std::unique_ptr<VectorInterface> best_v; // best non-accepted candidate
    bool accepted_this_cycle = false;

    // Candidates with a (numerically) zero right block, i.e. pure [x; 0]
    // directions, are skipped and do not count towards test_cap.
    std::size_t tested = 0;
    // Once a polish stalls at cg_maxit without reaching the threshold, the
    // remaining (worse-ranked) candidates of this cycle are not polished:
    // their residuals live on even smaller singular values, where CG is
    // slowest, so they would burn cg_maxit iterations each for nothing.
    bool polish_stalled = false;
    for (std::size_t t = 0; t < ord.size() && tested < test_cap; ++t) {
      const int jbest = ord[t];

      // candidate right vector v := right block of the Lanczos combination,
      // i.e. sum over even i of c_i y_{i/2}. Pure [x; 0] combinations
      // (the odd-only singular vectors of Tbar) are skipped before forming v.
      double c_even2 = 0.0;
      for (std::size_t i = 0; i < l; i += 2)
        c_even2 += tev.evecs(static_cast<int>(i), jbest) *
                   tev.evecs(static_cast<int>(i), jbest);
      if (!(c_even2 >= 1e-20))
        continue;
      auto v = vf.create_zero_of_size(n);
      for (std::size_t i = 0; i < l; i += 2)
        v->add_scaled(*Y[i / 2], tev.evecs(static_cast<int>(i), jbest));

      // GS vs accepted right-nullspace, normalize
      gs_against_basis(*v);
      double nv = v->norm();
      if (!std::isfinite(nv) || nv < 1e-8)
        continue;
      v->scale(1.0 / nv);
      ++tested;

      // Pre-screen: raw residual (avoid polishing hopeless candidates)
      A.multiply(*v, *tmp_m);
      double res = tmp_m->norm();
      if (!std::isfinite(res))
        continue;
      if (res > prescreen_thresh || polish_stalled) {
        if (res < best_res) {
          best_res = res;
          best_v = v->clone();
        }
        if (opt.verbose && !polish_stalled)
          std::cout << "[Cyclic] prescreen skip, ||A v||=" << res
                    << "  (thr=" << prescreen_thresh << ")\n";
        continue;
      }

      // POLISH: projector passes v <- v - A^T (A A^T)^{-1} A v with
      // (preconditioned) CG. An inexact CG solve can increase ||A v||, so a
      // pass is kept only if it reduces the residual of the normalized v.
      const double raw_res = res;
      std::size_t cg_its = 0;
      bool hit_maxit = false;
      for (int pass = 0;
           pass < std::max(1, opt.projector_passes) && res > accept_thresh;
           ++pass) {
        cg_its = cg_normal_eq_prec(A, vf, *tmp_m, *w_m, opt.precond_AAt,
                                   opt.cg_maxit, opt.cg_rel_tol,
                                   opt.cg_breakdown_tol)
                     .first;
        ++R.cg_solves;
        if (cg_its >= opt.cg_maxit) {
          ++R.cg_maxit_hits;
          hit_maxit = true;
        }
        A.multiply_transpose(*w_m, *tmp_n);
        auto vp = v->clone();
        vp->add_scaled(*tmp_n, -1.0);
        gs_against_basis(*vp);
        const double np = vp->norm();
        if (!std::isfinite(np) || np < 1e-14)
          break;
        vp->scale(1.0 / np);
        A.multiply(*vp, *w_m);
        const double res_p = w_m->norm();
        if (!(res_p < res))
          break;
        v = std::move(vp);
        res = res_p;
        tmp_m->copy_from(*w_m);
      }
      if (opt.verbose) {
        std::cout << "[Cyclic] restart=" << restart << "  ||A v|| raw="
                  << raw_res << "  (polished)=" << res
                  << "  accept_thr=" << accept_thresh << "  cg_its=" << cg_its
                  << "/" << opt.cg_maxit << "\n";
      }

      if (res <= accept_thresh) {
        basis.push_back(std::move(v));
        accepted_this_cycle = true;
        if (basis.size() >= k_target) {
          R.right_null = std::move(basis);
          R.converged_to_target = true;
          return R;
        }
        // earlier best_v is no longer orthogonal to the basis; it is
        // re-orthogonalized at the next restart
      } else {
        polish_stalled = polish_stalled || hit_maxit;
        if (res < best_res) {
          best_res = res;
          best_v = std::move(v);
        }
      }
    }

    // Stagnation guard
    if (!accepted_this_cycle) {
      ++restarts_since_accept;
      if (opt.stagnation_restarts > 0 &&
          restarts_since_accept >= opt.stagnation_restarts) {
        if (opt.verbose)
          std::cout << "[Cyclic] stopping due to stagnation ("
                    << restarts_since_accept << " restarts without accept)\n";
        break;
      }
    } else {
      restarts_since_accept = 0;
    }

    // ---- restart with the best remaining candidate (random if none)
    y0 = best_v ? std::move(best_v) : random_right();
  }

  // ---- robust fallback (randomized projector + gradient)
  if (basis.size() < k_target) {
    // Give up after a run of failures: when k exceeds the true nullity,
    // every try fails and each costs up to grad_max_iters gradient steps.
    const std::size_t max_consecutive_failures = 8;
    std::size_t failures = 0;
    const double tol_accept = accept_thresh;
    while (basis.size() < k_target && failures < max_consecutive_failures) {
      if (try_project_random_to_nullspace(
              A, vf, tol_accept, basis, opt.grad_max_iters, opt.frob_samples))
        failures = 0;
      else
        ++failures;
    }
    R.converged_to_target = basis.size() >= k_target;
  }

  if (opt.verbose && !R.converged_to_target) {
    std::cout << "[Cyclic] found " << basis.size() << " of " << k_target
              << " null vectors\n";
    if (R.cg_maxit_hits > 0)
      std::cout << "[Cyclic] " << R.cg_maxit_hits << " of " << R.cg_solves
                << " polishing CG solves stopped at cg_maxit=" << opt.cg_maxit
                << "; consider increasing cg_maxit or providing "
                   "precond_AAt\n";
  }

  R.right_null = std::move(basis);
  return R;
}

CyclicLanczosResult
find_nullspace_cyclic_lanczos_ex(const MatrixInterface &A,
                                 const VectorFactory &vf,
                                 const CyclicLanczosOptions &opt) {
  CyclicLanczosResult R = find_right_nullspace(A, vf, opt);
  if (!opt.return_left_space || A.rows() == 0 || A.cols() == 0)
    return R;

  // Left null space = right null space of A^T, computed by a separate run
  // (up to k vectors; with rank_known, nullity(A^T) = m - rank).
  CyclicLanczosOptions lopt;
  lopt.k = opt.k;
  lopt.ncv = opt.ncv;
  lopt.max_restarts = opt.max_restarts;
  lopt.tol_abs = opt.tol_abs;
  lopt.tol_rel = opt.tol_rel;
  lopt.reorth = opt.reorth;
  lopt.use_harmonic = opt.use_harmonic;
  lopt.target_tau = opt.target_tau;
  lopt.cg_maxit = opt.cg_maxit;
  lopt.cg_rel_tol = opt.cg_rel_tol;
  lopt.cg_breakdown_tol = opt.cg_breakdown_tol;
  lopt.projector_passes = opt.projector_passes;
  lopt.grad_max_iters = opt.grad_max_iters;
  lopt.frob_samples = opt.frob_samples;
  lopt.prescreen_factor = opt.prescreen_factor;
  lopt.stagnation_restarts = opt.stagnation_restarts;
  lopt.verbose = opt.verbose;
  lopt.rank_known = opt.rank_known;
  if (opt.rank_known >= 0) {
    const int left_nullity = static_cast<int>(A.rows()) - opt.rank_known;
    lopt.nullity_known =
        std::max(0, std::min(left_nullity, static_cast<int>(opt.k)));
  }
  // precond_AAt is for A A^T; the transposed run needs A^T A, so it is not
  // passed on.
  TransposedMatrix At(A);
  CyclicLanczosResult L = find_right_nullspace(At, vf, lopt);
  R.left_null = std::move(L.right_null);
  R.cg_solves += L.cg_solves;
  R.cg_maxit_hits += L.cg_maxit_hits;
  return R;
}

// Backwards-compatible wrapper
std::vector<std::unique_ptr<VectorInterface>>
find_nullspace_cyclic_lanczos(const MatrixInterface &A, const VectorFactory &vf,
                              const CyclicLanczosOptions &opt) {
  auto R = find_nullspace_cyclic_lanczos_ex(A, vf, opt);
  return std::move(R.right_null);
}

} // namespace krans::solvers
