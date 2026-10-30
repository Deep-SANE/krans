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

// ---------------- Block helpers: z = [x; y] ----------------------------------
struct BlockVec {
  std::unique_ptr<VectorInterface> top;    // x (size m)
  std::unique_ptr<VectorInterface> bottom; // y (size n)
};
static inline double block_dot(const BlockVec &a, const BlockVec &b) {
  return a.top->dot(*b.top) + a.bottom->dot(*b.bottom);
}
static inline double block_norm(const BlockVec &v) {
  return std::sqrt(block_dot(v, v));
}
static inline void block_axpy(BlockVec &y, const BlockVec &x, double a) {
  y.top->add_scaled(*x.top, a);
  y.bottom->add_scaled(*x.bottom, a);
}
static inline void block_scale(BlockVec &v, double s) {
  v.top->scale(s);
  v.bottom->scale(s);
}
static inline void block_normalize(BlockVec &v, double eps = 1e-16) {
  const double n = std::max(eps, block_norm(v));
  block_scale(v, 1.0 / n);
}

static inline void H_apply(const MatrixInterface &A, const BlockVec &z,
                           BlockVec &Hz) {
  A.multiply(*z.bottom, *Hz.top);           // x' = A y
  A.multiply_transpose(*z.top, *Hz.bottom); // y' = A^T x
}

// ---------------- Utilities -------------------------------------------------
static inline void reorth_block(BlockVec &w, const std::vector<BlockVec> &V,
                                const std::vector<BlockVec> &Locked,
                                ReorthMode mode) {
  if (mode == ReorthMode::None)
    return;

  auto pass = [&](const std::vector<BlockVec> &S) {
    for (const auto &s : S) {
      const double c = block_dot(s, w);
      block_axpy(w, s, -c);
    }
  };
  // First pass
  pass(Locked);
  pass(V);
  if (mode == ReorthMode::Double) {
    // Second pass
    pass(Locked);
    pass(V);
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
  if (rhsn == 0.0)
    return {0, 0.0};

  if (M_inv)
    M_inv(*r, *z);
  else
    z->copy_from(*r);
  p->copy_from(*z);

  double rz = r->dot(*z);
  double rnorm = std::sqrt(std::max(0.0, rz));
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

// ---------------- Tiny problems (plain & harmonic) ---------------------------
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

// harmonic interior (tau): K = (T - tau I); compute InvK ≈ K^{-1} by dense
// solve, eigendecompose sym(InvK). Harmonic Ritz θ ≈ tau + 1/mi; order by |mi|
// desc.
static TinyEV solve_tiny_harmonic(const std::vector<double> &alpha,
                                  const std::vector<double> &beta, double tau) {
  const int l = static_cast<int>(alpha.size());
  Eigen::MatrixXd K = Eigen::MatrixXd::Zero(l, l);
  for (int i = 0; i < l; ++i) {
    K(i, i) = alpha[(std::size_t)i] - tau;
    if (i + 1 < l) {
      const double b = beta[(std::size_t)i];
      K(i, i + 1) = b;
      K(i + 1, i) = b;
    }
  }
  Eigen::PartialPivLU<Eigen::MatrixXd> lu(K);
  Eigen::MatrixXd InvK = lu.solve(Eigen::MatrixXd::Identity(l, l));
  Eigen::MatrixXd InvKsym = 0.5 * (InvK + InvK.transpose());
  Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(InvKsym);
  return {es.eigenvalues(), es.eigenvectors()};
}

// ----------------------------- Main solver -----------------------------------
CyclicLanczosResult
find_nullspace_cyclic_lanczos_ex(const MatrixInterface &A,
                                 const VectorFactory &vf,
                                 const CyclicLanczosOptions &opt) {
  const std::size_t m = A.rows(), n = A.cols();
  CyclicLanczosResult R;

  if (n == 0 || m == 0)
    return R;

  const std::size_t k_target = static_cast<std::size_t>(
      (opt.nullity_known >= 0) ? opt.nullity_known : static_cast<int>(opt.k));
  const std::size_t ncv =
      std::max<std::size_t>(std::max<std::size_t>(2, opt.ncv), k_target + 2);

  auto make_block = [&](std::size_t mt, std::size_t nb) {
    return BlockVec{vf.create_zero_of_size(mt), vf.create_zero_of_size(nb)};
  };

  // Pre-compute ||A||_F estimate for scale-invariant thresholds
  R.frob_norm_est = estimate_fro_norm(A, vf, std::max(1, opt.frob_samples));
  const double accept_thresh =
      std::max(opt.tol_abs, opt.tol_rel * std::max(1.0, R.frob_norm_est));
  const double prescreen_thresh =
      std::max(0.0, opt.prescreen_factor) * std::max(1.0, R.frob_norm_est);

  // Seed z0
  BlockVec z0 = make_block(m, n);
  if (!opt.initial_right_space.empty() && opt.initial_right_space.front()) {
    z0.bottom = opt.initial_right_space.front()->clone();
    z0.top->set_zero();
  } else {
    z0.bottom = vf.create_random_of_size(n);
    z0.top = vf.create_random_of_size(m);
  }
  block_normalize(z0);

  std::vector<BlockVec> V;
  V.reserve(ncv + 1);
  V.push_back(BlockVec{z0.top->clone(), z0.bottom->clone()});

  std::vector<double> alpha;
  alpha.reserve(ncv);
  std::vector<double> beta;
  beta.reserve(ncv);

  // Hard-locked accepted eigenvectors (both blocks)
  std::vector<BlockVec> Locked;

  // Accepted right-nullspace basis
  std::vector<std::unique_ptr<VectorInterface>> basis;
  basis.reserve(k_target);

  // If returning left space, keep them too
  std::vector<std::unique_ptr<VectorInterface>> left_basis;

  auto tmp_m = vf.create_zero_of_size(m); // reused for ||A v||
  std::size_t restarts_since_accept = 0;

  for (std::size_t restart = 0; restart < opt.max_restarts; ++restart) {
    R.restarts = restart;
    alpha.clear();
    beta.clear();

    // ---- Krylov build up to l ≤ ncv
    BlockVec w = make_block(m, n);
    std::size_t local_steps = 0;
    for (std::size_t j = 0; j < ncv; ++j) {
      H_apply(A, V.back(), w);

      const double aj = block_dot(V.back(), w);
      block_axpy(w, V.back(), -aj);
      if (!beta.empty())
        block_axpy(w, V[V.size() - 2], -beta.back());

      reorth_block(w, V, Locked, opt.reorth);

      const double bj = block_norm(w);
      alpha.push_back(aj);

      if (bj < 1e-14 || V.size() >= ncv)
        break;

      beta.push_back(bj);
      BlockVec vnew{w.top->clone(), w.bottom->clone()};
      vnew.top->scale(1.0 / bj);
      vnew.bottom->scale(1.0 / bj);
      V.push_back(std::move(vnew));
      ++local_steps;
    }
    R.krylov_steps_total += local_steps;

    const std::size_t l = alpha.size();
    if (l == 0)
      break;

    // ---- tiny eigenproblem & interior selection (harmonic preferred)
    TinyEV tev = opt.use_harmonic
                     ? solve_tiny_harmonic(alpha, beta, opt.target_tau)
                     : solve_tiny_tridiag(alpha, beta);

    // ordering: harmonic => |mi| desc (closest to tau); plain => |lambda-tau|
    // asc
    std::vector<int> ord(tev.evals.size());
    std::iota(ord.begin(), ord.end(), 0);
    if (opt.use_harmonic) {
      std::stable_sort(ord.begin(), ord.end(), [&](int a, int b) {
        return std::abs(tev.evals[a]) > std::abs(tev.evals[b]);
      });
    } else {
      std::stable_sort(ord.begin(), ord.end(), [&](int a, int b) {
        return std::abs(tev.evals[a] - opt.target_tau) <
               std::abs(tev.evals[b] - opt.target_tau);
      });
    }

    // Reuse buffers across candidates
    auto w_m = vf.create_zero_of_size(m);
    auto tmp_n = vf.create_zero_of_size(n);

    const std::size_t test_cap =
        std::min<std::size_t>(l, std::max<std::size_t>(k_target, 8));
    double best_res = std::numeric_limits<double>::infinity();
    BlockVec zkeep{vf.create_zero_of_size(m), vf.create_zero_of_size(n)};
    bool accepted_this_cycle = false;

    for (std::size_t t = 0; t < test_cap; ++t) {
      const int jbest = ord[t];

      // z = V * y_j ; normalize
      BlockVec z{vf.create_zero_of_size(m), vf.create_zero_of_size(n)};
      for (int i = 0; i < static_cast<int>(l); ++i) {
        const double c = tev.evecs(i, jbest);
        z.top->add_scaled(*V[static_cast<std::size_t>(i)].top, c);
        z.bottom->add_scaled(*V[static_cast<std::size_t>(i)].bottom, c);
      }
      block_normalize(z);

      // candidate right vector v := bottom(z)
      auto v = z.bottom->clone();

      // GS vs accepted right-nullspace, normalize
      for (auto &q : basis) {
        const double c = v->dot(*q);
        v->add_scaled(*q, -c);
      }
      double nv = v->norm();
      if (nv < 1e-14)
        continue;
      v->scale(1.0 / nv);

      // Pre-screen: raw residual (avoid polishing hopeless candidates)
      A.multiply(*v, *tmp_m);
      const double rawRes = tmp_m->norm();
      if (rawRes > prescreen_thresh) {
        if (rawRes < best_res) {
          best_res = rawRes;
          zkeep = BlockVec{z.top->clone(), z.bottom->clone()};
        }
        if (opt.verbose)
          std::cout << "[Cyclic] prescreen skip, ||A v||=" << rawRes
                    << "  (thr=" << prescreen_thresh << ")\n";
        continue;
      }

      // POLISH: projector passes with (preconditioned) CG
      std::size_t cg_its = 0;
      for (int pass = 0; pass < std::max(1, opt.projector_passes); ++pass) {
        cg_its = cg_normal_eq_prec(A, vf, *tmp_m, *w_m, opt.precond_AAt,
                                   opt.cg_maxit, opt.cg_rel_tol,
                                   opt.cg_breakdown_tol)
                     .first;
        ++R.cg_solves;
        if (cg_its >= opt.cg_maxit)
          ++R.cg_maxit_hits;
        A.multiply_transpose(*w_m, *tmp_n);
        v->add_scaled(*tmp_n, -1.0);

        // re-GS + renorm
        for (auto &q : basis) {
          const double c = v->dot(*q);
          v->add_scaled(*q, -c);
        }
        nv = v->norm();
        if (nv < 1e-14)
          break;
        v->scale(1.0 / nv);

        // refresh rhs for next pass or final residual
        A.multiply(*v, *tmp_m);
      }
      if (nv < 1e-14)
        continue;

      const double resNull = tmp_m->norm();
      if (resNull < best_res) {
        best_res = resNull;
        zkeep = BlockVec{z.top->clone(), z.bottom->clone()};
      }
      if (opt.verbose) {
        std::cout << "[Cyclic] restart=" << restart
                  << "  ||A v|| (polished)=" << resNull
                  << "  accept_thr=" << accept_thresh << "  cg_its=" << cg_its
                  << "/" << opt.cg_maxit << "\n";
      }

      if (resNull <= accept_thresh) {
        // Accept: save right (and optional left) and hard-lock z
        if (opt.return_left_space) {
          R.left_null.push_back(z.top->clone());
          // Ensure left vector is normalized consistently
          const double nx = R.left_null.back()->norm();
          if (nx > 0.0)
            R.left_null.back()->scale(1.0 / nx);
        }
        basis.push_back(std::move(v));
        Locked.push_back(
            BlockVec{z.top->clone(), z.bottom->clone()}); // hard lock
        accepted_this_cycle = true;

        if ((opt.nullity_known >= 0 &&
             static_cast<int>(basis.size()) >= opt.nullity_known) ||
            basis.size() >= k_target) {
          R.right_null = std::move(basis);
          R.converged_to_target = true;
          return R;
        }
      }
    }

    // Stuck? Try probe-polish on best-so-far occasionally
    if (!accepted_this_cycle && (restarts_since_accept % 6 == 5)) {
      auto vprobe = zkeep.bottom->clone();
      for (auto &q : basis) {
        const double c = vprobe->dot(*q);
        vprobe->add_scaled(*q, -c);
      }
      double nv = vprobe->norm();
      if (nv >= 1e-14)
        vprobe->scale(1.0 / nv);

      auto w_m2 = vf.create_zero_of_size(m);
      auto tmp_n2 = vf.create_zero_of_size(n);
      A.multiply(*vprobe, *tmp_m);
      const std::size_t probe_cg_its =
          cg_normal_eq_prec(A, vf, *tmp_m, *w_m2, opt.precond_AAt,
                            opt.cg_maxit, opt.cg_rel_tol, opt.cg_breakdown_tol)
              .first;
      ++R.cg_solves;
      if (probe_cg_its >= opt.cg_maxit)
        ++R.cg_maxit_hits;
      A.multiply_transpose(*w_m2, *tmp_n2);
      vprobe->add_scaled(*tmp_n2, -1.0);

      for (auto &q : basis) {
        const double c = vprobe->dot(*q);
        vprobe->add_scaled(*q, -c);
      }
      nv = vprobe->norm();
      if (nv >= 1e-14)
        vprobe->scale(1.0 / nv);

      A.multiply(*vprobe, *tmp_m);
      const double rprobe = tmp_m->norm();
      if (rprobe < best_res) {
        best_res = rprobe;
        zkeep.bottom = vprobe->clone();
        A.multiply(*zkeep.bottom, *zkeep.top);
        block_normalize(zkeep);
      }
    }

    // stopping if rank/nullity imply no further nulls
    if (opt.nullity_known >= 0 &&
        static_cast<int>(basis.size()) >= opt.nullity_known) {
      R.right_null = std::move(basis);
      R.converged_to_target = true;
      return R;
    }
    if (opt.rank_known >= 0 && static_cast<int>(n) <= opt.rank_known) {
      // full column rank ⇒ no right nulls
      R.right_null = std::move(basis);
      return R;
    }

    // Stagnation guard
    if (!accepted_this_cycle) {
      ++restarts_since_accept;
      if (opt.stagnation_restarts > 0 &&
          restarts_since_accept >= opt.stagnation_restarts) {
        if (opt.verbose) {
          std::cout << "[Cyclic] stopping due to stagnation ("
                    << restarts_since_accept << " restarts without accept)\n";
          if (R.cg_maxit_hits > 0)
            std::cout << "[Cyclic] " << R.cg_maxit_hits << " of "
                      << R.cg_solves << " polishing CG solves stopped at cg_maxit="
                      << opt.cg_maxit
                      << "; consider increasing cg_maxit or providing "
                         "precond_AAt\n";
        }
        R.right_null = std::move(basis);
        return R;
      }
    } else {
      restarts_since_accept = 0;
    }

    // ---- restart with the best interior candidate
    if (accepted_this_cycle) {
        // If we just found a vector, don't restart with it!
        // Start fresh to find diffrent vectors.
        zkeep.top = vf.create_random_of_size(m);
        zkeep.bottom = vf.create_random_of_size(n);
    }
    
    // Orthogonalize zkeep against Locked to avoid rediscovery
    for (const auto &lz : Locked) {
        double c = block_dot(lz, zkeep);
        block_axpy(zkeep, lz, -c);
    }
    
    block_normalize(zkeep);
    V.clear();
    V.push_back(std::move(zkeep));
  }

  // ---- robust fallback (randomized projector + gradient)
  if (basis.size() < k_target) {
    const std::size_t need = k_target - basis.size();
    const std::size_t tries = std::max<std::size_t>(need * 8, need);
    std::size_t accepted = 0;
    const double tol_accept = accept_thresh;
    for (std::size_t t = 0; t < tries && accepted < need; ++t) {
      if (try_project_random_to_nullspace(
              A, vf, tol_accept, basis, opt.grad_max_iters, opt.frob_samples)) {
        ++accepted;
      }
    }
  }

  R.right_null = std::move(basis);
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
