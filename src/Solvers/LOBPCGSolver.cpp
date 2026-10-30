#include "sns/Solvers/LOBPCGSolver.hpp"

#include <Eigen/Dense>
#include <algorithm>
#include <cassert>
#include <cmath>
#include <functional>
#include <iostream>
#include <limits>
#include <numeric>
#include <utility>

#include "sns/Utils/GramSchmidt.hpp"
#include "sns/Utils/Projection.hpp"

namespace sns::solvers {

// ----------------- Small helpers for block algebra (on VectorInterface) -----

using VUPtr = std::unique_ptr<VectorInterface>;
using Block = std::vector<VUPtr>;

// Preconditioner signature (apply y = M^{-1} r)
using Precond = std::function<void(const VectorInterface &, VectorInterface &)>;

static inline Block make_block(const VectorFactory &vf, std::size_t n,
                               std::size_t cols) {
  Block B;
  B.reserve(cols);
  for (std::size_t j = 0; j < cols; ++j)
    B.push_back(vf.create_zero_of_size(n));
  return B;
}

static inline std::size_t cols(const Block &X) { return X.size(); }

// Z := [X | Y]
static inline Block hstack(const Block &X, const Block &Y) {
  Block Z;
  Z.reserve(cols(X) + cols(Y));
  for (const auto &p : X)
    Z.push_back(p->clone());
  for (const auto &p : Y)
    Z.push_back(p->clone());
  return Z;
}

// Z := [X | Y | P]
static inline Block hstack3(const Block &X, const Block &Y, const Block &P) {
  return hstack(hstack(X, Y), P);
}

// AX := N X, where N = A^T A (matrix-free: two matvecs per column).
static inline Block apply_normal(const MatrixInterface &A,
                                 const VectorFactory &vf, const Block &X) {
  const std::size_t m = A.rows(), n = A.cols();
  Block AX = make_block(vf, n, cols(X));
  auto tmp_m = vf.create_zero_of_size(m);
  for (std::size_t j = 0; j < cols(X); ++j) {
    A.multiply(*X[j], *tmp_m);
    A.multiply_transpose(*tmp_m, *AX[j]); // A^T (A x)
  }
  return AX;
}

// Gram matrices: S = Z^T Z, G = Z^T (N Z).  AZ must be N Z.
static inline void form_gram(const Block &Z, const Block &AZ,
                             Eigen::MatrixXd &S, Eigen::MatrixXd &G) {
  const int k = static_cast<int>(cols(Z));
  S.setZero(k, k);
  G.setZero(k, k);
  for (int j = 0; j < k; ++j) {
    for (int i = 0; i <= j; ++i) {
      const double sij = Z[(std::size_t)i]->dot(*Z[(std::size_t)j]);
      const double gij = Z[(std::size_t)i]->dot(*AZ[(std::size_t)j]);
      S(i, j) = S(j, i) = sij;
      G(i, j) = G(j, i) = gij;
    }
  }
}

// Y = argmin (smallest eigs) of G y = theta S y. Returns sorted (asc).
struct TinyGEP {
  Eigen::VectorXd theta; // ascending
  Eigen::MatrixXd Y;     // columns are generalized eigenvectors
};

static inline TinyGEP solve_tiny_gep(Eigen::MatrixXd &G, Eigen::MatrixXd &S) {
  // Symmetrize for safety
  G = 0.5 * (G + G.transpose());
  S = 0.5 * (S + S.transpose());

  // LLT with diagonal jitter for robustness on nearly-GS bases.
  const double s_norm = std::max(1.0, S.norm());
  const double jitters[] = {0.0, 1e-14, 1e-12, 1e-10};

  for (double jv : jitters) {
    Eigen::MatrixXd Sj = S;
    if (jv > 0.0)
      Sj.diagonal().array() += jv * s_norm;

    Eigen::LLT<Eigen::MatrixXd> chol(Sj);
    if (chol.info() != Eigen::Success)
      continue;

    Eigen::MatrixXd L = chol.matrixL();
    Eigen::MatrixXd Linv = L.inverse(); // tiny sizes → fine
    Eigen::MatrixXd C = Linv * G * Linv.transpose();
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(C);
    if (es.info() != Eigen::Success)
      continue;

    TinyGEP out;
    out.theta = es.eigenvalues();                 // ascending
    out.Y = Linv.transpose() * es.eigenvectors(); // Y = L^{-T} Q
    return out;
  }
  throw std::runtime_error("LOBPCG tiny GEP: S not SPD (after jitter)");
}

// X := sum_j Z_j * Y(j,i)  for i=0..p-1 (update p columns)
static inline Block combine_columns(const VectorFactory &vf, const Block &Z,
                                    const Eigen::MatrixXd &Y, std::size_t p) {
  const std::size_t n = Z.front()->size();
  Block X = make_block(vf, n, p);
  for (std::size_t i = 0; i < p; ++i) {
    for (int j = 0; j < Y.rows(); ++j) {
      const double c = Y(j, (int)i);
      if (c != 0.0)
        X[i]->add_scaled(*Z[(std::size_t)j], c);
    }
  }
  return X;
}

// Orthogonalize a block vs. a basis and among themselves using MGS.
static inline bool orthonormalize_block(Block &X,
                                        const std::vector<VUPtr> &vs_basis,
                                        double tol, bool reorth) {
  for (auto &x : X) {
    for (const auto &q : vs_basis) {
      const double c = x->dot(*q);
      x->add_scaled(*q, -c);
    }
    const double nrm = x->norm();
    if (nrm < tol)
      return false;
    x->scale(1.0 / nrm);
  }
  std::vector<std::unique_ptr<VectorInterface>> tmp;
  tmp.reserve(X.size());
  for (auto &p : X)
    tmp.push_back(p->clone());
  const bool indep = sns::utils::modified_gram_schmidt(tmp, tol, reorth);
  if (!indep)
    return false;
  for (std::size_t j = 0; j < X.size(); ++j)
    X[j]->copy_from(*tmp[j]);
  return true;
}

// Estimate ||A||_F using Hutchinson probes: sqrt(E ||A r||^2)
static inline double estimate_fro_norm(const MatrixInterface &A,
                                       const VectorFactory &vf,
                                       std::size_t samples) {
  samples = std::max<std::size_t>(1, samples);
  const std::size_t m = A.rows(), n = A.cols();
  auto r = vf.create_zero_of_size(n);
  auto Ar = vf.create_zero_of_size(m);
  double sum2 = 0.0;
  for (std::size_t s = 0; s < samples; ++s) {
    r = vf.create_random_of_size(n);
    A.multiply(*r, *Ar);
    const double nrm = Ar->norm();
    sum2 += nrm * nrm;
  }
  return std::sqrt(sum2 / (double)samples);
}

// Compute ||A v|| using tmp_m workspace.
static inline double residual_Av(const MatrixInterface &A,
                                 const VectorInterface &v,
                                 VectorInterface &tmp_m) {
  return sns::utils::residual_norm_Av(A, v, tmp_m);
}

// Orthogonalize a single vector vs. a basis (no normalization).
static inline void project_out(const std::vector<VUPtr> &basis,
                               VectorInterface &x) {
  for (const auto &q : basis) {
    const double c = x.dot(*q);
    if (c != 0.0)
      x.add_scaled(*q, -c);
  }
}

// -------------------- 2D Rayleigh–Ritz per-vector finisher -------------------

static inline bool
two_vec_rr_sharpen(const MatrixInterface &A, const VectorFactory &vf,
                   std::vector<VUPtr> &V, // full block (for projection)
                   std::size_t idx,       // which vector to sharpen
                   double target_residual,
                   const Precond *M_inv_opt, // may be null
                   int max_iters = 80) {
  if (idx >= V.size())
    return true;
  const std::size_t m = A.rows();

  auto Av = vf.create_zero_of_size(m);
  auto t = V[idx]->clone(); // N v
  auto d = V[idx]->clone(); // M^{-1} t -> projected
  auto Ad = vf.create_zero_of_size(m);

  std::vector<VUPtr> basis;
  basis.reserve(V.size() - 1);
  for (std::size_t j = 0; j < V.size(); ++j)
    if (j != idx)
      basis.push_back(V[j]->clone());

  for (int it = 0; it < max_iters; ++it) {
    A.multiply(*V[idx], *Av);
    const double res = Av->norm();
    if (res <= target_residual)
      return true;

    // t = N v = A^T (A v)
    A.multiply_transpose(*Av, *t);

    // d = M^{-1} t  (or t)
    if (M_inv_opt && *M_inv_opt) {
      (*M_inv_opt)(*t, *d);
    } else {
      d->copy_from(*t);
    }

    // project d vs span{basis}
    project_out(basis, *d);
    double nd = d->norm();
    if (nd <= 1e-14)
      return false;
    d->scale(1.0 / std::max(nd, 1e-300));

    // Nd
    A.multiply(*d, *Ad);
    auto Nd = t->clone();
    A.multiply_transpose(*Ad, *Nd);

    // 2x2 GEP with S ~ I (due to normalization); solve symmetric 2x2
    const double vNv = V[idx]->dot(*t);
    const double vNd = V[idx]->dot(*Nd);
    const double dNv = d->dot(*t);
    const double dNd = d->dot(*Nd);

    Eigen::Matrix2d G;
    G << vNv, vNd, dNv, dNd;
    G = 0.5 * (G + G.transpose());

    Eigen::SelfAdjointEigenSolver<Eigen::Matrix2d> es(G);
    if (es.info() != Eigen::Success)
      return false;

    Eigen::Vector2d y = es.eigenvectors().col(0); // smallest

    auto v_new = V[idx]->clone();
    v_new->scale(y(0));
    v_new->add_scaled(*d, y(1));

    project_out(basis, *v_new);
    const double nv = v_new->norm();
    if (nv <= 1e-14)
      return false;
    v_new->scale(1.0 / nv);

    V[idx]->copy_from(*v_new);
  }
  return false;
}

// ------------------------- Block final polish -------------------------------
//
// D := (I - V V^T) [M^{-1}] N V
// Z := [V | D], solve tiny GEP on Z (Eigen), take k smallest → V_new
// robust re-orthonormalize; stop when max ||A v_i|| <= target.
//
static inline void
block_rr_polish(const MatrixInterface &A, const VectorFactory &vf,
                std::vector<std::unique_ptr<VectorInterface>> &V,
                double target_residual, int max_sweeps,
                const Precond *M_inv_opt = nullptr) {
  if (V.empty())
    return;
  const std::size_t m = A.rows();
  const std::size_t k = V.size();

  auto r = vf.create_zero_of_size(m);   // A v
  auto g = V.front()->clone();          // direction (N v or M^{-1} N v)
  auto tmp = vf.create_zero_of_size(m); // A * direction

  (void)sns::utils::modified_gram_schmidt(V, 1e-14, true);

  for (int sweep = 0; sweep < max_sweeps; ++sweep) {
    double max_res = 0.0;
    for (std::size_t i = 0; i < k; ++i) {
      A.multiply(*V[i], *r);
      max_res = std::max(max_res, r->norm());
    }
    if (max_res <= target_residual)
      break;

    Block D;
    D.reserve(k);

    std::vector<VUPtr> V_basis;
    V_basis.reserve(k);
    for (std::size_t i = 0; i < k; ++i)
      V_basis.push_back(V[i]->clone());

    for (std::size_t j = 0; j < k; ++j) {
      // r = A v_j, g = A^T r = N v_j
      A.multiply(*V[j], *r);
      const double res = r->norm();
      if (res <= target_residual)
        continue;
      A.multiply_transpose(*r, *g);

      // precondition if available
      if (M_inv_opt && *M_inv_opt) {
        auto g_prec = g->clone();
        (*M_inv_opt)(*g, *g_prec);
        g.swap(g_prec);
      }

      project_out(V_basis, *g);
      const double gn = g->norm();
      if (gn <= 1e-14)
        continue;
      g->scale(1.0 / gn);

      bool dup = false;
      for (const auto &dcol : D) {
        if (std::fabs(g->dot(*dcol)) >= 1.0 - 1e-8) {
          dup = true;
          break;
        }
      }
      if (!dup)
        D.push_back(g->clone());
    }

    if (D.empty()) {
      // fallback: projected SD on each vector
      for (std::size_t i = 0; i < k; ++i) {
        for (int it = 0; it < 50; ++it) {
          A.multiply(*V[i], *r);
          const double res = r->norm();
          if (res <= target_residual)
            break;
          A.multiply_transpose(*r, *g);

          for (std::size_t t = 0; t < k; ++t) {
            if (t == i)
              continue;
            const double c = g->dot(*V[t]);
            if (c != 0.0)
              g->add_scaled(*V[t], -c);
          }
          const double gg = g->dot(*g);
          if (gg <= 0.0)
            break;

          A.multiply(*g, *tmp);
          const double Ag2 = tmp->dot(*tmp);
          if (Ag2 <= 0.0)
            break;
          const double alpha = gg / Ag2;

          V[i]->add_scaled(*g, -alpha);
          for (std::size_t t = 0; t < k; ++t)
            if (t != i) {
              const double c = V[i]->dot(*V[t]);
              if (c != 0.0)
                V[i]->add_scaled(*V[t], -c);
            }
          const double nv = V[i]->norm();
          if (nv > 0.0)
            V[i]->scale(1.0 / nv);
        }
      }
      (void)sns::utils::modified_gram_schmidt(V, 1e-14, true);

      // try to enrich so the RR step has leverage next sweep
      const int enrich = std::min<int>(2, (int)k);
      for (int e = 0; e < enrich; ++e) {
        auto q = vf.create_random_of_size(V.front()->size());
        project_out(V_basis, *q);
        const double nq = q->norm();
        if (nq > 1e-14) {
          q->scale(1.0 / nq);
          D.push_back(std::move(q));
        }
      }
      if (!D.empty())
        (void)orthonormalize_block(D, V_basis, 1e-14, true);
      else
        continue;
    }

    if (!orthonormalize_block(D, V_basis, 1e-14, true))
      continue;

    Block Z = hstack(V, D);
    Block AZ = apply_normal(A, vf, Z);
    Eigen::MatrixXd S, G;
    form_gram(Z, AZ, S, G);
    TinyGEP t = solve_tiny_gep(G, S);

    const int k_int = (int)k;
    Eigen::MatrixXd Yk = t.Y.leftCols(k_int);
    Block Vnew = combine_columns(vf, Z, Yk, k);

    std::vector<VUPtr> Vtmp;
    Vtmp.reserve(k);
    for (auto &p : Vnew)
      Vtmp.push_back(p->clone());
    (void)sns::utils::modified_gram_schmidt(Vtmp, 1e-14, true);
    for (std::size_t i = 0; i < k; ++i)
      V[i]->copy_from(*Vtmp[i]);

    max_res = 0.0;
    for (std::size_t i = 0; i < k; ++i) {
      A.multiply(*V[i], *r);
      max_res = std::max(max_res, r->norm());
    }
    if (max_res <= target_residual)
      break;
  }

  (void)sns::utils::modified_gram_schmidt(V, 1e-14, true);
}

// ----------------------------- Main solver -----------------------------------
LOBPCGResult find_nullspace_lobpcg_ex(const MatrixInterface &A,
                                      const VectorFactory &vf,
                                      const LOBPCGOptions &opt_in) {
  const LOBPCGOptions &opt = opt_in;
  LOBPCGResult R;

  const std::size_t m = A.rows(), n = A.cols();
  if (n == 0 || m == 0)
    return R;

  // derive target from hints
  std::size_t k_target = std::max<std::size_t>(1, opt.nev);
  if (opt.nullity_known >= 0) {
    const std::size_t nk = (std::size_t)opt.nullity_known;
    if (nk > n)
      throw std::invalid_argument("LOBPCG: nullity_known > n");
    k_target = nk;
  }
  if (opt.rank_known >= 0) {
    const std::size_t rk = (std::size_t)opt.rank_known;
    if (rk > n)
      throw std::invalid_argument("LOBPCG: rank_known > n");
    const std::size_t implied_null = n - rk;
    if (opt.nullity_known >= 0 && implied_null != k_target)
      throw std::invalid_argument(
          "LOBPCG: inconsistent rank_known vs nullity_known");
    if (opt.nullity_known < 0)
      k_target = implied_null;
    if (n <= rk) {
      R.frob_norm_est = 0.0;
      R.converged_to_target = (k_target == 0);
      return R;
    }
  }

  std::size_t bs = std::max<std::size_t>(1, opt.block_size);
  if (k_target > 0)
    bs = std::min<std::size_t>(bs, std::max<std::size_t>(1, k_target));

  const std::size_t guards =
      (bs == 1) ? 0
                : std::min<std::size_t>(
                      (std::size_t)((1.0 - opt.restart) * bs + 0.45), bs - 1);

  // thresholds
  R.frob_norm_est =
      estimate_fro_norm(A, vf, /*samples*/ std::max(1, opt.frob_samples));
  const double accept_thr =
      std::max(opt.tol_abs, opt.tol_rel * std::max(1.0, R.frob_norm_est));

  // locked set
  std::vector<VUPtr> Y_locked;
  Y_locked.reserve(k_target);

  // initial block
  Block X = make_block(vf, n, bs);
  std::size_t filled = 0;
  const std::size_t seeds = std::min<std::size_t>(bs, opt.initial_block.size());
  for (; filled < seeds; ++filled)
    X[filled] = opt.initial_block[filled]->clone();
  for (; filled < bs; ++filled)
    X[filled] = vf.create_random_of_size(n);

  if (!orthonormalize_block(X, Y_locked, opt.gs_tol, opt.gs_reorth))
    throw std::runtime_error("LOBPCG: initial block is dependent");

  // initial RR
  Block AX = apply_normal(A, vf, X);
  Eigen::MatrixXd S0, G0;
  form_gram(X, AX, S0, G0);
  TinyGEP init = solve_tiny_gep(G0, S0);
  Block Xrr = combine_columns(vf, X, init.Y, bs);
  X.swap(Xrr);
  AX = apply_normal(A, vf, X);

  std::vector<double> theta(bs, 0.0);
  const int lim =
      std::min<int>(static_cast<int>(init.theta.size()), static_cast<int>(bs));
  for (int j = 0; j < lim; ++j)
    theta[static_cast<std::size_t>(j)] = std::max(0.0, init.theta[j]);

  // residuals/directions
  Block Rb = make_block(vf, n, bs);
  Block W = make_block(vf, n, bs);
  Block P = make_block(vf, n, bs);

  auto tmp_m_v = vf.create_zero_of_size(m); // for ||A v||
  std::size_t its = 0, locked_total = 0;

  // main loop
  while (locked_total < k_target && its < opt.max_iters) {
    ++its;

    // residuals + acceptance
    std::size_t nconv = 0;
    for (std::size_t j = 0; j < bs; ++j) {
      Rb[j]->copy_from(*AX[j]);
      Rb[j]->add_scaled(*X[j], -theta[j]);

      const double Av_orig = residual_Av(A, *X[j], *tmp_m_v);
      if (Av_orig <= accept_thr) {
        auto v_acc = X[j]->clone();
        for (const auto &q : Y_locked) {
          const double c = v_acc->dot(*q);
          v_acc->add_scaled(*q, -c);
        }
        double nv = v_acc->norm();

        bool reject = false;
        if (nv <= std::max(1e-14, 10.0 * opt.gs_tol)) {
          reject = true;
        } else {
          v_acc->scale(1.0 / nv);
          for (const auto &q : Y_locked) {
            if (std::fabs(v_acc->dot(*q)) >= 1.0 - 1e-8) {
              reject = true;
              break;
            }
          }
        }

        double Av_acc = std::numeric_limits<double>::infinity();
        if (!reject) {
          Av_acc = residual_Av(A, *v_acc, *tmp_m_v);
          if (Av_acc > accept_thr)
            reject = true;
        }

        if (!reject) {
          Y_locked.push_back(v_acc->clone());
          R.right_null.push_back(std::move(v_acc));
          R.theta.push_back(theta[j]);
          R.residuals_Av.push_back(Av_acc);
          ++locked_total;
          ++nconv;
          if (opt.verbose)
            std::cout << "[LOBPCG] accept: ||A v||=" << Av_acc
                      << "  theta~=" << theta[j] << "  locked=" << locked_total
                      << "/" << k_target << "\n";
          if (locked_total >= k_target)
            break;
        } else {
          if (opt.verbose)
            std::cout << "[LOBPCG] reject candidate after ortho "
                         "(degenerate/duplicate or residual grew)"
                      << "  ||A v||_orig=" << Av_orig
                      << "  ||A v||_post=" << Av_acc << "\n";
        }
      } else if (opt.soft_locking && j == nconv) {
        // first unconverged ends the soft-locked prefix
      }
    }

    R.locked = locked_total;
    R.iters = its;
    if (locked_total >= k_target)
      break;

    const std::size_t ini = opt.soft_locking ? nconv : 0;

    // hard locking via guard vectors
    if (nconv >= bs - guards) {
      const std::size_t remain = bs - nconv;
      for (std::size_t j = 0; j < remain; ++j) {
        X[j]->copy_from(*X[nconv + j]);
        AX[j]->copy_from(*AX[nconv + j]);
        theta[j] = theta[nconv + j];
        P[j]->copy_from(*P[nconv + j]);
      }
      for (std::size_t j = remain; j < bs; ++j)
        X[j] = vf.create_random_of_size(n);
      if (!orthonormalize_block(X, Y_locked, opt.gs_tol, opt.gs_reorth))
        throw std::runtime_error("LOBPCG: restart orthogonalization failed");

      AX = apply_normal(A, vf, X);
      Eigen::MatrixXd S1, G1;
      form_gram(X, AX, S1, G1);
      TinyGEP t1 = solve_tiny_gep(G1, S1);
      std::fill(theta.begin(), theta.end(), 0.0);
      const int lim = std::min<int>(static_cast<int>(t1.theta.size()),
                                    static_cast<int>(bs));
      for (int j = 0; j < lim; ++j)
        theta[static_cast<std::size_t>(j)] = std::max(0.0, t1.theta[j]);

      if (opt.verbose)
        std::cout << "[LOBPCG] hard-lock/restart: soft=" << nconv
                  << " remain=" << remain << " guards=" << guards << "\n";
      continue;
    }

    // precondition residuals on active slice
    for (std::size_t j = ini; j < bs; ++j) {
      if (opt.M_inv)
        opt.M_inv(*Rb[j], *W[j]);
      else
        W[j]->copy_from(*Rb[j]);
    }

    // Orthonormalize W vs locked Y and internally
    {
      Block Wact;
      Wact.reserve(bs - ini);
      for (std::size_t j = ini; j < bs; ++j)
        Wact.push_back(W[j]->clone());
      if (!orthonormalize_block(Wact, Y_locked, opt.gs_tol, opt.gs_reorth)) {
        for (std::size_t j = ini; j < bs; ++j)
          W[j] = vf.create_random_of_size(n);
        for (std::size_t j = ini; j < bs; ++j) {
          for (const auto &q : Y_locked) {
            const double c = W[j]->dot(*q);
            W[j]->add_scaled(*q, -c);
          }
        }
      } else {
        for (std::size_t t = 0; t < Wact.size(); ++t)
          W[ini + t]->copy_from(*Wact[t]);
      }
    }

    // Orthonormalize P if its>1
    if (its > 1) {
      Block Pact;
      Pact.reserve(bs - ini);
      for (std::size_t j = ini; j < bs; ++j)
        Pact.push_back(P[j]->clone());
      (void)orthonormalize_block(Pact, Y_locked, opt.gs_tol, opt.gs_reorth);
      for (std::size_t t = 0; t < Pact.size(); ++t)
        P[ini + t]->copy_from(*Pact[t]);
    }

    // Build Z = [X_act | W_act | (P_act)]
    Block Xact, Wact, Pact;
    Xact.reserve(bs - ini);
    Wact.reserve(bs - ini);
    Pact.reserve((its > 1) ? (bs - ini) : 0);
    for (std::size_t j = ini; j < bs; ++j) {
      Xact.push_back(X[j]->clone());
      Wact.push_back(W[j]->clone());
      if (its > 1)
        Pact.push_back(P[j]->clone());
    }
    Block Z = (its > 1) ? hstack3(Xact, Wact, Pact) : hstack(Xact, Wact);

    const std::size_t nv = cols(Z);
    if (nv == 0)
      continue;
    if (opt.max_tiny_dim > 0 && nv > opt.max_tiny_dim)
      throw std::runtime_error("LOBPCG: tiny problem too large (nv cap)");

    Block AZ = apply_normal(A, vf, Z);
    Eigen::MatrixXd S, G;
    form_gram(Z, AZ, S, G);

    TinyGEP sol = solve_tiny_gep(G, S);

    const std::size_t take =
        std::min<std::size_t>(bs - ini, (std::size_t)sol.theta.size());
    const int take_i = (int)take;
    const int nx = (int)(bs - ini);
    const int nw = (int)(bs - ini);
    const int np = (its > 1) ? (int)(bs - ini) : 0;

    Eigen::MatrixXd Ysel = sol.Y.leftCols(take_i);
    Eigen::MatrixXd Yx = Ysel.block(0, 0, nx, take_i).eval();
    Eigen::MatrixXd Yw = Ysel.block(nx, 0, nw, take_i).eval();
    Eigen::MatrixXd Yp;
    if (its > 1)
      Yp = Ysel.block(nx + nw, 0, np, take_i).eval();

    Block Zwp;
    if (its > 1)
      Zwp = hstack(Wact, Pact);
    else
      Zwp = std::move(Wact);

    Eigen::MatrixXd Ywp;
    if (its > 1) {
      Ywp.resize(nw + np, take_i);
      for (int c = 0; c < take_i; ++c) {
        for (int r = 0; r < nw; ++r)
          Ywp(r, c) = Yw(r, c);
        for (int r = 0; r < np; ++r)
          Ywp(nw + r, c) = Yp(r, c);
      }
    } else {
      Ywp = Yw;
    }

    Block Pact_new = combine_columns(vf, Zwp, Ywp, take);
    for (std::size_t j = 0; j < take; ++j)
      P[ini + j]->copy_from(*Pact_new[j]);

    Block Xxy = combine_columns(vf, Xact, Yx, take);
    for (std::size_t j = 0; j < take; ++j) {
      X[ini + j]->copy_from(*Xxy[j]);
      X[ini + j]->add_scaled(*P[ini + j], 1.0);
    }

    // Normalize X_active and re-orth vs locked
    {
      Block Xslice;
      Xslice.reserve(take);
      for (std::size_t j = ini; j < ini + take; ++j)
        Xslice.push_back(X[j]->clone());
      if (!orthonormalize_block(Xslice, Y_locked, opt.gs_tol, opt.gs_reorth))
        throw std::runtime_error("LOBPCG: X slice orthonormalization failed");
      for (std::size_t t = 0; t < take; ++t)
        X[ini + t]->copy_from(*Xslice[t]);
    }

    // Full RR for ordering
    AX = apply_normal(A, vf, X);
    {
      Eigen::MatrixXd S2, G2;
      form_gram(X, AX, S2, G2);
      TinyGEP t2 = solve_tiny_gep(G2, S2);
      Block Xrr2 = combine_columns(vf, X, t2.Y, bs);
      X.swap(Xrr2);
      AX = apply_normal(A, vf, X);
      std::fill(theta.begin(), theta.end(), 0.0);
      const int lim = std::min<int>(static_cast<int>(t2.theta.size()),
                                    static_cast<int>(bs));
      for (int j = 0; j < lim; ++j)
        theta[static_cast<std::size_t>(j)] = std::max(0.0, t2.theta[j]);
    }
  }

  R.converged_to_target = (R.locked >= k_target);

  // Final polish: block polish + 2D RR per vector
  if (!R.right_null.empty()) {
    const double desired =
        std::max(opt.tol_abs, opt.tol_rel * std::max(1.0, R.frob_norm_est));
    const double target_residual =
        std::max(1e-14, std::min(1e-10, 0.01 * desired));

    const Precond *Mptr = opt.M_inv ? &opt.M_inv : nullptr;

    block_rr_polish(A, vf, R.right_null,
                    /*target_residual=*/target_residual,
                    /*max_sweeps=*/80,
                    /*M_inv_opt=*/Mptr);

    const std::size_t mloc = A.rows();
    auto tmp = vf.create_zero_of_size(mloc);
    for (std::size_t i = 0; i < R.right_null.size(); ++i) {
      A.multiply(*R.right_null[i], *tmp);
      if (tmp->norm() <= target_residual)
        continue;
      (void)two_vec_rr_sharpen(A, vf, R.right_null, i, target_residual, Mptr,
                               /*max_iters=*/120);
    }

    (void)sns::utils::modified_gram_schmidt(R.right_null, 1e-14, true);
  }

  return R;
}

// Backwards-compatible wrapper
std::vector<std::unique_ptr<VectorInterface>>
find_nullspace_lobpcg(const MatrixInterface &A, const VectorFactory &vf,
                      const LOBPCGOptions &opt) {
  auto R = find_nullspace_lobpcg_ex(A, vf, opt);
  return std::move(R.right_null);
}

} // namespace sns::solvers
