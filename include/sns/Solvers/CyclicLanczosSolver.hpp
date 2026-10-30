#pragma once
/*
  Cyclic/augmented right-nullspace solver (matrix-free)
  =====================================================

  GOAL
  ----
  Find nullspace vectors v of A (A v = 0), for large sparse A, without
  forming A^T*A or any dense matrices beyond tiny projected problems.

  IDEA (e.g., SLEPc CYCLIC / augmented SVD)
  -----------------------------------------
  Work with the symmetric augmented matrix

      H(A) = [ 0   A ]
             [ A^T 0 ]

  whose eigenvalues are +-sigma_i (singular values of A). Eigenvectors near
  lambda≈0 have bottom block equal to right singular vectors; if sigma=0 they
  are true null vectors. Target the interior near tau=0 with a Krylov method,
  matrix-free.

  METHOD
  ----------------------------
  1) Lanczos on H(A) (implicit multiply via A and A^T), with optional double
     reorthogonalization and **hard locking** of accepted vectors.
  2) Solve the tiny projected problem and use **harmonic interior selection**
     (tau=0 by default) to extract candidates z = [x; y].
  3) Candidate right vector v := y.  Gram–Schmidt vs. accepted basis.
  4) **Polish** by one or two passes of the **orthogonal projector** onto
  ker(A): v <- v − A^T (A A^T)^{-1} (A v), solved by short (preconditioned) CG
  on normal equations. 5) Accept if ||A v|| <= max(tol_abs, tol_rel*||A||_F).
  Restart with the best candidate if more vectors are needed. Fallback:
  randomized projector + gradient descent on f(v)=1/2||A v||^2 with safe steps
  via
  ||A||_F.

  OUTPUT
  ------
  Result struct with right-nullspace basis (and optionally left-nullspace),
  plus run diagnostics. A backwards-compatible wrapper returning only the
  right-null vectors is provided.

  DEPENDENCIES
  ------------
  MatrixInterface: multiply, multiply_transpose, rows, cols
  VectorInterface: dot, add_scaled, scale, norm, size, set_zero, copy_from,
  clone VectorFactory  : create_zero_of_size, create_random_of_size

  NOTE
  ----
  - Left-nullspace (ker(A^T)) vectors are available from the top block.
*/

#include <cstddef>
#include <functional>
#include <memory>
#include <vector>

#include "sns/MatrixInterface.hpp"
#include "sns/VectorFactory.hpp"
#include "sns/VectorInterface.hpp"

namespace sns::solvers {

/** Preconditioner for normal equations (A A^T) w = r applying y <- M^{-1} r. */
using AAtPreconditioner =
    std::function<void(const VectorInterface &r_m, VectorInterface &y_m)>;

/** Reorthogonalization mode for Lanczos on the augmented operator. */
enum class ReorthMode { None = 0, Single = 1, Double = 2 };

/** Options controlling cyclic Lanczos nullspace extraction on H(A). */
struct CyclicLanczosOptions {
  /** Target number of right-null vectors to compute (k >= 1). */
  std::size_t k = 1;
  /** Krylov subspace dimension ncv (recommend ncv >= k + 2). */
  std::size_t ncv = 48;
  /** Maximum number of outer restarts before giving up. */
  std::size_t max_restarts = 80;

  /** Absolute accept threshold on ||A v||_2. */
  double tol_abs = 1e-12;
  /** Relative accept threshold on ||A v||_2 scaled by ||A||_F. */
  double tol_rel = 1e-10;

  /** Reorthogonalization mode for Krylov vectors. */
  ReorthMode reorth = ReorthMode::Double;
  /** Use harmonic interior extraction when true, else plain Ritz. */
  bool use_harmonic = true;
  /** Interior target shift tau for harmonic selection (default 0). */
  double target_tau = 0.0;

  /** Known nullity to stop exactly when reached (-1 if unknown). */
  int nullity_known = -1;
  /** Known rank to short-circuit when n <= rank_known (-1 if unknown). */
  int rank_known = -1;

  /** Optional initial right-space seeds y_j (ownership consumed). */
  std::vector<std::unique_ptr<VectorInterface>> initial_right_space;

  /** Left preconditioner M^{-1} for (A A^T) in projector polishing. */
  AAtPreconditioner precond_AAt = nullptr;
  /** Maximum CG iterations for solving (A A^T) w = r in polishing. */
  std::size_t cg_maxit = 80;
  /** Relative CG tolerance for polishing solves. */
  double cg_rel_tol = 1e-10;
  /** Breakdown guard on p^T (A A^T) p during CG. */
  double cg_breakdown_tol = 1e-30;
  /** Number of projector polishing passes per candidate (1 or 2). */
  int projector_passes = 1;

  /** Maximum gradient steps for fallback descent on 1/2 ||A v||_2^2. */
  int grad_max_iters = 3000;
  /** Hutchinson probe count to estimate ||A||_F. */
  int frob_samples = 4;

  /** Prescreen factor: skip polishing if ||A v||_2 > factor * ||A||_F. */
  double prescreen_factor = 0.8;
  /** Stop after this many restarts without an acceptance (> 0 to enable). */
  std::size_t stagnation_restarts = 30;

  /** Also return left-null vectors from the top block when true. */
  bool return_left_space = false;
  /** Print verbose progress information when true. */
  bool verbose = false;
};

/** Result bundle from cyclic Lanczos nullspace extraction. */
struct CyclicLanczosResult {
  /** Accepted right-nullspace basis vectors (size <= k). */
  std::vector<std::unique_ptr<VectorInterface>> right_null;
  /** Accepted left-nullspace basis vectors if requested. */
  std::vector<std::unique_ptr<VectorInterface>> left_null;
  /** Number of restarts performed. */
  std::size_t restarts = 0;
  /** Total Lanczos steps accumulated across restarts. */
  std::size_t krylov_steps_total = 0;
  /** Estimated Frobenius norm of A used for scaling. */
  double frob_norm_est = 0.0;
  /** True if the target count was reached (k or nullity_known). */
  bool converged_to_target = false;
};

/** Compute right-nullspace via cyclic Lanczos on H(A); optional left space. */
CyclicLanczosResult
find_nullspace_cyclic_lanczos_ex(const MatrixInterface &A,
                                 const VectorFactory &vf,
                                 const CyclicLanczosOptions &opt);

/** Convenience wrapper returning only the right-nullspace basis. */
std::vector<std::unique_ptr<VectorInterface>>
find_nullspace_cyclic_lanczos(const MatrixInterface &A, const VectorFactory &vf,
                              const CyclicLanczosOptions &opt);

} // namespace sns::solvers
