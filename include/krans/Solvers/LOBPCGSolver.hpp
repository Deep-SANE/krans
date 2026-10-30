#pragma once
/*
  Block LOBPCG on A^T A for right-nullspace (matrix-free)
  =======================================================

  GOAL
  ----
  Compute a basis of right-nullspace vectors v of A (A v = 0) using a
  *block* Locally Optimal Preconditioned Conjugate Gradient (LOBPCG)
  iteration applied to the *normal* operator N := A^T A without forming N.

  IDEA (BLOPEX / SLEPc LOBPCG)
  ----------------------------
  Maintain a block of iterates X (n x bs) and update them by Rayleigh-Ritz
  minimization in the small subspace Z = [X | W | P], where
    - W are *preconditioned residuals* (approximate correction directions),
    - P are *conjugate directions* from the previous iteration.
  Solve the tiny generalized eigenproblem
      (Z^T N Z) y = θ (Z^T Z) y
  and update X := Z * Y where columns of Y are eigenvectors for the *smallest*
  eigenvalues θ (for nullspace, θ ≈ 0). Use *soft* and *hard* locking with
  guard vectors to stabilize progress.

  METHOD (high-level loop, mirroring SLEPc/BLOPEX)
  ------------------------------------------------
   1) Build/orthonormalize initial X (bs columns), apply constraints Y
  (hard-locked).
   2) Compute AX := N X by two matvecs per vector (A, then A^T).
   3) Rayleigh–Ritz in span{X} to initialize θ and X.
   4) Repeat until enough vectors accepted:
   5) Residuals R_j = AX_j - θ_j X_j for active (soft-unlocked) columns.
   6) Test convergence; update nconv (soft lock count).
      If nconv >= bs - guard -> perform *hard locking* (move to Y), refill X.
   7) Precondition residuals: W := M^{-1} R (or W := R if no M).
   8) Orthonormalize W vs Y and among themselves.
   9) If iter>1, orthonormalize P similarly (conjugate block).
  10) Assemble Z = [X_active | W | (P if iter>1)] and
      form G = Z^T N Z, S = Z^T Z (tiny dense symmetric matrices).
  11) Solve G y = θ S y (small GEP), take the (bs-ini) *smallest* θ.
  12) Partition y = [y_X; y_W; y_P]; set
      P_new := W y_W + P y_P
      X_act := X y_X + P_new
  13) Recompute AX := N X, continue.

  ACCEPTANCE / STOPPING
  ---------------------
  A candidate v is accepted (hard-locked) if ||A v|| <= max(tol_abs,
  tol_rel * ||A||_F_est). We estimate ||A||_F with a few Hutchinson probes.

  REFERENCES
  ----------
  - Knyazev, A.V. (2001) LOBPCG, SISC 23(2):517–541.
  - Knyazev et al. (2007) BLOPEX in hypre/PETSc.

*/

#include "krans/MatrixInterface.hpp"
#include "krans/VectorFactory.hpp"
#include "krans/VectorInterface.hpp"

#include <cstddef>
#include <functional>
#include <memory>
#include <vector>

namespace krans::solvers {

using NormalPreconditioner =
    std::function<void(const VectorInterface &r_n, VectorInterface &y_n)>;

struct LOBPCGOptions {
  /** Number of right-null vectors requested (used if nullity_known < 0). */
  std::size_t nev = 1;
  /** Block size (number of simultaneous vectors in the iterate X).
   * 0 = auto: min(16, number of requested vectors). Larger values are honored
   * (capped at cols(A)); the extra vectors act as guards and typically cut the
   * iteration count sharply on ill-conditioned A, at a higher cost per
   * iteration. */
  std::size_t block_size = 0;
  /** Maximum outer iterations. Unpreconditioned LOBPCG on A^T A may need far
   * more than the default for ill-conditioned A; check converged_to_target. */
  std::size_t max_iters = 1000;
  /** Absolute tolerance on ||A v||_2. */
  double tol_abs = 1e-12;
  /** Relative tolerance on ||A v||_2 scaled by ||A||_F estimate. */
  double tol_rel = 1e-10;
  /** Use soft locking (true) within the current block. */
  bool soft_locking = true;
  /** Restart parameter in (0.1..1]: fraction of converged vectors to trigger
   * hard-lock restart. */
  double restart = 0.9;
  /** Gram-Schmidt tolerance for dependence detection. */
  double gs_tol = 1e-14;
  /** Whether to use double reorthogonalization. */
  bool gs_reorth = true;
  /** Preconditioner M^{-1} acting in n-space (for N = A^T A), y <- M^{-1} r.
   * The standard way to speed up convergence on ill-conditioned A. */
  NormalPreconditioner M_inv = nullptr;
  /** Optional initial block seeds. */
  std::vector<std::unique_ptr<VectorInterface>> initial_block;
  /** Hutchinson samples to estimate ||A||_F. */
  int frob_samples = 4;
  /** Safety cap on tiny GEP dimension (0 = unlimited). */
  std::size_t max_tiny_dim = 0;
  /** Known nullity to stop exactly when reached (-1 if unknown). */
  int nullity_known = -1;
  /** Known rank to early exit when n <= rank_known (-1 if unknown). */
  int rank_known = -1;
  /** Print trace information (accept/reject events, periodic progress, and a
   * final summary that flags a non-converged run). */
  bool verbose = false;
  /** With verbose, print a progress line every this many iterations
   * (0 = no periodic progress). */
  std::size_t progress_every = 100;
};

struct LOBPCGResult {
  /** Accepted right-nullspace basis vectors (size ≤ requested target). */
  std::vector<std::unique_ptr<VectorInterface>> right_null;
  /** Rayleigh quotients θ ≈ v^T N v for accepted vectors (informative only). */
  std::vector<double> theta;
  /** ||A v||_2 of accepted vectors at acceptance. */
  std::vector<double> residuals_Av;
  /** Total LOBPCG iterations performed. */
  std::size_t iters = 0;
  /** Count of hard-locked vectors (equals right_null.size() at return). */
  std::size_t locked = 0;
  /** Estimated Frobenius norm of A used for thresholds. */
  double frob_norm_est = 0.0;
  /** Acceptance threshold on ||A v||_2 that was applied. */
  double accept_threshold = 0.0;
  /** Smallest ||A x||_2 among not-yet-accepted iterates in the last iteration
   * (shows how close a non-converged run got; 0 if nothing was left). */
  double best_unaccepted_residual = 0.0;
  /** Block size actually used. */
  std::size_t block_size = 0;
  /** True if the requested target (or known nullity) was reached. */
  bool converged_to_target = false;
};

/** Compute right-nullspace via LOBPCG on N=A^T A (matrix-free). */
LOBPCGResult find_nullspace_lobpcg_ex(const MatrixInterface &A,
                                      const VectorFactory &vf,
                                      const LOBPCGOptions &opt);

/** Convenience wrapper returning only the right-nullspace basis. */
std::vector<std::unique_ptr<VectorInterface>>
find_nullspace_lobpcg(const MatrixInterface &A, const VectorFactory &vf,
                      const LOBPCGOptions &opt);

} // namespace krans::solvers