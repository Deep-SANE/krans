#include "krans/Solvers/EigenCODSolver.hpp"
#include "krans/Solvers/CommonDenseUtils.hpp"

#include <Eigen/Dense>
#include <limits>

namespace krans::solvers {

std::vector<std::unique_ptr<VectorInterface>>
find_nullspace_cod(const MatrixInterface &A,
                   const VectorFactory &vf, double tol_rel) {
  if (A.rows() == 0 || A.cols() == 0)
    return {};

  // Dense realization
  Eigen::MatrixXd M = detail::dense_copy_via_matvecs_or_view(A, vf);
  const Eigen::Index m = M.rows();
  const Eigen::Index n = M.cols();

  const double scale = detail::matrix_rank_scale(M);
  const double base =
      (tol_rel > 0.0 ? tol_rel : std::numeric_limits<double>::epsilon());
  const double thr =
      base * static_cast<double>(std::max<Eigen::Index>(m, n)) * scale;

  Eigen::CompleteOrthogonalDecomposition<Eigen::MatrixXd> cod;
  cod.setThreshold(thr);
  cod.compute(M);

  const Eigen::Index r = cod.rank();
  const Eigen::Index k = n - r;
  if (k <= 0)
    return {};

  const Eigen::MatrixXd Z = cod.matrixZ();
  const auto &P = cod.colsPermutation();

  // The kernel is the rightmost k columns of P * Z*
  // Since A P = Q R Z => A = Q R Z P^T
  // Nullspace of R is [0; I_k]
  // Nullspace of A is P * Z^* * [0; I_k]
  // Z.bottomRows(k).adjoint() gives the last k columns of Z^*
  Eigen::MatrixXd N = P * Z.bottomRows(k).adjoint();

  for (Eigen::Index j = 0; j < N.cols(); ++j) {
    Eigen::VectorXd v = N.col(j);
    Eigen::VectorXd b = M * v;
    if (b.norm() > thr) {
      Eigen::VectorXd d = cod.solve(b);
      v -= d;
      N.col(j) = v;
    }
  }

  return detail::wrap_and_optionally_orthonormalize(N, vf, true);
}

} // namespace krans::solvers
