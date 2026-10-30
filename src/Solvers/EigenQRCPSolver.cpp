#include "sns/Solvers/EigenQRCPSolver.hpp"
#include "sns/Solvers/CommonDenseUtils.hpp"

#include <Eigen/Dense>
#include <limits>

namespace sns::solvers {

std::vector<std::unique_ptr<VectorInterface>>
find_nullspace_qrcp(const MatrixInterface &A,
                    const VectorFactory &vf, double tol_rel) {
  if (A.rows() == 0 || A.cols() == 0)
    return {};

  // Dense realization
  Eigen::MatrixXd M = detail::dense_copy_via_matvecs_or_view(A, vf);
  const Eigen::Index n = M.cols();
  const Eigen::Index m = M.rows();

  const double scale = detail::matrix_rank_scale(M);
  const double base =
      (tol_rel > 0.0 ? tol_rel : std::numeric_limits<double>::epsilon());
  const double thr =
      base * static_cast<double>(std::max<Eigen::Index>(m, n)) * scale;

  Eigen::ColPivHouseholderQR<Eigen::MatrixXd> qr(M);
  qr.setThreshold(thr);

  const Eigen::Index r = qr.rank();
  if (r >= n)
    return {};
  const Eigen::Index k = n - r;

  const auto &P = qr.colsPermutation();
  Eigen::MatrixXd R = qr.matrixR().topRows(std::min(m, n)); // upper trapezoid

  // Solve R11 X1 = -R12
  Eigen::MatrixXd X1 = -R.topRightCorner(r, k);
  R.topLeftCorner(r, r).template triangularView<Eigen::Upper>().solveInPlace(
      X1);

  // Build X = [X1; I_k] and unpermute: N = P * X
  Eigen::MatrixXd X = Eigen::MatrixXd::Zero(n, k);
  X.topRows(r) = X1;
  X.bottomRows(k) = Eigen::MatrixXd::Identity(k, k);

  Eigen::MatrixXd N = P * X;

  return detail::wrap_and_optionally_orthonormalize(N, vf, true);
}

} // namespace sns::solvers
