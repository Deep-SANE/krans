#include "sns/Solvers/EigenSVDSolver.hpp"
#include "sns/Solvers/CommonDenseUtils.hpp"

#include <Eigen/Dense>
#include <Eigen/SVD>
#include <limits>

namespace sns::solvers {

std::vector<std::unique_ptr<VectorInterface>>
find_nullspace_svd(const MatrixInterface &A,
                   const VectorFactory &vf, double tol_rel) {
  if (A.rows() == 0 || A.cols() == 0)
    return {};

  // Dense realization
  Eigen::MatrixXd M = detail::dense_copy_via_matvecs_or_view(A, vf);
  const Eigen::Index m = M.rows();
  const Eigen::Index n = M.cols();

  // Choose SVD variant (BDCSVD scales better for large problems)
  // Keeping Jacobi only for very small matrices
  const bool use_bdcs = (std::max(M.rows(), M.cols()) > 32);

  Eigen::VectorXd s;
  Eigen::MatrixXd V;
#if defined(__GNUC__) || defined(__clang__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#endif
  if (!use_bdcs) {
    Eigen::JacobiSVD<Eigen::MatrixXd> svd(M, Eigen::ComputeFullV);
    s = svd.singularValues();
    V = svd.matrixV();
  } else {
    Eigen::BDCSVD<Eigen::MatrixXd> svd(M, Eigen::ComputeFullV);
    s = svd.singularValues();
    V = svd.matrixV();
  }
#if defined(__GNUC__) || defined(__clang__)
#pragma GCC diagnostic pop
#endif

  if (s.size() == 0 || V.cols() != n || V.rows() != n)
    return {};

  const double smax = s.maxCoeff();
  const double base =
      (tol_rel > 0.0 ? tol_rel : std::numeric_limits<double>::epsilon());
  const double tau =
      base * static_cast<double>(std::max<Eigen::Index>(m, n)) * smax;

  Eigen::Index r = 0;
  for (Eigen::Index i = 0; i < s.size(); ++i)
    if (s(i) > tau)
      ++r;

  const Eigen::Index k = n - r;
  if (k <= 0)
    return {};
  if (k == n) {
    return detail::wrap_and_optionally_orthonormalize(V, vf, true);
  }

  // For FullV with sorted sigmas, the last k columns span the right nullspace
  Eigen::MatrixXd N = V.rightCols(k);

  return detail::wrap_and_optionally_orthonormalize(N, vf, true);
}

} // namespace sns::solvers
