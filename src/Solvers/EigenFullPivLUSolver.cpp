#include "sns/Solvers/EigenFullPivLUSolver.hpp"
#include "sns/Solvers/CommonDenseUtils.hpp"

#include <Eigen/Dense>
#include <limits>

namespace sns::solvers {

std::vector<std::unique_ptr<VectorInterface>>
find_nullspace_fullpivlu(const MatrixInterface &A,
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

  Eigen::FullPivLU<Eigen::MatrixXd> lu(M);
  lu.setThreshold(thr);

  // Early out for full column rank (nullity = 0).
  const Eigen::Index r = lu.rank();
  if (r >= n)
    return {};

  // Eigen provides a right-nullspace basis directly.
  Eigen::MatrixXd N = lu.kernel(); // n x (n - r)
  if (N.cols() == 0)
    return {};

  return detail::wrap_and_optionally_orthonormalize(N, vf, true);
}

} // namespace sns::solvers
