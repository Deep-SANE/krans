#include "krans/Utils/EigenSmallSVD.hpp"
#include <Eigen/Dense>
#include <stdexcept>

namespace krans::utils {

void EigenSmallSVD::svd_dense(
    int m, const std::vector<double> &Bflat, std::vector<double> &svals,
    std::vector<std::vector<double>> &Vhat_cols) const {
  if (m <= 0)
    throw std::invalid_argument("svd_dense: m must be > 0");
  if (static_cast<int>(Bflat.size()) != m * m)
    throw std::invalid_argument("svd_dense: Bflat size must be m*m");

  using RowMat =
      Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
  Eigen::Map<const RowMat> M(Bflat.data(), m, m);

  // We only need V (and singular values).
  Eigen::JacobiSVD<RowMat> svd(M, Eigen::ComputeFullV);

  const Eigen::VectorXd s = svd.singularValues(); // size m
  const Eigen::MatrixXd V = svd.matrixV();        // m x m

  svals.resize(m);
  for (int i = 0; i < m; ++i)
    svals[i] = s(i);

  Vhat_cols.clear();
  Vhat_cols.reserve(m);
  for (int j = 0; j < m; ++j) {
    std::vector<double> col(static_cast<std::size_t>(m));
    for (int i = 0; i < m; ++i)
      col[static_cast<std::size_t>(i)] = V(i, j);
    Vhat_cols.push_back(std::move(col));
  }
}

} // namespace krans::utils
