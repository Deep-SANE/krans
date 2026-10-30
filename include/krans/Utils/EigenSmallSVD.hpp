#pragma once
#include "krans/Utils/SmallDenseSVD.hpp"
#include <vector>

namespace krans::utils {

/** Small m x m dense SVD using Eigen (JacobiSVD), returning singular values and
 * V columns. */
class EigenSmallSVD final : public SmallDenseSVD {
public:
  /** Compute SVD of row-major m x m B; fill svals (size m) and Vhat_cols (m
   * columns of V). */
  void svd_dense(int m, const std::vector<double> &Bflat_row_major,
                 std::vector<double> &svals,
                 std::vector<std::vector<double>> &Vhat_cols) const override;
};

} // namespace krans::utils
