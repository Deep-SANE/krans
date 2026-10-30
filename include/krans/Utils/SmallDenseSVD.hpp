#pragma once
#include <vector>

namespace krans::utils {

// Interface for a tiny dense SVD on an m×m matrix.
class SmallDenseSVD {
public:
  virtual ~SmallDenseSVD() = default;
  virtual void svd_dense(int m, const std::vector<double> &B_rowmajor,
                         std::vector<double> &svals,
                         std::vector<std::vector<double>> &Vhat_cols) const = 0;
};

} // namespace krans::utils
