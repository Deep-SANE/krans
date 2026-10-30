#pragma once
#include "krans/MatrixInterface.hpp"
#include <Eigen/Sparse>

namespace krans {

/**
 * Common interface for Eigen-based sparse matrix adapters.
 * Provides access to the underlying Eigen::SparseMatrix.
 */
class EigenSparseMatrixAdapterInterface : public MatrixInterface {
public:
  using SparseMat = Eigen::SparseMatrix<double, Eigen::ColMajor, int>;

  virtual ~EigenSparseMatrixAdapterInterface() = default;

  /** Read-only access to the underlying Eigen sparse matrix. */
  virtual const SparseMat &view() const = 0;
};

} // namespace krans
