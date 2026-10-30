#pragma once
#include "sns/MatrixInterface.hpp"
#include "sns/adapters/eigen/EigenSparseMatrixAdapterInterface.hpp"
#include "sns/adapters/eigen/EigenVectorAdapter.hpp"
#include <Eigen/Sparse>
#include <cassert>
#include <stdexcept>

namespace sns {

/** MatrixInterface adapter backed by Eigen::SparseMatrix. */
class EigenSparseMatrixAdapter final : public EigenSparseMatrixAdapterInterface {
public:
  /** Scalar, index, and sparse matrix types for this adapter. */
  using Scalar = double;
  using Index = int;
  using SparseMat = Eigen::SparseMatrix<Scalar, Eigen::ColMajor, Index>;

  /** Construct from a sparse matrix. */
  explicit EigenSparseMatrixAdapter(const SparseMat &A) : A_(A) {
    A_.makeCompressed();
    AT_ = A_.transpose();
    AT_.makeCompressed();
  }
  /** Number of rows (m). */
  std::size_t rows() const override {
    return static_cast<std::size_t>(A_.rows());
  }
  /** Number of columns (n). */
  std::size_t cols() const override {
    return static_cast<std::size_t>(A_.cols());
  }
  /** Matrix–vector multiply: y <- A * x. */
  void multiply(const VectorInterface &x, VectorInterface &y) const override {
    const auto *ex = dynamic_cast<const EigenVectorAdapter *>(&x);
    auto *ey = dynamic_cast<EigenVectorAdapter *>(&y);
    if (!ex || !ey)
      throw std::invalid_argument(
          "EigenSparseMatrixAdapter::multiply backend mismatch");
    if (ex->view().size() != A_.cols() || ey->view().size() != A_.rows())
      throw std::invalid_argument(
          "EigenSparseMatrixAdapter::multiply size mismatch");
    ey->view().noalias() = A_ * ex->view();
  }
  /** Transpose multiply: y <- A^T * x. */
  void multiply_transpose(const VectorInterface &x,
                          VectorInterface &y) const override {
    const auto *ex = dynamic_cast<const EigenVectorAdapter *>(&x);
    auto *ey = dynamic_cast<EigenVectorAdapter *>(&y);
    if (!ex || !ey)
      throw std::invalid_argument(
          "EigenSparseMatrixAdapter::multiply_transpose backend mismatch");
    if (ex->view().size() != A_.rows() || ey->view().size() != A_.cols())
      throw std::invalid_argument(
          "EigenSparseMatrixAdapter::multiply_transpose size mismatch");
    ey->view().noalias() = AT_ * ex->view();
  }
  /** Read-only access to the underlying sparse matrix A. */
  const SparseMat &view() const override { return A_; }
  /** Read-only access to the cached transpose A^T. */
  const SparseMat &view_transpose() const { return AT_; }

private:
  /** Owned sparse matrix A in compressed column format. */
  SparseMat A_;
  /** Cached transpose Aᵀ in compressed column format. */
  SparseMat AT_;
};

} // namespace sns
