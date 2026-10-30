#pragma once
#include "sns/MatrixInterface.hpp"
#include "sns/adapters/eigen/EigenSparseMatrixAdapterInterface.hpp"
#include "sns/adapters/eigen/EigenVectorAdapter.hpp"
#include <Eigen/Sparse>
#include <cassert>
#include <stdexcept>

namespace sns {

/** 
 * MatrixInterface adapter backed by a reference to Eigen::SparseMatrix.
 * 
 * Unlike EigenSparseMatrixAdapter, this class does NOT own the matrix 
 * and does NOT store a precomputed transpose. It uses Eigen's .transpose() 
 * view for the multiply_transpose operation.
 */
class EigenSparseMatrixRefAdapter final : public EigenSparseMatrixAdapterInterface {
public:
  /** Scalar, index, and sparse matrix types for this adapter. */
  using Scalar = double;
  using Index = int;
  using SparseMat = Eigen::SparseMatrix<Scalar, Eigen::ColMajor, Index>;

  /** Construct from a reference to a sparse matrix. */
  explicit EigenSparseMatrixRefAdapter(const SparseMat &A) : A_(A) {}

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
          "EigenSparseMatrixRefAdapter::multiply backend mismatch");
    if (ex->view().size() != A_.cols() || ey->view().size() != A_.rows())
      throw std::invalid_argument(
          "EigenSparseMatrixRefAdapter::multiply size mismatch");
    ey->view().noalias() = A_ * ex->view();
  }
  /** Transpose multiply: y <- A^T * x. */
  void multiply_transpose(const VectorInterface &x,
                          VectorInterface &y) const override {
    const auto *ex = dynamic_cast<const EigenVectorAdapter *>(&x);
    auto *ey = dynamic_cast<EigenVectorAdapter *>(&y);
    if (!ex || !ey)
      throw std::invalid_argument(
          "EigenSparseMatrixRefAdapter::multiply_transpose backend mismatch");
    if (ex->view().size() != A_.rows() || ey->view().size() != A_.cols())
      throw std::invalid_argument(
          "EigenSparseMatrixRefAdapter::multiply_transpose size mismatch");
    // Use Eigen's transpose view. No extra storage!
    ey->view().noalias() = A_.transpose() * ex->view();
  }
  /** Read-only access to the underlying sparse matrix A. */
  const SparseMat &view() const override { return A_; }

private:
  /** Non-owning reference to the sparse matrix A. */
  const SparseMat &A_;
};

} // namespace sns
