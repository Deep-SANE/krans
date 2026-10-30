#pragma once
#include "krans/MatrixInterface.hpp"
#include "krans/adapters/eigen/EigenVectorAdapter.hpp"
#include <Eigen/Dense>
#include <cassert>
#include <stdexcept>

namespace krans {

/** MatrixInterface adapter backed by Eigen::MatrixXd. */
class EigenDenseMatrixAdapter final : public MatrixInterface {
public:
  /** Alias for the underlying dense matrix type. */
  using DenseMat = Eigen::MatrixXd;

  /** Construct from a const reference to an existing matrix. */
  explicit EigenDenseMatrixAdapter(const DenseMat &A) : A_(A) {}
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
          "EigenDenseMatrixAdapter::multiply backend mismatch");
    if (ex->view().size() != A_.cols() || ey->view().size() != A_.rows())
      throw std::invalid_argument(
          "EigenDenseMatrixAdapter::multiply size mismatch");
    ey->view().noalias() = A_ * ex->view();
  }
  /** Transpose multiply: y <- A^T * x. */
  void multiply_transpose(const VectorInterface &x,
                          VectorInterface &y) const override {
    const auto *ex = dynamic_cast<const EigenVectorAdapter *>(&x);
    auto *ey = dynamic_cast<EigenVectorAdapter *>(&y);
    if (!ex || !ey)
      throw std::invalid_argument(
          "EigenDenseMatrixAdapter::multiply_transpose backend mismatch");
    if (ex->view().size() != A_.rows() || ey->view().size() != A_.cols())
      throw std::invalid_argument(
          "EigenDenseMatrixAdapter::multiply_transpose size mismatch");
    ey->view().noalias() = A_.transpose() * ex->view();
  }
  /** Read-only access to the underlying dense matrix A. */
  const DenseMat &view() const { return A_; }

private:
  /** Non-owning reference to the user-supplied dense matrix A. */
  const DenseMat &A_;
};

} // namespace krans
