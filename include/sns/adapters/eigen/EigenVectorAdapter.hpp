#pragma once
#include "sns/VectorInterface.hpp"
#include <Eigen/Dense>
#include <cassert>
#include <memory>
#include <stdexcept>

namespace sns {

/** VectorInterface adapter backed by Eigen::VectorXd. */
class EigenVectorAdapter final : public VectorInterface {
public:
  /** Construct a zero-initialized vector of given size. */
  explicit EigenVectorAdapter(std::size_t size)
      : v_(static_cast<Eigen::Index>(size)) {
    v_.setZero();
  }
  /** Construct from an existing Eigen::VectorXd (copied). */
  explicit EigenVectorAdapter(const Eigen::VectorXd &vec) : v_(vec) {}
  /** Dot product with another EigenVectorAdapter. */
  double dot(const VectorInterface &other) const override {
    const auto *o = dynamic_cast<const EigenVectorAdapter *>(&other);
    if (!o)
      throw std::invalid_argument("EigenVectorAdapter::dot backend mismatch");
    if (v_.size() != o->v_.size())
      throw std::invalid_argument("EigenVectorAdapter::dot size mismatch");
    return v_.dot(o->v_);
  }
  /** In-place axpy: this <- this + alpha * other. */
  void add_scaled(const VectorInterface &other, double alpha) override {
    const auto *o = dynamic_cast<const EigenVectorAdapter *>(&other);
    if (!o)
      throw std::invalid_argument(
          "EigenVectorAdapter::add_scaled backend mismatch");
    if (v_.size() != o->v_.size())
      throw std::invalid_argument(
          "EigenVectorAdapter::add_scaled size mismatch");
    v_.noalias() += alpha * o->v_;
  }
  /** In-place scaling: this <- alpha * this. */
  void scale(double alpha) override { v_ *= alpha; }
  /** Normalize in place to unit 2-norm if norm > 0. */
  void normalize() override {
    const double n = v_.norm();
    if (n > 0.0)
      v_ /= n;
  }
  /** Euclidean (2) norm. */
  double norm() const override { return v_.norm(); }
  /** Number of entries. */
  std::size_t size() const override {
    return static_cast<std::size_t>(v_.size());
  }
  /** Set all entries to zero. */
  void set_zero() override { v_.setZero(); }
  /** Deep copy from another EigenVectorAdapter (backend/size checked). */
  void copy_from(const VectorInterface &other) override {
    const auto *o = dynamic_cast<const EigenVectorAdapter *>(&other);
    if (!o)
      throw std::invalid_argument(
          "EigenVectorAdapter::copy_from backend mismatch");
    v_ = o->v_;
  }
  /** Polymorphic clone (deep copy). */
  std::unique_ptr<VectorInterface> clone() const override {
    return std::make_unique<EigenVectorAdapter>(v_);
  }
  /** Raw data access. */
  double *data() override { return v_.data(); }
  /** Const raw data access. */
  const double *data() const override { return v_.data(); }
  /** Read-only access to the underlying Eigen::VectorXd. */
  const Eigen::VectorXd &view() const { return v_; }
  /** Mutable access to the underlying Eigen::VectorXd. */
  Eigen::VectorXd &view() { return v_; }

private:
  /** Underlying Eigen dense vector storage. */
  Eigen::VectorXd v_;
};

} // namespace sns
