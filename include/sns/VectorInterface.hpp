#pragma once
#include <cstddef>
#include <memory>

namespace sns {

/** Abstract vector interface used by SNS solvers. */
class VectorInterface {
public:
  /** Virtual destructor. */
  virtual ~VectorInterface() = default;
  /** Dot product with another vector. */
  virtual double dot(const VectorInterface &other) const = 0;
  /** In-place axpy: this ← this + alpha * other. */
  virtual void add_scaled(const VectorInterface &other, double alpha) = 0;
  /** In-place scaling: this <- alpha * this. */
  virtual void scale(double alpha) = 0;
  /** Normalize in place to unit 2-norm if norm > 0. */
  virtual void normalize() = 0;
  /** Euclidean (2) norm. */
  virtual double norm() const = 0;
  /** Number of entries. */
  virtual std::size_t size() const = 0;
  /** Set all entries to zero. */
  virtual void set_zero() = 0;
  /** Deep copy: this <- other. */
  virtual void copy_from(const VectorInterface &other) = 0;
  /** Polymorphic clone of this vector. */
  [[nodiscard]] virtual std::unique_ptr<VectorInterface> clone() const = 0;
  /** Raw data access if available (otherwise nullptr). */
  virtual double *data() = 0;
  /** Const raw data access if available (otherwise nullptr). */
  virtual const double *data() const = 0;
};

} // namespace sns