#pragma once
#include <cstddef>

namespace sns {

/** Forward declaration of the vector interface. */
class VectorInterface;

/** Abstract matrix interface. */
class MatrixInterface {
public:
  /** Virtual destructor. */
  virtual ~MatrixInterface() = default;
  /** Matrix–vector multiply: result ← A * x. */
  virtual void multiply(const VectorInterface &x,
                        VectorInterface &result) const = 0;
  /** Transpose multiply: result ← Aᵀ * x. */
  virtual void multiply_transpose(const VectorInterface &x,
                                  VectorInterface &result) const = 0;
  /** Number of rows (m). */
  virtual std::size_t rows() const = 0;
  /** Number of cols (n). */
  virtual std::size_t cols() const = 0;
};

} // namespace sns
