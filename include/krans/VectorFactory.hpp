#pragma once
#include "VectorInterface.hpp"
#include <cstddef>
#include <memory>

namespace krans {

/** Abstract factory for creating vectors compatible with a matrix/backend. */
class VectorFactory {
public:
  /** Virtual destructor. */
  virtual ~VectorFactory() = default;
  /** Create a new zero vector of the default/compatible size. */
  virtual std::unique_ptr<VectorInterface> create_zero() const = 0;
  /** Create a new random vector (e.g., i.i.d. normal) of the default size. */
  virtual std::unique_ptr<VectorInterface> create_random() const = 0;
  /** Create a new zero vector of explicit size n. */
  virtual std::unique_ptr<VectorInterface>
  create_zero_of_size(std::size_t n) const = 0;
  /** Create a new random vector of explicit size n. */
  virtual std::unique_ptr<VectorInterface>
  create_random_of_size(std::size_t n) const = 0;
};

} // namespace krans