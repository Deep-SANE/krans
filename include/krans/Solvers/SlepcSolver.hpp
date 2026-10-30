#pragma once
#include "krans/MatrixInterface.hpp"
#include "krans/VectorFactory.hpp"
#include "krans/VectorInterface.hpp"
#include <memory>
#include <vector>

namespace krans::solvers {

/**
 * Null space via SLEPc (Iterative Singular Value Solver).
 *
 * Find k smallest singular values of A.
 */
std::vector<std::unique_ptr<VectorInterface>>
find_nullspace_slepc(const MatrixInterface &A, const VectorFactory &vf,
                      std::size_t k, double tol = 1e-10);

} // namespace krans::solvers
