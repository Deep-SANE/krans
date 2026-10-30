#pragma once

#include "krans/MatrixInterface.hpp"
#include "krans/VectorFactory.hpp"
#include "krans/VectorInterface.hpp"
#include <memory>
#include <vector>

namespace krans::solvers {

/**
 * Null space via PRIMME (PRIMME_SVDS).
 * 
 * This uses the SVDS module of PRIMME to find the smallest singular values.
 */
std::vector<std::unique_ptr<VectorInterface>>
find_nullspace_primme(const MatrixInterface &A, const VectorFactory &vf,
                      std::size_t k, double tol = 1e-10);

} // namespace krans::solvers
