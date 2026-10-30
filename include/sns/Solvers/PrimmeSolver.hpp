#pragma once

#include "sns/MatrixInterface.hpp"
#include "sns/VectorFactory.hpp"
#include "sns/VectorInterface.hpp"
#include <memory>
#include <vector>

namespace sns::solvers {

/**
 * Null space via PRIMME (PRIMME_SVDS).
 * 
 * This uses the SVDS module of PRIMME to find the smallest singular values.
 */
std::vector<std::unique_ptr<VectorInterface>>
find_nullspace_primme(const MatrixInterface &A, const VectorFactory &vf,
                      std::size_t k, double tol = 1e-10);

} // namespace sns::solvers
