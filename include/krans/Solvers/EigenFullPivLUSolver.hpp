#pragma once
#include "krans/MatrixInterface.hpp"
#include "krans/VectorFactory.hpp"
#include "krans/VectorInterface.hpp"
#include <memory>
#include <vector>

namespace krans::solvers {

/**
 * Null space via Eigen::FullPivLU (kernel()).
 *
 * Use a scale-aware absolute threshold
 * tau = (tol_rel > 0 ? tol_rel : eps) * max(m,n) * max|A|.
 * With lu.setThreshold(tau), obtain the numerical rank r = lu.rank().
 * If r < n, the right nullspace basis is returned by lu.kernel() (size
 * n x (n−r)).
 */
std::vector<std::unique_ptr<VectorInterface>>
find_nullspace_fullpivlu(const MatrixInterface &A,
                         const VectorFactory &vf, double tol_rel = 0.0);

} // namespace krans::solvers
