#pragma once
#include "krans/MatrixInterface.hpp"
#include "krans/VectorFactory.hpp"
#include "krans/VectorInterface.hpp"
#include <memory>
#include <vector>

namespace krans::solvers {

/**
 * Null space via SVD.
 *
 * Compute M = U Σ V^T with FULL V (JacobiSVD or BDCSVD).
 * With a relative cutoff
 * tau = (tol_rel > 0 ? tol_rel : eps) * max(m,n) * sigma_max,
 * take r = #{ sigma_i > tau }, k = n − r; the right nullspace is spanned by the
 * trailing k columns of V.
 */
std::vector<std::unique_ptr<VectorInterface>>
find_nullspace_svd(const MatrixInterface &A, const VectorFactory &vf,
                   double tol_rel = 0.0);

} // namespace krans::solvers
