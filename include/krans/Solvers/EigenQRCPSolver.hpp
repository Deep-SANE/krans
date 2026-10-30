#pragma once
#include "krans/MatrixInterface.hpp"
#include "krans/VectorFactory.hpp"
#include "krans/VectorInterface.hpp"
#include <memory>
#include <vector>

namespace krans::solvers {

/**
 * Null space via QR with Column Pivoting (QRCP).
 *
 * Factorization: A P = Q R with qr.setThreshold(tau),
 * tau = (tol_rel > 0 ? tol_rel : eps) * max(m,n) * max|A|.
 * Let r = qr.rank(), k = n − r. Solve the upper-triangular system
 * R11 X1 = −R12, then build X = [X1; I_k] and unpermute: N = P X.
 */
std::vector<std::unique_ptr<VectorInterface>>
find_nullspace_qrcp(const MatrixInterface &A, const VectorFactory &vf,
                    double tol_rel = 0.0);

} // namespace krans::solvers
