#pragma once
#include "krans/MatrixInterface.hpp"
#include "krans/VectorFactory.hpp"
#include "krans/VectorInterface.hpp"
#include <memory>
#include <vector>

namespace krans::solvers {

/**
 * Null space via Eigen::CompleteOrthogonalDecomposition (COD).
 *
 * Build COD with a scale-aware absolute threshold
 * tau = (tol_rel > 0 ? tol_rel : eps) * max(m,n) * max|A|.
 * Let k = n - rank. Take the trailing k columns of Z and try the mappings
 * { Z_k,  P*Z_k,  P^T*Z_k }, pick the one with the smallest residual ||A N||,
 * then (optionally) refine each vector via v <- v − cod.solve(A v).
 */
std::vector<std::unique_ptr<VectorInterface>>
find_nullspace_cod(const MatrixInterface &A, const VectorFactory &vf,
                   double tol_rel = 0.0);

} // namespace krans::solvers
