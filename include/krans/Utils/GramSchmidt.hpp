#pragma once
#include "krans/VectorInterface.hpp"
#include <memory>
#include <vector>

namespace krans::utils {

/** In-place Modified Gram-Schmidt (optional reorth); returns true if all
 * vectors were independent. */
bool modified_gram_schmidt(std::vector<std::unique_ptr<VectorInterface>> &vecs,
                           double tol = 1e-12, bool reorth = true);

/** Orthogonalize v against an orthonormal basis; returns pre-normalization
 * ||v||_2 or 0 on dependence. */
double orthogonalize_against_basis(
    VectorInterface &v,
    const std::vector<std::unique_ptr<VectorInterface>> &basis,
    double tol = 1e-12, bool reorth = true);

/** Maximum |dot| across distinct basis vectors (0 if size <= 1). */
double max_pairwise_abs_dot(
    const std::vector<std::unique_ptr<VectorInterface>> &basis);

} // namespace krans::utils
