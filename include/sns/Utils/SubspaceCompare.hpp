#pragma once

#include "sns/VectorInterface.hpp"
#include <vector>
#include <memory>
#include <cmath>
#include <algorithm>
#include <stdexcept>

namespace sns::utils {

/**
 * Computes the Reduced Row Echelon Form (RREF) of a set of vectors.
 * The vectors are treated as rows of a matrix.
 * This is used to create a unique identifier for a subspace.
 */
inline void compute_rref(std::vector<std::unique_ptr<VectorInterface>>& rows, double tol = 1e-12) {
    if (rows.empty()) return;
    
    const std::size_t num_rows = rows.size();
    const std::size_t num_cols = rows[0]->size();
    std::size_t pivot_row = 0;

    for (std::size_t pivot_col = 0; pivot_col < num_cols && pivot_row < num_rows; ++pivot_col) {
        // Find the pivot
        std::size_t best_row = pivot_row;
        double max_val = 0.0;
        
        for (std::size_t r = pivot_row; r < num_rows; ++r) {
            double* data = rows[r]->data();
            if (!data) throw std::runtime_error("SubspaceCompare: Vector data access not available");
            double val = std::abs(data[pivot_col]);
            if (val > max_val) {
                max_val = val;
                best_row = r;
            }
        }

        if (max_val < tol) {
            // All values in this column (from pivot_row down) are effectively zero
            for (std::size_t r = pivot_row; r < num_rows; ++r) {
                rows[r]->data()[pivot_col] = 0.0;
            }
            continue;
        }

        // Swap rows to bring the pivot to the current pivot_row
        if (best_row != pivot_row) {
            std::swap(rows[pivot_row], rows[best_row]);
        }

        // Normalize pivot row
        double pivot_val = rows[pivot_row]->data()[pivot_col];
        rows[pivot_row]->scale(1.0 / pivot_val);
        rows[pivot_row]->data()[pivot_col] = 1.0; // Numerical stability

        // Eliminate other rows
        for (std::size_t r = 0; r < num_rows; ++r) {
            if (r != pivot_row) {
                double factor = rows[r]->data()[pivot_col];
                if (std::abs(factor) > tol) {
                    rows[r]->add_scaled(*rows[pivot_row], -factor);
                }
                rows[r]->data()[pivot_col] = 0.0; // Numerical stability
            }
        }
        pivot_row++;
    }

    // Clean up any remaining values that should be zero
    for (std::size_t r = 0; r < num_rows; ++r) {
        double* data = rows[r]->data();
        for (std::size_t c = 0; c < num_cols; ++c) {
            if (std::abs(data[c]) < tol) {
                data[c] = 0.0;
            }
        }
    }
}

/**
 * Computes the Frobenius distance between the RREF forms of two bases.
 * Returns a value close to 0 if the bases span the same subspace.
 * Both bases are assumed to be N x k (k vectors of size N).
 */
inline double subspace_diff(const std::vector<std::unique_ptr<VectorInterface>>& basis1,
                           const std::vector<std::unique_ptr<VectorInterface>>& basis2,
                           double tol = 1e-12) {
    if (basis1.empty() && basis2.empty()) return 0.0;
    if (basis1.size() != basis2.size()) return 1e9;; // Different dimensions
    if (basis1[0]->size() != basis2[0]->size()) return 1e9; // Different vector sizes

    // Clone bases to avoid modifying originals
    std::vector<std::unique_ptr<VectorInterface>> b1, b2;
    for (const auto& v : basis1) b1.push_back(v->clone());
    for (const auto& v : basis2) b2.push_back(v->clone());

    compute_rref(b1, tol);
    compute_rref(b2, tol);

    double sum_sq = 0.0;
    for (std::size_t i = 0; i < b1.size(); ++i) {
        // We compare the RREF rows directly
        b1[i]->add_scaled(*b2[i], -1.0);
        double n = b1[i]->norm();
        sum_sq += n * n;
    }
    
    return std::sqrt(sum_sq);
}

} // namespace sns::utils
