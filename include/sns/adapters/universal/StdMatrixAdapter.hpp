#pragma once

#include "sns/MatrixInterface.hpp" 
#include "StdVectorAdapter.hpp"
#include <vector>
#include <stdexcept> 
#include <functional>
#include <cstring>

namespace sns {
namespace universal {

/** 
 * MatrixInterface adapter backed by user-provided callbacks.
 * This allows wrapping any library's matrix-vector operations (Ax and A^T x)
 * without requiring specific matrix storage.
 */
class StdMatrixAdapter final : public MatrixInterface {
public:
    /** 
     * Callback type for matrix-vector operations.
     * Takes a reference to a source vector and a reference to a destination vector.
     */
    using MatVecOp = std::function<void(const std::vector<double>& x, std::vector<double>& y)>;

    /** 
     * Constructs a callback-based matrix adapter.
     */
    StdMatrixAdapter(std::size_t m, std::size_t n, MatVecOp op, MatVecOp op_t)
        : rows_(m), cols_(n), op_(std::move(op)), op_t_(std::move(op_t)) {
        if (!op_ || !op_t_) {
            throw std::invalid_argument("StdMatrixAdapter: Callbacks must be valid.");
        }
    }

    /** Number of rows (m). */
    std::size_t rows() const override {
        return rows_;
    }

    /** Number of columns (n). */
    std::size_t cols() const override {
        return cols_;
    }

    /** Matrix–vector multiply: y <- A * x. */
    void multiply(const VectorInterface& x, VectorInterface& y) const override {
        if (x.size() != cols_ || y.size() != rows_) {
            throw std::invalid_argument("StdMatrixAdapter::multiply size mismatch");
        }

        const auto* x_std = dynamic_cast<const StdVectorAdapter*>(&x);
        auto* y_std = dynamic_cast<StdVectorAdapter*>(&y);

        if (x_std && y_std) {
            // Direct use of underlying std::vector for maximum efficiency
            op_(x_std->view(), y_std->view());
        } else {
            // Fallback: Copy to temporary vectors if types don't match
            // This ensures compatibility with any VectorInterface implementation.
            std::vector<double> x_tmp(cols_);
            std::memcpy(x_tmp.data(), x.data(), sizeof(double) * cols_);
            std::vector<double> y_tmp(rows_, 0.0);
            op_(x_tmp, y_tmp);
            std::memcpy(y.data(), y_tmp.data(), sizeof(double) * rows_);
        }
    }

    /** Transpose multiply: y <- A^T * x. */
    void multiply_transpose(const VectorInterface& x, VectorInterface& y) const override {
        if (x.size() != rows_ || y.size() != cols_) {
            throw std::invalid_argument("StdMatrixAdapter::multiply_transpose size mismatch");
        }

        const auto* x_std = dynamic_cast<const StdVectorAdapter*>(&x);
        auto* y_std = dynamic_cast<StdVectorAdapter*>(&y);

        if (x_std && y_std) {
            op_t_(x_std->view(), y_std->view());
        } else {
            std::vector<double> x_tmp(rows_);
            std::memcpy(x_tmp.data(), x.data(), sizeof(double) * rows_);
            std::vector<double> y_tmp(cols_, 0.0);
            op_t_(x_tmp, y_tmp);
            std::memcpy(y.data(), y_tmp.data(), sizeof(double) * cols_);
        }
    }

private:
    std::size_t rows_;
    std::size_t cols_;
    MatVecOp op_;
    MatVecOp op_t_;
};

} // namespace universal
} // namespace sns
