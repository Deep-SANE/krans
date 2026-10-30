#pragma once

#include "sns/VectorInterface.hpp"
#include <cmath>     
#include <numeric>  
#include <vector>
#include <stdexcept> 
#include <memory>   

namespace sns {
namespace universal {

/** VectorInterface adapter universal backed. */
class StdVectorAdapter final : public VectorInterface {
public:
    /** Constructs a zero-initialized vector of given size. */
    explicit StdVectorAdapter(std::size_t size) : v_(size, 0.0) {}

    /** Constructs from an existing std::vector<double> (copied). */
    explicit StdVectorAdapter(const std::vector<double>& vec) : v_(vec) {}

    /**  Dot product with another StdVectorAdapter. */
    double dot(const VectorInterface& other) const override {
        const auto* o = dynamic_cast<const StdVectorAdapter*>(&other);
        if (!o) {
            throw std::invalid_argument("StdVectorAdapter::dot backend mismatch");
        }
        if (v_.size() != o->v_.size()) {
            throw std::invalid_argument("StdVectorAdapter::dot size mismatch");
        }
        double result = 0.0;
        for (std::size_t i = 0; i < v_.size(); ++i) {
            result += v_[i] * o->v_[i];
        }
        return result;
    }

    /** In-place axpy: this <- this + alpha * other. */
    void add_scaled(const VectorInterface& other, double alpha) override {
        const auto* o = dynamic_cast<const StdVectorAdapter*>(&other);
        if (!o) {
            throw std::invalid_argument("StdVectorAdapter::add_scaled backend mismatch");
        }
        if (v_.size() != o->v_.size()) {
            throw std::invalid_argument("StdVectorAdapter::add_scaled size mismatch");
        }
        for (std::size_t i = 0; i < v_.size(); ++i) {
            v_[i] += alpha * o->v_[i];
        }
    }

    /** In-place scaling: this <- alpha * this. */
    void scale(double alpha) override {
        for (double& val : v_) {
            val *= alpha;
        }
    }

    /**  Normalize in place to unit 2-norm if norm > 0. */
    void normalize() override {
        const double n = norm();
        if (n > 0.0) {
            scale(1.0 / n);
        }
    }

    /** Euclidean (2) norm. */
    double norm() const override {
        double sum_sq = 0.0;
        for (double val : v_) {
            sum_sq += val * val;
        }
        return std::sqrt(sum_sq);
    }

    /** Number of entries. */
    std::size_t size() const override {
        return v_.size();
    }

    /** Set all entries to zero. */
    void set_zero() override {
        std::fill(v_.begin(), v_.end(), 0.0);
    }

    /** Deep copy from another StdVectorAdapter (backend/size checked). */
    void copy_from(const VectorInterface& other) override {
        const auto* o = dynamic_cast<const StdVectorAdapter*>(&other);
        if (!o) {
            throw std::invalid_argument("StdVectorAdapter::copy_from backend mismatch");
        }
        v_ = o->v_; // std::vector deep copy
    }

    /** Polymorphic clone (deep copy). */
    std::unique_ptr<VectorInterface> clone() const override {
        return std::make_unique<StdVectorAdapter>(v_);
    }

    /** Raw data access. */
    double *data() override { return v_.data(); }

    /** Const raw data access. */
    const double *data() const override { return v_.data(); }

    /** Read-only access to the underlying std::vector<double>. */
    const std::vector<double>& view() const { return v_; }

    /** Mutable access to the underlying std::vector<double>. */
    std::vector<double>& view() { return v_; }

private:
    /** Underlying std::vector<double> storage. */
    std::vector<double> v_;
};

} // namespace universal
} // namespace sns
