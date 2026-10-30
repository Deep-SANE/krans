#pragma once

#include "krans/VectorFactory.hpp"
#include "StdVectorAdapter.hpp" 
#include <memory>               
#include <random>               
#include <vector>              

namespace krans {
namespace universal {

/** Factory producing universal backend vectors with zero/random initialization. */
class StdVectorFactory final : public VectorFactory {
public:
    /** Construct with a default vector size and RNG seed (0 -> random_device). */
    explicit StdVectorFactory(std::size_t default_size, unsigned seed = 0)
        : default_size_(default_size),
          seed_(seed ? seed : std::random_device{}()),
          gen_(seed_),
          dist_(0.0, 1.0) {}

    /** Create a zero vector of the default size. */
    std::unique_ptr<VectorInterface> create_zero() const override {
        return create_zero_of_size(default_size_);
    }

    /** Create a random (normal) vector of the default size. */
    std::unique_ptr<VectorInterface> create_random() const override {
        return create_random_of_size(default_size_);
    }

    /** Create a zero vector of explicit size n. */
    std::unique_ptr<VectorInterface> create_zero_of_size(std::size_t n) const override {
        return std::make_unique<StdVectorAdapter>(n); // StdVectorAdapter constructor initializes to zero
    }

    /** Create a random (normal) vector of explicit size n. */
    std::unique_ptr<VectorInterface> create_random_of_size(std::size_t n) const override {
        std::vector<double> v_data(n);
        for (std::size_t i = 0; i < n; ++i) {
            v_data[i] = dist_(gen_);
        }
        return std::make_unique<StdVectorAdapter>(v_data);
    }

    /** Return the RNG seed used by this factory. */
    unsigned seed() const { return seed_; }

private:
    /** Default size used by create_zero/create_random. */
    std::size_t default_size_;
    /** Seed used to initialize the RNG. */
    unsigned seed_;
    /** Mersenne Twister RNG (mutable for const creation methods). */
    mutable std::mt19937 gen_;
    /** Normal distribution N(0,1) for random vector entries. */
    mutable std::normal_distribution<double> dist_;
};

} // namespace universal
} // namespace krans
