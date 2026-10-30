# Integrating Custom Matrix and Vector Libraries with KraNS

The KraNS library is designed to be backend-agnostic through a set of abstract interfaces. This allows you to integrate your own matrix and vector types from other linear algebra libraries (e.g., PETSc, Trilinos, or a custom solution) without modifying the core solver algorithms.

This guide provides a step-by-step walkthrough for creating the necessary adapter classes to bridge your data structures with KraNS.

## Core Concepts

The abstraction layer consists of three key components defined in the `krans` namespace:

1.  **`VectorInterface`**: An abstract class representing a vector. It defines operations like dot products, scaling, norms, and raw data access via `data()`.
2.  **`MatrixInterface`**: An abstract class representing a matrix. Its role is to provide matrix-vector multiplication (`multiply`) and transpose-multiplication (`multiply_transpose`).
3.  **`VectorFactory`**: An abstract factory responsible for creating instances of your `VectorInterface` implementation. The solvers use this to generate temporary vectors and random seed vectors.

To integrate a new library, you must provide concrete implementations for these three interfaces.

## Step-by-Step Guide

Let's assume you want to integrate a library called `MyLinAlg`.

### Step 1: Implement `VectorInterface`

First, create a header file, for instance, `MyVectorAdapter.hpp`. This class will wrap your library's vector object and implement `krans::VectorInterface`.

**Key Responsibilities:**
-   Hold a reference or pointer to your library's native vector object.
-   Implement all pure virtual methods from `VectorInterface`.
-   Use `dynamic_cast` in methods like `dot` and `add_scaled` to safely access the underlying native vector from the `other` `VectorInterface` parameter.

```cpp
// include/krans/adapters/my_linalg/MyVectorAdapter.hpp
#pragma once

#include "krans/VectorInterface.hpp"
#include "MyLinAlg/Vector.hpp" // Your library's header
#include <stdexcept>

namespace krans {

class MyVectorAdapter final : public VectorInterface {
public:
    // --- Constructors ---
    explicit MyVectorAdapter(std::size_t size) : vec_(size) {}
    explicit MyVectorAdapter(const MyLinAlg::Vector& vec) : vec_(vec) {}

    // --- Access to Underlying Object ---
    MyLinAlg::Vector& view() { return vec_; }
    const MyLinAlg::Vector& view() const { return vec_; }

    // --- Raw Data Access (Required for Cross-Adapter Fallbacks) ---
    double* data() override { return vec_.data(); }
    const double* data() const override { return vec_.data(); }

    // --- krans::VectorInterface Implementation ---
    double dot(const VectorInterface& other) const override {
        const auto* other_adapter = dynamic_cast<const MyVectorAdapter*>(&other);
        if (!other_adapter) {
            throw std::invalid_argument("Backend mismatch in dot product.");
        }
        return vec_.dot(other_adapter->view());
    }

    void add_scaled(const VectorInterface& other, double alpha) override {
        const auto* other_adapter = dynamic_cast<const MyVectorAdapter*>(&other);
        if (!other_adapter) {
            throw std::invalid_argument("Backend mismatch in add_scaled.");
        }
        vec_.add_scaled(other_adapter->view(), alpha);
    }
    
    void scale(double alpha) override { vec_.scale(alpha); }
    void normalize() override { vec_.normalize(); }
    double norm() const override { return vec_.norm(); }
    std::size_t size() const override { return vec_.size(); }
    void set_zero() override { vec_.set_zero(); }

    void copy_from(const VectorInterface& other) override {
        const auto* other_adapter = dynamic_cast<const MyVectorAdapter*>(&other);
        if (!other_adapter) {
            throw std::invalid_argument("Backend mismatch in copy_from.");
        }
        vec_ = other_adapter->view();
    }

    std::unique_ptr<VectorInterface> clone() const override {
        return std::make_unique<MyVectorAdapter>(vec_);
    }

private:
    MyLinAlg::Vector vec_; // Your library's vector object
};

} // namespace krans
```

### Step 2: Implement `VectorFactory`

Next, create a factory that produces instances of your `MyVectorAdapter`. The solvers rely on this to create vectors of compatible types.

```cpp
// include/krans/adapters/my_linalg/MyVectorFactory.hpp
#pragma once

#include "krans/VectorFactory.hpp"
#include "MyVectorAdapter.hpp" // Your adapter header

namespace krans {

class MyVectorFactory final : public VectorFactory {
public:
    explicit MyVectorFactory(std::size_t default_size) : default_size_(default_size) {}

    std::unique_ptr<VectorInterface> create_zero_of_size(std::size_t n) const override {
        return std::make_unique<MyVectorAdapter>(n);
    }

    std::unique_ptr<VectorInterface> create_random_of_size(std::size_t n) const override {
        // Implementation depends on how your library creates random vectors
        MyLinAlg::Vector random_vec = MyLinAlg::create_random_vector(n);
        return std::make_unique<MyVectorAdapter>(random_vec);
    }

    std::unique_ptr<VectorInterface> create_zero() const override {
        return create_zero_of_size(default_size_);
    }

    std::unique_ptr<VectorInterface> create_random() const override {
        return create_random_of_size(default_size_);
    }

private:
    std::size_t default_size_;
};

} // namespace krans
```

### Step 3: Implement `MatrixInterface`

Finally, create an adapter for your matrix structure. This class is responsible for performing the matrix-vector products.

**Best Practice (The Fallback Pattern):**
While you should try to `dynamic_cast` the vectors to your specific adapter for maximum performance, it is highly recommended to provide a `memcpy` fallback using the `.data()` pointer. This ensures your matrix can still multiply vectors generated by other backends!

```cpp
// include/krans/adapters/my_linalg/MyMatrixAdapter.hpp
#pragma once

#include "krans/MatrixInterface.hpp"
#include "MyVectorAdapter.hpp" // Your vector adapter
#include "MyLinAlg/Matrix.hpp" // Your library's matrix header
#include <stdexcept>
#include <cstring>
#include <vector>

namespace krans {

class MyMatrixAdapter final : public MatrixInterface {
public:
    explicit MyMatrixAdapter(const MyLinAlg::Matrix& A) : A_(A) {}

    std::size_t rows() const override { return A_.rows(); }
    std::size_t cols() const override { return A_.cols(); }

    void multiply(const VectorInterface& x, VectorInterface& y) const override {
        const auto* x_adapter = dynamic_cast<const MyVectorAdapter*>(&x);
        auto* y_adapter = dynamic_cast<MyVectorAdapter*>(&y);
        
        if (x_adapter && y_adapter) {
            // Fast Path: Both vectors are native to this backend
            A_.multiply(x_adapter->view(), y_adapter->view());
        } else {
            // Universal Fallback: Safe cross-backend multiplication via raw memory
            std::vector<double> x_tmp(cols());
            std::memcpy(x_tmp.data(), x.data(), sizeof(double) * cols());
            
            std::vector<double> y_tmp(rows(), 0.0);
            
            // Assume your library has a way to multiply from raw std::vector/pointers
            A_.multiply_raw(x_tmp.data(), y_tmp.data()); 
            
            std::memcpy(y.data(), y_tmp.data(), sizeof(double) * rows());
        }
    }

    void multiply_transpose(const VectorInterface& x, VectorInterface& y) const override {
        // ... Implement multiply_transpose with the exact same fallback logic ...
    }

private:
    const MyLinAlg::Matrix& A_; // Reference to your matrix
};

} // namespace krans
```

### Step 4: Usage

Once the adapters are implemented, you can use them with the KraNS solvers.

## The "Universal" Callback Adapter

If you don't want to write a full `MatrixInterface` class, KraNS provides a highly flexible `StdMatrixAdapter` (in `include/krans/adapters/universal/`). 

Instead of wrapping a matrix object, it accepts **C++ lambda callbacks**. This allows you to quickly bridge any library's `A * x` and `A^T * x` operations with just two functions, making rapid prototyping incredibly easy. It natively uses `std::vector<double>` and includes the fallback mechanism described above.
