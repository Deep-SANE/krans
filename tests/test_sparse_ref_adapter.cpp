#include "krans/adapters/eigen/EigenSparseMatrixRefAdapter.hpp"
#include "krans/adapters/eigen/EigenVectorAdapter.hpp"
#include "krans/adapters/eigen/EigenVectorFactory.hpp"
#include <Eigen/Sparse>
#include <iostream>
#include <cassert>

using namespace krans;

int main() {
    // Create a simple sparse matrix
    Eigen::SparseMatrix<double> S(3, 3);
    S.insert(0, 0) = 1.0;
    S.insert(1, 1) = 2.0;
    S.insert(2, 2) = 3.0;
    S.insert(0, 2) = 4.0;
    S.makeCompressed();

    EigenSparseMatrixRefAdapter adapter(S);
    EigenVectorFactory vf(3);

    auto x = vf.create_zero_of_size(3);
    auto y = vf.create_zero_of_size(3);

    // Test multiply: y = A * x
    x->data()[0] = 1.0;
    x->data()[1] = 0.0;
    x->data()[2] = 1.0;
    // Expected A * x = [1+4, 0, 3] = [5, 0, 3]
    
    adapter.multiply(*x, *y);
    assert(y->data()[0] == 5.0);
    assert(y->data()[1] == 0.0);
    assert(y->data()[2] == 3.0);

    // Test multiply_transpose: y = A^T * x
    // A^T = [1 0 0; 0 2 0; 4 0 3]
    // Expected A^T * x = [1, 0, 4+3] = [1, 0, 7]
    adapter.multiply_transpose(*x, *y);
    assert(y->data()[0] == 1.0);
    assert(y->data()[1] == 0.0);
    assert(y->data()[2] == 7.0);

    std::cout << "test_sparse_ref_adapter OK\n";
    return 0;
}
