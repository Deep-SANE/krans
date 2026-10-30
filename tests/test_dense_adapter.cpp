#include "sns/adapters/eigen/EigenDenseMatrixAdapter.hpp"
#include "sns/adapters/eigen/EigenVectorAdapter.hpp"
#include <Eigen/Dense>
#include <cassert>
#include <cmath>
#include <iostream>

using namespace sns;

int main() {
  Eigen::MatrixXd A(2, 3);
  A << 1, 2, 0, 0, 1, 1;

  EigenDenseMatrixAdapter Amat(A);

  Eigen::VectorXd x_e(3);
  x_e << 1, 1, 1;
  EigenVectorAdapter x(x_e);

  EigenVectorAdapter y(2), z(3);

  // Compute with adapter
  Amat.multiply(x, y);
  Amat.multiply_transpose(y, z);

  // Reference with Eigen
  Eigen::VectorXd y_ref = A * x.view();
  Eigen::VectorXd z_ref = A.transpose() * y_ref;

  assert((y.view() - y_ref).norm() < 1e-12);
  assert((z.view() - z_ref).norm() < 1e-12);

  std::cout << "test_dense_adapter OK\n";
  return 0;
}