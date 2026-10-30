#include "krans/adapters/eigen/EigenSparseMatrixAdapter.hpp"
#include "krans/adapters/eigen/EigenVectorAdapter.hpp"
#include <Eigen/Sparse>
#include <cassert>
#include <cmath>
#include <iostream>
#include <vector>

using namespace krans;

int main() {
  using SpMat = Eigen::SparseMatrix<double, Eigen::ColMajor, int>;
  using T = Eigen::Triplet<double, int>;

  // A = [[ 3, 0, 0],
  //      [ 0, 4, 5]]
  std::vector<T> trips;
  trips.emplace_back(0, 0, 3.0);
  trips.emplace_back(1, 1, 4.0);
  trips.emplace_back(1, 2, 5.0);

  SpMat A(2, 3);
  A.setFromTriplets(trips.begin(), trips.end());

  EigenSparseMatrixAdapter Amat(A);

  Eigen::VectorXd x_e(3);
  x_e << 1, 2, 3; // x
  EigenVectorAdapter x(x_e);

  EigenVectorAdapter y(2), z(3); // results

  Amat.multiply(x, y); // y = [3*1, 4*2 + 5*3] = [3, 23]
  assert(std::fabs(y.view()(0) - 3.0) < 1e-12);
  assert(std::fabs(y.view()(1) - 23.0) < 1e-12);

  Amat.multiply_transpose(y, z); // z = A^T y = [3*y0, 4*y1, 5*y1]
  assert(std::fabs(z.view()(0) - 3.0 * 3.0) < 1e-12);  // 9
  assert(std::fabs(z.view()(1) - 4.0 * 23.0) < 1e-12); // 92
  assert(std::fabs(z.view()(2) - 5.0 * 23.0) < 1e-12); // 115

  Eigen::VectorXd y_ref = A * x.view();
  Eigen::VectorXd z_ref = A.transpose() * y_ref;
  assert((y.view() - y_ref).norm() < 1e-12);
  assert((z.view() - z_ref).norm() < 1e-12);

  std::cout << "test_sparse_adaptert OK\n";
  return 0;
}
