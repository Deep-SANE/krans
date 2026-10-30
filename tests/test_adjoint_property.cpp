#include "krans/adapters/eigen/EigenDenseMatrixAdapter.hpp"
#include "krans/adapters/eigen/EigenSparseMatrixAdapter.hpp"
#include "krans/adapters/eigen/EigenVectorAdapter.hpp"
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <cassert>
#include <cmath>
#include <iostream>
#include <vector>

using namespace krans;

static void check_dense(int m, int n) {
  Eigen::MatrixXd A = Eigen::MatrixXd::Random(m, n);
  EigenDenseMatrixAdapter M(A);
  EigenVectorAdapter x(n), y(m), Ax(m), ATy(n);
  x.view().setRandom();
  y.view().setRandom();
  M.multiply(x, Ax);            // Ax
  M.multiply_transpose(y, ATy); // A^T y
  double lhs = Ax.dot(y);       // <Ax, y>
  double rhs = x.dot(ATy);      // <x, A^T y>
  assert(std::fabs(lhs - rhs) < 1e-10);
}

static void check_sparse(int m, int n) {
  using Sp = Eigen::SparseMatrix<double, Eigen::ColMajor, int>;
  std::vector<Eigen::Triplet<double, int>> T;
  for (int j = 0; j < n; ++j) {
    T.emplace_back(j % m, j, 1.0 + (j % 3));
    if (m > 1)
      T.emplace_back((j + 1) % m, j, -0.5);
  }
  Sp A(m, n);
  A.setFromTriplets(T.begin(), T.end());
  EigenSparseMatrixAdapter M(A);
  EigenVectorAdapter x(n), y(m), Ax(m), ATy(n);
  x.view().setRandom();
  y.view().setRandom();
  M.multiply(x, Ax);
  M.multiply_transpose(y, ATy);
  double lhs = Ax.dot(y), rhs = x.dot(ATy);
  assert(std::fabs(lhs - rhs) < 1e-10);
}

int main() {
  check_dense(7, 5);
  check_dense(3, 9);
  check_sparse(8, 6);
  check_sparse(4, 11);
  std::cout << "test_adjoint_property OK\n";
  return 0;
}
