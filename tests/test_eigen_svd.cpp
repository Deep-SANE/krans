#include "sns/Utils/EigenSmallSVD.hpp"
#include <Eigen/Dense>
#include <cassert>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <vector>

using sns::utils::EigenSmallSVD;
using sns::utils::SmallDenseSVD;

static double max_abs_diff(const std::vector<double> &a,
                           const std::vector<double> &b) {
  assert(a.size() == b.size());
  double m = 0.0;
  for (std::size_t i = 0; i < a.size(); ++i)
    m = std::max(m, std::fabs(a[i] - b[i]));
  return m;
}

int main() {
  EigenSmallSVD svd;

  // 1) Identity: singular values all 1, V is identity (up to column signs)

  {
    const int m = 3;
    std::vector<double> B = {1, 0, 0, 0, 1, 0, 0, 0, 1}; // row-major
    std::vector<double> s;
    std::vector<std::vector<double>> Vcols;
    svd.svd_dense(m, B, s, Vcols);

    assert((int)s.size() == m);
    for (int i = 0; i < m; ++i)
      assert(std::fabs(s[i] - 1.0) < 1e-15);

    // V should be orthonormal: V^T V = I
    // quick check using pairwise dots
    for (int i = 0; i < m; ++i) {
      // norm ~ 1
      double ni = 0.0;
      for (int k = 0; k < m; ++k)
        ni += Vcols[i][k] * Vcols[i][k];
      assert(std::fabs(ni - 1.0) < 1e-14);
      for (int j = i + 1; j < m; ++j) {
        double dot = 0.0;
        for (int k = 0; k < m; ++k)
          dot += Vcols[i][k] * Vcols[j][k];
        assert(std::fabs(dot) < 1e-14);
      }
    }
  }

  // 2) Diagonal: known singular values (descending)

  {
    const int m = 3;
    // diag(5, 2, 0.3)
    std::vector<double> B = {5, 0, 0, 0, 2, 0, 0, 0, 0.3};
    std::vector<double> s;
    std::vector<std::vector<double>> Vcols;
    svd.svd_dense(m, B, s, Vcols);

    assert(s.size() == (std::size_t)m);
    // Eigen returns descending singular values
    std::vector<double> expected = {5.0, 2.0, 0.3};
    assert(max_abs_diff(s, expected) < 1e-14);
  }

  // 3) General case

  {
    const int m = 3;
    // B = R^T * diag(4,1,0.5) * Q with random-ish orthogonal Q,R
    Eigen::Matrix3d Q;
    Q = Eigen::AngleAxisd(0.4, Eigen::Vector3d::UnitX()) *
        Eigen::AngleAxisd(-0.7, Eigen::Vector3d::UnitY()) *
        Eigen::AngleAxisd(0.2, Eigen::Vector3d::UnitZ());
    Eigen::Matrix3d R;
    R = Eigen::AngleAxisd(-0.3, Eigen::Vector3d::UnitZ());
    Eigen::Matrix3d D = Eigen::DiagonalMatrix<double, 3>(4.0, 1.0, 0.5);
    Eigen::Matrix3d M = R.transpose() * D * Q;
    std::vector<double> B(m * m);
    for (int i = 0; i < m; i++)
      for (int j = 0; j < m; j++)
        B[i * m + j] = M(i, j); // row-major

    std::vector<double> s;
    std::vector<std::vector<double>> Vcols;
    svd.svd_dense(m, B, s, Vcols);

    // Check (B^T B) v = σ² v for each column v of V
    Eigen::Matrix3d BtB = M.transpose() * M;
    for (int j = 0; j < m; ++j) {
      Eigen::Vector3d vj;
      for (int i = 0; i < m; i++)
        vj(i) = Vcols[j][i];
      Eigen::Vector3d lhs = BtB * vj;
      Eigen::Vector3d rhs = (s[j] * s[j]) * vj;
      assert((lhs - rhs).norm() < 1e-10);
    }
  }

  // 4) Error handling: wrong m or size

  {
    bool threw = false;
    try {
      std::vector<double> badB = {1, 2, 3, 4}; // not m*m
      std::vector<double> s;
      std::vector<std::vector<double>> V;
      svd.svd_dense(3, badB, s, V);
    } catch (const std::invalid_argument &) {
      threw = true;
    }
    assert(threw);

    threw = false;
    try {
      std::vector<double> s;
      std::vector<std::vector<double>> V;
      svd.svd_dense(0, {}, s, V);
    } catch (const std::invalid_argument &) {
      threw = true;
    }
    assert(threw);
  }

  std::cout << "test_eigen_svd OK\n";
  return 0;
}
