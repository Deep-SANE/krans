#ifdef SNS_USE_SUITESPARSE

#include "sns/Solvers/SuiteSparseQRSolver.hpp"
#include "sns/adapters/eigen/EigenVectorAdapter.hpp"
#include <Eigen/Sparse>
#include <Eigen/SPQRSupport>
#include <vector>
#include <iostream>

namespace sns::solvers {

SuiteSparseQRSolver::SuiteSparseQRSolver(const EigenSparseMatrixAdapterInterface &A) : A_(A) {}

std::vector<std::unique_ptr<VectorInterface>>
SuiteSparseQRSolver::solve(const VectorFactory &vf, double tau_rel) {
  const auto &S = A_.view();
  const std::size_t m = A_.rows();
  const std::size_t n = A_.cols();

  double scale = 0.0;
  for (int k = 0; k < S.outerSize(); ++k) {
    for (Eigen::SparseMatrix<double>::InnerIterator it(S, k); it; ++it) {
      scale = std::max(scale, std::abs(it.value()));
    }
  }
  if (scale == 0.0) scale = 1.0;

  const double base = (tau_rel > 0.0 ? tau_rel : std::numeric_limits<double>::epsilon());
  const double tau_abs = base * static_cast<double>(std::max(m, n)) * scale;

  Eigen::SPQR<Eigen::SparseMatrix<double>> spqr;
  spqr.setSPQROrdering(SPQR_ORDERING_DEFAULT);
  spqr.setPivotThreshold(tau_abs);
  spqr.compute(S);

  const long rank = spqr.rank();
  if (rank >= static_cast<long>(n))
    return {};

  const long k = static_cast<long>(n) - rank;

  // R = [R11 R12; 0 0]
  // In Eigen::SPQR, matrixR() returns the upper trapezoidal part.
  // We need to extract R11 and R12.
  // However, Eigen::SPQR::matrixR() is not a simple sparse matrix sometimes.
  Eigen::SparseMatrix<double> R = spqr.matrixR();
  
  // R11 is rank x rank
  Eigen::SparseMatrix<double> R11 = R.topLeftCorner(rank, rank);
  // R12 is rank x k
  Eigen::SparseMatrix<double> R12 = R.topRightCorner(rank, k);

  // Solve R11 * X1 = -R12
  // R11 is upper triangular.
  Eigen::SparseMatrix<double> X1(rank, k);
  
  Eigen::SparseQR<Eigen::SparseMatrix<double>, Eigen::COLAMDOrdering<int>> solver;
  
  // Dense realization for X1 if k is small, or sparse if k is large.
  // For nullspace basis, typically k is small.
  Eigen::MatrixXd X1_dense = -Eigen::MatrixXd(R12);
  R11.template triangularView<Eigen::Upper>().solveInPlace(X1_dense);

  // X = [X1; I_k]
  Eigen::MatrixXd X = Eigen::MatrixXd::Zero(n, k);
  X.topRows(rank) = X1_dense;
  X.bottomRows(k) = Eigen::MatrixXd::Identity(k, k);

  // v = P * X
  const auto &P = spqr.colsPermutation();
  Eigen::MatrixXd N = P * X;

  // Wrap into VectorInterface.
  std::vector<std::unique_ptr<VectorInterface>> result;
  for (long j = 0; j < k; ++j) {
    auto v = vf.create_zero_of_size(n);
    std::memcpy(v->data(), N.col(j).data(), sizeof(double) * n);
    result.push_back(std::move(v));
  }

  return result;
}

std::vector<std::unique_ptr<VectorInterface>>
find_nullspace_suitesparse_qr(const MatrixInterface &A, const VectorFactory &vf,
                              double tol) {
  if (const auto *adapter = dynamic_cast<const EigenSparseMatrixAdapterInterface *>(&A)) {
    SuiteSparseQRSolver solver(*adapter);
    double tau = (tol < 0) ? 1e-10 : tol; 
    return solver.solve(vf, tau);
  } else {
    throw std::runtime_error("SuiteSparseQRSolver requires an Eigen sparse adapter (EigenSparseMatrixAdapterInterface)");
  }
}

} // namespace sns::solvers

#else // SNS_USE_SUITESPARSE

#include "sns/Solvers/SuiteSparseQRSolver.hpp"
#include <stdexcept>

namespace sns::solvers {

std::vector<std::unique_ptr<VectorInterface>>
find_nullspace_suitesparse_qr(const MatrixInterface&, const VectorFactory&, double) {
    throw std::runtime_error("find_nullspace_suitesparse_qr: SNS was built without SuiteSparse support.");
}

} // namespace sns::solvers

#endif // SNS_USE_SUITESPARSE

