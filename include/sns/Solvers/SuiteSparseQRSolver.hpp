#pragma once

#include "sns/MatrixInterface.hpp"
#include "sns/VectorFactory.hpp"
#include "sns/VectorInterface.hpp"
#include "sns/adapters/eigen/EigenSparseMatrixAdapterInterface.hpp"
#include <memory>
#include <vector>

namespace sns::solvers {

/**
 * Null space via SuiteSparseQR (SPQR).
 * This class wraps Eigen::SPQR to provide a direct sparse solver for the null space.
 */
class SuiteSparseQRSolver {
public:
  /** Construct from an EigenSparseMatrixAdapterInterface. */
  explicit SuiteSparseQRSolver(const EigenSparseMatrixAdapterInterface &A);

  /** Solve for the null space. */
  std::vector<std::unique_ptr<VectorInterface>> solve(const VectorFactory &vf,
                                                      double tau);

private:
  const EigenSparseMatrixAdapterInterface &A_;
};

/** Legacy free function. */
std::vector<std::unique_ptr<VectorInterface>>
find_nullspace_suitesparse_qr(const MatrixInterface &A, const VectorFactory &vf,
                              double tol = -1.0);

} // namespace sns::solvers
