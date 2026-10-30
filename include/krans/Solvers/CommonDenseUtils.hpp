#pragma once
#include "krans/MatrixInterface.hpp"
#include "krans/Utils/GramSchmidt.hpp"
#include "krans/VectorInterface.hpp"
#include "krans/VectorFactory.hpp"
#include "krans/adapters/eigen/EigenDenseMatrixAdapter.hpp"
#include "krans/adapters/eigen/EigenVectorAdapter.hpp"

#include <Eigen/Dense>
#include <algorithm>
#include <memory>
#include <stdexcept>
#include <vector>
#include <cstring>

namespace krans::solvers::detail {

/**
 * Make a dense copy of A.
 * - If A is an EigenDenseMatrixAdapter, return a direct copy of the backend.
 * - Otherwise, build a dense matrix via y = A e_j matvecs.
 */
inline Eigen::MatrixXd
dense_copy_via_matvecs_or_view(const MatrixInterface &A, const VectorFactory &vf) {
  if (const auto *ed = dynamic_cast<const krans::EigenDenseMatrixAdapter *>(&A)) {
    return ed->view(); // copy construct
  }
  const Eigen::Index m = static_cast<Eigen::Index>(A.rows());
  const Eigen::Index n = static_cast<Eigen::Index>(A.cols());
  Eigen::MatrixXd M(m, n);

  auto x = vf.create_zero_of_size(static_cast<std::size_t>(n));
  auto y = vf.create_zero_of_size(static_cast<std::size_t>(m));
  
  for (Eigen::Index j = 0; j < n; ++j) {
    x->set_zero();
    x->data()[j] = 1.0;
    A.multiply(*x, *y);
    // Copy result back to Eigen matrix column
    std::memcpy(M.col(j).data(), y->data(), sizeof(double) * m);
  }
  return M;
}

/**
 * Compute a scale for relative tolerances.
 */
inline double matrix_rank_scale(const Eigen::MatrixXd &M) {
  if (M.size() == 0)
    return 1.0;
  const double scale = M.cwiseAbs().maxCoeff();
  return (scale > 0.0 ? scale : 1.0); // avoid 0 scale
}

/**
 * Convert user-supplied relative tolerance to a method-internal absolute
 * threshold.
 */
inline double make_abs_threshold(double tol_rel, double scale) {
  const double eps = std::numeric_limits<double>::epsilon();
  const double rel = (tol_rel > 0.0 ? tol_rel : eps);
  return rel * scale;
}

/**
 * Wrap columns of a dense matrix N (n x k) into VectorInterface pointers and
 * optionally orthonormalize them with Modified Gram–Schmidt + reorth.
 * Uses the provided VectorFactory to ensure backend compatibility.
 */
inline std::vector<std::unique_ptr<VectorInterface>>
wrap_and_optionally_orthonormalize(const Eigen::MatrixXd &N,
                                   const VectorFactory &vf,
                                   bool orthonormalize = true) {
  std::vector<std::unique_ptr<VectorInterface>> basis;
  if (N.size() == 0)
    return basis;

  const std::size_t rows = static_cast<std::size_t>(N.rows());
  const std::size_t cols = static_cast<std::size_t>(N.cols());

  basis.reserve(cols);
  for (std::size_t j = 0; j < cols; ++j) {
    auto v = vf.create_zero_of_size(rows);
    // Copy data from Eigen column to the new vector
    std::memcpy(v->data(), N.col(static_cast<Eigen::Index>(j)).data(), sizeof(double) * rows);
    basis.emplace_back(std::move(v));
  }
  if (orthonormalize) {
    (void)krans::utils::modified_gram_schmidt(basis, 1e-12, true);
  }
  return basis;
}

} // namespace krans::solvers::detail
