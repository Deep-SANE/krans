#pragma once
#include "krans/MatrixInterface.hpp"
#include "krans/VectorInterface.hpp"
#include <functional>

namespace krans::utils {

/** ||Av||_2  (uses tmp_m as workspace of size m). Does not modify v. */
double residual_norm_Av(const MatrixInterface &A, const VectorInterface &v,
                        VectorInterface &tmp_m);

/** Steepest-descent step for f(v) = 1/2 ||A v||_2^2: v <- v - tau A^T(A v). */
inline void gradient_nullspace_step(const MatrixInterface &A,
                                    VectorInterface &v,
                                    VectorInterface &tmp_m, // size m
                                    VectorInterface &tmp_n, // size n
                                    double tau) {
  A.multiply(v, tmp_m);               // tmp_m = A v
  A.multiply_transpose(tmp_m, tmp_n); // tmp_n = A^T tmp_m
  v.add_scaled(tmp_n, -tau);          // v <- v - tau tmp_n
}

/** Convenience alias of the gradient step with tau = 1 (not an orthogonal
 * projector). */
inline void gradient_nullspace_step_unit_tau(const MatrixInterface &A,
                                             VectorInterface &v,
                                             VectorInterface &tmp_m,
                                             VectorInterface &tmp_n) {
  gradient_nullspace_step(A, v, tmp_m, tmp_n, 1.0);
}

/** Callback that solves (A A^T) w = rhs for w in R^m (normal equations). */
using NormalEqSolver =
    std::function<void(const VectorInterface &rhs, VectorInterface &w)>;

/** Orthogonal projection onto ker(A): v <- (I - A^T (A A^T)^{-1} A) v via
 * user-provided solver. */
void orthogonal_project_onto_nullspace(const MatrixInterface &A,
                                       VectorInterface &v,
                                       VectorInterface &tmp_m, // size m
                                       VectorInterface &w_m,   // size m
                                       VectorInterface &tmp_n, // size n
                                       const NormalEqSolver &solve_normal_eq);

} // namespace krans::utils
