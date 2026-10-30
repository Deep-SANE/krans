#include "krans/Utils/Projection.hpp"
#include <cassert>
#include <stdexcept>
#include <string>

namespace krans::utils {

double residual_norm_Av(const MatrixInterface &A, const VectorInterface &v,
                        VectorInterface &tmp_m) {
  if (A.rows() != tmp_m.size())
    throw std::invalid_argument("residual_norm_Av: tmp_m size mismatch");
  if (A.cols() != v.size())
    throw std::invalid_argument("residual_norm_Av: v size mismatch");

  A.multiply(v, tmp_m);
  return tmp_m.norm();
}

void orthogonal_project_onto_nullspace(const MatrixInterface &A,
                                       VectorInterface &v,
                                       VectorInterface &tmp_m,
                                       VectorInterface &w_m,
                                       VectorInterface &tmp_n,
                                       const NormalEqSolver &solve_normal_eq) {
  if (!solve_normal_eq)
    throw std::invalid_argument(
        "orthogonal_project_onto_nullspace: solver callback is null");
  if (A.rows() != tmp_m.size() || A.rows() != w_m.size())
    throw std::invalid_argument(
        "orthogonal_project_onto_nullspace: m-sized workspace mismatch");
  if (A.cols() != tmp_n.size() || A.cols() != v.size())
    throw std::invalid_argument(
        "orthogonal_project_onto_nullspace: n-sized workspace mismatch");

  // r = Av
  A.multiply(v, tmp_m);

  // Solve (AA^T) w = r
  solve_normal_eq(tmp_m, w_m);

  // t = A^T w
  A.multiply_transpose(w_m, tmp_n);

  // v <- v − t
  v.add_scaled(tmp_n, -1.0);
}

} // namespace krans::utils
