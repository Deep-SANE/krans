#include "krans/Utils/Projection.hpp"
#include "krans/adapters/eigen/EigenDenseMatrixAdapter.hpp"
#include "krans/adapters/eigen/EigenVectorAdapter.hpp"

#include <Eigen/Dense>
#include <cassert>
#include <cmath>
#include <iostream>

using namespace krans;

static const Eigen::VectorXd &view(const VectorInterface &v) {
  return dynamic_cast<const EigenVectorAdapter &>(v).view();
}
static Eigen::VectorXd &view(VectorInterface &v) {
  return dynamic_cast<EigenVectorAdapter &>(v).view();
}

int main() {
  // A = [[1,2,0],[0,1,1]]  (2x3), ker(A) is 1-dim, spanned by [-2,1,-1]
  Eigen::MatrixXd A_e(2, 3);
  A_e << 1, 2, 0, 0, 1, 1;
  EigenDenseMatrixAdapter A(A_e);

  // Workspaces
  EigenVectorAdapter v(3), tmp_m(2), tmp_n(3), w(2);

  // 1) residual_norm_Av matches direct computation
  v.view() << 1, -1, 2;
  double r1 = krans::utils::residual_norm_Av(A, v, tmp_m);
  double r1_ref = (A_e * view(v)).norm();
  assert(std::fabs(r1 - r1_ref) < 1e-15);

  // 2) gradient step strictly reduces ||Av|| with tau < 2 / ||A||_2^2
  {
    // estimate spectral norm via SVD (only in test)
    Eigen::JacobiSVD<Eigen::MatrixXd> svd(A_e, Eigen::ComputeThinU |
                                                   Eigen::ComputeThinV);
    const double smax = svd.singularValues()(0);
    const double tau = 0.9 / (smax * smax); // safely inside (0, 2/||A||^2)

    v.view() << 0.3, -0.7, 1.1;
    double before = krans::utils::residual_norm_Av(A, v, tmp_m);
    krans::utils::gradient_nullspace_step(A, v, tmp_m, tmp_n, tau);
    double after = krans::utils::residual_norm_Av(A, v, tmp_m);
    assert(after < before - 1e-14); // strict decrease
  }

  // 3) unit-tau gradient alias equals explicit gradient step with tau=1
  {
    EigenVectorAdapter v1(3), v2(3);
    v1.view() << 4, -5, 6;
    v2.view() << 4, -5, 6;

    krans::utils::gradient_nullspace_step_unit_tau(A, v1, tmp_m, tmp_n);
    krans::utils::gradient_nullspace_step(A, v2, tmp_m, tmp_n, 1.0);

    assert((view(v1) - view(v2)).norm() < 1e-15);
  }

  // 4) orthogonal projection
  {
    // normal-equations solver using dense LDLT on G = A A^T (SPD)
    auto solve_normal = [&](const VectorInterface &rhs,
                            VectorInterface &w_out) {
      const auto &b = view(rhs);
      auto &wv = view(w_out);
      Eigen::MatrixXd G = A_e * A_e.transpose(); // 2x2 SPD
      wv = G.ldlt().solve(b);
    };

    v.view() << 0.3, -2.1, 0.7;
    EigenVectorAdapter v_proj = v; // copy for explicit projector check

    // Apply library projector
    krans::utils::orthogonal_project_onto_nullspace(A, v, tmp_m, w, tmp_n,
                                                  solve_normal);

    // Check Av == 0
    double res = krans::utils::residual_norm_Av(A, v, tmp_m);
    assert(res < 1e-12);

    // Explicit projector P = I − A^T(AA^T)^{-1}A
    Eigen::MatrixXd G = A_e * A_e.transpose();
    Eigen::MatrixXd P =
        Eigen::MatrixXd::Identity(3, 3) - A_e.transpose() * G.ldlt().solve(A_e);
    Eigen::VectorXd v_exp = P * view(v_proj);

    assert((view(v) - v_exp).norm() < 1e-12);
  }

  // 5) Already-in-nullspace is a fixed point of orthogonal projection
  {
    Eigen::Vector3d z;
    z << -2, 1, -1; // null vector
    EigenVectorAdapter vz(z.size()), vz_before(z.size());
    view(vz) = z;
    view(vz_before) = z;

    auto solve_normal = [&](const VectorInterface &rhs,
                            VectorInterface &w_out) {
      const auto &b = view(rhs);
      auto &wv = view(w_out);
      Eigen::MatrixXd G = A_e * A_e.transpose();
      wv = G.ldlt().solve(b);
    };

    krans::utils::orthogonal_project_onto_nullspace(A, vz, tmp_m, w, tmp_n,
                                                  solve_normal);

    assert((view(vz) - view(vz_before)).norm() < 1e-12);
    assert(krans::utils::residual_norm_Av(A, vz, tmp_m) < 1e-15);

    // Idempotence (P^2 = P)
    EigenVectorAdapter vz2 = vz;
    krans::utils::orthogonal_project_onto_nullspace(A, vz2, tmp_m, w, tmp_n,
                                                  solve_normal);
    assert((view(vz2) - view(vz)).norm() < 1e-12);
  }

  std::cout << "test_projection OK\n";
  return 0;
}
