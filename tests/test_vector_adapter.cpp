#include "krans/adapters/eigen/EigenVectorAdapter.hpp"
#include <cassert>
#include <cmath>
#include <iostream>

using namespace krans;

int main() {
  // v = 0, u = [1,2,3]
  EigenVectorAdapter v(3);
  Eigen::VectorXd u_eig(3);
  u_eig << 1, 2, 3;
  EigenVectorAdapter u(u_eig);

  // size & zero
  assert(v.size() == 3);
  assert(std::fabs(v.norm()) < 1e-15);

  // add_scaled
  v.add_scaled(u, 1.0); // v = u
  assert(std::fabs(v.dot(u) - u.norm() * u.norm()) < 1e-12);

  // scale
  v.scale(0.5); // v = 0.5*u
  assert(std::fabs(v.norm() - (u.norm() * 0.5)) < 1e-12);

  // normalize -> unit
  v.normalize();
  assert(std::fabs(v.norm() - 1.0) < 1e-12);

  // copy_from + set_zero
  auto w = v.clone();
  w->set_zero();
  assert(std::fabs(w->norm()) < 1e-15);
  w->copy_from(v);
  assert(std::fabs(w->norm() - 1.0) < 1e-12);

  EigenVectorAdapter z(4); // starts as zero
  z.normalize();
  assert(std::fabs(z.norm()) < 1e-15);

  std::cout << "test_vector_adapter OK\n";
  return 0;
}
