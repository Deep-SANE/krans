#include "krans/adapters/eigen/EigenVectorAdapter.hpp"
#include "krans/adapters/eigen/EigenVectorFactory.hpp"
#include <cassert>
#include <cmath>
#include <iostream>

using namespace krans;

static bool approx_equal(const VectorInterface &a, const VectorInterface &b,
                         double tol = 1e-12) {
  if (a.size() != b.size()) return false;
  auto diff = a.clone();
  diff->add_scaled(b, -1.0);
  return diff->norm() <= tol;
}

int main() {
  const std::size_t N = 5;

  // Deterministic factories with same seed should match
  EigenVectorFactory F1(N, 12345);
  EigenVectorFactory F2(N, 12345);
  EigenVectorFactory F3(N, 111);
  EigenVectorFactory F4(N, 222);

  auto z1 = F1.create_zero();
  auto z2 = F2.create_zero();
  assert(z1->size() == N && z2->size() == N);
  assert(std::fabs(z1->norm()) < 1e-15 && std::fabs(z2->norm()) < 1e-15);

  auto r1 = F1.create_random();
  auto r2 = F2.create_random();
  // With the same seed and same sequence of calls, vectors should be identical
  assert(approx_equal(*r1, *r2));

  auto r3 = F3.create_random();
  auto r4 = F4.create_random();
  // If by extreme chance they match exactly, try one more draw
  bool same = approx_equal(*r3, *r4);
  if (same) {
    r3 = F3.create_random();
    r4 = F4.create_random();
    same = approx_equal(*r3, *r4);
  }
  assert(!same);

  // Sized variants
  auto z10 = F1.create_zero_of_size(10);
  auto r10 = F1.create_random_of_size(10);
  assert(z10->size() == 10 && r10->size() == 10);
  assert(std::fabs(z10->norm()) < 1e-15);

  // Random shouldn't be (exactly) zero with overwhelming probability
  assert(r10->norm() > 0.0);

  std::cout << "test_vector_factory OK\n";
  return 0;
}
