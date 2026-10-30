#include "sns/Utils/GramSchmidt.hpp"
#include "sns/adapters/eigen/EigenVectorAdapter.hpp"

#include <cassert>
#include <cmath>
#include <iostream>
#include <memory>
#include <random>
#include <vector>

using namespace sns;

// helper: build Eigen-backed vector from initializer list
static std::unique_ptr<VectorInterface>
make_vec(std::initializer_list<double> x) {
  Eigen::VectorXd v(static_cast<int>(x.size()));
  int i = 0;
  for (double a : x)
    v[i++] = a;
  return std::make_unique<EigenVectorAdapter>(v);
}

// helper: quick random vector (normal) of size n
static std::unique_ptr<VectorInterface> make_rand_vec(std::size_t n,
                                                      unsigned seed = 123) {
  std::mt19937 gen(seed);
  std::normal_distribution<double> dist(0.0, 1.0);
  Eigen::VectorXd v(static_cast<Eigen::Index>(n));
  for (std::size_t i = 0; i < n; ++i)
    v[static_cast<Eigen::Index>(i)] = dist(gen);
  return std::make_unique<EigenVectorAdapter>(v);
}

int main() {

  // 1) Independent set => produces an orthonormal basis of the same span
  {
    std::vector<std::unique_ptr<VectorInterface>> Q;
    Q.emplace_back(make_vec({1, 1, 0}));
    Q.emplace_back(make_vec({1, 0, 1}));
    Q.emplace_back(make_vec({0, 1, 1}));

    bool ok = sns::utils::modified_gram_schmidt(Q, 1e-12, true);
    assert(ok);

    // Orthonormality: ||q_i|| = 1 and <q_i, q_j> = 0 for i != j
    for (int i = 0; i < 3; ++i) {
      assert(std::fabs(Q[i]->norm() - 1.0) < 1e-12);
      for (int j = i + 1; j < 3; ++j) {
        const double d = Q[i]->dot(*Q[j]);
        assert(std::fabs(d) < 1e-12);
      }
    }

    // Diagnostic agrees (maximum off-diagonal inner product ~ 0)
    const double mx = sns::utils::max_pairwise_abs_dot(Q);
    assert(mx < 1e-12);
  }

  // 2) Dependent set => batch MGS reports failure
  {
    std::vector<std::unique_ptr<VectorInterface>> Dep;
    Dep.emplace_back(make_vec({1, 0, 0}));
    Dep.emplace_back(make_vec({2, 0, 0})); // dependent
    bool ok2 = sns::utils::modified_gram_schmidt(Dep, 1e-12, true);
    assert(!ok2);
  }

  // 3) Single-vector orthogonalization against an ONB
  {
    std::vector<std::unique_ptr<VectorInterface>> Basis;
    Basis.emplace_back(make_vec({1, 0, 0}));
    Basis[0]->normalize(); // ONB = {e1}

    // v = (1,2,3); distance to span{e1} is ||(0,2,3)|| = sqrt(13)
    auto v = make_vec({1, 2, 3});
    const double nrm =
        sns::utils::orthogonalize_against_basis(*v, Basis, 1e-12, true);
    assert(nrm > 0);
    assert(std::fabs(nrm - std::sqrt(13.0)) < 1e-12); // pre-normalization norm
    assert(std::fabs(v->dot(*Basis[0])) < 1e-12);     // orthogonal to basis
    assert(std::fabs(v->norm() - 1.0) < 1e-12);       // unit norm
  }

  // 4) Tolerance behavior on a near-dependent vector

  {
    std::vector<std::unique_ptr<VectorInterface>> Basis;
    Basis.emplace_back(make_vec({1, 0, 0}));
    Basis[0]->normalize();

    // v = e1 + eps * e2; after projection, residual norm ~ |eps|
    const double eps = 1e-13;
    auto v1 = make_vec({1.0, eps, 0.0});

    double n1 =
        sns::utils::orthogonalize_against_basis(*v1, Basis, 1e-12, true);
    assert(n1 == 0.0);

    auto v2 = make_vec({1.0, eps, 0.0});
    double n2 =
        sns::utils::orthogonalize_against_basis(*v2, Basis, 1e-14, true);
    assert(n2 > 0.0);
    assert(std::fabs(n2 - std::fabs(eps)) < 1e-15); // close to eps
    assert(std::fabs(v2->norm() - 1.0) < 1e-12);
  }

  // 5) Orthogonality diagnostic on a known ONB

  {
    std::vector<std::unique_ptr<VectorInterface>> E;
    E.emplace_back(make_vec({1, 0, 0}));
    E.emplace_back(make_vec({0, 1, 0}));
    E.emplace_back(make_vec({0, 0, 1}));
    for (auto &q : E)
      q->normalize(); // already ONB
    const double mx = sns::utils::max_pairwise_abs_dot(E);
    assert(mx < 1e-15);
  }

  // 6) Random smoke test: 5 random vectors in R^5 => MGS should succeed

  {
    const std::size_t n = 5, k = 5;
    std::vector<std::unique_ptr<VectorInterface>> V;
    for (std::size_t i = 0; i < k; ++i)
      V.emplace_back(make_rand_vec(n, 100 + static_cast<unsigned>(i)));

    bool ok = sns::utils::modified_gram_schmidt(V, 1e-12, true);
    assert(ok);

    // Check orthonormality with a slightly looser tolerance (random rounding)
    for (std::size_t i = 0; i < k; ++i) {
      assert(std::fabs(V[i]->norm() - 1.0) < 1e-10);
      for (std::size_t j = i + 1; j < k; ++j) {
        assert(std::fabs(V[i]->dot(*V[j])) < 1e-10);
      }
    }
  }

  std::cout << "test_gram_schmidt OK\n";
  return 0;
}
