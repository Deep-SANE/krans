#include "sns/Utils/GramSchmidt.hpp"
#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <vector>

namespace sns::utils {

// helper: validate all vectors share the same size (throws in release too)
static void
ensure_same_size(const std::vector<std::unique_ptr<VectorInterface>> &vecs) {
  if (vecs.empty())
    return;
  const std::size_t n = vecs.front()->size();
  for (const auto &p : vecs) {
    if (!p)
      throw std::invalid_argument("GramSchmidt: null vector");
    if (p->size() != n)
      throw std::invalid_argument("GramSchmidt: vector size mismatch");
  }
}

static double
mgs_pass(VectorInterface &v,
         const std::vector<std::unique_ptr<VectorInterface>> &basis) {
  for (const auto &q : basis) {
    const double c = v.dot(*q);
    v.add_scaled(*q, -c);
  }
  return v.norm();
}

double orthogonalize_against_basis(
    VectorInterface &v,
    const std::vector<std::unique_ptr<VectorInterface>> &basis, double tol,
    bool reorth) {
  // basis is assumed orthonormal
  double n1 = mgs_pass(v, basis);
  if (n1 < tol)
    return 0.0;

  if (reorth) {
    double n2 = mgs_pass(v, basis);
    if (n2 < tol)
      return 0.0;
    n1 = n2;
  }

  v.scale(1.0 / n1);
  return n1;
}

bool modified_gram_schmidt(std::vector<std::unique_ptr<VectorInterface>> &vecs,
                           double tol, bool reorth) {
  ensure_same_size(vecs);

  for (std::size_t i = 0; i < vecs.size(); ++i) {
    for (std::size_t j = 0; j < i; ++j) {
      const double c = vecs[i]->dot(*vecs[j]);
      vecs[i]->add_scaled(*vecs[j], -c);
    }
    double n1 = vecs[i]->norm();
    if (n1 < tol)
      return false;

    if (reorth) {
      for (std::size_t j = 0; j < i; ++j) {
        const double c = vecs[i]->dot(*vecs[j]);
        vecs[i]->add_scaled(*vecs[j], -c);
      }
      double n2 = vecs[i]->norm();
      if (n2 < tol)
        return false;
      n1 = n2;
    }

    vecs[i]->scale(1.0 / n1);
  }
  return true;
}

double max_pairwise_abs_dot(
    const std::vector<std::unique_ptr<VectorInterface>> &basis) {
  double m = 0.0;
  for (std::size_t i = 0; i < basis.size(); ++i)
    for (std::size_t j = i + 1; j < basis.size(); ++j)
      m = std::max(m, std::fabs(basis[i]->dot(*basis[j])));
  return m;
}

} // namespace sns::utils
