#ifdef KRANS_USE_SPECTRA

#include "krans/Solvers/SpectraSolver.hpp"
#include <Spectra/SymEigsSolver.h>
#include <Eigen/Core>
#include <vector>
#include <memory>
#include <cstring>
#include <iostream>
#include <algorithm> 
#include <stdexcept> 

namespace krans::solvers {

class SpectraNormalOperator {
private:
  const MatrixInterface &A_;
  mutable std::unique_ptr<VectorInterface> tmp_x_, tmp_y_, tmp_m_;

public:
  using Scalar = double;
  SpectraNormalOperator(const MatrixInterface &A, const VectorFactory &vf)
      : A_(A) {
    tmp_x_ = vf.create_zero_of_size(A.cols());
    tmp_y_ = vf.create_zero_of_size(A.cols());
    tmp_m_ = vf.create_zero_of_size(A.rows());
  }

  int rows() const { return static_cast<int>(A_.cols()); }
  int cols() const { return static_cast<int>(A_.cols()); }

  void perform_op(const double *x_in, double *y_out) const {
    if (!tmp_x_->data() || !tmp_y_->data() || !tmp_m_->data()) {
        throw std::runtime_error("Spectra wrapper requires contiguous vector memory.");
    }
    
    std::memcpy(tmp_x_->data(), x_in, sizeof(double) * A_.cols());
    A_.multiply(*tmp_x_, *tmp_m_);
    A_.multiply_transpose(*tmp_m_, *tmp_y_);
    std::memcpy(y_out, tmp_y_->data(), sizeof(double) * A_.cols());
  }
};

std::vector<std::unique_ptr<VectorInterface>>
find_nullspace_spectra(const MatrixInterface &A, const VectorFactory &vf,
                       std::size_t k, std::size_t ncv, double tol) {
  if (k == 0) return {};
  const std::size_t n = A.cols();
  SpectraNormalOperator op(A, vf);

  if (ncv <= k) ncv = std::max<std::size_t>(k + 10, std::min<std::size_t>(n, 2 * k + 20));
  if (ncv > n) ncv = n;
  if (ncv <= k) k = ncv - 1;

  Spectra::SymEigsSolver<SpectraNormalOperator> eigs(op, static_cast<int>(k), static_cast<int>(ncv));
  
  auto init_v = vf.create_random_of_size(n);
  if (!init_v->data()) {
      throw std::runtime_error("Spectra requires contiguous memory for initialization.");
  }
  eigs.init(init_v->data());

  // Spectra uses tol for its internal eigensolver residual, not the physical ||Av|| residual.
  int nconv = eigs.compute(Spectra::SortRule::SmallestAlge, 5000, tol);

  std::vector<std::unique_ptr<VectorInterface>> accepted_null_vectors;

  if (nconv > 0) {
    Eigen::MatrixXd evecs = eigs.eigenvectors();
    auto tmp_m = vf.create_zero_of_size(A.rows());
    
    for (int i = 0; i < std::min<int>(nconv, k); ++i) {
        auto candidate = vf.create_zero_of_size(n);
        std::memcpy(candidate->data(), evecs.col(i).data(), sizeof(double) * n);
        
        A.multiply(*candidate, *tmp_m);
        if (tmp_m->norm() <= tol) {
            accepted_null_vectors.push_back(std::move(candidate));
        }
    }
  }
  
  return accepted_null_vectors;
}

} // namespace krans::solvers

#else // KRANS_USE_SPECTRA

#include "krans/Solvers/SpectraSolver.hpp"
#include <stdexcept>

namespace krans::solvers {

std::vector<std::unique_ptr<VectorInterface>>
find_nullspace_spectra(const MatrixInterface &A, const VectorFactory &vf, std::size_t k, std::size_t ncv, double tol) {
    throw std::runtime_error("find_nullspace_spectra: KraNS was built without Spectra support.");
}

} // namespace krans::solvers

#endif // KRANS_USE_SPECTRA