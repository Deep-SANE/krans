#ifdef KRANS_USE_PRIMME

#include "krans/Solvers/PrimmeSolver.hpp"
#include "krans/Utils/GramSchmidt.hpp"
#include <primme.h>
#include <vector>
#include <memory>
#include <cstring>
#include <stdexcept>
#include <iostream>
#include <algorithm>
#include <random>

namespace krans::solvers {

struct PrimmeSvdsContext {
  const MatrixInterface *A;
  const VectorFactory *vf;
  
  // Vectors strictly sized for y = A * x
  mutable std::unique_ptr<VectorInterface> x_fwd;
  mutable std::unique_ptr<VectorInterface> y_fwd;
  
  // Vectors strictly sized for y = A^T * x
  mutable std::unique_ptr<VectorInterface> x_trans;
  mutable std::unique_ptr<VectorInterface> y_trans;

  PrimmeSvdsContext(const MatrixInterface *A, const VectorFactory *vf)
      : A(A), vf(vf) {
    // Forward sizes: x is cols, y is rows
    x_fwd = vf->create_zero_of_size(A->cols());
    y_fwd = vf->create_zero_of_size(A->rows());
    
    // Transpose sizes: x is rows, y is cols
    x_trans = vf->create_zero_of_size(A->rows());
    y_trans = vf->create_zero_of_size(A->cols());
  }
};

// The PRIMME_SVDS callback signature
static void primme_svds_matvec(void *x, PRIMME_INT *ldx, void *y, PRIMME_INT *ldy,
                               int *blockSize, int *transpose, 
                               struct primme_svds_params *primme, int *ierr) {
  PrimmeSvdsContext *ctx = static_cast<PrimmeSvdsContext *>(primme->matrix);
  double *x_ptr = static_cast<double *>(x);
  double *y_ptr = static_cast<double *>(y);

  if (!ctx->x_fwd->data() || !ctx->y_fwd->data() || 
      !ctx->x_trans->data() || !ctx->y_trans->data()) {
      *ierr = 1; // Signal error to PRIMME
      return;
  }

  for (int b = 0; b < *blockSize; ++b) {
    double *xb = x_ptr + b * (size_t)(*ldx);
    double *yb = y_ptr + b * (size_t)(*ldy);

    if (*transpose == 0) {
      // y = A * x (Forward)
      std::memcpy(ctx->x_fwd->data(), xb, sizeof(double) * ctx->A->cols());
      ctx->A->multiply(*ctx->x_fwd, *ctx->y_fwd);
      std::memcpy(yb, ctx->y_fwd->data(), sizeof(double) * ctx->A->rows());
    } else {
      // y = A^T * x (Transpose)
      std::memcpy(ctx->x_trans->data(), xb, sizeof(double) * ctx->A->rows());
      ctx->A->multiply_transpose(*ctx->x_trans, *ctx->y_trans);
      std::memcpy(yb, ctx->y_trans->data(), sizeof(double) * ctx->A->cols());
    }
  }
  *ierr = 0;
}

std::vector<std::unique_ptr<VectorInterface>>
find_nullspace_primme(const MatrixInterface &A, const VectorFactory &vf,
                      std::size_t k, double tol) {
  if (k == 0) return {};

  const int max_retries = 5;
  std::vector<std::unique_ptr<VectorInterface>> accumulated_basis;
  auto tmp_m = vf.create_zero_of_size(A.rows());

  for (int retry = 0; retry < max_retries && accumulated_basis.size() < k; ++retry) {
      primme_svds_params primme;
      primme_svds_initialize(&primme);

      // Ask for a bit more than what's remaining to find
      int num_needed = static_cast<int>(k - accumulated_basis.size());
      int num_request = std::min<int>(A.cols(), num_needed + 10);

      primme.m = (PRIMME_INT)A.rows();
      primme.n = (PRIMME_INT)A.cols();
      primme.numSvals = num_request; 
      primme.target = primme_svds_closest_abs;
      double shift = 0.0;
      primme.numTargetShifts = 1;
      primme.targetShifts = &shift;

      primme.eps = 1e-12; 
      primme.maxMatvecs = 30000;
      primme.matrixMatvec = primme_svds_matvec;
      primme.printLevel = 0;
      
      PrimmeSvdsContext ctx(&A, &vf);
      primme.matrix = &ctx;

      if (retry >= 2) {
          primme_svds_set_method(primme_svds_augmented, PRIMME_DEFAULT_METHOD, 
                                PRIMME_DEFAULT_METHOD, &primme);
      } else {
          primme_svds_set_method(primme_svds_default, PRIMME_DEFAULT_METHOD, 
                                PRIMME_DEFAULT_METHOD, &primme);
      }

      std::vector<double> svals(num_request, 0.0);
      std::vector<double> svecs((A.rows() + A.cols()) * num_request, 0.0);
      std::vector<double> resnorms(num_request, 0.0);

      int ret = dprimme_svds(svals.data(), svecs.data(), resnorms.data(), &primme);

      if (ret >= 0) {
        double* right_svecs = svecs.data() + (A.rows() * num_request);
        for (int i = 0; i < num_request; ++i) {
          if (svals[i] <= tol * 100.0 || resnorms[i] <= tol * 100.0) {
              auto v = vf.create_zero_of_size(A.cols());
              std::memcpy(v->data(), right_svecs + (i * A.cols()), sizeof(double) * A.cols());
              
              A.multiply(*v, *tmp_m);
              if (tmp_m->norm() <= tol) {
                  accumulated_basis.push_back(std::move(v));
              }
          }
        }
      }
      primme_svds_free(&primme);

      // Orthogonalize and remove duplicates/dependent vectors
      if (!accumulated_basis.empty()) {
          (void)krans::utils::modified_gram_schmidt(accumulated_basis, 1e-12, true);
          // MGS will shrink the vector if it's dependent, we should prune those
          accumulated_basis.erase(
              std::remove_if(accumulated_basis.begin(), accumulated_basis.end(),
                             [](const auto& v) { return v->norm() < 0.5; }),
              accumulated_basis.end()
          );
          // Renormalize after pruning
          for (auto& v : accumulated_basis) {
              double n = v->norm();
              if (n > 1e-15) v->scale(1.0/n);
          }
      }
  }

  if (accumulated_basis.size() > k) {
      accumulated_basis.resize(k);
  }
  
  return accumulated_basis;
}

} // namespace krans::solvers

#else // KRANS_USE_PRIMME

#include "krans/Solvers/PrimmeSolver.hpp"
#include <stdexcept>

namespace krans::solvers {

std::vector<std::unique_ptr<VectorInterface>>
find_nullspace_primme(const MatrixInterface&, const VectorFactory&, std::size_t, double) {
    throw std::runtime_error("find_nullspace_primme: KraNS was built without PRIMME support.");
}

} // namespace krans::solvers

#endif // KRANS_USE_PRIMME
