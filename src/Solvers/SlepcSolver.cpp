#ifdef KRANS_USE_SLEPC

#include "krans/Solvers/SlepcSolver.hpp"
#include <slepcsvd.h>
#include <vector>
#include <memory>
#include <iostream>
#include <cstring>
#include <stdexcept>

namespace krans::solvers {
    
class SlepcManager {
public:
  SlepcManager() {
    PetscBool initialized;
    PetscInitialized(&initialized);
    if (!initialized) {
      int argc = 0;
      char **argv = nullptr;
      SlepcInitialize(&argc, &argv, nullptr, nullptr);
    }
  }
  ~SlepcManager() {
    PetscBool initialized;
    PetscInitialized(&initialized);
    if (initialized) {
      SlepcFinalize();
    }
  }
};

struct SlepcMatContext {
  const MatrixInterface *A;
  const VectorFactory *vf;
  mutable std::unique_ptr<VectorInterface> x_vec;
  mutable std::unique_ptr<VectorInterface> y_vec;
  mutable std::unique_ptr<VectorInterface> m_vec;

  SlepcMatContext(const MatrixInterface *A, const VectorFactory *vf)
      : A(A), vf(vf) {
    x_vec = vf->create_zero_of_size(A->cols());
    y_vec = vf->create_zero_of_size(A->cols());
    m_vec = vf->create_zero_of_size(A->rows());
  }
};

static PetscErrorCode MyMatMult(Mat A_shell, Vec x, Vec y) {
  SlepcMatContext *ctx;
  MatShellGetContext(A_shell, &ctx);

  const double *x_ptr;
  double *y_ptr;
  VecGetArrayRead(x, &x_ptr);
  VecGetArray(y, &y_ptr);

  // x is n-dimensional (A cols)
  // y is m-dimensional (A rows)
  std::memcpy(ctx->x_vec->data(), x_ptr, sizeof(double) * ctx->A->cols());
  ctx->A->multiply(*ctx->x_vec, *ctx->m_vec);
  std::memcpy(y_ptr, ctx->m_vec->data(), sizeof(double) * ctx->A->rows());

  VecRestoreArrayRead(x, &x_ptr);
  VecRestoreArray(y, &y_ptr);
  return 0;
}

static PetscErrorCode MyMatMultTranspose(Mat A_shell, Vec x, Vec y) {
  SlepcMatContext *ctx;
  MatShellGetContext(A_shell, &ctx);

  const double *x_ptr;
  double *y_ptr;
  VecGetArrayRead(x, &x_ptr);
  VecGetArray(y, &y_ptr);

  // x is m-dimensional (A rows)
  // y is n-dimensional (A cols)
  std::memcpy(ctx->m_vec->data(), x_ptr, sizeof(double) * ctx->A->rows());
  ctx->A->multiply_transpose(*ctx->m_vec, *ctx->y_vec);
  std::memcpy(y_ptr, ctx->y_vec->data(), sizeof(double) * ctx->A->cols());

  VecRestoreArrayRead(x, &x_ptr);
  VecRestoreArray(y, &y_ptr);
  return 0;
}

std::vector<std::unique_ptr<VectorInterface>>
find_nullspace_slepc(const MatrixInterface &A, const VectorFactory &vf,
                      std::size_t k, double tol) {
  static SlepcManager manager; // Initialize once

  SlepcMatContext ctx(&A, &vf);
  Mat A_shell;
  MatCreateShell(PETSC_COMM_WORLD, A.rows(), A.cols(), A.rows(), A.cols(), &ctx,
                 &A_shell);
  MatShellSetOperation(A_shell, MATOP_MULT, (void (*)())MyMatMult);
  MatShellSetOperation(A_shell, MATOP_MULT_TRANSPOSE,
                       (void (*)())MyMatMultTranspose);

  SVD svd;
  SVDCreate(PETSC_COMM_WORLD, &svd);
  SVDSetOperators(svd, A_shell, NULL);
  SVDSetProblemType(svd, SVD_STANDARD);
  SVDSetWhichSingularTriplets(svd, SVD_SMALLEST);
  SVDSetType(svd, SVDTRLANCZOS); 
  
  // Set dimensions: request more than k to ensure convergence on smallest
  PetscInt ncv = std::max<PetscInt>(120, (PetscInt)k * 8);
  if (ncv > (PetscInt)A.cols()) ncv = (PetscInt)A.cols();
  SVDSetDimensions(svd, static_cast<PetscInt>(k + 15), ncv, PETSC_DEFAULT);
  
  SVDSetTolerances(svd, std::max(1e-9, tol), 5000);
  SVDSetFromOptions(svd);

  SVDSolve(svd);

  PetscInt nconv;
  SVDGetConverged(svd, &nconv);

  SVDConvergedReason reason;
  SVDGetConvergedReason(svd, &reason);
  if (reason < 0) {
      std::cerr << "Warning: SLEPc SVD did not converge. Reason: " << reason << std::endl;
  }

  std::vector<std::unique_ptr<VectorInterface>> result;
  Vec u, v;
  MatCreateVecs(A_shell, &v, &u); // v is right vector (n), u is left vector (m)

  for (int i = 0; i < nconv; ++i) {
    PetscReal sigma;
    SVDGetSingularTriplet(svd, i, &sigma, u, v);

    if (sigma < tol) { 
      const double *v_ptr;
      VecGetArrayRead(v, &v_ptr);
      auto vec = vf.create_zero_of_size(A.cols());

      if (!vec->data()) {
          throw std::runtime_error("SLEPc wrapper requires contiguous vector memory.");
      }

      std::memcpy(vec->data(), v_ptr, sizeof(double) * A.cols());
      VecRestoreArrayRead(v, &v_ptr);
      result.push_back(std::move(vec));
    }
    if (result.size() >= k)
      break;
  }
  VecDestroy(&u);
  VecDestroy(&v);

  SVDDestroy(&svd);
  MatDestroy(&A_shell);

  return result;
}

} // namespace krans::solvers

#else // KRANS_USE_SLEPC

#include "krans/Solvers/SlepcSolver.hpp"
#include <stdexcept>

namespace krans::solvers {

std::vector<std::unique_ptr<VectorInterface>>
find_nullspace_slepc(const MatrixInterface&, const VectorFactory&, std::size_t, double) {
    throw std::runtime_error("find_nullspace_slepc: KraNS was built without SLEPc support.");
}

} // namespace krans::solvers

#endif // KRANS_USE_SLEPC
