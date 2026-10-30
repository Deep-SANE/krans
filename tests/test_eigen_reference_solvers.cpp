#include "sns/Solvers/EigenCODSolver.hpp"
#include "sns/Solvers/EigenFullPivLUSolver.hpp"
#include "sns/Solvers/EigenQRCPSolver.hpp"
#include "sns/Solvers/EigenSVDSolver.hpp"

#include "sns/adapters/eigen/EigenDenseMatrixAdapter.hpp"
#include "sns/adapters/eigen/EigenSparseMatrixAdapter.hpp"
#include "sns/adapters/eigen/EigenVectorAdapter.hpp"

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <cassert>
#include <cmath>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <vector>

using namespace sns;

static const Eigen::VectorXd &view(const VectorInterface &v) {
  return dynamic_cast<const EigenVectorAdapter &>(v).view();
}

static double
max_offdiag_dot(const std::vector<std::unique_ptr<VectorInterface>> &basis) {
  double m = 0.0;
  for (std::size_t i = 0; i < basis.size(); ++i)
    for (std::size_t j = i + 1; j < basis.size(); ++j)
      m = std::max(m, std::fabs(basis[i]->dot(*basis[j])));
  return m;
}

struct SolverEntry {
  std::string name;
  std::vector<std::unique_ptr<VectorInterface>> (*fn)(const MatrixInterface &,
                                                      const VectorFactory &,
                                                      double /*tol_rel*/);
};

struct DummyFactory : VectorFactory {
  std::unique_ptr<VectorInterface> create_zero() const override {
    throw std::logic_error("unused");
  }
  std::unique_ptr<VectorInterface> create_random() const override {
    throw std::logic_error("unused");
  }
  std::unique_ptr<VectorInterface>
  create_zero_of_size(std::size_t) const override {
    throw std::logic_error("unused");
  }
  std::unique_ptr<VectorInterface>
  create_random_of_size(std::size_t) const override {
    throw std::logic_error("unused");
  }
};

// same scale we use in solvers: cheap and consistent
static double matrix_rank_scale(const Eigen::MatrixXd &M) {
  const double s = M.cwiseAbs().maxCoeff();
  return s > 0.0 ? s : 1.0;
}

static void run_and_check(const std::string &label, const MatrixInterface &A,
                          int expected_nullity,
                          const std::vector<SolverEntry> &solvers,
                          double tol_rel, double res_tol = 1e-10,
                          double ortho_tol = 1e-10) {
  std::cout << "---- " << label << " ----\n";
  DummyFactory dummy;

  // Build a *dense* copy to compute the same scale as the solvers
  Eigen::MatrixXd M;
  if (const auto *ed = dynamic_cast<const EigenDenseMatrixAdapter *>(&A)) {
    M = ed->view();
  } else {
    const int m = static_cast<int>(A.rows());
    const int n = static_cast<int>(A.cols());
    M.resize(m, n);
    EigenVectorAdapter x(n), y(m);
    for (int j = 0; j < n; ++j) {
      x.view().setZero();
      x.view()(j) = 1.0;
      A.multiply(x, y);
      M.col(j) = y.view();
    }
  }
  const double scale = matrix_rank_scale(M);

  for (const auto &s : solvers) {
    auto basis = s.fn(A, dummy, tol_rel);

    std::cout << "  [" << s.name << "] nullity=" << basis.size() << "\n";
    assert(static_cast<int>(basis.size()) == expected_nullity);

    // Residuals and unit norms
    EigenVectorAdapter y(A.rows());
    for (auto &v : basis) {
      const double nv = v->norm();
      assert(std::fabs(nv - 1.0) <= 1e-12);
      A.multiply(*v, y);
      const double r = y.norm();

      // Allowance proportional to both an absolute floor (res_tol*scale)
      // and a looser cushion driven by tol_rel*scale
      const double allowed = std::max(res_tol * scale, 10.0 * tol_rel * scale);
      assert(r <= allowed);
    }

    // Orthonormality among returned basis vectors (if >1)
    if (basis.size() > 1) {
      const double mmax = max_offdiag_dot(basis);
      assert(mmax <= ortho_tol);
    }
  }
}

int main() {
  std::vector<SolverEntry> solvers = {
      {"fullpivlu", &solvers::find_nullspace_fullpivlu},
      {"svd", &solvers::find_nullspace_svd},
      {"cod", &solvers::find_nullspace_cod},
      {"qrcp", &solvers::find_nullspace_qrcp},
  };

  // ---------------------------------------------------------------------------
  // Case A: 2x3 with known 1D nullspace
  // ---------------------------------------------------------------------------
  {
    Eigen::MatrixXd A_e(2, 3);
    A_e << 1, 2, 0, 0, 1, 1;
    EigenDenseMatrixAdapter A(A_e);
    run_and_check("A: 2x3 (nullity 1)", A,
                  /*expected_nullity=*/1, solvers,
                  /*tol_rel=*/1e-12,
                  /*res_tol=*/1e-12, /*ortho_tol=*/1e-10);
  }

  // ---------------------------------------------------------------------------
  // Case B: Full column-rank 2x2 => nullity = 0
  // (tol_rel=0 falls back to eps internally)
  // ---------------------------------------------------------------------------
  {
    Eigen::MatrixXd A_e(2, 2);
    A_e << 3, 0, 0, 2;
    EigenDenseMatrixAdapter A(A_e);
    run_and_check("B: 2x2 full rank (nullity 0)", A, 0, solvers,
                  /*tol_rel=*/0.0,
                  /*res_tol=*/1e-12, /*ortho_tol=*/1e-10);
  }

  // ---------------------------------------------------------------------------
  // Case C: Rank-1 1x3 => nullity = 2
  // ---------------------------------------------------------------------------
  {
    Eigen::MatrixXd A_e(1, 3);
    A_e << 1, 1, 1;
    EigenDenseMatrixAdapter A(A_e);
    run_and_check("C: 1x3 rank 1 (nullity 2)", A, 2, solvers,
                  /*tol_rel=*/1e-12,
                  /*res_tol=*/1e-12, /*ortho_tol=*/1e-10);
  }

  // ---------------------------------------------------------------------------
  // Case D: Zero matrix => nullity = n
  // ---------------------------------------------------------------------------
  {
    const int m = 2, n = 3;
    Eigen::MatrixXd A_e = Eigen::MatrixXd::Zero(m, n);
    EigenDenseMatrixAdapter A(A_e);
    run_and_check("D: zero matrix (nullity n)", A, n, solvers,
                  /*tol_rel=*/1e-12,
                  /*res_tol=*/1e-15, /*ortho_tol=*/1e-10);
  }

  // ---------------------------------------------------------------------------
  // Case E: Sparse adapter path for the 2x3 example (nullity = 1)
  // ---------------------------------------------------------------------------
  {
    Eigen::SparseMatrix<double> S(2, 3);
    S.insert(0, 0) = 1.0;
    S.insert(0, 1) = 2.0;
    S.insert(1, 1) = 1.0;
    S.insert(1, 2) = 1.0;
    S.makeCompressed();

    EigenSparseMatrixAdapter A(S);
    run_and_check("E: sparse adapter 2x3 (nullity 1)", A, 1, solvers,
                  /*tol_rel=*/1e-12,
                  /*res_tol=*/1e-12, /*ortho_tol=*/1e-10);
  }

  // ---------------------------------------------------------------------------
  // Case F: Tolerance sensitivity on nearly rank-deficient 2x2
  // ---------------------------------------------------------------------------
  {
    const double eps = 1e-10;
    Eigen::MatrixXd A_e(2, 2);
    A_e << 1, 1, 1, 1 + eps;
    EigenDenseMatrixAdapter A(A_e);

    // Tight tol -> full rank
    run_and_check("F1: near-RD tight tol (nullity 0)", A, 0, solvers,
                  /*tol_rel=*/1e-12,
                  /*res_tol=*/1e-10, /*ortho_tol=*/1e-10);

    // Looser tol -> rank-1
    run_and_check("F2: near-RD loose tol (nullity 1)", A, 1, solvers,
                  /*tol_rel=*/1e-6,
                  /*res_tol=*/1e-6, /*ortho_tol=*/1e-10);
  }

  // ---------------------------------------------------------------------------
  // Case G: Tall - 5x3 full column rank => nullity 0
  // ---------------------------------------------------------------------------
  {
    Eigen::MatrixXd A_e = Eigen::MatrixXd::Random(5, 3);
    A_e.col(2) += 0.3 * A_e.col(1); // keep likely full col rank
    EigenDenseMatrixAdapter A(A_e);
    run_and_check("Tall 5x3 (nullity 0)", A, 0, solvers, 1e-12);
  }

  // ---------------------------------------------------------------------------
  // Case H: Wide - 3x5 with constructed 2D nullspace
  // ---------------------------------------------------------------------------
  {
    Eigen::MatrixXd A_e(3, 5);
    A_e << 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0; // rank 2 -> nullity 3
    EigenDenseMatrixAdapter A(A_e);
    run_and_check("Wide 3x5 (nullity 3)", A, 3, solvers, 1e-12);
  }

  // ---------------------------------------------------------------------------
  // Case I: Start full rank 3x3, make two tiny singular values
  // ---------------------------------------------------------------------------
  {
    Eigen::MatrixXd U =
        Eigen::HouseholderQR<Eigen::MatrixXd>(Eigen::MatrixXd::Random(3, 3))
            .householderQ();
    Eigen::MatrixXd V =
        Eigen::HouseholderQR<Eigen::MatrixXd>(Eigen::MatrixXd::Random(3, 3))
            .householderQ();
    Eigen::Vector3d s(1.0, 1e-9, 1e-9);
    Eigen::MatrixXd A_e = U * s.asDiagonal() * V.transpose();
    EigenDenseMatrixAdapter A(A_e);

    run_and_check("Two tiny singulars (tight tol -> nullity 0)", A, 0, solvers,
                  1e-12, 1e-10, 1e-10);
    run_and_check("Two tiny singulars (loose tol -> nullity 2)", A, 2, solvers,
                  1e-6, 1e-6, 1e-10);
  }

  // ---------------------------------------------------------------------------
  // Case G: Large matrix with nullspace = 3
  // ---------------------------------------------------------------------------
  {
    const int m = 1200, n = 600; // tall: m >= n
    Eigen::MatrixXd A_e = Eigen::MatrixXd::Random(m, n);
    A_e.rightCols(3).setZero(); // kills 3 columns → rank ≈ n-3
    EigenDenseMatrixAdapter A(A_e);
    run_and_check("Large BDCSVD path (nullity 3)", A, 3, solvers, 1e-10, 1e-8,
                  1e-8);
  }

  std::cout << "test_eigen_reference_solvers OK\n";
  return 0;
}
