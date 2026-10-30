#include "sns/IO/MatrixLoader.hpp"
#include "sns/Solvers/CyclicLanczosSolver.hpp"
#include "sns/Utils/GramSchmidt.hpp"
#include "sns/Utils/Projection.hpp"
#include "sns/adapters/eigen/EigenSparseMatrixAdapter.hpp"
#include "sns/adapters/eigen/EigenVectorFactory.hpp"
#include <Eigen/Sparse>

#include <cassert>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

using namespace sns;
using Clock = std::chrono::steady_clock;

// A lightweight wrapper that counts multiplies while delegating to an existing
// MatrixInterface.
class CountingMatrixAdapter final : public MatrixInterface {
public:
  explicit CountingMatrixAdapter(const MatrixInterface &underlying)
      : U_(underlying) {}

  std::size_t rows() const override { return U_.rows(); }
  std::size_t cols() const override { return U_.cols(); }

  void multiply(const VectorInterface &x, VectorInterface &y) const override {
    ++spmv_; // A * x
    U_.multiply(x, y);
  }

  void multiply_transpose(const VectorInterface &x,
                          VectorInterface &y) const override {
    ++spmv_t_; // A^T * x
    U_.multiply_transpose(x, y);
  }

  std::size_t spmv() const { return spmv_; }    // # of A*x calls
  std::size_t spmvT() const { return spmv_t_; } // # of A^T*x calls
  std::size_t totalSpMV() const { return spmv_ + spmv_t_; }

private:
  const MatrixInterface &U_;
  mutable std::size_t spmv_ = 0;
  mutable std::size_t spmv_t_ = 0;
};

// helpers
static void
check_unit_norms(const std::vector<std::unique_ptr<VectorInterface>> &basis,
                 double tol = 1e-12) {
  for (const auto &q : basis) {
    const double n = q->norm();
    assert(std::fabs(n - 1.0) <= tol);
  }
}

static void
project_onto_span(const std::vector<std::unique_ptr<VectorInterface>> &basis,
                  VectorInterface &dst, const VectorInterface &v_src) {
  dst.set_zero();
  for (const auto &q : basis) {
    const double c = v_src.dot(*q);
    dst.add_scaled(*q, c);
  }
}

static const char *reorth_name(sns::solvers::ReorthMode m) {
  switch (m) {
  case sns::solvers::ReorthMode::None:
    return "none";
  case sns::solvers::ReorthMode::Single:
    return "single";
  case sns::solvers::ReorthMode::Double:
    return "double";
  }
  return "?";
}

static void print_header(const std::string &name, std::size_t m, std::size_t n,
                         std::size_t nnz,
                         const sns::solvers::CyclicLanczosOptions &opt) {
  std::cout << "---- " << name
            << " ---------------------------------------------\n";
  std::cout << "dims: " << m << " x " << n << "   nnz=" << nnz << "\n";
  std::cout << "opts: k=" << opt.k << "  ncv=" << opt.ncv
            << "  max_restarts=" << opt.max_restarts
            << "  tol_abs=" << std::scientific << opt.tol_abs
            << "  tol_rel=" << opt.tol_rel << "  cg_maxit=" << std::defaultfloat
            << opt.cg_maxit << "  cg_rel_tol=" << std::scientific
            << opt.cg_rel_tol << "  harmonic=" << std::boolalpha
            << opt.use_harmonic << "  reorth=" << reorth_name(opt.reorth)
            << "  projector_passes=" << std::noboolalpha << opt.projector_passes
            << "\n";
}

static void print_perf(const CountingMatrixAdapter &C, double seconds) {
  const std::size_t spmv = C.spmv();
  const std::size_t spmvT = C.spmvT();
  const std::size_t tot = C.totalSpMV();
  const double t_per_spmv = (tot > 0) ? seconds / double(tot) : 0.0;

  std::cout << std::fixed << std::setprecision(6);
  std::cout << "perf: time=" << seconds << " s"
            << "  SpMV=" << spmv << "  SpMV^T=" << spmvT << "  total=" << tot
            << "  time/SpMV=" << t_per_spmv << " s\n";
}

int main() {
  using S = Eigen::SparseMatrix<double, Eigen::ColMajor, int>;

  // ==================== Test 1: small analytic nullspace (2x3) ===============
  {
    S A(2, 3);
    std::vector<Eigen::Triplet<double>> T = {
        {0, 0, 1.0}, {0, 2, 1.0}, {1, 1, 1.0}, {1, 2, 1.0}};
    A.setFromTriplets(T.begin(), T.end());
    A.makeCompressed();

    EigenSparseMatrixAdapter M(A);
    CountingMatrixAdapter C(M);
    EigenVectorFactory F(static_cast<std::size_t>(A.cols()));

    sns::solvers::CyclicLanczosOptions opt;
    opt.k = 1;
    opt.ncv = 20;
    opt.max_restarts = 50;
    opt.tol_abs = 1e-12;
    opt.tol_rel = 1e-10;
    opt.verbose = true;
    opt.use_harmonic = true;
    opt.reorth = sns::solvers::ReorthMode::Double;
    opt.nullity_known = 1;
    opt.rank_known = 2;
    opt.cg_maxit = 40;
    opt.cg_rel_tol = 1e-10;

    print_header("Small (2x3)", (std::size_t)A.rows(), (std::size_t)A.cols(),
                 (std::size_t)A.nonZeros(), opt);

    const auto t0 = Clock::now();
    auto basis = sns::solvers::find_nullspace_cyclic_lanczos(C, F, opt);
    const auto t1 = Clock::now();
    const double sec = std::chrono::duration<double>(t1 - t0).count();
    print_perf(C, sec);

    assert(!basis.empty());
    assert(basis.size() == 1);
    check_unit_norms(basis, 1e-12);

    auto Av = F.create_zero_of_size((std::size_t)A.rows());
    C.multiply(*basis[0], *Av);
    const double res = Av->norm();
    std::cout << "[Small] ||A v|| = " << std::scientific << res << "\n";
    assert(res <= 1e-10);

    std::cout << "test_cyclic_lanczos (small) OK\n\n";
  }

  // ==================== Test 2: IJA mat1.ija (rank=100, nullity=5) ==========
  {
    try {
      S A1 = sns::io::load_ija_triplet_matrix("data/mat1.ija", -1, -1);
      A1.makeCompressed();

      EigenSparseMatrixAdapter M1(A1);
      CountingMatrixAdapter C1(M1);
      EigenVectorFactory F1((std::size_t)A1.cols());

      sns::solvers::CyclicLanczosOptions opt1;
      opt1.k = 5;
      opt1.ncv = std::max<std::size_t>(opt1.k + 40, 60);
      opt1.max_restarts = 200;
      opt1.tol_abs = 1e-12;
      opt1.tol_rel = 1e-10;
      opt1.verbose = true;
      opt1.use_harmonic = true;
      opt1.reorth = sns::solvers::ReorthMode::Double;
      opt1.nullity_known = 5;
      opt1.rank_known = 100;
      opt1.cg_maxit = 60;
      opt1.cg_rel_tol = 1e-8;

      opt1.initial_right_space.push_back(
          F1.create_random_of_size((std::size_t)A1.cols()));
      opt1.initial_right_space.push_back(
          F1.create_random_of_size((std::size_t)A1.cols()));
      opt1.precond_AAt = [&](const VectorInterface &r_m, VectorInterface &y_m) {
        y_m.copy_from(r_m);
      };

      print_header("mat1.ija", (std::size_t)A1.rows(), (std::size_t)A1.cols(),
                   (std::size_t)A1.nonZeros(), opt1);

      const auto t0 = Clock::now();
      auto basis1 = sns::solvers::find_nullspace_cyclic_lanczos(C1, F1, opt1);
      const auto t1 = Clock::now();
      const double sec = std::chrono::duration<double>(t1 - t0).count();
      print_perf(C1, sec);

      std::cout << "[mat1] returned " << basis1.size() << " vectors\n";
      assert(basis1.size() == 5);
      check_unit_norms(basis1, 1e-10);

      auto tmp_m1 = F1.create_zero_of_size((std::size_t)A1.rows());
      double worst = 0.0;
      for (std::size_t i = 0; i < basis1.size(); ++i) {
        C1.multiply(*basis1[i], *tmp_m1);
        const double ri = tmp_m1->norm();
        worst = std::max(worst, ri);
        std::cout << "[mat1] ||A v_" << i << "|| = " << std::scientific << ri
                  << "\n";
        assert(ri <= 1e-8);
      }
      const double ortho1 = sns::utils::max_pairwise_abs_dot(basis1);
      std::cout << "[mat1] max pairwise |dot| = " << std::scientific << ortho1
                << "\n";
      assert(ortho1 <= 1e-8);

      auto v_rand1 = F1.create_random_of_size((std::size_t)A1.cols());
      auto v_proj1 = F1.create_zero_of_size((std::size_t)A1.cols());
      project_onto_span(basis1, *v_proj1, *v_rand1);
      C1.multiply(*v_proj1, *tmp_m1);
      const double r_proj1 = tmp_m1->norm();
      std::cout << "[mat1] ||A (Proj_N v_rand)|| = " << std::scientific
                << r_proj1 << "\n";
      assert(r_proj1 <= 1e-8);

      std::cout << "test_cyclic_lanczos (mat1) OK\n\n";
    } catch (const std::exception &e) {
      std::cerr << "[mat1] Skipping mat1 test: " << e.what() << "\n\n";
    }
  }

  // ==================== Test 3: identity (nullity=0) ========================
  {
    S I(3, 3);
    I.setIdentity();

    EigenSparseMatrixAdapter MI(I);
    CountingMatrixAdapter CI(MI);
    EigenVectorFactory F(3);

    sns::solvers::CyclicLanczosOptions opt;
    opt.k = 1;
    opt.ncv = 20;
    opt.max_restarts = 30;
    opt.tol_abs = 1e-12;
    opt.tol_rel = 1e-10;
    opt.verbose = false;
    opt.nullity_known = 0; // no right nullspace
    opt.rank_known = 3;
    opt.reorth = sns::solvers::ReorthMode::Double;

    print_header("Identity (3x3)", (std::size_t)I.rows(), (std::size_t)I.cols(),
                 (std::size_t)I.nonZeros(), opt);

    const auto t0 = Clock::now();
    auto basis = sns::solvers::find_nullspace_cyclic_lanczos(CI, F, opt);
    const auto t1 = Clock::now();
    const double sec = std::chrono::duration<double>(t1 - t0).count();
    print_perf(CI, sec);

    std::cout << "[Identity] returned " << basis.size() << " vectors\n";
    assert(basis.empty());
    std::cout << "test_cyclic_lanczos (identity) OK\n\n";
  }

  // ==================== Test 4: IJA mat2.ija (rank=5425, nullity=15) ========
  // nnz - 27,440. Stronger settings + a tiny SD preconditioner for A A^T.
  {
    try {
      S A2 = sns::io::load_ija_triplet_matrix("data/mat2.ija", -1, -1);
      A2.makeCompressed();

      EigenSparseMatrixAdapter M2(A2);
      CountingMatrixAdapter C2(M2);
      EigenVectorFactory F2((std::size_t)A2.cols());

      const int rank_known = 5425;
      const int nullity_known = 15;

      sns::solvers::CyclicLanczosOptions opt2;
      opt2.k = nullity_known;
      opt2.ncv = std::max<std::size_t>(nullity_known + 80, 128);
      opt2.max_restarts = 400;
      opt2.projector_passes = 2;
      opt2.cg_maxit = 200;
      opt2.cg_rel_tol = 1e-10;
      opt2.prescreen_factor = 4.0;
      opt2.stagnation_restarts = 0;
      opt2.tol_abs = 1e-12;
      opt2.tol_rel = 1e-10;
      opt2.use_harmonic = true;
      opt2.reorth = sns::solvers::ReorthMode::Double;
      opt2.nullity_known = nullity_known;
      opt2.rank_known = rank_known;
      opt2.verbose = true;

      // Two random seeds
      opt2.initial_right_space.push_back(
          F2.create_random_of_size((std::size_t)A2.cols()));
      opt2.initial_right_space.push_back(
          F2.create_random_of_size((std::size_t)A2.cols()));

      // ---- Simple SD preconditioner for (A A^T) y = r (k small iterations)
      // y_0 = 0; r_0 = r; for t=0..k-1:
      //   Ap = A A^T p; alpha = (p*p) / (p*Ap); y += alpha p; r -= alpha Ap;
      //   p=r
      // This is matrix-free and uses only VectorInterface ops.
      const int k_sd = 8; // small, fixed
      opt2.precond_AAt = [&](const VectorInterface &r_m, VectorInterface &y_m) {
        const std::size_t m = (std::size_t)A2.rows();
        const std::size_t n = (std::size_t)A2.cols();
        auto p = F2.create_zero_of_size(m);
        auto Ap = F2.create_zero_of_size(m);
        auto tmpn = F2.create_zero_of_size(n);
        auto res = F2.create_zero_of_size(m);

        y_m.set_zero();
        res->copy_from(r_m);
        p->copy_from(*res);

        for (int it = 0; it < k_sd; ++it) {
          // Ap = A A^T p
          C2.multiply_transpose(*p, *tmpn);
          C2.multiply(*tmpn, *Ap);

          const double pAp = p->dot(*Ap);
          if (std::fabs(pAp) < 1e-30)
            break;

          const double p2 = p->dot(*p);
          const double alpha = p2 / pAp;

          y_m.add_scaled(*p, alpha);
          res->add_scaled(*Ap, -alpha);

          const double rnorm2 = res->dot(*res);
          if (rnorm2 <= 1e-30)
            break;

          p->copy_from(*res);
        }
      };

      print_header("mat2.ija", (std::size_t)A2.rows(), (std::size_t)A2.cols(),
                   (std::size_t)A2.nonZeros(), opt2);

      const auto t0 = Clock::now();
      auto R = sns::solvers::find_nullspace_cyclic_lanczos_ex(C2, F2, opt2);
      const auto t1 = Clock::now();
      const double sec = std::chrono::duration<double>(t1 - t0).count();
      print_perf(C2, sec);

      std::cout << "[mat2] returned " << R.right_null.size()
                << " vectors; restarts=" << R.restarts
                << "  krylov_steps_total=" << R.krylov_steps_total
                << "  ||A||_F~" << std::scientific << R.frob_norm_est << "\n";

      assert(R.right_null.size() == (std::size_t)nullity_known);
      check_unit_norms(R.right_null, 1e-12);

      auto tmp_m2 = F2.create_zero_of_size((std::size_t)A2.rows());
      double worst = 0.0;
      for (std::size_t i = 0; i < R.right_null.size(); ++i) {
        C2.multiply(*R.right_null[i], *tmp_m2);
        const double ri = tmp_m2->norm();
        worst = std::max(worst, ri);
        std::cout << "[mat2] ||A v_" << i << "|| = " << std::scientific << ri
                  << "\n";
        assert(ri <= 1e-8);
      }
      const double ortho2 = sns::utils::max_pairwise_abs_dot(R.right_null);
      std::cout << "[mat2] max pairwise |dot| = " << std::scientific << ortho2
                << "\n";
      assert(ortho2 <= 1e-8);

      auto v_rand2 = F2.create_random_of_size((std::size_t)A2.cols());
      auto v_proj2 = F2.create_zero_of_size((std::size_t)A2.cols());
      project_onto_span(R.right_null, *v_proj2, *v_rand2);
      C2.multiply(*v_proj2, *tmp_m2);
      const double r_proj2 = tmp_m2->norm();
      std::cout << "[mat2] ||A (Proj_N v_rand)|| = " << std::scientific
                << r_proj2 << "\n";
      assert(r_proj2 <= 1e-8);

      std::cout << "test_cyclic_lanczos (mat2) OK\n\n";
    } catch (const std::exception &e) {
      std::cerr << "[mat2] Skipping mat2 test: " << e.what() << "\n\n";
    }
  }

  // ==================== Test 5: 2x2 rank-1 (both left & right null) =========
  {
    // A = [[1,1],[1,1]] has rank 1.
    // Right null = span([-1,1]); Left null = span([1,-1]).
    S A(2, 2);
    std::vector<Eigen::Triplet<double>> T = {
        {0, 0, 1.0}, {0, 1, 1.0}, {1, 0, 1.0}, {1, 1, 1.0}};
    A.setFromTriplets(T.begin(), T.end());
    A.makeCompressed();

    EigenSparseMatrixAdapter M(A);
    CountingMatrixAdapter C(M);
    EigenVectorFactory F(2);

    sns::solvers::CyclicLanczosOptions opt;
    opt.k = 1;
    opt.ncv = 20;
    opt.max_restarts = 50;
    opt.tol_abs = 1e-12;
    opt.tol_rel = 1e-10;
    opt.use_harmonic = true;
    opt.reorth = sns::solvers::ReorthMode::Double;
    opt.nullity_known = 1;
    opt.rank_known = 1;
    opt.return_left_space = true;
    opt.verbose = true;

    print_header("2x2 rank-1 (both nullspaces)", (std::size_t)A.rows(),
                 (std::size_t)A.cols(), (std::size_t)A.nonZeros(), opt);

    const auto t0 = Clock::now();
    auto R = sns::solvers::find_nullspace_cyclic_lanczos_ex(C, F, opt);
    const auto t1 = Clock::now();
    const double sec = std::chrono::duration<double>(t1 - t0).count();
    print_perf(C, sec);

    assert(R.right_null.size() == 1);
    check_unit_norms(R.right_null, 1e-12);

    // Check right: ||A v|| small
    auto Av = F.create_zero_of_size(2);
    C.multiply(*R.right_null[0], *Av);
    const double res_r = Av->norm();
    std::cout << "[2x2] right ||A v|| = " << std::scientific << res_r << "\n";
    assert(res_r <= 1e-10);

    // Check left if returned: ||A^T u|| small
    if (!R.left_null.empty()) {
      check_unit_norms(R.left_null, 1e-12);
      auto Atu = F.create_zero_of_size(2);
      C.multiply_transpose(*R.left_null[0], *Atu);
      const double res_l = Atu->norm();
      std::cout << "[2x2] left ||A^T u|| = " << std::scientific << res_l
                << "\n";
      assert(res_l <= 1e-10);
    }

    std::cout << "test_cyclic_lanczos (2x2 both nullspaces) OK\n\n";
  }

  // ==================== Test 6: scale invariance =============================
  {
    // Same 2x3 from Test 1, but scaled by a large factor s.
    const double s = 1e6;
    S A(2, 3);
    std::vector<Eigen::Triplet<double>> T = {
        {0, 0, s * 1.0}, {0, 2, s * 1.0}, {1, 1, s * 1.0}, {1, 2, s * 1.0}};
    A.setFromTriplets(T.begin(), T.end());
    A.makeCompressed();

    EigenSparseMatrixAdapter M(A);
    CountingMatrixAdapter C(M);
    EigenVectorFactory F(static_cast<std::size_t>(A.cols()));

    sns::solvers::CyclicLanczosOptions opt;
    opt.k = 1;
    opt.ncv = 20;
    opt.max_restarts = 50;
    opt.tol_abs = 1e-12;
    opt.tol_rel = 1e-10;
    opt.verbose = false;
    opt.use_harmonic = true;
    opt.reorth = sns::solvers::ReorthMode::Double;
    opt.nullity_known = 1;
    opt.rank_known = 2;
    opt.cg_maxit = 40;
    opt.cg_rel_tol = 1e-10;

    print_header("Scale invariance (2x3 scaled by 1e6)", (std::size_t)A.rows(),
                 (std::size_t)A.cols(), (std::size_t)A.nonZeros(), opt);

    const auto t0 = Clock::now();
    auto R = sns::solvers::find_nullspace_cyclic_lanczos_ex(C, F, opt);
    const auto t1 = Clock::now();
    const double sec = std::chrono::duration<double>(t1 - t0).count();
    print_perf(C, sec);

    assert(!R.right_null.empty());
    assert(R.right_null.size() == 1);
    check_unit_norms(R.right_null, 1e-12);

    auto Av = F.create_zero_of_size((std::size_t)A.rows());
    C.multiply(*R.right_null[0], *Av);
    const double res = Av->norm();
    const double fn = std::max(1.0, R.frob_norm_est);
    const double rel = res / fn;
    std::cout << "[Scale] ||A v||=" << std::scientific << res
              << "   ||A||_F~=" << R.frob_norm_est << "   rel=" << rel << "\n";
    // Relative residual should be ~ tol_rel level (allow some slack)
    assert(rel <= 1e-8);

    std::cout << "test_cyclic_lanczos (scale invariance) OK\n\n";
  }

  std::cout << "ALL TESTS OK\n";
  return 0;
}
