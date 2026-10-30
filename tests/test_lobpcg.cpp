// test_lobpcg.cpp
#include "sns/IO/MatrixLoader.hpp"
#include "sns/Solvers/LOBPCGSolver.hpp"
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

// --------- helpers ---------
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

static void print_header(const std::string &name, std::size_t m, std::size_t n,
                         std::size_t nnz,
                         const sns::solvers::LOBPCGOptions &o) {
  std::cout << "---- " << name
            << " ---------------------------------------------\n";
  std::cout << "dims: " << m << " x " << n << "   nnz=" << nnz << "\n";
  std::cout << "opts: nev=" << o.nev << "  bs=" << o.block_size
            << "  max_iters=" << o.max_iters << "  restart=" << o.restart
            << "  soft_locking=" << std::boolalpha << o.soft_locking
            << "  tol_abs=" << std::scientific << o.tol_abs
            << "  tol_rel=" << o.tol_rel << "  gs_reorth=" << std::boolalpha
            << o.gs_reorth << std::noboolalpha << "\n";
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

    sns::solvers::LOBPCGOptions opt;
    opt.nev = 1;
    opt.block_size = 4;
    opt.max_iters = 200;
    opt.tol_abs = 1e-12;
    opt.tol_rel = 1e-10;
    opt.soft_locking = true;
    opt.restart = 0.9;
    opt.gs_reorth = true;
    opt.nullity_known = 1;
    opt.rank_known = 2;
    opt.verbose = true;

    print_header("LOBPCG Small (2x3)", (std::size_t)A.rows(),
                 (std::size_t)A.cols(), (std::size_t)A.nonZeros(), opt);

    const auto t0 = Clock::now();
    auto R = sns::solvers::find_nullspace_lobpcg_ex(C, F, opt);
    const auto t1 = Clock::now();
    const double sec = std::chrono::duration<double>(t1 - t0).count();
    print_perf(C, sec);

    assert(!R.right_null.empty());
    assert(R.right_null.size() == 1);
    check_unit_norms(R.right_null, 1e-12);

    auto Av = F.create_zero_of_size((std::size_t)A.rows());
    C.multiply(*R.right_null[0], *Av);
    const double res = Av->norm();
    std::cout << "[Small] ||A v|| = " << std::scientific << res << "\n";
    assert(res <= 1e-10);

    std::cout << "test_lobpcg (small) OK\n\n";
  }

  // ==================== Test 2: IJA mat1.ija (rank=100, nullity=5) ==========
  {
    try {
      S A1 = sns::io::load_ija_triplet_matrix("data/mat1.ija", -1, -1);
      A1.makeCompressed();

      EigenSparseMatrixAdapter M1(A1);
      CountingMatrixAdapter C1(M1);
      EigenVectorFactory F1((std::size_t)A1.cols());

      sns::solvers::LOBPCGOptions opt1;
      opt1.nev = 5;
      opt1.block_size = 12; // bs >= nev; small 3k subspace → fast tiny GEPs
      opt1.max_iters = 500;
      opt1.tol_abs = 1e-12;
      opt1.tol_rel = 1e-10;
      opt1.soft_locking = true;
      opt1.restart = 0.9;
      opt1.gs_reorth = true;
      opt1.nullity_known = 5;
      opt1.rank_known = 100;
      opt1.verbose = true;

      // a couple of random initial guesses
      opt1.initial_block.push_back(
          F1.create_random_of_size((std::size_t)A1.cols()));
      opt1.initial_block.push_back(
          F1.create_random_of_size((std::size_t)A1.cols()));

      // simple k-step SD preconditioner on N = A^T A (matrix-free)
      const int k_sd = 6;
      opt1.M_inv = [&](const VectorInterface &r_n, VectorInterface &y_n) {
        const std::size_t m = (std::size_t)A1.rows();
        const std::size_t n = (std::size_t)A1.cols();
        auto p = F1.create_zero_of_size(n);
        auto Np = F1.create_zero_of_size(n);
        auto tmp = F1.create_zero_of_size(m);

        y_n.set_zero();
        p->copy_from(r_n);
        for (int it = 0; it < k_sd; ++it) {
          // N p = A^T (A p)
          C1.multiply(*p, *tmp);
          C1.multiply_transpose(*tmp, *Np);
          const double pNp = p->dot(*Np);
          if (std::fabs(pNp) < 1e-30)
            break;
          const double p2 = p->dot(*p);
          const double alpha = p2 / pNp;
          y_n.add_scaled(*p, alpha);
          // r ← r - α N p  (store back into p as the next direction)
          p->add_scaled(*Np, -alpha);
          const double r2 = p->dot(*p);
          if (r2 <= 1e-30)
            break;
        }
      };

      print_header("LOBPCG mat1.ija", (std::size_t)A1.rows(),
                   (std::size_t)A1.cols(), (std::size_t)A1.nonZeros(), opt1);

      const auto t0 = Clock::now();
      auto R = sns::solvers::find_nullspace_lobpcg_ex(C1, F1, opt1);
      const auto t1 = Clock::now();
      const double sec = std::chrono::duration<double>(t1 - t0).count();
      print_perf(C1, sec);

      std::cout << "[mat1] returned " << R.right_null.size()
                << " vectors; iters=" << R.iters << "  ||A||_F~"
                << std::scientific << R.frob_norm_est << "\n";

      assert(R.right_null.size() == 5);
      check_unit_norms(R.right_null, 1e-10);

      auto tmp_m1 = F1.create_zero_of_size((std::size_t)A1.rows());
      double worst = 0.0;
      for (std::size_t i = 0; i < R.right_null.size(); ++i) {
        C1.multiply(*R.right_null[i], *tmp_m1);
        const double ri = tmp_m1->norm();
        worst = std::max(worst, ri);
        std::cout << "[mat1] ||A v_" << i << "|| = " << std::scientific << ri
                  << "\n";
        assert(ri <= 1e-8);
      }
      const double ortho1 = sns::utils::max_pairwise_abs_dot(R.right_null);
      std::cout << "[mat1] max pairwise |dot| = " << std::scientific << ortho1
                << "\n";
      assert(ortho1 <= 1e-8);

      auto v_rand1 = F1.create_random_of_size((std::size_t)A1.cols());
      auto v_proj1 = F1.create_zero_of_size((std::size_t)A1.cols());
      project_onto_span(R.right_null, *v_proj1, *v_rand1);
      C1.multiply(*v_proj1, *tmp_m1);
      const double r_proj1 = tmp_m1->norm();
      std::cout << "[mat1] ||A (Proj_N v_rand)|| = " << std::scientific
                << r_proj1 << "\n";
      assert(r_proj1 <= 1e-8);

      std::cout << "test_lobpcg (mat1) OK\n\n";
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

    sns::solvers::LOBPCGOptions opt;
    opt.nev = 1;
    opt.block_size = 4;
    opt.max_iters = 100;
    opt.tol_abs = 1e-12;
    opt.tol_rel = 1e-10;
    opt.soft_locking = true;
    opt.restart = 0.9;
    opt.gs_reorth = true;
    opt.nullity_known = 0; // no right nullspace
    opt.rank_known = 3;
    opt.verbose = false;

    print_header("LOBPCG Identity (3x3)", (std::size_t)I.rows(),
                 (std::size_t)I.cols(), (std::size_t)I.nonZeros(), opt);

    const auto t0 = Clock::now();
    auto R = sns::solvers::find_nullspace_lobpcg_ex(CI, F, opt);
    const auto t1 = Clock::now();
    const double sec = std::chrono::duration<double>(t1 - t0).count();
    print_perf(CI, sec);

    std::cout << "[Identity] returned " << R.right_null.size() << " vectors\n";
    assert(R.right_null.empty());
    std::cout << "test_lobpcg (identity) OK\n\n";
  }

  // ==================== Test 4: IJA mat2.ija (rank=5425, nullity=15) ========
  {
    try {
      S A2 = sns::io::load_ija_triplet_matrix("data/mat2.ija", -1, -1);
      A2.makeCompressed();

      EigenSparseMatrixAdapter M2(A2);
      CountingMatrixAdapter C2(M2);
      EigenVectorFactory F2((std::size_t)A2.cols());

      const int rank_known = 5425;
      const int nullity_known = 15;

      sns::solvers::LOBPCGOptions opt2;
      opt2.nev = nullity_known;
      opt2.block_size = std::max<std::size_t>(nullity_known, 24);
      opt2.max_iters = 1200;
      opt2.tol_abs = 1e-12;
      opt2.tol_rel = 1e-10;
      opt2.soft_locking = true;
      opt2.restart = 0.85; // a bit more aggressive: more guard vectors
      opt2.gs_reorth = true;
      opt2.nullity_known = nullity_known;
      opt2.rank_known = rank_known;
      opt2.verbose = false;

      // two random seeds
      opt2.initial_block.push_back(
          F2.create_random_of_size((std::size_t)A2.cols()));
      opt2.initial_block.push_back(
          F2.create_random_of_size((std::size_t)A2.cols()));

      // k-step SD preconditioner for N = A^T A
      const int k_sd = 10;
      opt2.M_inv = [&](const VectorInterface &r_n, VectorInterface &y_n) {
        const std::size_t m = (std::size_t)A2.rows();
        const std::size_t n = (std::size_t)A2.cols();
        auto p = F2.create_zero_of_size(n);
        auto Np = F2.create_zero_of_size(n);
        auto tmp = F2.create_zero_of_size(m);

        y_n.set_zero();
        p->copy_from(r_n);
        for (int it = 0; it < k_sd; ++it) {
          C2.multiply(*p, *tmp);
          C2.multiply_transpose(*tmp, *Np);
          const double pNp = p->dot(*Np);
          if (std::fabs(pNp) < 1e-30)
            break;
          const double p2 = p->dot(*p);
          const double alpha = p2 / pNp;
          y_n.add_scaled(*p, alpha);
          p->add_scaled(*Np, -alpha); // r ← r - α N p
          const double r2 = p->dot(*p);
          if (r2 <= 1e-30)
            break;
        }
      };

      print_header("LOBPCG mat2.ija", (std::size_t)A2.rows(),
                   (std::size_t)A2.cols(), (std::size_t)A2.nonZeros(), opt2);

      const auto t0 = Clock::now();
      auto R = sns::solvers::find_nullspace_lobpcg_ex(C2, F2, opt2);
      const auto t1 = Clock::now();
      const double sec = std::chrono::duration<double>(t1 - t0).count();
      print_perf(C2, sec);

      std::cout << "[mat2] returned " << R.right_null.size()
                << " vectors; iters=" << R.iters << "  ||A||_F~"
                << std::scientific << R.frob_norm_est << "\n";

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
        assert(ri <= 1e-9);
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

      std::cout << "test_lobpcg (mat2) OK\n\n";
    } catch (const std::exception &e) {
      std::cerr << "[mat2] Skipping mat2 test: " << e.what() << "\n\n";
    }
  }

  // ==================== Test 5: scale invariance =============================
  {
    // Same 2x3 as Test 1, but scaled by a large factor s.
    const double s = 1e6;
    S A(2, 3);
    std::vector<Eigen::Triplet<double>> T = {
        {0, 0, s * 1.0}, {0, 2, s * 1.0}, {1, 1, s * 1.0}, {1, 2, s * 1.0}};
    A.setFromTriplets(T.begin(), T.end());
    A.makeCompressed();

    EigenSparseMatrixAdapter M(A);
    CountingMatrixAdapter C(M);
    EigenVectorFactory F(static_cast<std::size_t>(A.cols()));

    sns::solvers::LOBPCGOptions opt;
    opt.nev = 1;
    opt.block_size = 6;
    opt.max_iters = 300;
    opt.tol_abs = 1e-12;
    opt.tol_rel = 1e-10;
    opt.soft_locking = true;
    opt.restart = 0.9;
    opt.gs_reorth = true;
    opt.nullity_known = 1;
    opt.rank_known = 2;
    opt.verbose = false;

    print_header("LOBPCG Scale invariance (2x3 scaled by 1e6)",
                 (std::size_t)A.rows(), (std::size_t)A.cols(),
                 (std::size_t)A.nonZeros(), opt);

    const auto t0 = Clock::now();
    auto R = sns::solvers::find_nullspace_lobpcg_ex(C, F, opt);
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

    std::cout << "test_lobpcg (scale invariance) OK\n\n";
  }

  std::cout << "ALL LOBPCG TESTS OK\n";
  return 0;
}
