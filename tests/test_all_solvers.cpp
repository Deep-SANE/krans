#include "krans/Solvers/EigenCODSolver.hpp"
#include "krans/Solvers/EigenFullPivLUSolver.hpp"
#include "krans/Solvers/EigenQRCPSolver.hpp"
#include "krans/Solvers/EigenSVDSolver.hpp"
#include "krans/Solvers/CyclicLanczosSolver.hpp"
#include "krans/Solvers/LOBPCGSolver.hpp"
#include "krans/IO/MatrixLoader.hpp"

#ifdef KRANS_USE_SPECTRA
#include "krans/Solvers/SpectraSolver.hpp"
#endif
#ifdef KRANS_USE_SLEPC
#include "krans/Solvers/SlepcSolver.hpp"
#endif
#ifdef KRANS_USE_SUITESPARSE
#include "krans/Solvers/SuiteSparseQRSolver.hpp"
#endif
#ifdef KRANS_USE_PRIMME
#include "krans/Solvers/PrimmeSolver.hpp"
#endif

#include "krans/Utils/SubspaceCompare.hpp"
#include "krans/adapters/eigen/EigenDenseMatrixAdapter.hpp"
#include "krans/adapters/eigen/EigenSparseMatrixAdapter.hpp"
#include "krans/adapters/eigen/EigenVectorAdapter.hpp"
#include "krans/adapters/eigen/EigenVectorFactory.hpp"

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <iostream>
#include <memory>
#include <vector>
#include <cassert>
#include <functional>

using namespace krans;

struct SolverEntry {
    std::string name;
    std::function<std::vector<std::unique_ptr<VectorInterface>>(const MatrixInterface&, const VectorFactory&, std::size_t, double)> solve;
};

void run_test_on_matrix(const std::string& path, std::size_t k_target, const std::vector<SolverEntry>& solvers) {
    std::cout << "\n>>> Testing matrix: " << path << " (Target nullity: " << k_target << ")\n";
    
    Eigen::SparseMatrix<double> S;
    try {
        if (path.find(".mtx") != std::string::npos) S = io::load_matrix_market(path);
        else S = io::load_ija_triplet_matrix(path);
    } catch (const std::exception &e) {
        std::cerr << "Failed to load " << path << ": " << e.what() << "\n";
        return;
    }

    EigenSparseMatrixAdapter A(S);
    EigenVectorFactory vf(A.cols());

    struct Result {
        std::string name;
        std::vector<std::unique_ptr<VectorInterface>> basis;
    };
    std::vector<Result> all_results;

    for (const auto& s : solvers) {
        std::cout << "  - [" << s.name << "] ";
        auto basis = s.solve(A, vf, k_target, 1e-8);
        std::cout << "found " << basis.size() << " vectors. ";
        
        if (basis.size() < k_target && k_target != static_cast<size_t>(-1)) {
            std::cout << "UNDER-COUNTED ";
        }

        double max_res = 0.0;
        auto tmp_m = vf.create_zero_of_size(A.rows());
        for (const auto& v : basis) {
            A.multiply(*v, *tmp_m);
            max_res = std::max(max_res, tmp_m->norm());
        }
        std::cout << "Max res: " << max_res << "\n";

        all_results.push_back({s.name, std::move(basis)});
    }

    // --- Cross-compare solvers that found the same number of vectors ---
    std::cout << "  - [Subspace Cross-Comparison]\n";
    bool all_match = true;
    for (size_t i = 0; i < all_results.size(); ++i) {
        for (size_t j = i + 1; j < all_results.size(); ++j) {
            const auto& r1 = all_results[i];
            const auto& r2 = all_results[j];
            
            if (r1.basis.size() > 0 && r1.basis.size() == r2.basis.size()) {
                double diff = krans::utils::subspace_diff(r1.basis, r2.basis, 1e-7);
                if (diff > 1e-5) {
                    std::cout << "! WARNING: Subspace mismatch between " << r1.name 
                              << " and " << r2.name << " (diff=" << diff << ")\n";
                    all_match = false;
                } else {
                    // Only print if verbose or if we want to confirm matches
                    // std::cout << "    OK: " << r1.name << " == " << r2.name << "\n";
                }
            }
        }
    }
    if(all_match) {
        std::cout << "All solvers with the same number of vectors found consistent subspaces.\n";
    }
}

int main() {
    std::vector<SolverEntry> solvers;
    
    solvers.push_back({"svd", [](const MatrixInterface& A, const VectorFactory& vf, std::size_t, double tol) {
        return solvers::find_nullspace_svd(A, vf, tol);
    }});
    solvers.push_back({"cod", [](const MatrixInterface& A, const VectorFactory& vf, std::size_t, double tol) {
        return solvers::find_nullspace_cod(A, vf, tol);
    }});
    solvers.push_back({"qrcp", [](const MatrixInterface& A, const VectorFactory& vf, std::size_t, double tol) {
        return solvers::find_nullspace_qrcp(A, vf, tol);
    }});
    solvers.push_back({"fullpivlu", [](const MatrixInterface& A, const VectorFactory& vf, std::size_t, double tol) {
        return solvers::find_nullspace_fullpivlu(A, vf, tol);
    }});
    solvers.push_back({"cyclic", [](const MatrixInterface& A, const VectorFactory& vf, std::size_t k, double tol) {
        solvers::CyclicLanczosOptions opt;
        opt.k = k;
        opt.tol_rel = tol;
        opt.max_restarts = 200;
        opt.ncv = std::max<std::size_t>(60, k + 10);
        if (opt.ncv > A.cols() + A.rows()) opt.ncv = A.cols() + A.rows();
        return solvers::find_nullspace_cyclic_lanczos(A, vf, opt);
    }});
    solvers.push_back({"lobpcg", [](const MatrixInterface& A, const VectorFactory& vf, std::size_t k, double tol) {
        solvers::LOBPCGOptions opt;
        opt.nev = k;
        opt.tol_rel = tol;
        opt.block_size = std::min<std::size_t>(std::min<std::size_t>(A.cols(), 64), std::max<std::size_t>(16, k));
        opt.max_iters = 1000;
        return solvers::find_nullspace_lobpcg(A, vf, opt);
    }});
#ifdef KRANS_USE_SPECTRA
    solvers.push_back({"spectra", [](const MatrixInterface& A, const VectorFactory& vf, std::size_t k, double tol) {
        std::size_t n = A.cols();
        std::size_t nev = std::min<std::size_t>(k, n - 1);
        if (nev == 0 && k > 0) nev = 1;
        std::size_t ncv = std::min<std::size_t>(n, std::max<std::size_t>(k * 6, 80));
        return solvers::find_nullspace_spectra(A, vf, nev, ncv, tol);
    }});
#endif
#ifdef KRANS_USE_SLEPC
    solvers.push_back({"slepc", [](const MatrixInterface& A, const VectorFactory& vf, std::size_t k, double tol) {
        return solvers::find_nullspace_slepc(A, vf, k, tol);
    }});
#endif
#ifdef KRANS_USE_SUITESPARSE
    solvers.push_back({"suitesparse", [](const MatrixInterface& A, const VectorFactory& vf, std::size_t, double tol) {
        return solvers::find_nullspace_suitesparse_qr(A, vf, tol);
    }});
#endif

#ifdef KRANS_USE_PRIMME
    solvers.push_back({"primme", [](const MatrixInterface& A, const VectorFactory& vf, std::size_t k, double tol) {
        return solvers::find_nullspace_primme(A, vf, k, tol);
    }});
#endif

    // Run tests (TODO: add more test matrices with known nullity)
    run_test_on_matrix("data/mat1.ija", 5, solvers);
    //run_test_on_matrix("data/mat2.ija", 15, solvers);
    run_test_on_matrix("data/mat3.ija", 10, solvers);
    run_test_on_matrix("data/mat4.ija", 15, solvers);
    run_test_on_matrix("data/mat5.ija", 20, solvers);
    run_test_on_matrix("data/mat6.ija", 25, solvers);

    return 0;
}
