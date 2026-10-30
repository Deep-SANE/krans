#include <iostream>
#include <vector>
#include <string>
#include <memory>
#include <chrono>
#include <Eigen/Dense>

#include "krans/MatrixInterface.hpp"
#include "krans/VectorInterface.hpp"
#include "krans/VectorFactory.hpp"
#include "krans/IO/MatrixLoader.hpp"
#include "krans/adapters/eigen/EigenSparseMatrixRefAdapter.hpp"
#include "krans/adapters/eigen/EigenVectorFactory.hpp"
#include "krans/Solvers/LOBPCGSolver.hpp"
#include "krans/Solvers/CyclicLanczosSolver.hpp"

using namespace krans;

// Helper to check orthogonality
double check_orthogonality(const std::vector<std::unique_ptr<VectorInterface>>& Z) {
    if (Z.empty()) return 1.0;
    int k = Z.size();
    Eigen::MatrixXd ZtZ = Eigen::MatrixXd::Zero(k, k);
    for (int i = 0; i < k; ++i) {
        for (int j = 0; j <= i; ++j) {
            ZtZ(i, j) = ZtZ(j, i) = Z[i]->dot(*Z[j]);
        }
    }
    Eigen::MatrixXd I = Eigen::MatrixXd::Identity(k, k);
    return (ZtZ - I).norm();
}

int main(int argc, char** argv) {
    if (argc != 4) return 1;
    
    std::string filename = argv[1];
    std::string solver_name = argv[2];
    std::size_t param_val = std::stoull(argv[3]);

    Eigen::SparseMatrix<double> S;
    try {
        if (filename.find(".ija") != std::string::npos) {
            // Passing -1, -1 allows the loader to infer the dimensions from the IJA header
            S = io::load_ija_triplet_matrix(filename, -1, -1);
        } else {
            S = io::load_matrix_market(filename);
        }
    } 
    catch (const std::exception& e) { 
        std::cerr << "Matrix Load Error: " << e.what() << "\n";
        // Return 0 so Python cleanly catches the fallback result instead of a hard crash
        std::cout << "RESULT 0.0 1.0 1.0 0\n"; 
        return 0; 
    }

    EigenSparseMatrixRefAdapter A(S);
    EigenVectorFactory vf(A.cols(), 42);
    
    // Explicitly ask for 15 vectors for this unstructured matrix
    std::size_t k = 15;

    std::vector<std::unique_ptr<VectorInterface>> basis;
    auto start = std::chrono::high_resolution_clock::now();
    
    try {
        if (solver_name == "LOBPCG") {
            solvers::LOBPCGOptions opt; 
            opt.nev = k; 
            opt.block_size = param_val; 
            opt.max_iters = 15000;
            opt.tol_rel = 1e-10;
            opt.tol_abs = 1e-12;
            opt.gs_reorth = true; 
            
            basis = solvers::find_nullspace_lobpcg(A, vf, opt);
            
        } else if (solver_name == "CyclicLanczos") {
            solvers::CyclicLanczosOptions opt; 
            opt.k = k; 
            opt.ncv = param_val; 
            opt.max_restarts = 400;
            opt.projector_passes = 2;
            opt.cg_maxit = 200;
            opt.cg_rel_tol = 1e-10;
            opt.prescreen_factor = 4.0;
            opt.stagnation_restarts = 0;
            opt.tol_abs = 1e-12;
            opt.tol_rel = 1e-10;
            opt.use_harmonic = true;
            opt.reorth = krans::solvers::ReorthMode::Double;

            basis = solvers::find_nullspace_cyclic_lanczos(A, vf, opt);
        }

        auto end = std::chrono::high_resolution_clock::now();
        double time_ms = std::chrono::duration<double, std::milli>(end - start).count();

        double max_res = 0.0;
        auto tmp_m = vf.create_zero_of_size(A.rows());
        for (const auto& v : basis) {
            A.multiply(*v, *tmp_m);
            max_res = std::max(max_res, tmp_m->norm());
        }
        double ortho_err = check_orthogonality(basis);
        std::size_t found_k = basis.size();

        // Print: RESULT <time> <max_residual> <ortho_error> <found_vectors>
        std::cout << "RESULT " << time_ms << " " << max_res << " " << ortho_err << " " << found_k << "\n";

    } catch (...) {
        std::cout << "RESULT 0.0 1.0 1.0 0\n";
    }
    return 0;
}