#include <iostream>
#include <vector>
#include <string>
#include <chrono>
#include <memory>
#include <cmath>
#include <algorithm>
#include <Eigen/Dense>
#include <Eigen/Sparse>

#include "sns/MatrixInterface.hpp"
#include "sns/VectorInterface.hpp"
#include "sns/VectorFactory.hpp"
#include "sns/IO/MatrixLoader.hpp"
#include "sns/adapters/eigen/EigenSparseMatrixRefAdapter.hpp"
#include "sns/adapters/eigen/EigenVectorFactory.hpp"
#include "sns/Solvers/LOBPCGSolver.hpp"
#include "sns/Solvers/CyclicLanczosSolver.hpp"

#ifdef SNS_USE_SPECTRA
#include "sns/Solvers/SpectraSolver.hpp"
#endif
#ifdef SNS_USE_PRIMME
#include "sns/Solvers/PrimmeSolver.hpp"
#endif

using namespace sns;

int main(int argc, char** argv) {
    if (argc != 4) {
        std::cerr << "Usage: ./conditioning_worker <matrix_file> <target_k> <tolerance>\n";
        return 1;
    }
    std::string filename = argv[1];
    std::size_t target_k = std::stoull(argv[2]);
    double target_tol = std::stod(argv[3]);

    Eigen::SparseMatrix<double> S;
    try {
        if (filename.find(".ija") != std::string::npos) {
            S = io::load_ija_triplet_matrix(filename, -1, -1);
        } else {
            S = io::load_matrix_market(filename);
        }
    } catch (const std::exception& e) { 
        std::cout << "ERROR LOAD: " << e.what() << "\n"; 
        return 1; 
    } catch (...) {
        std::cout << "ERROR LOAD: Unknown exception\n";
        return 1;
    }

    // The core abstraction
    EigenSparseMatrixRefAdapter A(S);
    EigenVectorFactory vf(A.cols(), 42);

    auto run_test = [&](const std::string& name) {
        auto start = std::chrono::high_resolution_clock::now();
        std::vector<std::unique_ptr<VectorInterface>> basis;
        try {
            if (name == "External_Spectra") {
#ifdef SNS_USE_SPECTRA
                std::size_t ncv = std::max<std::size_t>(60, target_k * 4);
                basis = solvers::find_nullspace_spectra(A, vf, target_k, ncv, target_tol);
#else
                throw std::runtime_error("Spectra not enabled");
#endif
            } else if (name == "External_PRIMME") {
#ifdef SNS_USE_PRIMME
                basis = solvers::find_nullspace_primme(A, vf, target_k, target_tol);
#else
                throw std::runtime_error("PRIMME not enabled");
#endif
            } else if (name == "Custom_Lanczos_H") {
                solvers::CyclicLanczosOptions opt; 
                opt.k = target_k; 
                opt.ncv = 80;                               
                opt.max_restarts = 500;
                opt.projector_passes = 2;                    
                opt.cg_maxit = 200;                          
                opt.tol_abs = target_tol;                    
                opt.tol_rel = 0.0;
                opt.cg_rel_tol = target_tol * 0.1;
                opt.use_harmonic = true;                     
                opt.reorth = sns::solvers::ReorthMode::Double;
                opt.prescreen_factor = 100.0;
                opt.stagnation_restarts = 50;
                opt.verbose = false;
                
                basis = solvers::find_nullspace_cyclic_lanczos(A, vf, opt);
                
            } else if (name == "Custom_LOBPCG_AtA") {
                solvers::LOBPCGOptions opt; 
                opt.nev = target_k; 
                opt.block_size = 16;                          
                opt.max_iters = 20000;
                opt.tol_abs = target_tol;
                opt.tol_rel = 0.0;
                opt.gs_reorth = true;                         
                
                basis = solvers::find_nullspace_lobpcg(A, vf, opt);
            }
            
            auto end = std::chrono::high_resolution_clock::now();
            double time_s = std::chrono::duration<double>(end - start).count();

            // Strict Validation
            double max_res = 1.0; 
            double avg_res = 1.0;
            if (!basis.empty()) {
                max_res = 0.0;
                avg_res = 0.0;
                auto tmp = vf.create_zero_of_size(A.rows());
                for (const auto& v : basis) {
                    A.multiply(*v, *tmp);
                    double r = tmp->norm();
                    max_res = std::max(max_res, r);
                    avg_res += r;
                }
                avg_res /= basis.size();
            }
            
            std::cout << "MASTER_RES " << name << " " << time_s << " " << max_res << " " << basis.size() << " " << avg_res << "\n";
        } catch (const std::exception& e) {
            std::cout << "MASTER_RES " << name << " FAILED 1.0 0 (" << e.what() << ")\n";
        } catch (...) {
            std::cout << "MASTER_RES " << name << " FAILED 1.0 0\n";
        }
    };

    run_test("External_Spectra");
    run_test("External_PRIMME");
    run_test("Custom_Lanczos_H");
    run_test("Custom_LOBPCG_AtA");

    return 0;
}