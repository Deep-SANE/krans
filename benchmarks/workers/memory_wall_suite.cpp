#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <chrono>
#include <memory>
#include <cmath>
#include <iomanip>
#include <sys/resource.h>
#include <stdexcept>

#include "sns/MatrixInterface.hpp"
#include "sns/VectorInterface.hpp"
#include "sns/VectorFactory.hpp"
#include "sns/IO/MatrixLoader.hpp"
#include "sns/adapters/eigen/EigenSparseMatrixAdapter.hpp"
#include "sns/adapters/eigen/EigenSparseMatrixRefAdapter.hpp"
#include "sns/adapters/eigen/EigenVectorFactory.hpp"

#include "sns/Solvers/EigenSVDSolver.hpp"
#include "sns/Solvers/EigenFullPivLUSolver.hpp"
#include "sns/Solvers/EigenCODSolver.hpp"
#include "sns/Solvers/EigenQRCPSolver.hpp"
#include "sns/Solvers/LOBPCGSolver.hpp"
#include "sns/Solvers/CyclicLanczosSolver.hpp"

#ifdef SNS_USE_SLEPC
#include "sns/Solvers/SlepcSolver.hpp"
#endif
#ifdef SNS_USE_SUITESPARSE
#include "sns/Solvers/SuiteSparseQRSolver.hpp"
#endif

using namespace sns;

// Cross-platform Peak RAM usage in Megabytes
double get_peak_memory_mb() {
    struct rusage r_usage;
    getrusage(RUSAGE_SELF, &r_usage);
#ifdef __APPLE__
    return static_cast<double>(r_usage.ru_maxrss) / (1024.0 * 1024.0);
#else
    return static_cast<double>(r_usage.ru_maxrss) / 1024.0;
#endif
}

int main(int argc, char** argv) {
    if (argc != 4) {
        std::cerr << "Usage: ./memory_wall_worker <matrix_file> <adapter> <solver>\n";
        return 1;
    }
    
    std::string filename = argv[1];
    std::string adapter_name = argv[2];
    std::string solver_name = argv[3];

    Eigen::SparseMatrix<double> S;
    try {
        S = io::load_matrix_market(filename);
    } catch (...) {
        std::cout << "RESULT 0.0 0.0 ERROR_LOADING_MATRIX\n";
        return 1;
    }

    std::unique_ptr<MatrixInterface> A;
    if (adapter_name == "Owned") {
        A = std::make_unique<EigenSparseMatrixAdapter>(S);
    } else {
        A = std::make_unique<EigenSparseMatrixRefAdapter>(S);
    }

    EigenVectorFactory vf(A->cols(), 42);
    std::size_t k = 1;
    
    // Lower tolerance for memory wall to ensure we get results before OOM
    const double tol_val = 1e-5; 

    std::vector<std::unique_ptr<VectorInterface>> basis;
    auto start = std::chrono::high_resolution_clock::now();
    
    try {
        if (solver_name == "LOBPCG") {
            solvers::LOBPCGOptions opt; 
            opt.nev = k; opt.tol_rel = tol_val; opt.block_size = 16; 
            opt.max_iters = 6000;
            basis = solvers::find_nullspace_lobpcg(*A, vf, opt);
        } else if (solver_name == "CyclicLanczos") {
            solvers::CyclicLanczosOptions opt; 
            opt.k = k; opt.tol_rel = tol_val; 
            opt.ncv = std::max<std::size_t>(150, k + 100); 
            opt.max_restarts = 2000; 
            opt.stagnation_restarts = 500;
            opt.projector_passes = 2;
            opt.cg_maxit = 2000; 
            opt.use_harmonic = true;
            opt.reorth = sns::solvers::ReorthMode::Double;
            opt.grad_max_iters = 5000; 
            basis = solvers::find_nullspace_cyclic_lanczos(*A, vf, opt);
        } else if (solver_name == "Slepc") {
#ifdef SNS_USE_SLEPC
            basis = solvers::find_nullspace_slepc(*A, vf, k, tol_val);
#else
            throw std::runtime_error("DISABLED");
#endif
        } else if (solver_name == "SuiteSparseQR") {
#ifdef SNS_USE_SUITESPARSE
            basis = solvers::find_nullspace_suitesparse_qr(*A, vf, tol_val);
#else
            throw std::runtime_error("DISABLED");
#endif
        } else if (solver_name == "EigenSVD") {
            basis = solvers::find_nullspace_svd(*A, vf, tol_val);
        } else if (solver_name == "EigenFullPivLU") {
            basis = solvers::find_nullspace_fullpivlu(*A, vf, tol_val);
        } else if (solver_name == "EigenCOD") {
            basis = solvers::find_nullspace_cod(*A, vf, tol_val);
        } else if (solver_name == "EigenQRCP") {
            basis = solvers::find_nullspace_qrcp(*A, vf, tol_val);
        } else {
            throw std::runtime_error("UNKNOWN_SOLVER");
        }

        auto end = std::chrono::high_resolution_clock::now();
        double time_ms = std::chrono::duration<double, std::milli>(end - start).count();
        
        // OS Peak RAM Isolation
        double mem_mb = get_peak_memory_mb();

        std::string status = (basis.size() < k) ? "FAILED_CONVERGENCE" : "OK";
        std::cout << "RESULT " << time_ms << " " << mem_mb << " " << status << "\n";

    } catch (const std::bad_alloc&) {
        std::cout << "RESULT 0.0 " << get_peak_memory_mb() << " OOM\n";
    } catch (const std::exception& e) {
        std::string msg = e.what();
        if (msg == "DISABLED") std::cout << "RESULT 0.0 0.0 DISABLED\n";
        else std::cout << "RESULT 0.0 " << get_peak_memory_mb() << " ERROR\n";
    }

    return 0;
}