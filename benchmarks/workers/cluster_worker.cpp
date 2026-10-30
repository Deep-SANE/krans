#include <iostream>
#include <vector>
#include <string>
#include <chrono>
#include <memory>
#include <cmath>
#include <stdexcept>
#include <sys/resource.h>
#include <map>

#include <Eigen/Dense>
#include <Eigen/Sparse>

#include "krans/MatrixInterface.hpp"
#include "krans/VectorInterface.hpp"
#include "krans/VectorFactory.hpp"
#include "krans/IO/MatrixLoader.hpp"
#include "krans/adapters/eigen/EigenSparseMatrixRefAdapter.hpp"
#include "krans/adapters/eigen/EigenVectorFactory.hpp"

#include "krans/Solvers/LOBPCGSolver.hpp"
#include "krans/Solvers/CyclicLanczosSolver.hpp"
#include "krans/Solvers/EigenQRCPSolver.hpp"
#include "krans/Solvers/EigenSVDSolver.hpp"

#ifdef KRANS_USE_SPECTRA
#include "krans/Solvers/SpectraSolver.hpp"
#endif
#ifdef KRANS_USE_PRIMME
#include "krans/Solvers/PrimmeSolver.hpp"
#endif
#ifdef KRANS_USE_SLEPC
#include "krans/Solvers/SlepcSolver.hpp"
#endif

using namespace krans;

// ==========================================
// PEAK MEMORY TRACKER (Gigabytes)
// ==========================================
double get_peak_memory_gb() {
    struct rusage r_usage;
    getrusage(RUSAGE_SELF, &r_usage);
#ifdef __APPLE__
    return static_cast<double>(r_usage.ru_maxrss) / (1024.0 * 1024.0 * 1024.0);
#else
    return static_cast<double>(r_usage.ru_maxrss) / (1024.0 * 1024.0); // Linux uses KB
#endif
}

// ==========================================
// MATRIX-FREE SpMV COUNTER
// ==========================================
class CountingMatrixAdapter : public MatrixInterface {
    const MatrixInterface& base_;
    mutable std::size_t spmv_count_ = 0;
public:
    explicit CountingMatrixAdapter(const MatrixInterface& base) : base_(base) {}
    
    void multiply(const VectorInterface& x, VectorInterface& y) const override {
        base_.multiply(x, y);
        spmv_count_++;
    }
    
    void multiply_transpose(const VectorInterface& x, VectorInterface& y) const override {
        base_.multiply_transpose(x, y);
        spmv_count_++;
    }
    
    std::size_t rows() const override { return base_.rows(); }
    std::size_t cols() const override { return base_.cols(); }
    std::size_t get_count() const { return spmv_count_; }
};

// ==========================================
// MAIN EXECUTION
// ==========================================
int main(int argc, char** argv) {
    std::map<std::string, std::string> args;
    for (int i = 1; i < argc - 1; i += 2) {
        args[argv[i]] = argv[i+1];
    }

    std::string filename = args.count("--matrix") ? args["--matrix"] : "";
    std::string solver   = args.count("--solver") ? args["--solver"] : "";
    std::size_t target_k = args.count("--k") ? std::stoull(args["--k"]) : 1;
    
    // locked in configuration 
    const double tol_rel = 1e-10;
    const double tol_abs = 1e-12;
    const int proj_passes = 2;
    const int cg_maxiters = 200;
    const bool use_harmonic = true;

    if (filename.empty() || solver.empty()) {
        std::cerr << "Usage: ./titan_worker --matrix <path> --solver <name> --k <nullity>\n";
        return 1;
    }

    // Load Matrix (Zero-Copy View to survive the 10M unknown Titans)
    Eigen::SparseMatrix<double> S;
    try {
        if (filename.find(".ija") != std::string::npos || filename.find(".aij") != std::string::npos) {
            S = io::load_ija_triplet_matrix(filename, -1, -1);
        } else {
            S = io::load_matrix_market(filename);
        }
    } catch (const std::exception& e) {
       std::cout << "0.0,0.0,-,-,0,0\n"; // Python CSV format on failure
        return 1;
    }

    EigenSparseMatrixRefAdapter BaseAdapter(S);
    CountingMatrixAdapter A(BaseAdapter);
    EigenVectorFactory vf(A.cols(), 42); // Locked seed for strictly equitable comparisons

    std::vector<std::unique_ptr<VectorInterface>> basis;
    auto start_time = std::chrono::high_resolution_clock::now();

    // Execute Selected Solver
    try {
        if (solver == "lobpcg") {
            solvers::LOBPCGOptions opt; 
            opt.nev = target_k;
            opt.nullity_known = target_k;
            opt.block_size = (target_k == 1) ? 2 : std::min<std::size_t>(16, target_k);
            opt.max_iters = 20000;
            opt.tol_rel = tol_rel;
            opt.tol_abs = tol_abs;
            opt.gs_tol = 1e-14;
            opt.gs_reorth = true; // "Twice is enough" Double Reorth
            opt.verbose = false;
            basis = solvers::find_nullspace_lobpcg(A, vf, opt);

        } else if (solver == "lanczos") {
            solvers::CyclicLanczosOptions opt; 
            opt.k = target_k; 
            opt.nullity_known = target_k;
            opt.ncv = std::max<std::size_t>(80, target_k * 3); // Adaptive Krylov capacity
            opt.prescreen_factor = 1.0;
            opt.cg_maxit = cg_maxiters;  
            opt.projector_passes = proj_passes;
            opt.max_restarts = 5000; 
            opt.stagnation_restarts = 50;
            opt.tol_rel = tol_rel;
            opt.tol_abs = tol_abs;
            
            opt.use_harmonic = use_harmonic;
            opt.reorth = krans::solvers::ReorthMode::Double; // "Twice is enough"
            basis = solvers::find_nullspace_cyclic_lanczos(A, vf, opt);

        } else if (solver == "qrcp") {
            basis = solvers::find_nullspace_qrcp(A, vf, tol_rel);

        } else if (solver == "svd") {
            basis = solvers::find_nullspace_svd(A, vf, tol_rel);

        } else if (solver == "slepc") {
#ifdef KRANS_USE_SLEPC
            // Pass the strict 1e-10 relative tolerance to SLEPc
            basis = solvers::find_nullspace_slepc(A, vf, target_k, tol_rel);
#else
            throw std::runtime_error("SLEPc not compiled");
#endif
        } else if (solver == "primme") {
#ifdef KRANS_USE_PRIMME
            // Pass the strict 1e-10 relative tolerance to PRIMME
            basis = solvers::find_nullspace_primme(A, vf, target_k, tol_rel);
#else
            throw std::runtime_error("PRIMME not compiled");
#endif
        } else if (solver == "spectra") {
#ifdef KRANS_USE_SPECTRA
            std::size_t ncv = std::max<std::size_t>(60, target_k * 3);
            basis = solvers::find_nullspace_spectra(A, vf, target_k, ncv, tol_rel);
#else
            throw std::runtime_error("Spectra not compiled");
#endif
        } else {
            throw std::runtime_error("Unknown solver requested.");
        }

        auto end_time = std::chrono::high_resolution_clock::now();
        double time_s = std::chrono::duration<double>(end_time - start_time).count();

       // --- SOFT VALIDATION (Thesis Mode) ---
        double max_residual = 0.0;
        double ortho_error = 0.0;

        // Validation only if we actually found something
        if (!basis.empty()) {
            auto tmp = vf.create_zero_of_size(A.rows());
            for (const auto& v : basis) {
                A.multiply(*v, *tmp);
                max_residual = std::max(max_residual, tmp->norm());
            }

            double sum_sq = 0.0;
            for (std::size_t i = 0; i < basis.size(); ++i) {
                for (std::size_t j = 0; j < basis.size(); ++j) {
                    double dot_val = basis[i]->dot(*basis[j]);
                    double expected = (i == j) ? 1.0 : 0.0;
                    double diff = dot_val - expected;
                    sum_sq += diff * diff;
                }
            }
            ortho_error = std::sqrt(sum_sq);
        }

        // Final Output including basis.size() so we know if it hit target_k
        // Format: Time, Mem, Res, Ortho, SpMV, FoundK
        std::cout << time_s << "," 
                  << get_peak_memory_gb() << "," 
                  << max_residual << "," 
                  << ortho_error << "," 
                  << A.get_count() << ","
                  << basis.size() << "\n";

    } catch (const std::bad_alloc&) {
        // Out of Memory Catcher
        std::cout << "OOM," << get_peak_memory_gb() << ",-,-,-\n";
    } catch (const std::exception& e) {
        // Output failure signature that can be easily parsed
        std::cerr << "Worker Error: " << e.what() << "\n";
        std::cout << "ERROR," << get_peak_memory_gb() << ",-,-,0,0\n";
        return 1;
    }

    return 0;
}