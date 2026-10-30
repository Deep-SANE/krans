#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <chrono>
#include <memory>
#include <cmath>
#include <iomanip>
#include <sstream>
#include <map>
#include <functional>
#include <algorithm>

#include "sns/MatrixInterface.hpp"
#include "sns/VectorInterface.hpp"
#include "sns/VectorFactory.hpp"
#include "sns/IO/MatrixLoader.hpp"
#include "sns/adapters/eigen/EigenSparseMatrixAdapter.hpp"
#include "sns/adapters/eigen/EigenVectorFactory.hpp"
#include "sns/Solvers/EigenSVDSolver.hpp"
#include "sns/Solvers/EigenFullPivLUSolver.hpp"
#include "sns/Solvers/EigenCODSolver.hpp"
#include "sns/Solvers/EigenQRCPSolver.hpp"
#include "sns/Solvers/LOBPCGSolver.hpp"
#include "sns/Solvers/CyclicLanczosSolver.hpp"

#ifdef SNS_USE_SPECTRA
#include "sns/Solvers/SpectraSolver.hpp"
#endif
#ifdef SNS_USE_PRIMME
#include "sns/Solvers/PrimmeSolver.hpp"
#endif
#ifdef SNS_USE_SLEPC
#include "sns/Solvers/SlepcSolver.hpp"
#endif
#ifdef SNS_USE_SUITESPARSE
#include "sns/Solvers/SuiteSparseQRSolver.hpp"
#endif

#include "sns/Solvers/CommonDenseUtils.hpp"
#include "sns/Utils/SubspaceCompare.hpp"

using namespace sns;

// Matrix Wrapper for SpMV counting 
class CountingMatrixAdapter final : public MatrixInterface {
public:
    explicit CountingMatrixAdapter(const MatrixInterface &underlying)
        : U_(underlying) {}
    std::size_t rows() const override { return U_.rows(); }
    std::size_t cols() const override { return U_.cols(); }
    void multiply(const VectorInterface &x, VectorInterface &y) const override {
        ++spmv_; U_.multiply(x, y);
    }
    void multiply_transpose(const VectorInterface &x, VectorInterface &y) const override {
        ++spmv_t_; U_.multiply_transpose(x, y);
    }
    std::size_t spmv() const { return spmv_; }
    std::size_t spmvT() const { return spmv_t_; }
    void reset_counters() { spmv_ = 0; spmv_t_ = 0; }
private:
    const MatrixInterface &U_;
    mutable std::size_t spmv_ = 0, spmv_t_ = 0;
};

// Solver Result storage for a series of runs
struct SolverBenchmark {
    std::string name;
    std::vector<double> times_ms;
    std::vector<double> residuals;
    std::vector<double> orthogonalities;
    std::vector<double> rref_distances;
    std::vector<std::size_t> spmv_counts;
    std::vector<std::size_t> spmv_t_counts;
    std::vector<std::unique_ptr<VectorInterface>> representative_basis;
};

// Compute the max physical residual
double compute_max_residual(const MatrixInterface& A, const std::vector<std::unique_ptr<VectorInterface>>& basis, const VectorFactory& vf) {
    if (basis.empty()) return 1e9;
    double max_res = 0.0;
    auto Ax = vf.create_zero_of_size(A.rows());
    for (const auto& v : basis) {
        A.multiply(*v, *Ax);
        max_res = std::max(max_res, Ax->norm());
    }
    return max_res;
}

// Compute orthogonality error
double compute_orthogonality_error(const std::vector<std::unique_ptr<VectorInterface>>& basis) {
    if (basis.empty()) return 1.0;
    std::size_t k = basis.size();
    double sum_sq = 0.0;
    for (std::size_t i = 0; i < k; ++i) {
        for (std::size_t j = 0; j < k; ++j) {
            double dot = basis[i]->dot(*basis[j]);
            double target = (i == j) ? 1.0 : 0.0;
            double err = dot - target;
            sum_sq += err * err;
        }
    }
    return std::sqrt(sum_sq);
}

// Helper to format a vector as a JSON array string
template<typename T>
std::string to_json_array(const std::vector<T>& v) {
    std::ostringstream oss;
    oss << "[";
    for (size_t i = 0; i < v.size(); ++i) {
        if constexpr (std::is_floating_point_v<T>) {
            oss << std::scientific << std::setprecision(8) << v[i];
        } else {
            oss << v[i];
        }
        if (i + 1 < v.size()) oss << ", ";
    }
    oss << "]";
    return oss.str();
}

void run_verification() {
    const double tol_val = 1e-7; 
    const int iterative_runs = 5;

    struct MatrixTask {
        std::string filename;
        std::size_t k;
        bool enabled;
    };

    // test matrices and their expected nullity 
    std::vector<MatrixTask> tasks = {
        {"data/mat1.ija", 5,  true},
        {"data/mat2.ija", 15, true},
        {"data/mat3.ija", 10, true},
        {"data/mat4.ija", 15, true},
        {"data/mat5.ija", 20, true},
        {"data/mat6.ija", 25, true},
        {"data/mat7.mtx", 1,  true},
        {"data/mat8.mtx", 3,  true}
    };

    std::ostringstream json_out;
    json_out << "{\n";

    bool first_matrix = true;
    for (size_t m_idx = 0; m_idx < tasks.size(); ++m_idx) {
        const auto& task = tasks[m_idx];
        if (!task.enabled) continue;

        std::cout << "\n========================================================================\n";
        std::cout << "MATRIX: " << task.filename << " (Target Nullity k=" << task.k << ")\n";
        std::cout << "========================================================================\n";

        Eigen::SparseMatrix<double> S;
        try {
            if (task.filename.find(".mtx") != std::string::npos) S = io::load_matrix_market(task.filename);
            else S = io::load_ija_triplet_matrix(task.filename);
        } catch (const std::exception& e) {
            std::cerr << "CRITICAL: Could not load " << task.filename << ": " << e.what() << std::endl;
            continue;
        }

        EigenSparseMatrixAdapter A_underlying(S);
        CountingMatrixAdapter A_counting(A_underlying);
        EigenVectorFactory vf(A_underlying.cols(), 42); 

        std::map<std::string, SolverBenchmark> benchmarks;
        std::vector<std::string> solver_names_in_order;

        // Reference: EigenSVD 
        std::cout << "[EigenSVD] (Reference)... " << std::flush;
        std::vector<std::unique_ptr<VectorInterface>> Z_ref;
        {
            auto start = std::chrono::high_resolution_clock::now();

            // Dense realization for reference singular values
            Eigen::MatrixXd M = solvers::detail::dense_copy_via_matvecs_or_view(A_underlying, vf);
            Eigen::JacobiSVD<Eigen::MatrixXd> svd_diag(M);
            auto s = svd_diag.singularValues();
            std::cout << " (Smallest svals: ";
            for (int i = 0; i < std::min<int>(5, (int)s.size()); ++i) 
                std::cout << s(s.size()-1-i) << (i==4?"":", ");
            std::cout << ") " << std::flush;

            Z_ref = solvers::find_nullspace_svd(A_underlying, vf, tol_val);
            auto end = std::chrono::high_resolution_clock::now();
            
            double time = std::chrono::duration<double, std::milli>(end - start).count();
            auto& b = benchmarks["EigenSVD"];
            b.name = "EigenSVD";
            b.times_ms.push_back(time);
            b.residuals.push_back(compute_max_residual(A_underlying, Z_ref, vf));
            b.orthogonalities.push_back(compute_orthogonality_error(Z_ref));
            b.rref_distances.push_back(0.0);
            b.spmv_counts.push_back(0);
            b.spmv_t_counts.push_back(0);
            for(const auto& v : Z_ref) b.representative_basis.push_back(v->clone());
            solver_names_in_order.push_back("EigenSVD");
            std::cout << "Done (" << time << " ms, k_found=" << Z_ref.size() << ")" << std::endl;
        }

        // Shared Runner Lambda
        auto run_solver = [&](const std::string& name, bool direct, auto solve_func) {
            std::cout << "[" << name << "] " << (direct ? "Direct..." : "Iterative...") << std::flush;
            auto& b = benchmarks[name];
            b.name = name;
            int runs = direct ? 1 : iterative_runs;

            for (int r = 0; r < runs; ++r) {
                try {
                    A_counting.reset_counters();
                    auto start = std::chrono::high_resolution_clock::now();
                    auto basis = solve_func();
                    auto end = std::chrono::high_resolution_clock::now();

                    double t_ms = std::chrono::duration<double, std::milli>(end - start).count();
                    b.times_ms.push_back(t_ms);
                    b.residuals.push_back(compute_max_residual(A_underlying, basis, vf));
                    b.orthogonalities.push_back(compute_orthogonality_error(basis));
                    b.rref_distances.push_back(utils::subspace_diff(basis, Z_ref, 1e-10));
                    b.spmv_counts.push_back(A_counting.spmv());
                    b.spmv_t_counts.push_back(A_counting.spmvT());
                    
                    if (r == 0) {
                        for(const auto& v : basis) b.representative_basis.push_back(v->clone());
                    }
                    if (!direct) std::cout << "." << std::flush;
                } catch (const std::exception& e) {
                    std::cerr << "\nERROR in " << name << ": " << e.what() << std::endl;
                }
            }
            solver_names_in_order.push_back(name);
            std::cout << " Done (" << b.times_ms[0] << " ms, k_found=" << b.representative_basis.size() << ")" << std::endl;
        };

        run_solver("EigenFullPivLU", true, [&]() { return solvers::find_nullspace_fullpivlu(A_underlying, vf, tol_val); });
        run_solver("EigenCOD", true, [&]() { return solvers::find_nullspace_cod(A_underlying, vf, tol_val); });
        run_solver("EigenQRCP", true, [&]() { return solvers::find_nullspace_qrcp(A_underlying, vf, tol_val); });
#ifdef SNS_USE_SUITESPARSE
        run_solver("SuiteSparseQR", true, [&]() { return solvers::find_nullspace_suitesparse_qr(A_underlying, vf, tol_val); });
#endif

        run_solver("LOBPCG", false, [&]() {
            solvers::LOBPCGOptions opt;
            opt.nev = task.k;
            opt.tol_rel = tol_val;
            opt.block_size = std::min<std::size_t>(std::min<std::size_t>(A_counting.cols(), 64), std::max<std::size_t>(16, task.k));
            opt.max_iters = 1000;
            return solvers::find_nullspace_lobpcg(A_counting, vf, opt);
        });

        run_solver("CyclicLanczos", false, [&]() {
            solvers::CyclicLanczosOptions opt;
            opt.k = task.k;
            opt.tol_rel = tol_val;
            opt.max_restarts = 500;
            opt.ncv = std::max<std::size_t>(60, task.k + 30);
            opt.cg_maxit = 200;
            if (opt.ncv > A_counting.cols() + A_counting.rows()) opt.ncv = A_counting.cols() + A_counting.rows();
            return solvers::find_nullspace_cyclic_lanczos(A_counting, vf, opt);
        });

#ifdef SNS_USE_SPECTRA
        run_solver("Spectra", false, [&]() { 
            std::size_t n = A_counting.cols();
            std::size_t nev = std::min<std::size_t>(task.k, n - 1);
            if (nev == 0 && task.k > 0) nev = 1;
            std::size_t ncv = std::min<std::size_t>(n, std::max<std::size_t>(task.k * 6, 80));
            return solvers::find_nullspace_spectra(A_counting, vf, nev, ncv, tol_val);
        });
#endif
#ifdef SNS_USE_PRIMME
       run_solver("Primme", false, [&]() { return solvers::find_nullspace_primme(A_counting, vf, task.k, tol_val); });
#endif
#ifdef SNS_USE_SLEPC
        run_solver("Slepc", false, [&]() { return solvers::find_nullspace_slepc(A_counting, vf, task.k, tol_val); });
#endif

        // --- Summary Verification ---
        std::cout << "\n[Consistency Verification]\n";
        bool all_k_correct = true;
        for (const auto& name : solver_names_in_order) {
            size_t found = benchmarks[name].representative_basis.size();
            if (found != task.k) {
                std::cout << "  ! FAILED: " << name << " found " << found << " instead of " << task.k << "\n";
                all_k_correct = false;
            } else {
                double dist = benchmarks[name].rref_distances[0];
                if (dist > 1e-5) {
                    std::cout << "  ! WARNING: " << name << " subspace distance is " << dist << "\n";
                }
            }
        }
        if (all_k_correct) std::cout << "  OK: All solvers identified the full nullspace for this matrix.\n";
        else {
            std::cout << "  FATAL: Verification failed for matrix " << task.filename << "\n";
        }

        std::string m_id = task.filename.substr(task.filename.find_last_of('/') + 1);
        m_id = m_id.substr(0, m_id.find('.'));
        if (!first_matrix) json_out << ",\n";
        json_out << "  \"" << m_id << "\": {\n";
        json_out << "    \"path\": \"" << task.filename << "\",\n";
        json_out << "    \"target_k\": " << task.k << ",\n";
        
        size_t s_idx = 0;
        for (auto const& name : solver_names_in_order) {
            auto const& b = benchmarks[name];
            json_out << "    \"" << name << "\": {\n";
            json_out << "      \"time_ms\": " << to_json_array(b.times_ms) << ",\n";
            json_out << "      \"residual\": " << to_json_array(b.residuals) << ",\n";
            json_out << "      \"orthogonality\": " << to_json_array(b.orthogonalities) << ",\n";
            json_out << "      \"rref_distance\": " << to_json_array(b.rref_distances) << ",\n";
            json_out << "      \"spmv\": " << to_json_array(b.spmv_counts) << ",\n";
            json_out << "      \"spmv_t\": " << to_json_array(b.spmv_t_counts) << "\n";
            json_out << "    }" << (++s_idx == solver_names_in_order.size() ? "" : ",") << "\n";
        }
        json_out << "  }";
        first_matrix = false;
    }

    json_out << "\n}\n";

    std::ofstream fout("verification_results.json");
    if (fout.is_open()) {
        fout << json_out.str();
        std::cout << "\nResults saved to verification_results.json" << std::endl;
    }
}

int main(int argc, char** argv) {
    std::cout << "SNS Iterative Verification Suite" << std::endl;
    try {
        run_verification();
    } catch (const std::exception& e) {
        std::cerr << "FATAL ERROR: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}
