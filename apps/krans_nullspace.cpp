#include "krans/IO/MatrixLoader.hpp"
#include "krans/Utils/ExportIO.hpp"
#include "krans/Utils/Timer.hpp"

// Direct (reference) solvers
#include "krans/Solvers/EigenCODSolver.hpp"
#include "krans/Solvers/EigenFullPivLUSolver.hpp"
#include "krans/Solvers/EigenQRCPSolver.hpp"
#include "krans/Solvers/EigenSVDSolver.hpp"

// Iterative cyclic/augmented solver
#include "krans/Solvers/CyclicLanczosSolver.hpp"
#include "krans/Solvers/LOBPCGSolver.hpp"
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

#include "krans/VectorFactory.hpp"
#include "krans/adapters/eigen/EigenSparseMatrixAdapter.hpp"
#include "krans/adapters/eigen/EigenVectorFactory.hpp"

#include <Eigen/Sparse>
#include <cctype>
#include <functional>
#include <iostream>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

// ---- Small helpers ---------------------------------------------------------

std::string to_lower(std::string s) {
  for (auto &c : s)
    c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
  return s;
}

void usage(const char *argv0) {
  std::cerr
      << "KraNS Nullspace CLI\n"
      << "---------------------\n"
      << "Usage:\n"
      << "  " << argv0 << " --solver [svd|qrcp|cod|fullpivlu|cyclic|lobpcg|spectra|primme|slepc|suitesparse]\n"
      << "           --input <matrix.(mtx|ija)>\n"
      << "           [--out N.csv] [--report report.json]\n"
      << "           [--tol <rel_tol>]           # for direct solvers (default "
         "1e-12)\n"
      << "           [--seed <int>]              # RNG seed for iterative "
         "solvers (default 1; 0 = random)\n"
      << "\n"
      << "Cyclic solver options (optional):\n"
      << "  --k <int>              # number of right null vectors (default 1)\n"
      << "  --ncv <int>            # subspace size (default 48)\n"
      << "  --max-restarts <int>   # outer restarts (default 80)\n"
      << "  --tol-abs <float>      # accept if ||A v|| <= max(tol_abs, "
         "tol_rel*||A||_F), default 1e-12\n"
      << "  --tol-rel <float>      # relative part in acceptance, default "
         "1e-10\n"
      << "  --proj-passes <int>    # projector polish passes (1 or 2), default "
         "1\n"
      << "  --cg-maxit <int>       # CG iters for normal eqs, default 80\n"
      << "  --cg-rel <float>       # CG relative tol, default 1e-10\n"
      << "  --prescreen <float>    # skip polishing if ||A v|| > f*||A||_F, "
         "default 0.8\n"
      << "  --stagnation <int>     # stop after this many restarts without "
         "progress (0 = off), default 30\n"
      << "  --verbose              # print progress\n"
      << "  --left                 # also export left nullspace to N_left.csv\n"
      << "\n"
      << "LOBPCG options (optional):\n"
      << "  --k <int>                  # number of right null vectors (default "
         "1)\n"
      << "  --lobpcg-block <int>       # block size (default min(16,k))\n"
      << "  --lobpcg-max-it <int>      # max outer iterations (default 1000)\n"
      << "  --lobpcg-restart <float>   # restart fraction in (0.1..1], default "
         "0.9\n"
      << "  --lobpcg-soft-lock <0|1>   # soft locking on/off (default 1)\n"
      << "  --tol-abs <float>          # accept if ||A v|| <= max(tol_abs, "
         "tol_rel*||A||_F), default 1e-12\n"
      << "  --tol-rel <float>          # relative part in acceptance, default "
         "1e-10\n"
      << "  --verbose                  # print progress\n"
      << std::endl;
}

// Registry type for direct solvers (uniform signature)
using DirectSolverFn =
    std::function<std::vector<std::unique_ptr<krans::VectorInterface>>(
        const krans::MatrixInterface &, const krans::VectorFactory &, double)>;

std::map<std::string, DirectSolverFn> make_direct_registry() {
  using namespace krans::solvers;
  return {
      {"svd", &find_nullspace_svd},
      {"qrcp", &find_nullspace_qrcp},
      {"cod", &find_nullspace_cod},
      {"fullpivlu", &find_nullspace_fullpivlu},
  };
}

} // namespace

int main(int argc, char **argv) {
  // ---- Defaults -------------------------------------------------------------
  std::string solver_name = "svd";
  std::string input_path;
  std::string out_csv = "N.csv";
  std::string report_json = "report.json";

  // Direct solvers relative tolerance
  double tol_rel_direct = 1e-12;

  // Cyclic options (sensible defaults)
  int cyc_k = 1;
  int cyc_ncv = 48;
  int cyc_max_restarts = 80;
  double cyc_tol_abs = 1e-12;
  double cyc_tol_rel = 1e-10;
  int cyc_proj_passes = 1;
  int cyc_cg_maxit = 80;
  double cyc_cg_rel = 1e-10;
  double cyc_prescreen = 0.8;
  int cyc_stagnation = 30;
  bool cyc_verbose = false;
  bool cyc_left = false;

  // Fixed default seed so runs are reproducible
  unsigned seed = 1;

  // LOBPCG options
  int lob_k = 1;
  int lob_block = 0; // 0 -> auto min(16, k)
  int lob_max_it = 1000;
  double lob_restart = 0.9; // (0.1..1]
  int lob_soft_lock = 1;    // 1=true
  double lob_tol_abs = 1e-12;
  double lob_tol_rel = 1e-10;
  bool lob_verbose = false;

  // ---- Parse args -----------------------------------------------------------
  auto need = [&](int i, int k) -> bool { return (i + k >= argc); };

  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    if (a == "--solver") {
      if (need(i, 1)) {
        usage(argv[0]);
        return 2;
      }
      solver_name = to_lower(argv[++i]);
    } else if (a == "--input") {
      if (need(i, 1)) {
        usage(argv[0]);
        return 2;
      }
      input_path = argv[++i];
    } else if (a == "--out") {
      if (need(i, 1)) {
        usage(argv[0]);
        return 2;
      }
      out_csv = argv[++i];
    } else if (a == "--report") {
      if (need(i, 1)) {
        usage(argv[0]);
        return 2;
      }
      report_json = argv[++i];
    } else if (a == "--tol") { // direct solvers only
      if (need(i, 1)) {
        usage(argv[0]);
        return 2;
      }
      tol_rel_direct = std::stod(argv[++i]);
    }
    // Cyclic/LOBPCG shared flags
    else if (a == "--k") {
      if (need(i, 1)) {
        usage(argv[0]);
        return 2;
      }
      cyc_k = std::stoi(argv[++i]);
    } else if (a == "--tol-abs") {
      if (need(i, 1)) {
        usage(argv[0]);
        return 2;
      }
      cyc_tol_abs = std::stod(argv[++i]);
    } else if (a == "--tol-rel") {
      if (need(i, 1)) {
        usage(argv[0]);
        return 2;
      }
      cyc_tol_rel = std::stod(argv[++i]);
    } else if (a == "--verbose") {
      cyc_verbose = true;
    }
    // Cyclic-only flags
    else if (a == "--ncv") {
      if (need(i, 1)) {
        usage(argv[0]);
        return 2;
      }
      cyc_ncv = std::stoi(argv[++i]);
    } else if (a == "--max-restarts") {
      if (need(i, 1)) {
        usage(argv[0]);
        return 2;
      }
      cyc_max_restarts = std::stoi(argv[++i]);
    } else if (a == "--proj-passes") {
      if (need(i, 1)) {
        usage(argv[0]);
        return 2;
      }
      cyc_proj_passes = std::stoi(argv[++i]);
    } else if (a == "--cg-maxit") {
      if (need(i, 1)) {
        usage(argv[0]);
        return 2;
      }
      cyc_cg_maxit = std::stoi(argv[++i]);
    } else if (a == "--cg-rel") {
      if (need(i, 1)) {
        usage(argv[0]);
        return 2;
      }
      cyc_cg_rel = std::stod(argv[++i]);
    } else if (a == "--prescreen") {
      if (need(i, 1)) {
        usage(argv[0]);
        return 2;
      }
      cyc_prescreen = std::stod(argv[++i]);
    } else if (a == "--stagnation") {
      if (need(i, 1)) {
        usage(argv[0]);
        return 2;
      }
      cyc_stagnation = std::stoi(argv[++i]);
    } else if (a == "--seed") {
      if (need(i, 1)) {
        usage(argv[0]);
        return 2;
      }
      seed = static_cast<unsigned>(std::stoul(argv[++i]));
    } else if (a == "--left") {
      cyc_left = true;
    }
    // LOBPCG-only flags
    else if (a == "--lobpcg-block") {
      if (need(i, 1)) {
        usage(argv[0]);
        return 2;
      }
      lob_block = std::stoi(argv[++i]);
    } else if (a == "--lobpcg-max-it") {
      if (need(i, 1)) {
        usage(argv[0]);
        return 2;
      }
      lob_max_it = std::stoi(argv[++i]);
    } else if (a == "--lobpcg-restart") {
      if (need(i, 1)) {
        usage(argv[0]);
        return 2;
      }
      lob_restart = std::stod(argv[++i]);
    } else if (a == "--lobpcg-soft-lock") {
      if (need(i, 1)) {
        usage(argv[0]);
        return 2;
      }
      lob_soft_lock = std::stoi(argv[++i]); // 0/1
    } else if (a == "--help" || a == "-h") {
      usage(argv[0]);
      return 0;
    } else {
      std::cerr << "Unknown argument: " << a << "\n";
      usage(argv[0]);
      return 2;
    }
  }

  if (input_path.empty()) {
    std::cerr << "Error: --input is required.\n";
    usage(argv[0]);
    return 2;
  }

  // Share the common flags with LOBPCG
  lob_k = cyc_k;
  lob_tol_abs = cyc_tol_abs;
  lob_tol_rel = cyc_tol_rel;
  lob_verbose = cyc_verbose;

  // ---- Load sparse matrix (.mtx / .ija) ------------------------------------
  krans::utils::Timer timer;
  double t_load_ms = 0.0;
  double t_solve_ms = 0.0;

  const auto dot = input_path.find_last_of('.');
  const std::string ext =
      (dot == std::string::npos) ? "" : to_lower(input_path.substr(dot + 1));

  Eigen::SparseMatrix<double> S;
  try {
    if (ext == "mtx") {
      S = krans::io::load_matrix_market(input_path);
    } else if (ext == "ija") {
      // rows/cols are read from the file header internally
      S = krans::io::load_ija_triplet_matrix(input_path, -1, -1);
    } else {
      std::cerr << "Unsupported input format: expected .mtx or .ija\n";
      return 2;
    }
  } catch (const std::exception &e) {
    std::cerr << "Matrix load error: " << e.what() << "\n";
    return 2;
  }
  S.makeCompressed();

  t_load_ms = timer.elapsed_seconds() * 1000.0;

  // Wrap in adapter
  krans::EigenSparseMatrixAdapter A(S);
  const std::size_t m = A.rows();
  const std::size_t n = A.cols();

  // Real vector factory (default size = n)
  krans::EigenVectorFactory vf(n, seed);

  // ---- Solve ----------------------------------------------------------------
  timer.reset();

  std::vector<std::unique_ptr<krans::VectorInterface>> right_basis;
  bool wrote_left = false;

  const auto direct_registry = make_direct_registry();

  if (solver_name == "cyclic") {
    // Configure and run cyclic solver
    krans::solvers::CyclicLanczosOptions opt;
    opt.k = (cyc_k > 0 ? static_cast<std::size_t>(cyc_k) : 1);
    opt.ncv = (cyc_ncv > 0 ? static_cast<std::size_t>(cyc_ncv) : 48);
    opt.max_restarts =
        (cyc_max_restarts >= 0 ? static_cast<std::size_t>(cyc_max_restarts)
                               : 80);
    opt.tol_abs = cyc_tol_abs;
    opt.tol_rel = cyc_tol_rel;
    opt.projector_passes = std::max(1, cyc_proj_passes);
    opt.cg_maxit =
        (cyc_cg_maxit >= 0 ? static_cast<std::size_t>(cyc_cg_maxit) : 80);
    opt.cg_rel_tol = cyc_cg_rel;
    opt.prescreen_factor = cyc_prescreen;
    opt.stagnation_restarts =
        static_cast<std::size_t>(std::max(0, cyc_stagnation));
    opt.reorth = krans::solvers::ReorthMode::Double;
    opt.use_harmonic = true;
    opt.target_tau = 0.0;
    opt.return_left_space = cyc_left;
    opt.verbose = cyc_verbose;

    auto R = krans::solvers::find_nullspace_cyclic_lanczos_ex(A, vf, opt);
    right_basis = std::move(R.right_null);

    // Optionally export left nullspace
          if (cyc_left && !R.left_null.empty()) {
            try {
              if (!krans::utils::write_csv_basis("N_left.csv", R.left_null)) {
                std::cerr << "Warning: failed to write N_left.csv\n";
              } else {          wrote_left = true;
        }
      } catch (const std::exception &e) {
        std::cerr << "Warning: left-basis export failed: " << e.what() << "\n";
      }
    }
  } else if (solver_name == "lobpcg") {
    // Configure and run LOBPCG solver
    krans::solvers::LOBPCGOptions opt;
    opt.nev = (lob_k > 0 ? static_cast<std::size_t>(lob_k) : 1);
    opt.block_size = (lob_block > 0 ? static_cast<std::size_t>(lob_block)
                                    : std::min<std::size_t>(16, opt.nev));
    opt.max_iters =
        (lob_max_it > 0 ? static_cast<std::size_t>(lob_max_it) : 1000);
    opt.restart = (lob_restart > 0.1 && lob_restart <= 1.0) ? lob_restart : 0.9;
    opt.soft_locking = (lob_soft_lock != 0);
    opt.tol_abs = lob_tol_abs;
    opt.tol_rel = lob_tol_rel;
    opt.gs_tol = 1e-14;
    opt.gs_reorth = true;
    opt.frob_samples = 4;
    opt.verbose = lob_verbose;

    right_basis = krans::solvers::find_nullspace_lobpcg(A, vf, opt);
#ifdef KRANS_USE_SPECTRA
  } else if (solver_name == "spectra") {
    right_basis = krans::solvers::find_nullspace_spectra(A, vf, cyc_k, cyc_ncv, cyc_tol_rel);
#endif
#ifdef KRANS_USE_SLEPC
  } else if (solver_name == "slepc") {
    right_basis = krans::solvers::find_nullspace_slepc(A, vf, cyc_k, cyc_tol_rel);
#endif
#ifdef KRANS_USE_SUITESPARSE
  } else if (solver_name == "suitesparse") {
    right_basis = krans::solvers::find_nullspace_suitesparse_qr(A, vf, cyc_tol_rel);
#endif
#ifdef KRANS_USE_PRIMME
  } else if (solver_name == "primme") {
    right_basis = krans::solvers::find_nullspace_primme(A, vf, cyc_k, cyc_tol_rel);
#endif
  } else {
    // Direct methods from registry
    auto it = direct_registry.find(solver_name);
    if (it == direct_registry.end()) {
      std::cerr << "Unknown solver: " << solver_name << "\n";
      usage(argv[0]);
      return 2;
    }
    right_basis = it->second(A, vf, tol_rel_direct);
  }

  t_solve_ms = timer.elapsed_seconds() * 1000.0;

  const std::size_t nullity = right_basis.size();
  const std::size_t rank_est = (n >= nullity ? n - nullity : 0);

  // ---- Export N.csv + report.json ------------------------------------------
  try {
    if (!krans::utils::write_csv_basis(out_csv, right_basis)) {
      std::cerr << "Failed to write " << out_csv << "\n";
      return 3;
    }
  } catch (const std::exception &e) {
    std::cerr << "Basis export failed: " << e.what() << "\n";
    return 3;
  }

  // Diagnostics
  auto diag = krans::utils::diagnose_basis(A, right_basis, vf);
  double rel_for_report = tol_rel_direct;
  if (solver_name == "cyclic")
    rel_for_report = cyc_tol_rel;
  if (solver_name == "lobpcg")
    rel_for_report = lob_tol_rel;

  if (!krans::utils::write_json_report(report_json, m, n, rank_est, nullity,
                                     solver_name, rel_for_report, t_load_ms,
                                     t_solve_ms, diag)) {
    std::cerr << "Failed to write " << report_json << "\n";
    return 4;
  }

  // ---- Summary --------------------------------------------------------------
  std::cout << "OK [" << solver_name << "]  m=" << m << " n=" << n << " rank≈"
            << rank_est << " nullity=" << nullity << " | load=" << t_load_ms
            << " ms, solve=" << t_solve_ms << " ms\n"
            << "Wrote: " << out_csv << " and " << report_json
            << (wrote_left ? " (plus N_left.csv)" : "") << "\n";

  return 0;
}
