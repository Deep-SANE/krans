#pragma once
#include "krans/MatrixInterface.hpp"
#include "krans/VectorInterface.hpp"
#include "krans/VectorFactory.hpp"

#include <fstream>
#include <iomanip>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>
#include <cmath>

namespace krans::utils {

/** Write a basis (vector of VectorInterface pointers) to CSV (no headers); returns true on success. */
inline bool write_csv_basis(const std::string &path,
                            const std::vector<std::unique_ptr<krans::VectorInterface>> &basis,
                            int precision = 17) {
  std::ofstream out(path);
  if (!out)
    return false;
  out.setf(std::ios::fixed, std::ios::floatfield);
  out << std::setprecision(precision);
  
  if (basis.empty())
    return true;
    
  const std::size_t rows = basis.front()->size();
  const std::size_t cols = basis.size();
  
  for (std::size_t i = 0; i < rows; ++i) {
    for (std::size_t j = 0; j < cols; ++j) {
      const double* data = basis[j]->data();
      if (!data) throw std::runtime_error("ExportIO requires direct data pointer access.");
      out << data[i];
      if (j + 1 < cols)
        out << ',';
    }
    out << '\n';
  }
  return static_cast<bool>(out);
}

/** Per-basis diagnostics. */
struct BasisDiagnostics {
  /** Residuals per basis vector. */
  std::vector<double> residuals;
  /** Maximum |dot| between distinct basis vectors (orthogonality check). */
  double max_offdiag_abs_dot = 0.0;
};

/** Compute BasisDiagnostics for a right-nullspace basis under A. */
inline BasisDiagnostics diagnose_basis(
    const krans::MatrixInterface &A,
    const std::vector<std::unique_ptr<krans::VectorInterface>> &basis,
    const krans::VectorFactory &vf) {
  BasisDiagnostics d;
  const size_t k = basis.size();
  d.residuals.resize(k, 0.0);

  auto y = vf.create_zero_of_size(A.rows());
  for (size_t j = 0; j < k; ++j) {
    A.multiply(*basis[j], *y);
    d.residuals[j] = y->norm();
  }
  double m = 0.0;
  for (size_t i = 0; i < k; ++i)
    for (size_t j = i + 1; j < k; ++j)
      m = std::max(m, std::fabs(basis[i]->dot(*basis[j])));
  d.max_offdiag_abs_dot = m;
  return d;
}

/** Write compact JSON report. */
inline bool write_json_report(const std::string &path, std::size_t m,
                              std::size_t n, std::size_t rank,
                              std::size_t nullity,
                              const std::string &solver_name, double tol_rel,
                              double t_load_ms, double t_solve_ms,
                              const BasisDiagnostics &d) {
  std::ofstream out(path);
  if (!out)
    return false;
  out << std::setprecision(17);
  out << "{\n";
  out << "  \"solver\": \"" << solver_name << "\",\n";
  out << "  \"rows\": " << m << ",\n";
  out << "  \"cols\": " << n << ",\n";
  out << "  \"rank\": " << rank << ",\n";
  out << "  \"nullity\": " << nullity << ",\n";
  out << "  \"tol_rel\": " << tol_rel << ",\n";
  out << "  \"timings_ms\": { \"load\": " << t_load_ms
      << ", \"solve\": " << t_solve_ms << " },\n";
  out << "  \"residuals\": [";
  for (size_t i = 0; i < d.residuals.size(); ++i) {
    if (i)
      out << ", ";
    out << d.residuals[i];
  }
  out << "],\n";
  out << "  \"max_offdiag_abs_dot\": " << d.max_offdiag_abs_dot << "\n";
  out << "}\n";
  return static_cast<bool>(out);
}

} // namespace krans::utils
