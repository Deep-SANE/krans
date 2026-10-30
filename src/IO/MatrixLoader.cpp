#include "krans/IO/MatrixLoader.hpp"
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <unsupported/Eigen/SparseExtra>
#include <vector>

namespace krans::io {

// === Matrix Market (.mtx) ===
Eigen::SparseMatrix<double> load_matrix_market(const std::string &path) {
  Eigen::SparseMatrix<double> mat;
  if (!Eigen::loadMarket(mat, path)) {
    throw std::runtime_error("Failed to load Matrix Market file: " + path);
  }
  return mat;
}

// === IJA-style triplet loader ===
Eigen::SparseMatrix<double> load_ija_triplet_matrix(const std::string &path,
                                                    int rows, int cols) {
  std::ifstream in(path);
  if (!in) {
    throw std::runtime_error("Cannot open file: " + path);
  }

  // Read dimensions and nnz
  int nrows = 0, ncols = 0, nnz = 0;
  if (!(in >> nrows >> ncols >> nnz)) {
    throw std::runtime_error("Invalid IJA file header: " + path);
  }

  if (rows < 0)
    rows = nrows;
  if (cols < 0)
    cols = ncols;

  std::vector<Eigen::Triplet<double>> triplets;
  triplets.reserve(nnz);

  int i, j;
  double value;
  while (in >> i >> j >> value) {
    triplets.emplace_back(i, j, value);
  }

  if ((int)triplets.size() != nnz) {
    std::cerr << "Warning: Expected " << nnz << " entries but read "
              << triplets.size() << "\n";
  }

  Eigen::SparseMatrix<double> mat(rows, cols);
  mat.setFromTriplets(triplets.begin(), triplets.end());
  return mat;
}

} // namespace krans::io
