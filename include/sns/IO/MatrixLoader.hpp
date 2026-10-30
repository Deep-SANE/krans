#pragma once

#include <Eigen/Sparse>
#include <string>

namespace sns::io {

/** Loads a sparse matrix from a Matrix Market file (.mtx). */
Eigen::SparseMatrix<double> load_matrix_market(const std::string &path);

/** Loads a sparse matrix from an .ija-style triplet file. */
Eigen::SparseMatrix<double>
load_ija_triplet_matrix(const std::string &path, int rows = -1, int cols = -1);

} // namespace sns::io
