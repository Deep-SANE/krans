#include "sns/IO/MatrixLoader.hpp"
#include "sns/adapters/eigen/EigenSparseMatrixAdapter.hpp"
#include "sns/adapters/eigen/EigenVectorAdapter.hpp"

#include <Eigen/Dense>
#include <iostream>

using namespace sns;

int main() {
  try {
    auto mat = io::load_ija_triplet_matrix("data/mat1.ija");

    std::cout << "Loaded matrix: " << mat.rows() << " x " << mat.cols()
              << " with " << mat.nonZeros() << " non-zero entries.\n";

    EigenSparseMatrixAdapter adapter(mat);

    Eigen::VectorXd x(mat.cols());
    x.setOnes();

    EigenVectorAdapter input(x);
    EigenVectorAdapter result(mat.rows());

    adapter.multiply(input, result);

    std::cout << "A * [1, ..., 1] = " << result.view().transpose() << "\n";
  } catch (const std::exception &e) {
    std::cerr << "Error: " << e.what() << "\n";
    return 1;
  }
  return 0;
}
