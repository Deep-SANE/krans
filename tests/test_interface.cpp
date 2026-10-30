#include "sns/MatrixInterface.hpp"
#include "sns/VectorInterface.hpp"
#include <cassert>
#include <cmath>
#include <iostream>
#include <memory>
#include <vector>

using namespace sns;

// === Simple Mock Vector ===
class MockVector : public VectorInterface {
public:
  std::vector<double> data;

  explicit MockVector(std::size_t size) : data(size, 0.0) {}

  double dot(const VectorInterface &other) const override {
    const MockVector &o = dynamic_cast<const MockVector &>(other);
    double sum = 0.0;
    for (size_t i = 0; i < data.size(); ++i)
      sum += data[i] * o.data[i];
    return sum;
  }

  void scale(double alpha) override {
    for (auto &val : data)
      val *= alpha;
  }

  void add_scaled(const VectorInterface &other, double alpha) override {
    const MockVector &o = dynamic_cast<const MockVector &>(other);
    for (size_t i = 0; i < data.size(); ++i)
      data[i] += alpha * o.data[i];
  }

  double norm() const override { return std::sqrt(this->dot(*this)); }

  void normalize() override {
    double n = norm();
    if (n > 0.0)
      scale(1.0 / n);
  }

  std::size_t size() const override { return data.size(); }

  std::unique_ptr<VectorInterface> clone() const override {
    auto out = std::make_unique<MockVector>(data.size());
    out->data = data;
    return out;
  }
};

// === Simple 2x2 Matrix ===
class MockMatrix : public MatrixInterface {
public:
  double a, b, c, d;

  MockMatrix(double a, double b, double c, double d) : a(a), b(b), c(c), d(d) {}

  void multiply(const VectorInterface &x,
                VectorInterface &result) const override {
    const MockVector &v = dynamic_cast<const MockVector &>(x);
    MockVector &r = dynamic_cast<MockVector &>(result);

    r.data[0] = a * v.data[0] + b * v.data[1];
    r.data[1] = c * v.data[0] + d * v.data[1];
  }

  void multiply_transpose(const VectorInterface & /*x*/,
                          VectorInterface & /*result*/) const override {}

  std::size_t rows() const override { return 2; }
  std::size_t cols() const override { return 2; }
};

// === Test Runner ===
int main() {
  MockVector x(2);
  x.data[0] = 3.0;
  x.data[1] = 4.0;

  assert(std::abs(x.norm() - 5.0) < 1e-8);
  x.normalize();
  assert(std::abs(x.norm() - 1.0) < 1e-8);

  MockMatrix A(1.0, 2.0, 0.0, 1.0);
  MockVector y(2);
  A.multiply(x, y);

  std::cout << "A * x = [" << y.data[0] << ", " << y.data[1] << "]"
            << std::endl;
  return 0;
}
