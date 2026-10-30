#include "sns/Utils/SubspaceCompare.hpp"
#include "sns/adapters/universal/StdVectorAdapter.hpp"
#include <cassert>
#include <iostream>
#include <vector>
#include <memory>

using namespace sns::universal;
using namespace sns::utils;

int main() {
    std::cout << "Running tests for SubspaceCompare..." << std::endl;

    // Test 1: Identity comparison
    {
        std::cout << "Test 1: Identical bases" << std::endl;
        std::vector<std::unique_ptr<sns::VectorInterface>> basis1;
        basis1.push_back(std::make_unique<StdVectorAdapter>(std::vector<double>{1.0, 0.0, 0.0}));
        basis1.push_back(std::make_unique<StdVectorAdapter>(std::vector<double>{0.0, 1.0, 0.0}));

        std::vector<std::unique_ptr<sns::VectorInterface>> basis2;
        basis2.push_back(std::make_unique<StdVectorAdapter>(std::vector<double>{1.0, 0.0, 0.0}));
        basis2.push_back(std::make_unique<StdVectorAdapter>(std::vector<double>{0.0, 1.0, 0.0}));

        double diff = subspace_diff(basis1, basis2);
        std::cout << "  diff = " << diff << std::endl;
        assert(diff < 1e-12);
    }

    // Test 2: Different bases for the same subspace
    {
        std::cout << "Test 2: Different bases for the same subspace" << std::endl;
        std::vector<std::unique_ptr<sns::VectorInterface>> basis1;
        basis1.push_back(std::make_unique<StdVectorAdapter>(std::vector<double>{1.0, 0.0, 0.0}));
        basis1.push_back(std::make_unique<StdVectorAdapter>(std::vector<double>{0.0, 1.0, 0.0}));

        std::vector<std::unique_ptr<sns::VectorInterface>> basis2;
        // v1' = v1 + v2, v2' = v1 - v2
        basis2.push_back(std::make_unique<StdVectorAdapter>(std::vector<double>{1.0, 1.0, 0.0}));
        basis2.push_back(std::make_unique<StdVectorAdapter>(std::vector<double>{1.0, -1.0, 0.0}));

        double diff = subspace_diff(basis1, basis2);
        std::cout << "  diff = " << diff << std::endl;
        assert(diff < 1e-12);
    }

    // Test 3: Different subspaces
    {
        std::cout << "Test 3: Different subspaces" << std::endl;
        std::vector<std::unique_ptr<sns::VectorInterface>> basis1;
        basis1.push_back(std::make_unique<StdVectorAdapter>(std::vector<double>{1.0, 0.0, 0.0}));
        basis1.push_back(std::make_unique<StdVectorAdapter>(std::vector<double>{0.0, 1.0, 0.0}));

        std::vector<std::unique_ptr<sns::VectorInterface>> basis2;
        basis2.push_back(std::make_unique<StdVectorAdapter>(std::vector<double>{1.0, 0.0, 0.0}));
        basis2.push_back(std::make_unique<StdVectorAdapter>(std::vector<double>{0.0, 0.0, 1.0}));

        double diff = subspace_diff(basis1, basis2);
        std::cout << "  diff = " << diff << std::endl;
        assert(diff > 0.1);
    }

    // Test 4: Higher dimension and scaling
    {
        std::cout << "Test 4: Higher dimension and scaling" << std::endl;
        std::vector<std::unique_ptr<sns::VectorInterface>> basis1;
        basis1.push_back(std::make_unique<StdVectorAdapter>(std::vector<double>{2.0, 0.0, 4.0, 0.0}));
        basis1.push_back(std::make_unique<StdVectorAdapter>(std::vector<double>{0.0, 5.0, 0.0, 1.0}));

        std::vector<std::unique_ptr<sns::VectorInterface>> basis2;
        // Scale vectors
        basis2.push_back(std::make_unique<StdVectorAdapter>(std::vector<double>{1.0, 0.0, 2.0, 0.0}));
        basis2.push_back(std::make_unique<StdVectorAdapter>(std::vector<double>{0.0, 10.0, 0.0, 2.0}));

        double diff = subspace_diff(basis1, basis2);
        std::cout << "  diff = " << diff << std::endl;
        assert(diff < 1e-12);
    }

    // Test 5: Linear dependence in input
    {
        std::cout << "Test 5: Linear dependence in input (both span same plane but have 3 vectors)" << std::endl;
        std::vector<std::unique_ptr<sns::VectorInterface>> basis1;
        basis1.push_back(std::make_unique<StdVectorAdapter>(std::vector<double>{1.0, 0.0, 0.0}));
        basis1.push_back(std::make_unique<StdVectorAdapter>(std::vector<double>{0.0, 1.0, 0.0}));
        basis1.push_back(std::make_unique<StdVectorAdapter>(std::vector<double>{1.0, 1.0, 0.0}));

        std::vector<std::unique_ptr<sns::VectorInterface>> basis2;
        basis2.push_back(std::make_unique<StdVectorAdapter>(std::vector<double>{2.0, 0.0, 0.0}));
        basis2.push_back(std::make_unique<StdVectorAdapter>(std::vector<double>{0.0, 2.0, 0.0}));
        basis2.push_back(std::make_unique<StdVectorAdapter>(std::vector<double>{2.0, 2.0, 0.0}));

        double diff = subspace_diff(basis1, basis2);
        std::cout << "  diff = " << diff << std::endl;
        assert(diff < 1e-12);
    }

    std::cout << "All SubspaceCompare tests passed!" << std::endl;
    return 0;
}
