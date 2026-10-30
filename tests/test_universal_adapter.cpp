#include "krans/adapters/universal/StdVectorAdapter.hpp"
#include "krans/adapters/universal/StdVectorFactory.hpp"
#include "krans/adapters/universal/StdMatrixAdapter.hpp"
#include "krans/VectorInterface.hpp"
#include "krans/Solvers/LOBPCGSolver.hpp" 

#include <cassert>
#include <cmath>
#include <iostream>
#include <vector>
#include <numeric> 

int main() {
    const double TOL = 1e-10;
    std::cout << "Running tests for universal adapters...\n";

    // --- End-to-end test with LOBPCGSolver and callback-based StdMatrixAdapter ---
    {
        std::cout << "Testing end-to-end with LOBPCGSolver and StdMatrixAdapter...\n";
        
        // Matrix A = [[1, 0, 1],
        //             [0, 1, 1]] (2x3 matrix)
        // Nullspace should be spanned by [-1, -1, 1]^T
        
        auto mult = [](const std::vector<double>& x, std::vector<double>& y) {
            y[0] = x[0] + x[2];
            y[1] = x[1] + x[2];
        };
        
        auto mult_t = [](const std::vector<double>& x, std::vector<double>& y) {
            y[0] = x[0];
            y[1] = x[1];
            y[2] = x[0] + x[1];
        };

        krans::universal::StdMatrixAdapter A_std(2, 3, mult, mult_t);
        krans::universal::StdVectorFactory vf_e2e(3);

        krans::solvers::LOBPCGOptions opt;
        opt.nev = 1;
        opt.block_size = 4;
        opt.max_iters = 1000;
        opt.tol_rel = TOL;
        opt.verbose = false;

        krans::solvers::LOBPCGResult result = 
            krans::solvers::find_nullspace_lobpcg_ex(A_std, vf_e2e, opt);

        assert(result.right_null.size() == 1);
        
        // Check residual: A * v = 0
        krans::universal::StdVectorAdapter tmp_m(2);
        A_std.multiply(*result.right_null[0], tmp_m);
        assert(tmp_m.norm() < 100 * TOL);

        const auto* res_vec_adapter = dynamic_cast<const krans::universal::StdVectorAdapter*>(result.right_null[0].get());
        const std::vector<double>& v_view = res_vec_adapter->view();
        std::cout << "Found null vector: [" << v_view[0] << ", " << v_view[1] << ", " << v_view[2] << "]\n";

        // Reference check (proportionality)
        double val = v_view[2];
        assert(std::fabs(v_view[0] - (-val)) < 1e-8);
        assert(std::fabs(v_view[1] - (-val)) < 1e-8);
    }
    std::cout << "StdMatrixAdapter tests passed successfully.\n";

    return 0;
}
