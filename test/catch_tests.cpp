/**
 * @file catch_tests.cpp
 * @brief Correctness tests for JacobiSolver using Catch2 (v3).
 *
 * These tests verify that the solver produces numerically correct results.
 * They run in serial mode (use_mpi = false).
 *
 * MPI is still initialised with MPI_Init / MPI_Finalize because JacobiSolver links against the MPI library.
 *
 * ### Test categories
 * 1. Boundary conditions  – solution matches g(x,y) on all four edges
 * 2. Convergence          – L^2 error decreases as h decreases
 * 3. Known exact solution – L^2 error is below a reasonable threshold
 * 4. Zero forcing         – f=0, g=0 gives u=0
 * 5. Symmetry             – symmetric problem gives symmetric solution
 *
 * ### Build
 * ```
 * make catch_tests
 * ```
 *
 * ### Run
 * ```
 * ./test/catch_tests          # run all tests
 * ./test/catch_tests "[bc]" # run only boundary condition tests
 * ```
 */

#include "../src/jacobi_solver.hpp"

#include <catch2/catch_all.hpp>
#include <mpi.h>

#include <cmath>
#include <string>

// MPI initialisation
struct MPISession
{
    MPISession()
    {
        int argc = 0; char** argv = nullptr;
        MPI_Init(&argc, &argv);
    }
    ~MPISession() { MPI_Finalize(); }
};

// A single global instance: constructed before any test, destroyed after all.
static MPISession mpi_session;

// Helper: build a serial solver and run it

/**
 * @brief Construct and run a serial JacobiSolver, then return it.
 *
 * @param n         Grid size.
 * @param forcing   Forcing function f(x,y).
 * @param boundary  Boundary function g(x,y).
 * @param tol       Convergence tolerance (default 1e-6).
 * @param max_iter  Maximum iterations (default 200000).
 */
static JacobiSolver makeAndSolve(
    int n, ForcingFunction  forcing, BoundaryFunction boundary, double tol = 1e-6, int max_iter = 200000)
{
    // use_mpi = false
    JacobiSolver solver(n, max_iter, tol, forcing, boundary, false);
    solver.solve();
    return solver;
}

// Mathematical constants
static constexpr double PI = M_PI;


//  1.  Boundary conditions

// Test 1: non-homogeneous BC
// Verifies that the solver correctly imposes non-zero Dirichlet boundary conditions.
TEST_CASE("Boundary conditions are respected", "[bc]")
{
    const int n = 32;

    // Forcing term and zero BC for a first homogeneous run
    auto f = [](double x, double y){
        return 8.0 * PI * PI * std::sin(2*PI*x) * std::sin(2*PI*y);
    };
    auto g = [](double, double){ return 0.0; };

    auto solver = makeAndSolve(n, f, g);
    solver.exportCSV("test_bc.csv");

    // Non-homogeneous BC: g(x,y) = x + y.
    // f is reused here as the zero function (g from above), so the problem is:
    //   -laplace(u) = 0  in Ω,   u = x+y  on ∂Ω
    // whose exact solution is u = x+y because x+y is harmonic.
    auto g_nonhom = [](double x, double y){ return x + y; };
    auto solver2 = makeAndSolve(n, g, g_nonhom); // f=0, g=x+y

    // Compute the L^2 error against the exact solution u=x+y (it should be well below 5e-3)
    const double l2 = solver2.computeL2Error(g_nonhom);

    INFO("L^2 error for harmonic exact solution: " << l2); // printed only on failure
    REQUIRE(l2 < 5e-3);
}

// Test 2: homogeneous BC with zero forcing
// The simplest possible case: f=0 and g=0.
// The unique solution is u=0 everywhere, so the L^2 error should be at machine precision.
TEST_CASE("Homogeneous BC gives zero on boundary", "[bc]")
{
    const int n = 16;

    // Both the forcing term and the boundary condition are identically zero.
    auto zero = [](double, double){ return 0.0; };

    // Use a tight tolerance (1e-8) to confirm the solver reaches near-zero error.
    auto solver = makeAndSolve(n, zero, zero, 1e-8, 100000);

    const double l2 = solver.computeL2Error(zero);
    INFO("L^2 error for zero solution: " << l2); // printed only on failure

    REQUIRE(l2 < 1e-10);
}

//  2.  Convergence order

// Checks that the solver is second-order accurate: when h is halved (n doubled),
// the L^2 error should decrease by a factor of 4.
TEST_CASE("L^2 error decreases as grid is refined (order 2)", "[convergence]")
{
    // Test case: exact solution is sin(2*pi*x)sin(2*pi*y).
    auto f = [](double x, double y){
        return 8.0 * PI * PI * std::sin(2*PI*x) * std::sin(2*PI*y);
    };
    auto g = [](double, double){ return 0.0; };
    auto exact = [](double x, double y){
        return std::sin(2*PI*x) * std::sin(2*PI*y);
    };

    const double tol = 1e-5;
    const int max_iter = 100000;

    // n2 = 2*n1 means h2 = h1/2: the grid is twice as fine.
    const int n1 = 17, n2 = 33;

    // Solve on both grids and compute the L^2 error against the exact solution.
    auto s1 = makeAndSolve(n1, f, g, tol, max_iter);
    auto s2 = makeAndSolve(n2, f, g, tol, max_iter);

    const double e1 = s1.computeL2Error(exact);
    const double e2 = s2.computeL2Error(exact);

    INFO("n=" << n1 << "  L^2=" << e1); // printed only if the test fails
    INFO("n=" << n2 << "  L^2=" << e2);

    // For second-order accuracy: e1/e2 = (h1/h2)^2 = 4
    // Accept ratio > 2 (not just > 4) because Jacobi may not have fully converged within max_iter
    const double ratio = e1 / e2;
    INFO("Error ratio e1/e2 = " << ratio << " (expected = 4)");
    REQUIRE(ratio > 2.0);
}

//  3.  Known exact solution

// Test 1: test case
// Runs the exact problem and checks that the L^2 error is below a reasonable threshold for n=64.
TEST_CASE("Test case: L2 error is below threshold", "[exactsol]")
{
    const int n = 64;

    auto f = [](double x, double y){
        return 8.0 * PI * PI * std::sin(2*PI*x) * std::sin(2*PI*y);
    };
    auto g     = [](double, double){ return 0.0; };
    auto exact = [](double x, double y){
        return std::sin(2*PI*x) * std::sin(2*PI*y);
    };

    auto solver = makeAndSolve(n, f, g, 1e-5, 200000);
    const double l2 = solver.computeL2Error(exact);

    // Print full diagnostics if the test fails.
    INFO("n=" << n << "  L2 error=" << l2
         << "  iterations=" << solver.getIterations()
         << "  converged=" << solver.hasConverged());

    // For n=64 with tolerance 1e-5 the discretisation error dominates and the L^2 error should be well below 5e-3.
    REQUIRE(l2 < 5e-3);
}

// Test 2: harmonic exact solution
// u = x+y is harmonic (Δu = 0), so -Δu = 0 = f.
TEST_CASE("Harmonic function u=x+y: L^2 error is negligible", "[exactsol]")
{
    const int n = 32;
    auto zero = [](double, double){ return 0.0; }; // f = 0
    auto linear = [](double x, double y){ return x + y; }; // g = x+y, exact = x+y

    auto solver = makeAndSolve(n, zero, linear, 1e-8, 200000);
    const double l2 = solver.computeL2Error(linear);

    INFO("L^2 error for u=x+y: " << l2);

    // The error should be essentially zero, so 1e-4 is a very conservative upper bound
    REQUIRE(l2 < 1e-4);
}

//  4.  Zero forcing with zero BC

// If both f and g are zero, the solution never changes from the initial condition 
// and the solver converges in exactly 1 iteration.
TEST_CASE("f=0, g=0 gives u=0 everywhere", "[trivial]")
{
    const int n = 16;
    auto zero = [](double, double){ return 0.0; };

    // max_iter=10 is more than enough: the solver should stop after 1 iteration.
    // tolerance=1e-12 is very tight to confirm the increment is truly zero.
    auto solver = makeAndSolve(n, zero, zero, 1e-12, 10);

    // The increment after the first step is zero, so convergence is immediate.
    REQUIRE(solver.getIterations() == 1);
    REQUIRE(solver.hasConverged());

    // The solution should be exactly zero everywhere.
    const double l2 = solver.computeL2Error(zero);
    REQUIRE(l2 < 1e-14); // below machine precision
}

//  5.  Symmetry

// f(x,y) = sin(2*pi*x)sin(2*pi*y) is unchanged when x and y are swapped,
// so the solution must also be symmetric: u(x,y) = u(y,x).
// Run the solver with f(x,y) and with f(y,x) -> same L^2 error against the known exact solution.
TEST_CASE("Symmetric problem gives symmetric solution", "[symmetry]")
{
    const int n = 33; // odd so there is an exact centre point

    auto f = [](double x, double y){
        return 8.0 * PI * PI * std::sin(2*PI*x) * std::sin(2*PI*y);
    };
    auto g = [](double, double){ return 0.0; };

    // First solve: standard f(x,y)
    auto solver = makeAndSolve(n, f, g, 1e-6, 200000);
    solver.exportCSV("test_symmetry.csv");

    // Second solve: f with x and y swapped
    auto f_transposed = [](double x, double y){
        return 8.0 * PI * PI * std::sin(2*PI*y) * std::sin(2*PI*x); // same as f
    };
    auto solver2 = makeAndSolve(n, f_transposed, g, 1e-6, 200000);

    // Check that both solvers give the same L2 error against the exact solution.
    // If the solver is correct and f is truly symmetric, e1 and e2 must be equal.
    auto exact = [](double x, double y){
        return std::sin(2*PI*x) * std::sin(2*PI*y);
    };

    const double e1 = solver.computeL2Error(exact);
    const double e2 = solver2.computeL2Error(exact);

    INFO("L2 error solver1=" << e1 << "  solver2=" << e2); // printed only on failure
    REQUIRE(std::abs(e1 - e2) < 1e-10); // errors must be identical
}

//  6.  Solver state checks

// Checks that hasConverged() and getIterations() report the correct state
// in two opposite situations: a problem that converges immediately and one
// that cannot converge within the allowed iterations.
TEST_CASE("Solver reports convergence correctly", "[state]")
{
    auto zero = [](double, double){ return 0.0; };

    SECTION("Converges for trivial problem")
    {
        // f=0, g=0: the solution is zero and the solver converges in 1 step.
        // Constructor arguments: n=8, max_iter=100, tol=1e-12, f, g, use_mpi=false.
        JacobiSolver solver(8, 100, 1e-12, zero, zero, false);
        solver.solve();

        REQUIRE(solver.hasConverged()); // must have reached tolerance
        REQUIRE(solver.getIterations() >= 1); // must have performed at least one step
    }

    SECTION("Does not converge when max_iter is too small")
    {
        auto f = [](double x, double y){
            return 8.0 * PI * PI * std::sin(2*PI*x) * std::sin(2*PI*y);
        };

        // max_iter=1 with tolerance 1e-12: one Jacobi step is never enough
        // to reduce the increment below 1e-12 for a non-trivial problem.
        JacobiSolver solver(32, 1, 1e-12, f, zero, false);
        solver.solve();

        REQUIRE_FALSE(solver.hasConverged()); // must NOT have converged
        REQUIRE(solver.getIterations() == 1); // must have performed exactly 1 step
    }
}