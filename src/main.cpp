/**
 * @file main.cpp
 * @brief Driver for the Jacobi solver (parallel and serial modes).
 *
 * ### Usage
 * ```
 * # Parallel (MPI + OpenMP):
 * OMP_NUM_THREADS=<t> mpirun -np <nprocs> ./jacobi_solver n max_iter tol f_expr [g_expr [exact_expr]]
 *
 * # Serial (no MPI communication, single process):
 * OMP_NUM_THREADS=1 mpirun -np 1 ./jacobi_solver --serial n max_iter tol f_expr [g_expr [exact_expr]]
 * ```
 *
 * The `--serial` flag disables all MPI communication inside JacobiSolver.
 * MPI is still initialised but no collective operations are performed.
 *
 * Argument,     Example,                           Description
 * - n,          64,                                Grid points along each axis
 * - max_iter,   100000,                            Maximum Jacobi iterations
 * - tolerance,  1e-6,                              Stopping threshold (L^2 increment norm)
 * - f_expr,     "8*pi^2*sin(2*pi*x)*sin(2*pi*y)",  Forcing term  f(x,y)  as a math string
 * - g_expr,     "0",                               Dirichlet BC  g(x,y)  (default: "0")
 * - exact_expr, "sin(2*pi*x)*sin(2*pi*y)",         Exact solution  u(x,y)  (default: "0"), when provided, the discrete L^2 error is computed
 *
 * ### Supported muParser syntax
 * - Variables : x, y
 * - Constants : pi, e
 * - Functions : sin, cos, tan, exp, log, log2, log10, sqrt, abs, ...
 * - Operators : +  -  *  /  ^  
 *
 * ### Example invocations
 * ```bash
 * # Test case:
 * OMP_NUM_THREADS=2 mpirun -np 4 ./jacobi_solver 64 100000 1e-6 \
 *     "8*pi^2*sin(2*pi*x)*sin(2*pi*y)" "0" "sin(2*pi*x)*sin(2*pi*y)"
 *
 * # Serial baseline, same problem:
 * OMP_NUM_THREADS=1 mpirun -np 1 ./jacobi_solver --serial 64 100000 1e-6 \
 *     "8*pi^2*sin(2*pi*x)*sin(2*pi*y)" "0" "sin(2*pi*x)*sin(2*pi*y)"
 *
 * # Generic forcing, no exact solution available:
 * OMP_NUM_THREADS=2 mpirun -np 2 ./jacobi_solver 64 100000 1e-6 "4*x*y" "0"
 *
 * # Non-homogeneous BC, exact solution provided:
 * OMP_NUM_THREADS=2 mpirun -np 2 ./jacobi_solver 64 100000 1e-6 \
 *     "0" "sin(pi*x)*sin(pi*y)" "sin(pi*x)*sin(pi*y)"
 * ```
 *
 * ### Output files (written by rank 0)
 * - `solution_n<n>.csv`  – comma-separated (x, y, u) triplets
 * - `solution_n<n>.vtk`  – VTK file for ParaView
 */

#include "jacobi_solver.hpp"

#include <mpi.h>
#include <omp.h>
#include <muParser.h> // mu::Parser

#include <chrono> // steady_clock: wall-clock timing in serial mode
#include <cmath> // M_PI, M_E
#include <cstdlib> // std::atoi, std::atof
#include <iostream> // std::cout, std::cerr
#include <optional> // std::optional, std::nullopt
#include <string> // std::string
#include <vector> // std::vector

//  muParser helpers

/**
 * @brief Evaluate a muParser expression string at a single point (x,y).
 *
 * Creates a temporary parser, evaluates it once, and returns the result.
 * Used for pre-computing grid values and for validation.
 *
 * @param expr  Math expression string.
 * @param x     x coordinate at which to evaluate the expression.
 * @param y     y coordinate at which to evaluate the expression.
 * @return      Value of expr at (x,y).
 */
static double evalExpr(const std::string& expr, double x, double y)
{
    mu::Parser p;

    // Register x and y
    p.DefineVar("x", &x);
    p.DefineVar("y", &y);

    // Register the two standard mathematical constants
    p.DefineConst("pi", M_PI);
    p.DefineConst("e",  M_E);

    // Give the parser the expression string.
    // SetExpr() parses and compiles the expression; Eval() executes it.
    p.SetExpr(expr);
    return p.Eval();
}

/**
 * @brief Pre-evaluate an expression on the full n×n grid and return a flat vector.
 *
 * It evaluates f(x,y) and g(x,y) once, sequentially, before the
 * solver starts. The results are stored in a flat vector that the solver
 * reads during the Jacobi iteration
 *
 * @param expr  Math expression string.
 * @param n     Grid size.
 * @param h     Mesh spacing h = 1/(n-1).
 * @return      Flat row-major vector of size n*n with expr(x_j, y_i).
 */
static std::vector<double> preEval(const std::string& expr, int n, double h)
{
    std::vector<double> vals(static_cast<std::size_t>(n) * n);

    // Evaluate the expression at every grid point and store in row-major order
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j)
            vals[static_cast<std::size_t>(i) * n + j] = evalExpr(expr, j * h, i * h); // x = j*h, y = i*h
    return vals;
}

/**
 * @brief Check that a math expression is syntactically valid.
 *
 * Evaluates @p expr at (0.5, 0.5) to trigger muParser's internal parser.
 * Called on rank 0 before the solver starts so that syntax errors are
 * caught immediately with a clear message.
 * On error, all MPI processes are terminated via MPI_Abort so no rank hangs 
 * waiting for a collective operation that will never come.
 *
 * @param expr   The expression string to check.
 * @param label  Short name shown in the error message (e.g. "f(x,y)").
 */
static void validateExpr(const std::string& expr, const std::string& label)
{
    try 
    { 
        evalExpr(expr, 0.5, 0.5); 
    }
    catch (const mu::Parser::exception_type& e)
    {
        // Print which expression failed and why, then stop all MPI processes
        std::cerr << "Error in " << label << " expression \""
                  << expr << "\": " << e.GetMsg() << "\n";
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }
}

//  printResults

/**
 * @brief Print L^2 error and write output files (called only on rank 0).
 *
 * Called only on rank 0 after solve() has completed, because U_global_
 * (the full assembled solution) only exists on rank 0.
 *
 * @param solver      Completed solver (solve() already called).
 * @param exact_expr  Optional exact-solution expression string.
 * @param n           Grid size (used to build the output filename).
 * @param serial      True when running in serial mode.
 * @param h           Mesh spacing.
 */
static void printResults(const JacobiSolver& solver, const std::optional<std::string>& exact_expr, int n, bool serial, double h)
{
    if (exact_expr)
    {
        // Pre-evaluate the exact solution on the grid
        std::vector<double> exact_vals = preEval(*exact_expr, n, h);

        // Wrap the pre-computed values in a lambda for computeL2Error().
        // Given a physical point (x,y), recover the grid indices by rounding x/h and y/h to the nearest integer.
        const auto exact_fn = [&](double x, double y) -> double {
            const int j = static_cast<int>(std::round(x / h)); // column index
            const int i = static_cast<int>(std::round(y / h)); // row index
            return exact_vals[static_cast<std::size_t>(i) * n + j];
        };

        const double l2 = solver.computeL2Error(exact_fn);
        std::cout << "  L2 error      : " << l2 << "\n"
                  << "    (||u_h - u_exact||_2 = sqrt(h * sum_ij diff^2))\n";
    }
    else
    {
        // No exact solution was given: print a clear placeholder
        std::cout << "  L2 error      : (exact solution not provided)\n";
    }
    std::cout << "\n";

    // Choose the filename prefix based on the execution mode (never overwrite)
    const std::string prefix = serial ? "solution_serial_n" : "solution_n";
    const std::string base   = prefix + std::to_string(n);
    solver.exportCSV(base + ".csv");
    solver.exportVTK(base + ".vtk");
    std::cout << "Output written to " << base << ".csv and " << base << ".vtk\n";
}

//  main
int main(int argc, char* argv[])
{
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank); // index of this process
    MPI_Comm_size(MPI_COMM_WORLD, &size); // total number of MPI processes

    // Check for --serial flag
    // Scan argv for "--serial" and remove it from the argument list .
    // use_mpi = false disables all MPI communication inside JacobiSolver.
    bool use_mpi = true;
    std::vector<std::string> args;
    for (int i = 1; i < argc; ++i)
    {
        if (std::string(argv[i]) == "--serial")
            use_mpi = false; // found the flag: disable MPI communication
        else
            args.push_back(argv[i]); // keep all other arguments
    }

    // Argument parsing
    // After removing --serial, I expect 4 to 6 positional arguments:
    //   n  max_iter  tol  f_expr  [g_expr  [exact_expr]]
    if (args.size() < 4 || args.size() > 6)
    {
        // Only rank 0 prints the error
        if (rank == 0)
            std::cerr << "Usage: mpirun -np <np> " << argv[0]
                      << " [--serial] n max_iter tol f_expr [g_expr [exact_expr]]\n";
        MPI_Finalize();
        return EXIT_FAILURE;
    }

    const int n = std::atoi(args[0].c_str()); // grid size (string -> int)
    const int max_iter = std::atoi(args[1].c_str()); // max iterations (string -> int)
    const double tol = std::atof(args[2].c_str()); // tolerance (string -> double)
    const double h = 1.0 / (n - 1); // mesh spacing
    const std::string f_expr(args[3]); // forcing term expression
    const std::string g_expr(args.size() >= 5 ? args[4] : "0"); // BC expression; default "0"
    
    // exact_expr is wrapped in std::optional, if absent: std::nullopt
    const std::optional<std::string> exact_expr = (args.size() == 6) ? std::optional<std::string>(args[5]) : std::nullopt;

    // A grid with n < 3 has no interior points so the Jacobi loop would have nothing to update
    if (n < 3)
    {
        if (rank == 0) std::cerr << "Error: n must be >= 3.\n";
        MPI_Finalize();
        return EXIT_FAILURE;
    }

    // Validate expressions (rank 0 only) 
    if (rank == 0)
    {
        validateExpr(f_expr, "f(x,y)");
        validateExpr(g_expr, "g(x,y)");
        if (exact_expr) validateExpr(*exact_expr, "exact(x,y)");
    }
    // Wait until rank 0 has finished validation before any rank proceeds
    MPI_Barrier(MPI_COMM_WORLD);

    // Pre-evaluate f and g on the full grid 
    // This avoids calling muParser inside the solver.
    // Each value f(x_j, y_i) is stored at index i*n+j (row-major).
    const std::vector<double> f_vals = preEval(f_expr, n, h);
    const std::vector<double> g_vals = preEval(g_expr, n, h);

    // Wrap the pre-computed vectors in lambdas that match the ForcingFunction
    // and BoundaryFunction signatures expected by JacobiSolver.
    const ForcingFunction f = [&f_vals, n, h](double x, double y) -> double {
        const int j = static_cast<int>(std::round(x / h));
        const int i = static_cast<int>(std::round(y / h));
        return f_vals[static_cast<std::size_t>(i) * n + j];
    };
    const BoundaryFunction g = [&g_vals, n, h](double x, double y) -> double {
        const int j = static_cast<int>(std::round(x / h));
        const int i = static_cast<int>(std::round(y / h));
        return g_vals[static_cast<std::size_t>(i) * n + j];
    };

    // Print run parameters
    if (rank == 0)
    {
        std::cout << "========================================\n"
                  << (use_mpi ? " Parallel" : " Serial")
                  << " Jacobi solver\n"
                  << "========================================\n";
        if (use_mpi)
            std::cout << "  MPI processes : " << size << "\n"
                      << "  OMP threads   : " << omp_get_max_threads() << "\n";
        std::cout << "  n             : " << n << "\n"
                  << "  h             : " << h << "\n"
                  << "  max_iter      : " << max_iter << "\n"
                  << "  tolerance     : " << tol << "\n"
                  << "  f(x,y)        : " << f_expr << "\n"
                  << "  g(x,y)        : " << g_expr << "\n"
                  << "  exact u(x,y)  : " << (exact_expr ? *exact_expr : "(not provided)") << "\n"
                  << "\n";
    }

    // Build and run the solver
    JacobiSolver solver(n, max_iter, tol, f, g, use_mpi);

    // Use MPI_Wtime in parallel mode (already initialised) and std::chrono in serial mode.
    double wall_time = 0.0;
    if (use_mpi)
    {
        const double t0 = MPI_Wtime();
        solver.solve();
        wall_time = MPI_Wtime() - t0;
    }
    else
    {
        const auto t0 = std::chrono::steady_clock::now();
        solver.solve();
        wall_time = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    }

    // Post-processing (rank 0 only)
    if (rank == 0)
    {
        std::cout << "  Iterations    : " << solver.getIterations() << "\n"
                  << "  Converged     : " << (solver.hasConverged() ? "yes" : "no") << "\n"
                  << "  Wall time     : " << wall_time << " s\n";
        printResults(solver, exact_expr, n, !use_mpi, h);
    }

    MPI_Finalize();
    return EXIT_SUCCESS;
}