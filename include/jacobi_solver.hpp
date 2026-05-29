#ifndef JACOBI_SOLVER_HPP
#define JACOBI_SOLVER_HPP

#include <vector>
#include <functional>
#include <string>
#include <cmath>

/**
 * @file jacobi_solver.hpp
 * @brief Declaration of the JacobiSolver class for the serial and parallel solution of the Laplace equation.
 *
 * The solver implements a matrix-free Jacobi iteration on a uniform
 * Cartesian grid of size n×n over the unit square Ω = (0,1)^2.
 * The domain rows are distributed among MPI ranks.
 * OpenMP threads further parallelize the per-rank inner loops.
 *
 * Passing `use_mpi = false` at construction disables every MPI call, so the
 * same class and the same main executable can be used for both the parallel
 * benchmark and the sequential baseline.
 */

// Type aliases

/// 2-D forcing function  f(x, y)
using ForcingFunction  = std::function<double(double, double)>;

/// Dirichlet boundary-condition function  g(x, y)
using BoundaryFunction = std::function<double(double, double)>;

//JacobiSolver

/**
 * @class JacobiSolver
 * @brief Serial or parallel (MPI + OpenMP) Jacobi solver for the Laplace equation
 *        with homogeneous or non-homogeneous Dirichlet boundary conditions.
 *
 ** ### Execution modes
 * `use_mpi`,  Behaviour
 * `true`,     Full MPI domain decomposition + OpenMP inner loops.
 * `false`,    Single-process, no MPI calls, no OpenMP.               |
 *
 * ### Algorithm outline (parallel)
 * 1. The n×n grid is partitioned into contiguous row blocks, one per MPI rank.
 *    Ranks with a smaller index receive one extra row when n−2 is not exactly
 *    divisible by the number of processes (load balancing).
 * 2. Each rank owns `local_rows` interior rows plus two *ghost rows* (halo),
 *    one above and one below, which hold values from neighbouring ranks.
 * 3. At every Jacobi step the halo rows are exchanged with
 *    MPI_Sendrecv before the local update is applied.
 * 4. OpenMP parallelises the inner loops over the local rows and columns.
 * 5. Convergence is checked locally; an MPI_Allreduce (MAX) then broadcasts
 *    the global convergence flag to all ranks.
 * 6. Rank 0 gathers the full solution, writes a CSV file and a VTK file.
 *
 * ### Serial mode (use_mpi = false)
 * - The whole n×n grid lives in a single flat vector on the one process.
 * - No ghost rows, no communication, no OpenMP directives active.
 */
class JacobiSolver
{
    public:
        // Construction

        /**
        * @brief Construct a JacobiSolver.
        *
        * @param n           Number of grid points along each axis (including
        *                    boundary nodes). The interior grid has (n-2)^2 unknowns.
        * @param max_iter    Maximum number of Jacobi iterations allowed.
        * @param tolerance   Stopping threshold for the discrete L^2 norm.
        * @param forcing     Forcing function  f(x,y).
        * @param boundary    Dirichlet boundary function  g(x,y).
        *                    Defaults to the zero function (homogeneous BC).
        * @param use_mpi     If true, use MPI+OpenMP parallelism.
        *                    If false, run as a plain sequential solver (no MPI calls).
        */
        JacobiSolver(int n,
                    int max_iter,
                    double tolerance,
                    ForcingFunction  forcing,
                    BoundaryFunction boundary = [](double, double){ return 0.0; },
                    bool use_mpi = true);

        // Public interface

        /**
        * @brief Run the Jacobi iteration until convergence or max_iter is reached.
        *
        * Must be called after construction and before any output routine.
        * The method is collective: all MPI ranks must call it together.
        */
        void solve();

        /**
        * @brief Compute and return the discrete L^2 error against an exact solution.
        *
        * @param exact  Known exact solution  u(x,y).
        * @return       sqrt( h * sum_i,j (U(i,j) − exact(x_i,y_j))^2 )
        *
        * The result is meaningful only on rank 0 after solve() has been called.
        * On other ranks the return value is 0.
        * When use_mpi = false, always valid after solve().
        */
        double computeL2Error(const std::function<double(double,double)>& exact) const;

        /**
        * @brief Export the global solution (gathered on rank 0) to a CSV file.
        *
        * @param filename  Output file path (written only by rank 0).
        */
        void exportCSV(const std::string& filename) const;

        /**
        * @brief Export the global solution to a VTK file readable by ParaView.
        *
        * @param filename  Output file path (written only by rank 0).
        */
        void exportVTK(const std::string& filename) const;

        /**
        * @brief Return the number of iterations actually performed.
        */
        int getIterations() const { return iterations_performed_; }

        /**
        * @brief Return true if the solver converged before hitting max_iter.
        */
        bool hasConverged() const { return converged_; }

    private:
        // Grid parameters
        int    n_;          ///< Total number of grid points per axis
        double h_;          ///< Mesh spacing  h = 1/(n-1)
        int    max_iter_;   ///< Maximum iterations
        double tol_;        ///< Convergence tolerance
        bool   use_mpi_;    ///< Whether MPI communication is active

        // Problem functions
        ForcingFunction  f_;  ///< Forcing term
        BoundaryFunction g_;  ///< Dirichlet boundary data

        // MPI variables
        int rank_;             ///< MPI rank of this process (0 in serial mode)
        int size_;             ///< Total number of MPI processes (1 in serial mode)
        int local_rows_;       ///< Number of interior rows owned by this rank
        int global_row_start_; ///< Global row index of the first owned interior row

        // Local data 
        /// Parallel mode: (local_rows_ + 2) × n_ patch, row-major.
        ///   Row 0: top ghost row.
        ///   Rows 1,...,local_rows_: interior rows owned by this rank.
        ///   Row local_rows_+1: bottom ghost row.
        ///   Serial mode: full n_ × n_ grid (no ghost rows).
        std::vector<double> U_local_;   ///< Current solution patch
        std::vector<double> U_new_;     ///< Scratch buffer for the updated patch

        // Global solution (rank 0 only)
        std::vector<double> U_global_;  ///< Full n×n solution (valid on rank 0)

        // Solver state
        int  iterations_performed_; ///< Iterations actually performed
        bool converged_;            ///< True if tolerance was reached

        // Private helpers

        /**
        * @brief Initialise the local patch (or full grid if serial) with boundary values and zeros.
        */
        void initLocalPatch();

        /**
        * @brief Exchange halo rows with neighbouring MPI ranks using MPI_Sendrecv (parallel only).
        */
        void exchangeHalos();

        /**
        * @brief Perform one Jacobi update step on the local interior rows.
        * @return Local L^2 increment  sqrt( h * sum(U_new − U_old)^2 )  over owned rows.
        */
        double localJacobiStep();

        /**
        * @brief Gather the distributed solution onto rank 0 (parallel only).
        */
        void gatherSolution();

        /**
        * @brief Map flat local index (row_in_patch, col) -> pointer into U_local_.
        * @param r  Row index in the patch (0 = top ghost, 1,...,local_rows_ = interior).
        * @param c  Column index  0,...,n_-1.
        */
        inline double& at(std::vector<double>& v, int r, int c) const
        {
            return v[static_cast<std::size_t>(r) * n_ + c];
        }
        inline double at(const std::vector<double>& v, int r, int c) const
        {
            return v[static_cast<std::size_t>(r) * n_ + c];
        }
};

#endif // JACOBI_SOLVER_HPP