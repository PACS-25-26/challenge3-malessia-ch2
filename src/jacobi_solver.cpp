/**
 * @file jacobi_solver.cpp
 * @brief Implementation of the parallel and serial Jacobi solver for the Laplace equation.
 *
 * Every function checks `use_mpi_` before issuing any MPI call, so the same
 * code path compiles and runs correctly whether MPI is active or not.
 *
 * Parallelism strategy
 * - **MPI (inter-node / inter-process)**: the n×n grid is decomposed into
 *   contiguous row blocks.  Each rank owns `local_rows_` interior rows.
 *   Before every Jacobi step the top/bottom ghost rows are exchanged with
 *   the neighbouring ranks via MPI_Sendrecv.
 * - **OpenMP (intra-process)**: the loops over local rows and columns are
 *   parallelised with `#pragma omp parallel for` + `reduction` clauses.
 *
 * ### Memory layout
 * **Parallel mode**: U_local_ has shape (local_rows_ + 2) × n_:
 *   row 0: top ghost row  (from rank above, or top BC)
 *   rows 1,...,local_rows_: interior rows owned by this rank
 *   row local_rows_+1: bottom ghost row (from rank below, or bottom BC)
 *
 * **Serial mode**: U_local_ has shape n_ × n_:
 *   rows 0 and n_-1: top and bottom boundary rows
 *   rows 1,...,n_-2: interior rows
 *   No ghost rows needed because there is no neighbouring rank.
 *
 */

#include "../include/jacobi_solver.hpp"

#include <mpi.h>
#include <omp.h>

#include <algorithm>   // std::min
#include <cmath>       // std::sqrt
#include <fstream>     // std::ofstream (CSV and VTK file output)
#include <iostream>    // std::cout, std::cerr
#include <stdexcept>   // std::invalid_argument, std::runtime_error

//  Constructor
JacobiSolver::JacobiSolver(int n,
                           int max_iter,
                           double tolerance,
                           ForcingFunction  forcing,
                           BoundaryFunction boundary,
                           bool use_mpi)
    : n_(n), h_(1.0 / (n - 1)), max_iter_(max_iter), tol_(tolerance), use_mpi_(use_mpi),
      f_(std::move(forcing)), g_(std::move(boundary)), rank_(0), size_(1),
      local_rows_(0), global_row_start_(0), iterations_performed_(0), converged_(false)
{
    if (n_ < 3)
        throw std::invalid_argument("n must be at least 3 (need at least one interior point).");

    if (use_mpi_)
    {
        MPI_Comm_rank(MPI_COMM_WORLD, &rank_);
        MPI_Comm_size(MPI_COMM_WORLD, &size_);

        // Row distribution (load balancing)
        // There are (n_-2) interior rows to distribute among `size_` ranks.
        // Ranks 0,...,remainder-1 get one extra row.
        const int interior_rows = n_ - 2;
        const int base = interior_rows / size_;
        const int remainder = interior_rows % size_;

        local_rows_ = base + (rank_ < remainder ? 1 : 0);
        global_row_start_ = rank_ * base + std::min(rank_, remainder) + 1;
        // +1 because row indices 0 and n_-1 are boundary rows

        // Allocate local patch + scratch buffer 
        const std::size_t patch_size = static_cast<std::size_t>(local_rows_ + 2) * n_;
        U_local_.assign(patch_size, 0.0);
        U_new_.assign(patch_size, 0.0);
    }
     else
    {
        // Serial mode: the whole grid lives on one process
        // rank_ and size_ stay at their initialised values (0 and 1).
        // local_rows_ covers all interior rows; global_row_start_ is 1.
        local_rows_ = n_ - 2;
        global_row_start_ = 1;
 
        const std::size_t full_size = static_cast<std::size_t>(n_) * n_;
        U_local_.assign(full_size, 0.0);
        U_new_.assign(full_size, 0.0);
    }

    initLocalPatch();
}


//  initLocalPatch
void JacobiSolver::initLocalPatch()
{
    // Interior nodes -> 0 (already done by assign)
    // Left and right boundary columns of every row owned by this rank:
    for (int r = 1; r <= local_rows_; ++r)
    {
        // I skip r=0 and r=local_rows_+1 because those are ghost rows,
        // not owned by this rank (except for ranks 0 and size-1, handled below)
        const int global_row = global_row_start_ + (r - 1); // convert local row index to global row index
        const double y = global_row * h_; // physical y-coordinate of this row

        // Set the leftmost and rightmost columns to the Dirichlet boundary value g(x,y).
        // These columns correspond to x=0 and x=1 and never change during the iteration
        at(U_local_, r, 0) = g_(0.0, y); // left boundary  (x=0)
        at(U_local_, r, n_ - 1) = g_(1.0, y); // right boundary (x=1)
    }

    // Only rank 0 owns the bottom boundary of the domain (y = 0, global row index 0)
    // if serial: always fill row 0
    if (!use_mpi_ || rank_ == 0)
    {
        for (int c = 0; c < n_; ++c)
        {
            const double x = c * h_; // physical x-coordinate of this column
            at(U_local_, 0, c) = g_(x, 0.0); // y = 0  (bottom of domain, row 0)
        }
    }

    // Only the last rank owns the top boundary of the domain (y = 1)
    // if serial: always fill row n_-1
    if (!use_mpi_ || rank_ == size_ - 1)
    {
        for (int c = 0; c < n_; ++c)
        {
            const double x = c * h_; // physical x-coordinate of this column
            at(U_local_, local_rows_ + 1, c) = g_(x, 1.0); // y = 1  (top of domain)
        }
    }

    // Initialise U_new_ with the same values
    U_new_ = U_local_;
}


//  exchangeHalos (parallel only)
void JacobiSolver::exchangeHalos()
{
    // In serial mode there are no neighbours to communicate with
    if (!use_mpi_) return;

    const int tag_down = 0; // tag for messages going to higher-rank neighbour
    const int tag_up   = 1; // tag for messages going to lower-rank  neighbour

    // Returns a raw pointer to the first element of row r in U_local_
    // Row r starts at offset r * n_ because the storage is row-major
    auto row_ptr = [&](int r) { return U_local_.data() + static_cast<std::size_t>(r) * n_; };

    MPI_Status status;

    // Every rank except the last sends its bottom interior row to the rank below,
    // and simultaneously receives the top interior row of the rank below into its bottom ghost slot
    if (rank_ < size_ - 1)
    {
        MPI_Sendrecv(
            row_ptr(local_rows_), n_, MPI_DOUBLE, rank_ + 1, tag_down, // SEND: last interior row -> rank+1
            row_ptr(local_rows_ + 1), n_, MPI_DOUBLE, rank_ + 1, tag_up, // RECV: bottom ghost <- rank+1
            MPI_COMM_WORLD, &status);
    }

    // Every rank except the first sends its top interior row to the rank above,
    // and simultaneously receives the bottom interior row of the rank above into its top ghost slot
    if (rank_ > 0)
    {
        MPI_Sendrecv(
            row_ptr(1), n_, MPI_DOUBLE, rank_ - 1, tag_up, // SEND: first interior row -> rank-1
            row_ptr(0), n_, MPI_DOUBLE, rank_ - 1, tag_down, // RECV: top ghost <- rank-1
            MPI_COMM_WORLD, &status);
    }
}


//  localJacobiStep
double JacobiSolver::localJacobiStep()
{
    const double h2 = h_ * h_;
    double local_sq_sum = 0.0; // accumulates the sum of squared increments over all interior points owned by this rank

    // OpenMP is active only when use_mpi_ = true and more than one thread is available
    // Each thread gets a contiguous chunk of rows (schedule(static)).
    // The reduction clause ensures that each thread accumulates its own private copy of local_sq_sum; 
    // the copies are summed together automatically at the end of the parallel region.
    #pragma omp parallel for reduction(+:local_sq_sum) schedule(static) if(use_mpi_)
        for (int r = 1; r <= local_rows_; ++r)
        {
            const int global_row = global_row_start_ + (r - 1); // local row r -> global row index
            const double y = global_row * h_; // global row index -> physical y coordinate

            for (int c = 1; c < n_ - 1; ++c) // skip c=0 and c=n_-1 (left and right boundary columns)
            {
                const double x = c * h_; // column index -> physical x coordinate

                // Jacobi update: new value = average of the four neighbours + source term.
                // The four neighbours are the ghost/interior rows above (r-1) and below (r+1),
                // and the columns to the left (c-1) and right (c+1).
                // All four values are taken from U_local_ (the OLD solution, not yet updated)
                const double new_val =
                    0.25 * (at(U_local_, r - 1, c) +  // neighbour above
                            at(U_local_, r + 1, c) +  // neighbour below
                            at(U_local_, r, c - 1) +  // neighbour left
                            at(U_local_, r, c + 1) +  // neighbour right
                            h2 * f_(x, y));           // source term contribution

                const double diff = new_val - at(U_local_, r, c); // increment at this point
                local_sq_sum += diff * diff; // accumulate squared increment

                at(U_new_, r, c) = new_val; // write result into the scratch buffer
            }
        }

    // Copy updated interior back into U_local_ (boundary cols unchanged)
    for (int r = 1; r <= local_rows_; ++r)
        for (int c = 1; c < n_ - 1; ++c)
            at(U_local_, r, c) = at(U_new_, r, c);

    // Local discrete L^2 increment:  sqrt( h * sum_ij diff^2 )
    // This is only the contribution from the rows owned by this rank;
    // solve() will combine it with the other ranks via MPI_Allreduce
    return std::sqrt(h_ * local_sq_sum);
}


//  solve
void JacobiSolver::solve()
{
    for (int iter = 0; iter < max_iter_; ++iter)
    {
        exchangeHalos(); // refresh ghost rows from neighbouring ranks

        const double local_err = localJacobiStep(); // perform one Jacobi update and compute the local L^2 increment

        // Global convergence check (all ranks must agree in parallel)
        // We reduce with MAX so that convergence is declared only when every rank has a local error below the tolerance.
        double global_err = local_err;
        if (use_mpi_)
            MPI_Allreduce(&local_err, &global_err, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);

        ++iterations_performed_;

        // All ranks see the same global_err, so they all take the same branch
        if (global_err < tol_)
        {
            converged_ = true;
            if (rank_ == 0)
                std::cout << "Converged at iteration " << iter << "  ||err||_2 = " << global_err << std::endl;
            break; // all ranks break together
        }
    }

    // Warn if the solver hit the iteration limit without converging (rank 0 only)
    if (!converged_ && rank_ == 0)
        std::cerr << "Warning: solver did not converge within " << max_iter_ << " iterations.\n";

    gatherSolution(); // collect the distributed solution onto rank 0 for output
}


//  gatherSolution
void JacobiSolver::gatherSolution()
{
    if (!use_mpi_)
    {
        // Serial mode: U_local_ already contains the full n×n grid.
        // Just alias it to U_global_ so that exportCSV/VTK work unchanged.
        U_global_ = U_local_;
        return;
    }

    // Parallel mode: each rank packs its owned interior rows (r=1,...,local_rows_) into a contiguous buffer (no ghost rows) for sending to rank 0.
    // I use MPI_Gatherv so ranks with different local_rows_ are handled.

    // Build send buffer: interior rows only 
    std::vector<double> send_buf(static_cast<std::size_t>(local_rows_) * n_);
    for (int r = 0; r < local_rows_; ++r)
        for (int c = 0; c < n_; ++c)
            send_buf[static_cast<std::size_t>(r) * n_ + c] = at(U_local_, r + 1, c); // r+1 because row 0 in U_local_ is the top ghost row

    // Only rank 0 needs to know how many elements to expect from each sender and where to place them in U_global_
    std::vector<int> recv_counts(size_), displs(size_);
    if (rank_ == 0)
    {
        const int interior_rows = n_ - 2; // total interior rows in the global grid
        const int base = interior_rows / size_; // minimum rows per rank
        const int rem = interior_rows % size_; // ranks 0..rem-1 have one extra row
        int offset = 0;
        for (int p = 0; p < size_; ++p)
        {
            int rows_p = base + (p < rem ? 1 : 0); // actual row count for rank p
            recv_counts[p] = rows_p * n_; // number of doubles expected from rank p
            displs[p] = offset; // offset in the receive buffer where rank p's data will be written
            offset += recv_counts[p];
        }
        // Allocate the full n*n solution buffer on rank 0 and zero-initialise it
        U_global_.assign(static_cast<std::size_t>(n_) * n_, 0.0);

        // Fill boundary rows with g_ values
        for (int c = 0; c < n_; ++c)
        {
            const double x = c * h_; 
            U_global_[c] = g_(x, 0.0); // row 0
            U_global_[static_cast<std::size_t>(n_ - 1) * n_ + c]   = g_(x, 1.0); // row n-1
        }
        // Fill boundary columns with g_ values
        for (int r = 0; r < n_; ++r)
        {
            const double y = r * h_;
            U_global_[static_cast<std::size_t>(r) * n_] = g_(0.0, y); // col 0
            U_global_[static_cast<std::size_t>(r) * n_ + (n_ - 1)] = g_(1.0, y); // col n-1
        }
    }

    // Gather interior rows into U_global_ on rank 0starting at offset n_ (skip row 0)
    MPI_Gatherv(send_buf.data(),                                // send buffer (all ranks)
                static_cast<int>(send_buf.size()),              // number of elements to send
                MPI_DOUBLE,                                     // send type
                rank_ == 0 ? U_global_.data() + n_ : nullptr,   // receive buffer (rank 0 only; +n_ skips row 0)
                rank_ == 0 ? recv_counts.data() : nullptr,      // how many elements from each rank
                rank_ == 0 ? displs.data() : nullptr,           // where to place each rank's data
                MPI_DOUBLE,                                     // receive type
                0,                                              // root rank  
                MPI_COMM_WORLD);
}


//  computeL2Error
double JacobiSolver::computeL2Error(const std::function<double(double,double)>& exact) const
{
    // U_global_ is only populated on rank 0, all other ranks have nothing to compute and return a dummy value
    if (use_mpi_ && rank_ != 0) return 0.0;

    // accumulates the sum of squared pointwise errors over all n*n nodes
    double sq_sum = 0.0;
    for (int i = 0; i < n_; ++i) // i iterates over rows
    {
        const double y = i * h_; // row index -> physical y coordinate
        for (int j = 0; j < n_; ++j) // j iterates over columns
        {
            const double x  = j * h_; // column index -> physical x coordinate
            // Pointwise error: difference between the numerical solution stored in U_global_ 
            // and the exact solution evaluated at the same physical point
            const double diff = U_global_[static_cast<std::size_t>(i) * n_ + j] - exact(x, y); // row-major: (i,j) -> i*n_ + j
            sq_sum += diff * diff; // accumulate squared error
        }
    }
    return std::sqrt(h_ * sq_sum);
}


//  exportCSV
void JacobiSolver::exportCSV(const std::string& filename) const
{
    // U_global_ is only available on rank 0; all other ranks skip this function
    if (use_mpi_ && rank_ != 0) return;

    std::ofstream ofs(filename); // open the file for writing (creates it if it does not exist)
    if (!ofs) throw std::runtime_error("Cannot open " + filename); // fail early with a clear message if the path is invalid or not writable

    ofs << "x,y,u\n"; // CSV header row: three columns, physical coordinates and solution value
    for (int i = 0; i < n_; ++i) // i iterates over rows
    {
        const double y = i * h_; // row index -> physical y coordinate
        for (int j = 0; j < n_; ++j) // j iterates over columns
        {
            const double x = j * h_; // column index -> physical x coordinate
            // Write one line per grid point: x, y, and the numerical solution value
            ofs << x << ',' << y << ','
                << U_global_[static_cast<std::size_t>(i) * n_ + j] << '\n';
        }
    }
}

//  exportVTK
void JacobiSolver::exportVTK(const std::string& filename) const
{
    // U_global_ is only available on rank 0; all other ranks skip this function
    if (use_mpi_ && rank_ != 0) return;

    // Same as exportCSV: open the file and check for errors
    std::ofstream ofs(filename);
    if (!ofs) throw std::runtime_error("Cannot open " + filename);

    // Header
    ofs << "# vtk DataFile Version 3.0\n"                               // mandatory first line: VTK format version
        << "Jacobi solution of Laplace equation\n"                      // text description
        << "ASCII\n"                                                    // data encoding (ASCII or binary)
        << "DATASET STRUCTURED_POINTS\n"                                // grid type: uniform Cartesian grid
        << "DIMENSIONS " << n_ << " " << n_ << " 1\n"                   // number of points along x, y, z (z=1 for 2D)
        << "ORIGIN 0.0 0.0 0.0\n"                                       // physical coordinates of the first point
        << "SPACING " << h_ << " " << h_ << " 1.0\n"                    // distance between adjacent points
        << "POINT_DATA " << static_cast<long long>(n_) * n_ << "\n"     // total number of points
        << "SCALARS u double 1\n"                                       // declare a scalar field named "u" of type double, 1 component per point
        << "LOOKUP_TABLE default\n";                                    // use the default ParaView colour map

    // Data (x-fastest, then y)
    // VTK structured-points: x varies fastest; my storage is row=y, col=x.
    for (int i = 0; i < n_; ++i)
        for (int j = 0; j < n_; ++j)
            ofs << U_global_[static_cast<std::size_t>(i) * n_ + j] << '\n';
}