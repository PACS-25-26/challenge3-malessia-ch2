# Challenge 3 – A matrix-free parallel solver for the Laplace equation

**Course**: Advanced Programming for Scientific Computing (PACS) – A.Y. 2025-26  
**Author**: *Alessia Meroni*

## 1. Problem description

I solve the Laplace equation with homogeneous Dirichlet boundary conditions
on the unit square:

```
−Δu = f(x,y)   in  Ω = (0,1)²
 u  = g(x,y)   on  ∂Ω
```

The forcing term used is
```
f(x,y) = 8π² sin(2πx) sin(2πy)
```
whose exact solution is `u(x,y) = sin(2πx) sin(2πy)` (with boundary conditions g(x,y) = 0).

---

## 2. Mathematical background

### Discretisation

The domain is discretised with a uniform Cartesian grid of `n × n` points.
The mesh spacing is `h = 1/(n-1)`. The centred finite-difference Laplacian
gives the Jacobi update formula:
```
U^(k+1)(i,j) = 1/4 * [ U^(k)(i-1,j) + U^(k)(i+1,j) + U^(k)(i,j-1) + U^(k)(i,j+1) + h^2 f(x_i, y_j) ]
```

### Convergence criterion

The iteration stops when the discrete L^2 norm of the increment is below
the prescribed tolerance `ε`:
```
e^(k) = sqrt( h * sum_i_j (U^(k+1)(i,j) − U^(k)(i,j))^2 ) < ε
```

### Error measurement

```
||u_h − u||_2 = sqrt( h * sum_i_j (U(i,j) − u(x_i,y_j))^2 )
```
The expected convergence rate is second order in h (O(h^2)).

---

## 3. Parallelisation strategy

### MPI – domain decomposition
The grid is split into contiguous row blocks, one per MPI rank. Before every
Jacobi step, ghost (halo) rows are exchanged with `MPI_Sendrecv`. Convergence
is checked globally with `MPI_Allreduce` (MAX).

### OpenMP – loop parallelism
The inner double loop over (row, column) is parallelised with
`#pragma omp parallel for reduction(+:local_sq_sum) schedule(static) if(use_mpi_)`.

### Serial mode
Passing `--serial` on the command line disables all MPI communication and
OpenMP threading, giving a clean sequential baseline for performance comparison.

---

## 4. Repository structure

```
challenge3/
├── include/
│   └── jacobi_solver.hpp   # class declaration
├── src/
│   ├── jacobi_solver.cpp   # implementation (MPI + OpenMP + serial mode)
│   └── main.cpp            # driver: CLI parsing, solve, export, L^ error
├── test/
│   ├── catch_tests.cpp     # Catch2 correctness test suite
│   ├── run_scalability.cpp # scalability and convergence test driver
│   ├── hw.info             # hardware description
│   ├── README.md           # how to reproduce the test results
│   └── data/               # auto-generated output logs and summary tables
├── Makefile
├── README.md               # this file
├── RESULT.md               # discussion of numerical results
└── .gitignore
```

### File descriptions

File: description
- `include/jacobi_solver.hpp`: declares `JacobiSolver` with full Doxygen comments. The `use_mpi` flag in the constructor controls whether MPI communication is active.
- `src/jacobi_solver.cpp`: implements construction, halo exchange, Jacobi step (OpenMP), convergence check, solution gathering, CSV and VTK export. Every MPI call is guarded by `use_mpi_`.
- `src/main.cpp`: parses CLI arguments (n, max_iter, tolerance, f, g, exact), builds muParser callables, constructs and runs the solver, prints results and writes output files.
- `Makefile`: builds `jacobi_solver` (MPI+OpenMP), `test/catch_tests` (Catch2), and `test/run_scalability`. MPI paths are detected automatically from `mpicxx`.
- `test/catch_tests.cpp`: eight Catch2 tests covering boundary conditions, convergence order, known exact solutions, the trivial zero case, symmetry, and solver state.
- `test/run_scalability.cpp`: launches the solver via `popen()` for serial, 1-, 2-, and 4-process runs. Prints and saves strong-scaling and convergence tables. |
- `RESULT.md`: tables and discussion of convergence order, scaling efficiency, and observed limitations of the plain Jacobi method.

---

## 5. Build instructions

```bash
# Build everything (solver + test runner + Catch2 tests)
make

# Build only the solver
make jacobi_solver

# Build only the Catch2 tests
make catch_tests

# Build only the scalability test driver
make test_runner

# Clean
make clean        # removes executables and object files
make distclean    # also removes CSV/VTK output files
```

---

## 6. Usage

``` 
OMP_NUM_THREADS=<t> mpirun -np <nprocs> ./jacobi_solver [--serial] n max_iter tol f_expr [g_expr [exact_expr]]
```

| Argument | Default | Description |
|----------|---------|-------------|
| `OMP_NUM_THREADS` | system default | OpenMP threads per MPI process |
| `<nprocs>` | — | Number of MPI processes |       
| `--serial` | *(absent)* | Disable MPI communication (sequential baseline) |
| `n` | — | Grid points along each axis (>= 3) |
| `max_iter` | — | Maximum Jacobi iterations |
| `tol` | — | Convergence threshold for the L^2 increment norm |
| `f_expr` | — | Forcing term f(x,y) as a muParser string |
| `g_expr` | `"0"` | Dirichlet BC g(x,y) |
| `exact_expr` | *(absent)* | Exact solution u(x,y); if provided, L² error is printed |

### Examples

```bash
# Parallel, test case with L^2 error:
OMP_NUM_THREADS=2 mpirun -np 4 ./jacobi_solver 64 100000 1e-6 \
    "8*pi^2*sin(2*pi*x)*sin(2*pi*y)" "0" "sin(2*pi*x)*sin(2*pi*y)"

# Serial baseline, same problem:
OMP_NUM_THREADS=1 mpirun -np 1 ./jacobi_solver --serial 64 100000 1e-6 \
    "8*pi^2*sin(2*pi*x)*sin(2*pi*y)" "0" "sin(2*pi*x)*sin(2*pi*y)"

# Hybrid: 2 MPI processes × 4 OpenMP threads:
OMP_NUM_THREADS=4 mpirun -np 2 ./jacobi_solver 128 200000 1e-6 \
    "8*pi^2*sin(2*pi*x)*sin(2*pi*y)" "0" "sin(2*pi*x)*sin(2*pi*y)"
```

---

## 7. Output files

Two files are written by rank 0 after the solver completes:

- `solution_n<n>.csv`: comma-separated `x, y, u` triplets (parallel mode)
- `solution_serial_n<n>.csv`: same, serial mode
- `solution_n<n>.vtk` / `solution_serial_n<n>.vtk`: VTK file, open with ParaView

---

## 8. Tests

### Correctness (Catch2)

```bash
make catch_tests
./test/catch_tests              # run all 8 tests
./test/catch_tests "[bc]"    # only boundary condition tests
./test/catch_tests "[convergence]"
./test/catch_tests "[exactsol]"
```

### Scalability and convergence

```bash
make test_runner
./test/run_scalability
```

See `test/README.md` for details.

---

## 9. Cluster (PoliMi HPC)

### Environment

Tests were also run on the PoliMi HPC cluster (Intel Cascade Lake nodes,
112 logical CPUs, 503 GB RAM per node).
The software environment uses Spack-managed packages:

```bash
source /software/spack/share/spack/setup-env.sh
spack load gcc@15.2.0
spack load openmpi@5.0.8
```

### Why Catch2 is not available on the cluster

Catch2 is a header-only testing library that must be installed system-wide
or manually. It is not available on the cluster and cannot be installed
without administrator privileges. The correctness tests (`catch_tests`) were
therefore run only on the development machine; the cluster is used
exclusively for performance and scalability testing.

### Build instructions on the cluster

The cluster uses OpenMPI 5.x (via Spack), which has different library paths
from the system OpenMPI on WSL2. The Makefile requires the MPI paths to be
overridden at build time:

```bash
make jacobi_solver test_runner \
    MPI_INC=/software/spack-v1.0/opt/spack/linux-cascadelake/openmpi-5.0.8-anjjgjce24doaxseqzzzhmfohoog7tpk/include \
    MPI_LIB=/software/spack-v1.0/opt/spack/linux-cascadelake/openmpi-5.0.8-anjjgjce24doaxseqzzzhmfohoog7tpk/lib \
    MPI_LIBS="-lmpi"
```

Note: OpenMPI 5.x dropped the separate `libmpi_cxx` library (the C++
bindings were removed), so `-lmpi_cxx` is replaced by `-lmpi` alone.

Once compiled, the solver and scalability driver are launched exactly as on
the local machine, for example:

```bash
OMP_NUM_THREADS=2 mpirun -np 4 ./jacobi_solver 64 200000 1e-6 \
    "8*pi^2*sin(2*pi*x)*sin(2*pi*y)" "0" "sin(2*pi*x)*sin(2*pi*y)"

./test/run_scalability
```

---

## 10. GitHub workflow

Files committed: `src/`, `test/` (sources only), `Makefile`, `README.md`,
`RESULT.md`, `.gitignore`.

Files excluded by `.gitignore`: executables, `obj/`, `*.csv`, `*.vtk`,
log files in `test/data/`.