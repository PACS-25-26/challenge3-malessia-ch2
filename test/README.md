# Test folder

## Contents

```
test/
├── catch_tests.cpp      # Catch2 correctness tests (8 test cases)
├── run_scalability.cpp  # scalability and convergence test driver
├── hw.info              # hardware description
├── data/                # auto-generated logs and summary tables
└── README.md            # this file
```

---

## How to build and run

From the **project root**:

```bash
# Build everything (solver + both test targets)
make

# Correctness tests (Catch2)
./test/catch_tests                     # run all 8 tests
./test/catch_tests "[bc]"              # boundary condition tests only
./test/catch_tests "[convergence]"     # convergence order test only
./test/catch_tests "[exactsol]"        # known exact solution tests
./test/catch_tests "[trivial]"         # zero forcing / zero BC test
./test/catch_tests "[symmetry]"        # symmetry test
./test/catch_tests "[state]"           # solver state reporting tests

# Scalability and convergence driver
./test/run_scalability
```

---

## Catch2 correctness tests (`catch_tests.cpp`)

All tests run in **serial mode** (`use_mpi = false`).
MPI is still initialised because `JacobiSolver` links against the MPI library,
but no collective operations are performed.

### `[bc]` – Boundary conditions

| Test case | What it checks |
|-----------|----------------|
| *"Boundary conditions are respected"* | Solves with a non-homogeneous BC `g(x,y) = x+y`. The exact solution is `u = x+y` (harmonic, `Δu = 0`). Asserts L^2 error < 5 × 10^(-3). |
| *"Homogeneous BC gives zero on boundary"* | Sets `f = 0`, `g = 0`. The unique solution is `u = 0`. Asserts L^2 error < 10^(-10) with tolerance 10^-8. |

### `[convergence]` – Second-order accuracy

| Test case | What it checks |
|-----------|----------------|
| *"L^2 error decreases as grid is refined (order 2)"* | Solves the test case (`f = 8π² sin(2πx)sin(2πy)`, exact `u = sin(2πx)sin(2πy)`) on grids `n = 17` and `n = 33` (h halved). Asserts that the L^2 error ratio `e(n=17) / e(n=33)` is greater than 2. For a second-order method the expected ratio is 4; the threshold is relaxed to 2 because Jacobi may not have fully converged within `max_iter`. |

### `[exactsol]` – Known exact solution

| Test case | What it checks |
|-----------|----------------|
| *"Test case: L^2 error is below threshold"* | Runs the standard test case on a 64×64 grid with tolerance 10^(-5) and up to 200 000 iterations. Asserts L^2 error < 5 × 10^(-3). |
| *"Harmonic function u=x+y: L^2 error is negligible"* | Solves `−Δu = 0` with `g = x+y` on a 32×32 grid. The exact solution is `u = x+y`. Asserts L^2 error < 10^(-4) (the error should be essentially at machine precision). |

### `[trivial]` – Zero forcing, zero BC

| Test case | What it checks |
|-----------|----------------|
| *"f=0, g=0 gives u=0 everywhere"* | The initial condition is already the exact solution, so the increment after the first Jacobi step is zero. Asserts that the solver converges in **exactly 1 iteration** and that the L^2 error is below 10^(-14). |

### `[symmetry]` – Symmetric problem

| Test case | What it checks |
|-----------|----------------|
| *"Symmetric problem gives symmetric solution"* | `f(x,y) = 8π² sin(2πx)sin(2πy)` is unchanged when x and y are swapped. Solves once with `f(x,y)` and once with `f(y,x)` on a 33×33 grid. Asserts that the two L^2 errors against the exact solution differ by less than 10^(-10). |

### `[state]` – Solver state reporting

| Test case | What it checks |
|-----------|----------------|
| *"Converges for trivial problem"* | Trivial `f=g=0` problem, `max_iter = 100`. Asserts `hasConverged() == true` and `getIterations() >= 1`. |
| *"Does not converge when max_iter is too small"* | Non-trivial forcing, `max_iter = 1`, tolerance `10^(-12)`. One Jacobi step is never enough to satisfy such a tight tolerance. Asserts `hasConverged() == false` and `getIterations() == 1`. |

---

## Scalability and convergence driver (`run_scalability.cpp`)

The driver builds shell commands and launches the solver via `popen()`,
capturing stdout, saving it to a log file, and parsing the metrics
(wall time, iterations, L^2 error, convergence flag) for table printing.

### Hard-coded parameters

| Parameter | Value |
|-----------|-------|
| Forcing term `f(x,y)` | `8π² sin(2πx)sin(2πy)` |
| Boundary condition `g(x,y)` | `0` (homogeneous) |
| Exact solution | `sin(2πx)sin(2πy)` |
| Max iterations | 200 000 |
| Tolerance | 10^(-6) |
| Grid size for strong-scaling | `n = 64` |
| Grid sizes for convergence study | `n = 16, 32, 64, 128, 256` |
| MPI process counts (strong scaling) | 1, 2, 4 |
| OMP threads (parallel runs) | 2 |
| OMP threads (serial baseline) | 1 |

### Phase 1 – Serial baseline

Runs the solver once with `--serial` and `OMP_NUM_THREADS=1` on `n = 64`.
This gives a clean single-threaded reference time `t_serial` used to compute
absolute speed-ups in the scaling table.

Output log: `test/data/serial_n64.log`

### Phase 2 – Strong-scaling test

Runs the solver in **parallel mode** with 1, 2, and 4 MPI processes, all on
the same fixed grid size `n = 64` (`OMP_NUM_THREADS=2` for each run).
Keeps the problem size constant while increasing the number of processes.

For each run the table reports:
- wall time
- speed-up relative to the 1-process parallel run (`su_vs_par1`)
- speed-up relative to the serial baseline (`su_vs_serial`)
- parallel efficiency `= su_vs_par1 / nprocs × 100 %`
- number of Jacobi iterations
- L^2 error

Output logs: `test/data/scaling_np{1,2,4}_n64.log`  
Summary table: `test/data/scaling_N64.txt`

### Phase 3 – Convergence study

Runs both the serial and 1-process parallel solver for each grid size in
`{16, 32, 64, 128, 256}`. For each size records h, L^2 error, and wall time.
If the discretisation is second-order accurate, the L^2 error should decrease
by a factor of =4 each time n doubles (h is halved).

Output logs: `test/data/conv_serial_n{16,32,64,128,256}.log`  
             `test/data/conv_par_n{16,32,64,128,256}.log`  
Summary table: `test/data/convergence.txt`

---

## Output files in `test/data/`

| File | Contents |
|------|----------|
| `serial_n64.log` | Raw solver stdout for the serial baseline (n = 64) |
| `scaling_np1_n64.log` | Raw output, 1 MPI process, n = 64 |
| `scaling_np2_n64.log` | Raw output, 2 MPI processes, n = 64 |
| `scaling_np4_n64.log` | Raw output, 4 MPI processes, n = 64 |
| `scaling_N64.txt` | Strong-scaling summary table (space-separated) |
| `conv_serial_n16.log` … `conv_serial_n256.log` | Raw serial output for each grid size |
| `conv_par_n16.log` … `conv_par_n256.log` | Raw parallel output for each grid size |
| `convergence.txt` | Convergence study summary table (space-separated) |

All files in `test/data/` are auto-generated and **git-ignored**.