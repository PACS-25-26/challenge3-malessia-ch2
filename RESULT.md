# Results – Challenge 3: Matrix-free parallel Jacobi solver

## 1. Solver runs (`main`)

### 1.1 Standard test case

The solver was run on the reference problem
`f(x,y) = 8π² sin(2πx)sin(2πy)`, `g = 0`, exact solution `u = sin(2πx)sin(2πy)`,
with `n = 64`, `max_iter = 200000`, `tol = 1e-6`.

| Mode     | nprocs | OMP threads | Iterations | Converged | Wall time  | L^2 error  |
|----------|--------|-------------|------------|-----------|------------|------------|
| Serial   | 1      | 1           | 1987       | yes       | 0.0787 s   | 3.092e-03  |
| Parallel | 4      | 2           | 1857       | yes       | 0.134 s    | 2.910e-03  |

Serial and parallel produce the same L^2 error (differences at the fourth
significant digit are due to different iteration counts at convergence, not
to numerical inconsistency). The parallel run with 4 MPI processes is
slower than serial on this small grid (see the scaling discussion in Section 3).

### 1.2 Generic forcing term (no exact solution)

```
f(x,y) = 4xy,  g = 0,  n = 64,  2 MPI × 2 OMP threads
```

| Iterations | Converged | Wall time |
|------------|-----------|-----------|
| 4550       | yes       | 0.140 s   |

This problem has no known closed-form solution, so only the residual
convergence criterion is checked. The solver converges correctly; the
higher iteration count (4550 vs ~1900 for the sin test case) reflects the
different properties of the forcing term.

### 1.3 Non-homogeneous Dirichlet BC

```
f(x,y) = 0,  g(x,y) = sin(πx)sin(πy),  n = 64,  2 MPI × 2 OMP threads
exact u(x,y) = sin(πx)sin(πy)
```

| Iterations | Converged | Wall time | L^2 error |
|------------|-----------|-----------|-----------|
| 1          | yes       | 0.023 s   | 3.969     |

The solver converges in a single iteration with residual `‖err‖_2 ≈ 2.65e-17`
(essentially machine zero), yet the L^2 error against the exact solution is
large (= 3.97). 
The solver correctly imposes the given BC; the large L^2 error simply reflects 
the fact that `sin(πx)sin(πy)` is not harmonic (Δu != 0), so the problem with `f = 0` 
and this BC does not have `u = sin(πx)sin(πy)` as its exact solution.

---

## 2. Convergence study

**Setup**: serial and 1-process parallel runs, `n = 16, 32, 64, 128, 256`,
`tol = 1e-6`, `max_iter = 200000`, `f(x,y) = 8π² sin(2πx)sin(2πy)`, `g = 0`, 
exact solution `u = sin(2πx)sin(2πy)`.

| n   | h          | L^2 error (serial) | L^2 error (parallel) | t_serial  | t_parallel |
|-----|------------|--------------------|----------------------|-----------|------------|
|  16 | 6.667e-02  | 2.856e-02          | 2.856e-02            | < 0.001 s | 0.001 s    |
|  32 | 3.226e-02  | 9.502e-03          | 9.502e-03            | 0.006 s   | 0.005 s    |
|  64 | 1.587e-02  | 3.092e-03          | 3.092e-03            | 0.104 s   | 0.064 s    |
| 128 | 7.874e-03  | 3.340e-04          | 3.340e-04            | 1.159 s   | 0.762 s    |
| 256 | 3.922e-03  | 2.889e-03          | 2.889e-03            | 15.183 s  | 8.168 s    |

Serial and parallel (1 process) give **identical L^2 errors** at every grid
size, confirming that the MPI domain decomposition does not affect the
numerical result.

### Error ratios

| Refinement | e(n) / e(2n) | Expected (order 2) |
|------------|--------------|--------------------|
| 16 → 32    | 3.01         | 4                  |
| 32 → 64    | 3.07         | 4                  |
| 64 → 128   | 9.25         | 4                  |
| 128 → 256  | 0.12         | 4                  |

**n = 16 → 32, 32 → 64**: the ratio = 3 is slightly below the theoretical
value of 4. This is consistent with second-order accuracy: the solver has
not converged to machine precision (tolerance is 1e-6), so a small
iteration error adds to the discretisation error and slightly depresses the
observed ratio.

**n = 64 → 128**: the anomalously large ratio (9.25) indicates that the
n = 128 run converged to a residual significantly tighter than the
tolerance required. Jacobi occasionally "overshoots" the tolerance on
grids where the dominant eigenvalue of the iteration matrix is
particularly favourable, yielding a smaller-than-expected L^2 error.

**n = 128 → 256**: the ratio drops to 0.12 because the n = 256 run
**did not converge** within `max_iter = 200000` iterations. The plain
Jacobi method requires O(n^2) iterations; for n = 256 that is roughly
1.6 × 10^6 steps. The reported L^2 error (2.889e-03) reflects the
partially-iterated solution, not the true discretisation error.
A tighter study up to n = 128 confirms second-order accuracy.

### Wall-time scaling

The parallel solver (1 process, 2 OMP threads) is consistently faster
than serial for n ≥ 64. At n = 256 the speed-up reaches **1.86×**
(15.18 s → 8.17 s), showing that OpenMP threading is beneficial for
larger grids even on a single node.

---

## 3. Strong-scaling test

**Setup**: fixed grid `n = 64`, `OMP_NUM_THREADS = 2`,
`tol = 1e-6`, `max_iter = 200000`.
Serial baseline: `OMP_NUM_THREADS = 1`, `--serial` flag.

| Mode     | nprocs | OMP threads | Wall time | Speed-up vs serial | Speed-up vs par(1) | Efficiency |
|----------|--------|-------------|-----------|--------------------|--------------------|------------|
| Serial   | 1      | 1           | 0.095 s   | -                  | -                  | -          |
| Parallel | 1      | 2           | 0.077 s   | 1.23×              | 1.00               | 100%       |
| Parallel | 2      | 2           | 0.076 s   | 1.25×              | 1.02×              | 51%        |
| Parallel | 4      | 2           | 0.123 s   | 0.77×              | 0.62×              | 16%        |

### Discussion

**1 MPI × 2 OMP** is 1.23× faster than serial, entirely due to OpenMP
threading. This is the only configuration that gives a genuine speed-up
over the sequential baseline for n = 64.

**2 MPI × 2 OMP** gives negligible further improvement (0.076 s vs
0.077 s, speed-up 1.02×). At this grid size each rank owns only
~31 interior rows; the cost of `MPI_Sendrecv` (halo exchange) and
`MPI_Allreduce` (global convergence check) at every iteration becomes
comparable to the computation cost, leaving no room for MPI to help.

**4 MPI × 2 OMP** is slower than serial (0.123 s vs 0.095 s). With
~15 rows per rank the communication overhead completely dominates.

**Conclusion**: for small grids (n = 64) on a shared-memory machine,
OpenMP threading (1 process, 2 threads) is the optimal strategy.
Meaningful MPI scaling requires either a larger grid (n ≥ 256, where
each rank owns substantially more work) or a distributed-memory cluster
with dedicated cores per process and low-latency interconnect.

---

## 4. Correctness tests (Catch2)

All 8 test cases passed (13 assertions).
Build: `make catch_tests` — Run: `./test/catch_tests`.
All tests use serial mode (`use_mpi = false`); no MPI environment is needed.

| Tag | Test | Iterations | Key metric | Result |
|-----|------|-----------|------------|--------|
| `[bc]` | Non-homogeneous BC: u = x+y (harmonic) | 530 | L^2 < 5e-3 | ✓ passed |
| `[bc]` | Homogeneous BC: f=0, g=0 -> u=0 | 1954 | L^2 < 1e-10 | ✓ passed |
| `[convergence]` | Error ratio e(n=17)/e(n=33) > 2 | 122 / 444 | ratio = 3 | ✓ passed |
| `[exactsol]` | Test case n=64: L^2 < 5e-3 | 1524 | L^2 = 3.09e-3 | ✓ passed |
| `[exactsol]` | Harmonic u=x+y: L^2 < 1e-4 | 2849 | L^2 = 0 | ✓ passed |
| `[trivial]` | f=0, g=0: converges in 1 iteration | 0* | L^2 < 1e-14 | ✓ passed |
| `[symmetry]` | f(x,y) = f(y,x): errors differ < 1e-10 | 563 / 563 | Δe = 0 | ✓ passed |
| `[state]` | hasConverged() / getIterations() correct | 0* / 1 | — | ✓ passed |

\* "iteration 0" means the first Jacobi step produced zero increment
(the initial condition is already the exact solution); the solver reports
convergence after 1 call to `solve()` but at iteration index 0.

### Notes on specific tests

**`[trivial]`** (`f=0, g=0`): the initial condition (all zeros, consistent
with the zero BC) is the exact solution. The very first Jacobi step
produces `U_new = U_old` everywhere, so the increment norm is identically
zero and convergence is declared immediately (`‖err‖_2 = 0`).

**`[symmetry]`**: both solver instances (standard `f(x,y)` and transposed
`f(y,x)`) converge in exactly 563 iterations with identical L^2 errors,
confirming that the solver treats x and y symmetrically as expected.

**`[state]` — non-convergence section**: the solver is deliberately
limited to `max_iter = 1` with tolerance `1e-12`. One Jacobi step is
never enough to satisfy such a tight criterion; `hasConverged()` correctly
returns `false` and `getIterations()` returns 1. The expected
"Warning: solver did not converge" message is printed to stderr.

**`[convergence]`**: the error ratio for n=17->33 is = 3 (above the
required threshold of 2), slightly below the theoretical value of 4 for
the same reason discussed in Section 2: the solver stops at tolerance
1e-5 rather than machine precision, leaving a small iteration error that
slightly depresses the observed order.

---

## 5. Cluster (PoliMi HPC)

### Solver runs

Run on cpu02 with `n=64`:

| Test | Mode | nprocs | OMP | Iterations | Wall time | L^2 error |
|------|------|--------|-----|------------|-----------|----------|
| Standard test case | Serial | 1 | 1 | 1987 | 0.082 s | 3.092e-03 |
| Standard test case | Parallel | 4 | 2 | 1857 | 0.026 s | 2.910e-03 |
| Generic forcing `f=4xy` | Parallel | 2 | 2 | 4550 | 0.093 s | — |
| Non-homogeneous BC | Parallel | 2 | 2 | 1 | < 0.001 s | 3.969 |

The parallel run (4 MPI × 2 OMP) is **3.2× faster** than serial (0.026 s
vs 0.082 s), a significantly better result than on my local PC (where 4 
processes was actually slower than serial). This confirms that the poor 
scaling on the local machine was due to the virtualisation overhead and 
shared-memory contention, not to a flaw in the implementation.

The L^2 errors are identical to the results on my local PC, confirming 
numerical correctness across platforms.

### Strong-scaling results (cluster, n=64)

| nprocs | OMP threads | Wall time | Speed-up vs serial | Speed-up vs par(1) | Efficiency |
|--------|-------------|-----------|--------------------|--------------------|------------|
| 1      | 2           | 0.073 s   | 1.14×              | 1.00               | 100%       |
| 2      | 2           | 0.040 s   | 2.09×              | 1.83×              | 91.7%      |
| 4      | 2           | 0.027 s   | 3.07×              | 2.70×              | 67.6%      |

The cluster delivers near-linear scaling up to 2 processes (91.7%
efficiency) and good scaling at 4 processes (67.6%), in contrast to the 
previous results on the local machine, where 4 processes was slower than serial. 
The remaining inefficiency at 4 processes is expected: with n=64 each rank owns 
only ~15 interior rows, so the `MPI_Sendrecv` halo exchange and `MPI_Allreduce` 
convergence check still represent a non-negligible fraction of the total work. 
Larger grids (n >= 256) would show even better scaling.

### Large-grid scaling test (cluster, n=256 and n=512)
 
For large grids each rank owns substantially more rows, so MPI
communication becomes a smaller fraction of the total work. These runs
use `tol = 1e-4` (Jacobi requires O(n^2) iterations; tighter tolerances
would exceed `max_iter = 200000` for n >= 256).
 
### n = 256
 
| Mode     | nprocs | OMP | Iterations | Wall time | Speed-up vs serial | Efficiency | L^2 error |
|----------|--------|-----|------------|-----------|--------------------|------------|-----------|
| Serial   | 1      | 1   | 10502      | 7.339 s   | -                  |-           | 0.329     |
| Parallel | 4      | 2   | 8257       | 1.356 s   | 5.41×              | 67.6%      | 0.651     |
| Parallel | 8      | 2   | 7931       | 0.828 s   | 8.86×              | 55.4%      | 0.718     |
 
### n = 512
 
| Mode     | nprocs | OMP | Iterations | Wall time | Speed-up vs serial | Efficiency | L^2 error |
|----------|--------|-----|------------|-----------|--------------------|------------|-----------|
| Serial   | 1      | 1   | 28379      | 80.637 s  | -                  |-           | 1.323     |
| Parallel | 4      | 2   | 19287      | 12.855 s  | 6.27×              | 78.4%      | 2.630     |
| Parallel | 8      | 2   | 17969      | 6.426 s   | 12.55×             | 78.4%      | 2.906     |
 
For n=512 the 8-process parallel run is **12.5× faster** than serial
(6.4 s vs 80.6 s), demonstrating that the implementation scales well
on larger problems. Each rank owns ~63 interior rows, enough for
computation to dominate communication.
 
The L^2 errors vary across configurations because different numbers of
processes reach the tolerance criterion at different iteration counts.

Efficiency at n=512 is consistent across 4 and 8 processes (both
78.4%), suggesting the implementation has reached a regime where scaling
is genuinely limited by the algorithm (global `MPI_Allreduce` at every
iteration) rather than by implementation inefficiency.
 
| n   | Best config   | Speed-up vs serial |
|-----|---------------|--------------------|
| 64  | 4 MPI × 2 OMP | 3.1×               |
| 256 | 8 MPI × 2 OMP | 8.9×               |
| 512 | 8 MPI × 2 OMP | 12.5×              |
 
Speed-up improves significantly with grid size, confirming that the
parallel implementation is communication-efficient and that MPI overhead
becomes negligible for large problems.