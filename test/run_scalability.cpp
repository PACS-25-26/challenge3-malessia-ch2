/**
 * @file run_scalability.cpp
 * @brief Automated strong-scaling and convergence test for the Jacobi solver.
 *
 * ### What this program does
 * 1. **Serial baseline**: runs the solver with `--serial` for each grid size
 *    n = 2^k, k = 4,...,8, recording wall time, iterations and L^2 error.
 * 2. **Strong-scaling test**: fixed grid size n = N_SCALING, MPI process
 *    counts {1, 2, 4}.  Computes speed-up and efficiency relative to both
 *    the 1-process parallel run and the serial baseline.
 * 3. **Convergence study**: 1 MPI process (parallel mode), grid sizes
 *    n = 2^k, k = 4,...,8, records L^2 error vs h alongside the serial times.
 * 4. Writes summary tables to test/data/ and prints them to stdout.
 *
 * ### Build
 * ```
 * make test_runner
 * ```
 *
 * ### Run (from root after `make all`)
 * ```
 * ./test/run_scalability [path/to/jacobi_solver]
 * ```
 */
 
#include <array> // std::array
#include <cstdio> // popen, pclose, fgets 
#include <cstdlib> // EXIT_FAILURE, EXIT_SUCCESS
#include <filesystem> // std::filesystem::path, exists, create_directories
#include <fstream> // std::ofstream
#include <iomanip> // std::setw, std::setprecision, std::fixed, std::scientific
#include <iostream> // std::cout, std::cerr
#include <sstream> // std::ostringstream, std::istringstream
#include <stdexcept> // std::runtime_error
#include <string> // std::string
#include <vector> // std::vector

/// Alias to avoid writing std::filesystem
namespace fs = std::filesystem;

// Configuration

/// Grid size used for the strong-scaling benchmark
static constexpr int N_SCALING = 64;
/// Grid sizes for the convergence study
static const std::vector<int> CONVERGENCE_SIZES = {16, 32, 64, 128, 256};
/// Number of MPI processes tested in the strong-scaling run
static const std::vector<int> MPI_COUNTS = {1, 2, 4};
/// Maximum Jacobi iterations
static constexpr int MAX_ITER = 200000;
/// Convergence tolerance 
static constexpr double TOLERANCE = 1e-6;
/// Forcing term
static const std::string F_EXPR = "8*pi^2*sin(2*pi*x)*sin(2*pi*y)";
/// Homogeneous Dirichlet BC (zero on all boundaries)
static const std::string G_EXPR = "0";
/// Exact solution used for error computation
static const std::string EXACT_EXPR = "sin(2*pi*x)*sin(2*pi*y)";
 
// Result struct
/**
 * @brief Stores all metrics extracted from one solver run.
 *
 * One instance is created for each (nprocs, n) combination tested.
 * The fields are filled by parsing the solver's stdout after the run.
 */
struct RunResult
{
    int nprocs = 0; ///< number of MPI processes used
    int n = 0; ///< grid size
    double h = 0.0; ///< mesh spacing h = 1/(n-1)
    double wall_time = 0.0; ///< elapsed wall-clock time in seconds
    int iterations = 0; ///< number of Jacobi iterations performed
    double l2_error = 0.0; ///< discrete L2 error against the exact solution
    bool converged = false; ///< true if the solver reached the tolerance before max_iter
};
 
// Utility

/**
 * @brief Convert a double to a formatted string.
 *
 * Used to build table cells with a consistent number of decimal digits.
 *
 * @param v    The value to format.
 * @param prec Number of digits after the decimal point.
 * @param sci  If true, use scientific notation, if false, use fixed notation.
 * @return     The formatted string.
 */
static std::string fmtDouble(double v, int prec, bool sci = false)
{
    std::ostringstream ss; // temporary string buffer
    if (sci) 
        ss << std::scientific; 
    else 
        ss << std::fixed;
    ss << std::setprecision(prec) << v;
    return ss.str();
}

/**
 * @brief Run a shell command and return everything it prints to stdout.
 *
 * Uses popen() to open a pipe to the process launched by @p cmd,
 * reads its output line by line into a buffer, and returns the
 * complete output as a single string.
 *
 * @param cmd Shell command to execute (e.g. "mpirun -np 4 ./jacobi_solver ...").
 * @return    Everything the command printed to stdout.
 * @throws    std::runtime_error if popen() fails (e.g. shell not found).
 */
static std::string runCommand(const std::string& cmd)
{
    // popen() forks a shell, runs cmd, and returns a FILE* connected to the process's stdout
    FILE* pipe = popen(cmd.c_str(), "r"); // "r" means read from it
    if (!pipe) 
        throw std::runtime_error("popen() failed: " + cmd);
    std::string out;

    // Read the output 512 bytes at a time into buf.
    // fgets stops at a newline or when the buffer is full
    std::array<char, 512> buf{};
    while (fgets(buf.data(), static_cast<int>(buf.size()), pipe)) // The loop continues until fgets returns nullptr
        out += buf.data(); // append this chunk to the accumulated output string

    pclose(pipe);
    return out;
}

/**
 * @brief Find a labelled value in the solver output and parse it as a double.
 *
 * Searches for the first occurrence of @p label in @p s, then reads the
 * first floating-point number that follows the colon on that line.
 *
 * @param s        The full solver output to search.
 * @param label    The label to look for (e.g. "Wall time", "L2 error").
 * @param fallback Value to return if the label is not found.
 * @return         The parsed double, or @p fallback if parsing fails.
 */
static double parseDouble(const std::string& s, const std::string& label, double fallback = 0.0)
{
    // Find the position of the label in the string.
    // std::string::npos is returned when find() finds nothing.
    const auto pos = s.find(label);
    if (pos == std::string::npos) return fallback; // label not found

    // Advance to the colon that separates the label from its value
    const auto colon = s.find(':', pos);
    if (colon == std::string::npos) return fallback; // malformed line

    // Extract the substring after the colon and parse the first number from it
    double v = fallback;
    std::istringstream ss(s.substr(colon + 1));
    ss >> v;
    return v;
}
 
/**
 * @brief Find an integer value in the solver output and parse it.
 *
 * Identical logic to parseDouble but reads an int instead of a double.
 * Used to parse iteration counts from the solver output.
 *
 * @param s        The full solver output to search.
 * @param label    The label to look for (e.g. "Iterations").
 * @param fallback Value to return if the label is not found.
 * @return         The parsed integer, or @p fallback if parsing fails.
 */
static int parseInt(const std::string& s, const std::string& label,
                     int fallback = 0)
{
    const auto pos = s.find(label);
    if (pos == std::string::npos) return fallback;

    const auto colon = s.find(':', pos);
    if (colon == std::string::npos) return fallback;

    int v = fallback;
    std::istringstream ss(s.substr(colon + 1));
    ss >> v;
    return v;
}

/** 
 * @brief Find a boolean value in the solver output and parse it.
 *
 * Searches for the first occurrence of @p label in @p s, then checks if
 * the substring after the colon contains "yes".
 *
 * @param s        The full solver output to search.
 * @param label    The label to look for (e.g. "Converged").
 * @return         true if the label is found and contains "yes", false otherwise.
 */
static bool parseBool(const std::string& s, const std::string& label)
{
    const auto pos = s.find(label);
    if (pos == std::string::npos) return false;

    const auto colon = s.find(':', pos);
    if (colon == std::string::npos) return false;

    // Search for the word "yes" anywhere after the colon
    return s.find("yes", colon) != std::string::npos;
}
 
// Run helpers
 
/**
 * @brief Launch the solver in parallel mode and return the measured metrics.
 *
 * Builds a mpirun command string, executes it with runCommand(),
 * saves the raw output to a log file, and parses the metrics
 * (wall time, iterations, L2 error, convergence) from the output.
 *
 * @param exe    Path to the jacobi_solver executable.
 * @param nprocs Number of MPI processes to use.
 * @param n      Grid size.
 * @param log    Path to the log file where the raw output is saved.
 * @return       A RunResult with all metrics filled in.
 */
static RunResult runParallel(const std::string& exe, int nprocs, int n, const fs::path& log)
{
    // Build the command string piece by piece (\" to have a single statement)
    std::ostringstream cmd;
    cmd << "OMP_NUM_THREADS=2 mpirun -np " << nprocs << " " << exe
        << " " << n << " " << MAX_ITER
        << " " << std::scientific << TOLERANCE
        << " \"" << F_EXPR << "\"" // forcing term f(x,y)
        << " \"" << G_EXPR << "\"" // boundary condition g(x,y)
        << " \"" << EXACT_EXPR << "\"" // exact solution for L^2 error
        << " 2>&1";
 
    // Print the command to be run, execute it, and save the output to a log file
    std::cout << "  $ " << cmd.str() << "\n";
    const std::string out = runCommand(cmd.str());
    { std::ofstream ofs(log); if (ofs) ofs << out; }
 
    // Parse each metric from the captured output and store it in a RunResult
    RunResult r;
    r.nprocs = nprocs;
    r.n = n;
    r.h = 1.0 / (n - 1);
    r.wall_time = parseDouble(out, "Wall time");
    r.iterations = parseInt(out, "Iterations");
    r.l2_error = parseDouble(out, "L2 error");
    r.converged = parseBool(out, "Converged");
    return r;
}
 
/**
 * @brief Launch the solver in serial mode and return the measured metrics.
 *
 * Identical to runParallel but adds two things:
 * - the --serial flag, which disables all MPI communication inside the solver
 * - OMP_NUM_THREADS=1, which disables OpenMP so the run is truly single-threaded
 *
 * This gives a clean sequential baseline that can be fairly compared
 * against the parallel runs in the scaling table.
 *
 * @param exe  Path to the jacobi_solver executable.
 * @param n    Grid size.
 * @param log  Path to the log file where the raw output is saved.
 * @return     A RunResult with all metrics filled in.
 */
static RunResult runSerial(const std::string& exe, int n, const fs::path& log)
{
    // --serial disables MPI communication inside the solver.
    // OMP_NUM_THREADS=1 ensures no OpenMP threads are used either.
    std::ostringstream cmd;
    cmd << "OMP_NUM_THREADS=1 mpirun -np 1 " << exe
        << " --serial"
        << " " << n << " " << MAX_ITER
        << " " << std::scientific << TOLERANCE
        << " \"" << F_EXPR << "\""
        << " \"" << G_EXPR << "\""
        << " \"" << EXACT_EXPR << "\""
        << " 2>&1";
 
    std::cout << "  $ " << cmd.str() << "\n";
    const std::string out = runCommand(cmd.str());
    { std::ofstream ofs(log); if (ofs) ofs << out; }
 
    RunResult r;
    r.nprocs = 1; // nprocs is hardcoded to 1
    r.n = n;
    r.h = 1.0 / (n - 1);
    r.wall_time = parseDouble(out, "Wall time");
    r.iterations = parseInt(out, "Iterations");
    r.l2_error = parseDouble(out, "L2 error");
    r.converged = parseBool(out, "Converged");
    return r;
}
 
// Table printers
 
/**
 * @brief Print the strong-scaling results as a formatted table.
 *
 * For each parallel run shows:
 * - wall time
 * - speed-up relative to the 1-process parallel run (su_vs_par1)
 * - speed-up relative to the serial baseline (su_vs_serial)
 * - parallel efficiency (su_vs_par1 / nprocs * 100%)
 * - number of iterations
 * - L^2 error
 *
 * @param par_results  One RunResult per parallel run.
 * @param serial_ref   The serial baseline run for the same grid size.
 * @param out_path     File path where the summary table is saved.
 */
static void printScalingTable(const std::vector<RunResult>& par_results, const RunResult& serial_ref, const fs::path& out_path)
{
    // Reference wall times used to compute speed-ups
    const double t_par1 = par_results.empty() ? 1.0 : par_results[0].wall_time;
    const double t_serial = serial_ref.wall_time;
 
    // Print the header to stdout.
    std::cout << "\nStrong-scaling results (n = " << N_SCALING << ")\n"
              << "  Serial baseline : " << fmtDouble(t_serial, 4) << " s\n\n"
              << "  nprocs | wall_time(s) | su_vs_par1 | su_vs_serial | efficiency | iters | L2_error\n"
              << "         |              |            |              |            |       |         \n";
 
    // Open the output file and write a comment header.
    std::ofstream ofs(out_path);
    if (ofs)
        ofs << "# nprocs  wall_time_s  speedup_vs_par1  speedup_vs_serial"
               "  efficiency_%  iterations  L2_error  converged\n"
            << "# serial_baseline: " << t_serial << " s\n";
 
    // Print one row per parallel run
    for (const auto& r : par_results)
    {
        // Speed-up = reference_time / this_run_time
        // Perfect linear scaling would give su_par1 = nprocs
        const double su_par1 = t_par1 / r.wall_time;
        const double su_serial = t_serial / r.wall_time;

        // Efficiency = how well the extra processes are used (100% = perfect)
        const double eff = su_par1 / r.nprocs * 100.0;

        // Print to stdout using | separators for readability
        std::cout << "  "
                  << r.nprocs                         << " | "
                  << fmtDouble(r.wall_time, 4)        << " | "
                  << fmtDouble(su_par1, 2)            << " | "
                  << fmtDouble(su_serial, 2)          << " | "
                  << fmtDouble(eff, 1)                << " %" << " | "
                  << r.iterations                     << " | "
                  << fmtDouble(r.l2_error, 3, true)   << "\n";

        // Write the same data to file as space-separated values
        if (ofs)
            ofs << r.nprocs << "  " << r.wall_time << "  "
                << su_par1 << "  " << su_serial   << "  "
                << eff << "  " << r.iterations << "  "
                << r.l2_error << "  " << (r.converged ? "yes" : "no") << "\n";
    }
    std::cout << "\n";
}
 
/**
 * @brief Print the convergence study results comparing serial and parallel runs.
 *
 * For each grid size shows 
 * - the mesh spacing h 
 * - the L2 error for both the serial and the 1-process parallel run
 * - wall times.
 * If the solver is second-order accurate, the L2 error should roughly
 * divide by 4 each time n doubles (h is halved).
 * The table is printed to stdout and saved to @p out_path.
 *
 * @param par_results  One RunResult per grid size, parallel mode (1 process).
 * @param ser_results  One RunResult per grid size, serial mode.
 * @param out_path     File where the summary table is saved.
 */
static void printConvergenceTable(const std::vector<RunResult>& par_results, const std::vector<RunResult>& ser_results, const fs::path& out_path)
{
    // Print the header to stdout
    std::cout << "\n=== Convergence study (serial vs parallel, 1 process) ===\n\n"
              << "  n   | h          | L2_serial  | L2_parallel | t_serial | t_parallel\n"
              << "      |            |            |             |          |           \n";

    // Open the output file and write a comment header
    std::ofstream ofs(out_path);
    if (ofs)
        ofs << "# n  h  L2_serial  L2_parallel  t_serial_s  t_parallel_s  iterations\n";

    // Iterate over both result vectors together
    const std::size_t sz = std::min(ser_results.size(), par_results.size());
    for (std::size_t k = 0; k < sz; ++k)
    {
        const auto& s = ser_results[k];  // serial result for this grid size
        const auto& p = par_results[k];  // parallel result for the same grid size

        // Print one row to stdout using | separators.
        std::cout << "  "
                  << s.n                                  << " | "
                  << fmtDouble(s.h, 8)                    << " | "
                  << fmtDouble(s.l2_error, 4, true)       << " | "
                  << fmtDouble(p.l2_error, 4, true)       << " | "
                  << fmtDouble(s.wall_time, 3) << " s"    << " | "
                  << fmtDouble(p.wall_time, 3) << " s"    << "\n";

        // Write the same data to file as space-separated values.
        if (ofs)
            ofs << s.n << "  " << s.h << "  "
                << s.l2_error << "  " << p.l2_error << "  "
                << s.wall_time << "  " << p.wall_time << "  "
                << s.iterations << "\n";
    }
    std::cout << "\n";
}
 
// main
 
int main(int argc, char* argv[])
{
    // Locate the solver executable
    // The solver is expected to be in the project root as "jacobi_solver".
    // If the user passes a path as the first argument, that is used instead.
    const fs::path bin_dir = fs::path(argv[0]).parent_path().parent_path();
    const fs::path exe = (argc > 1) ? fs::path(argv[1]) : bin_dir / "jacobi_solver";

    if (!fs::exists(exe))
    {
        std::cerr << "Error: executable not found: " << exe
                  << "\nRun `make all` in the project root first.\n";
        return EXIT_FAILURE;
    }

    // Create the output directory
    // All log files and summary tables are written to test/data/.
    // create_directories() does nothing if the directory already exists.
    const fs::path data_dir = fs::path(argv[0]).parent_path() / "data";
    fs::create_directories(data_dir);

    // Print a summary of what is about to run.
    std::cout << "\n Jacobi solver, scalability test\n"
              << "  Solver       : " << exe << "\n"
              << "  Data output  : " << data_dir << "\n"
              << "  MAX_ITER     : " << MAX_ITER << "\n"
              << "  Tolerance    : " << TOLERANCE << "\n"
              << "  f(x,y)       : " << F_EXPR << "\n"
              << "  exact u(x,y) : " << EXACT_EXPR << "\n\n";

    // Serial baseline
    // Run the solver once in serial mode on the scaling grid size.
    // This gives the reference time t_serial used to compute speed-ups in the scaling table.
    std::cout << "Serial baseline (n = " << N_SCALING << ")\n";
    const RunResult serial_ref = runSerial(exe.string(), N_SCALING, data_dir / ("serial_n" + std::to_string(N_SCALING) + ".log"));

    // Strong-scaling test
    // Run the solver in parallel mode with 1, 2 and 4 MPI processes, all on the same fixed grid size N_SCALING.
    std::cout << "\nStrong-scaling test (n = " << N_SCALING << ")\n";
    std::vector<RunResult> scaling_results;
    for (int np : MPI_COUNTS)
    {
        std::cout << "\n[parallel, " << np << " process(es)]\n";
        scaling_results.push_back(runParallel(exe.string(), np, N_SCALING, 
                                  data_dir / ("scaling_np" + std::to_string(np) + "_n" + std::to_string(N_SCALING) + ".log")));
        // Log filename encodes both the process count and the grid size (never overwrite)
    }

    // Print and save the scaling table
    const fs::path scaling_file = data_dir / ("scaling_N" + std::to_string(N_SCALING) + ".txt");
    printScalingTable(scaling_results, serial_ref, scaling_file);
    std::cout << "Scaling summary written to " << scaling_file << "\n";

    // Convergence study
    // Run both the serial and the 1-process parallel solver for each grid size in CONVERGENCE_SIZES.  
    // For each size record the L^2 error and the wall time.
    std::cout << "\nConvergence study\n";
    std::vector<RunResult> conv_par, conv_ser;
    for (int n : CONVERGENCE_SIZES)
    {
        // Run serial first, then parallel with 1 process.
        std::cout << "\n[serial,   n = " << n << "]\n";
        conv_ser.push_back(runSerial(exe.string(), n, data_dir / ("conv_serial_n" + std::to_string(n) + ".log")));

        std::cout << "[parallel, n = " << n << "]\n";
        conv_par.push_back(runParallel(exe.string(), 1, n, data_dir / ("conv_par_n" + std::to_string(n) + ".log")));
    }

    // Print and save the convergence table
    const fs::path conv_file = data_dir / "convergence.txt";
    printConvergenceTable(conv_par, conv_ser, conv_file);
    std::cout << "Convergence summary written to " << conv_file << "\n";

    std::cout << "\nAll tests completed. Results are in " << data_dir << "/\n";
    return EXIT_SUCCESS;
}