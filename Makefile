# Makefile – serial and parallel Jacobi solver for the Laplace equation
# 
# Targets
#   make / make all   – build solver + test runner
#   make test_runner  – build only the test driver
#   make catch_tests   – build the Catch2 correctness test suite
#   make clean        – remove object files and executables
#   make distclean    – also remove output CSV/VTK files
#
# ===========================================================================

# Compilers
MPICXX := mpicxx
CXX := g++

MPI_INC  := /usr/lib/x86_64-linux-gnu/openmpi/include
MPI_LIB  := /usr/lib/x86_64-linux-gnu/openmpi/lib
MPI_LIBS := -lmpi_cxx -lmpi

STD_FLAGS = -std=c++17 -O2 -Wall -Wextra

SOLVER_CXXFLAGS = $(STD_FLAGS) -fopenmp -I$(MPI_INC)
SOLVER_LDFLAGS = -fopenmp -fno-lto
 
MUPARSER_INC ?=
MUPARSER_LIB ?=
ifneq ($(MUPARSER_INC),)
    SOLVER_CXXFLAGS += -I$(MUPARSER_INC)
endif
MUPARSER_LDFLAG :=
ifneq ($(MUPARSER_LIB),)
    MUPARSER_LDFLAG += -L$(MUPARSER_LIB)
endif
MUPARSER_LDFLAG += -lmuparser
 
# Solver
SRCDIR = src
OBJDIR = obj
SRCS = $(SRCDIR)/main.cpp $(SRCDIR)/jacobi_solver.cpp
OBJS = $(patsubst $(SRCDIR)/%.cpp, $(OBJDIR)/%.o, $(SRCS))
TARGET = jacobi_solver
 
# Test runner
TEST_RUNNER = test/run_scalability
TEST_SRC = test/run_scalability.cpp
 
# Catch2 correctness tests
CATCH_TARGET = test/catch_tests
CATCH_SRC = test/catch_tests.cpp
CATCH_OBJ = $(OBJDIR)/catch_jacobi_solver.o
CATCH_CXXFLAGS = $(STD_FLAGS) -fopenmp -I$(MPI_INC)
CATCH_LDFLAGS = -fopenmp -fno-lto \
                 -L$(MPI_LIB) $(MPI_LIBS) \
                 -lCatch2Main -lCatch2 \
                 $(MUPARSER_LDFLAG)
 
.PHONY: all test_runner catch_tests clean distclean
 
all: $(TARGET) test_runner catch_tests
 
# Solver link
$(TARGET): $(OBJS)
	$(MPICXX) $(SOLVER_LDFLAGS) -o $@ $^ $(MUPARSER_LDFLAG)
 
$(OBJDIR)/%.o: $(SRCDIR)/%.cpp | $(OBJDIR)
	$(MPICXX) $(SOLVER_CXXFLAGS) -c $< -o $@
 
# Test runner link
test_runner: $(TEST_RUNNER)
 
$(TEST_RUNNER): $(TEST_SRC)
	$(CXX) $(STD_FLAGS) -o $@ $^
 
# Catch2 tests
catch_tests: $(CATCH_TARGET)
 
$(CATCH_OBJ): $(SRCDIR)/jacobi_solver.cpp | $(OBJDIR)
	$(CXX) $(CATCH_CXXFLAGS) -c $< -o $@
 
$(CATCH_TARGET): $(CATCH_SRC) $(CATCH_OBJ)
	$(CXX) $(CATCH_CXXFLAGS) -o $@ $^ $(CATCH_LDFLAGS)
 
$(OBJDIR):
	mkdir -p $(OBJDIR)
 
# Cleanup 
clean:
	rm -rf $(OBJDIR) $(TARGET) $(TEST_RUNNER) $(CATCH_TARGET)
 
distclean: clean
	rm -f solution_n*.csv solution_n*.vtk solution_serial_n*.csv solution_serial_n*.vtk
	rm -f test_bc.csv test_symmetry.csv
 