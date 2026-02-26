/**
 * Minimal test to debug distributed transpose segfault
 */

#include <ndarray/ndarray.hh>
#include <ndarray/transpose.hh>
#include <mpi.h>
#include <iostream>

using namespace ftk;

int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);

  int rank, nprocs;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

  if (rank == 0) {
    std::cout << "Starting minimal distributed transpose test with " << nprocs << " ranks" << std::endl;
  }

  try {
    if (rank == 0) std::cout << "Step 1: Creating distributed array..." << std::endl;

    // Create simple 100x80 array, decompose along dimension 0
    ndarray<double> arr;
    arr.decompose(MPI_COMM_WORLD, {100, 80}, nprocs, {static_cast<size_t>(nprocs), 0}, {0, 0});

    if (rank == 0) {
      std::cout << "Step 2: Array created. Global dims: "
                << arr.global_lattice().size(0) << "x"
                << arr.global_lattice().size(1) << std::endl;
      std::cout << "Step 3: Decomposition: "
                << arr.decomp_pattern()[0] << "x"
                << arr.decomp_pattern()[1] << std::endl;
    }

    if (rank == 0) std::cout << "Step 4: Initializing array..." << std::endl;

    // Simple initialization
    for (size_t i = 0; i < arr.nelem(); i++) {
      arr[i] = rank * 1000.0 + i;
    }

    if (rank == 0) std::cout << "Step 5: Checking partitioner..." << std::endl;

    // Check partitioner
    const auto& part = arr.partitioner();
    if (rank == 0) {
      std::cout << "Partitioner has " << part.np() << " partitions" << std::endl;
    }

    for (int r = 0; r < nprocs; r++) {
      if (rank == r) {
        std::cout << "  Rank " << r << " core: ";
        const auto& core = part.get_core(r);
        std::cout << "[" << core.start(0) << ":" << core.start(0)+core.size(0) << ", "
                  << core.start(1) << ":" << core.start(1)+core.size(1) << "]" << std::endl;
      }
      MPI_Barrier(MPI_COMM_WORLD);
    }

    if (rank == 0) std::cout << "Step 6: Calling transpose..." << std::endl;

    // Transpose
    try {
      auto transposed = ftk::transpose(arr, {1, 0});

      if (rank == 0) {
        std::cout << "Step 7: Transpose completed!" << std::endl;
        std::cout << "Transposed dims: "
                  << transposed.global_lattice().size(0) << "x"
                  << transposed.global_lattice().size(1) << std::endl;
      }

      if (rank == 0) std::cout << "SUCCESS: Test passed!" << std::endl;
    } catch (const std::exception& e) {
      std::cerr << "[Rank " << rank << "] EXCEPTION in transpose: " << e.what() << std::endl;
      throw;
    }

  } catch (const std::exception& e) {
    std::cerr << "[Rank " << rank << "] Exception: " << e.what() << std::endl;
    MPI_Abort(MPI_COMM_WORLD, 1);
  }

  MPI_Finalize();
  return 0;
}
