#include <ndarray/ndarray.hh>
#include <iostream>

#if NDARRAY_HAVE_MPI
#include <mpi.h>
#endif

/**
 * MPI parallel operations example for ndarray
 *
 * This example demonstrates:
 * - Distributed array processing with MPI
 * - Parallel I/O operations
 * - Domain decomposition
 *
 * Compile with MPI support: -DNDARRAY_USE_MPI=ON
 * Run with: mpirun -np 4 ./parallel_mpi
 */

int main(int argc, char** argv) {
#if NDARRAY_HAVE_MPI
  MPI_Init(&argc, &argv);

  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  if (rank == 0) {
    std::cout << "=== ndarray MPI Parallel Example ===" << std::endl;
    std::cout << "Running with " << size << " MPI processes" << std::endl << std::endl;
  }

  // 1. Create local data on each process
  if (rank == 0) {
    std::cout << "1. Creating distributed data:" << std::endl;
  }

  // Each process creates its own local array
  const size_t local_size = 100;
  ftk::ndarray<double> local_data;
  local_data.reshapef(local_size, 10);

  // Fill with rank-specific values
  for (size_t i = 0; i < local_data.size(); i++) {
    local_data[i] = rank * 1000.0 + static_cast<double>(i);
  }

  if (rank == 0) {
    std::cout << "   - Each process has array: " << local_data.dimf(0)
              << " x " << local_data.dimf(1) << std::endl;
  }
  MPI_Barrier(MPI_COMM_WORLD);
  std::cout << "   - Rank " << rank << ": local data[0] = " << local_data[0] << std::endl;
  MPI_Barrier(MPI_COMM_WORLD);

  if (rank == 0) std::cout << std::endl;

  // 2. Perform local computations
  if (rank == 0) {
    std::cout << "2. Performing local computations:" << std::endl;
  }

  // Each rank processes its local data
  double local_sum = 0.0;
  for (size_t i = 0; i < local_data.size(); i++) {
    local_data[i] *= 2.0;  // Scale values
    local_sum += local_data[i];
  }

  double local_mean = local_sum / local_data.size();
  std::cout << "   - Rank " << rank << ": local mean = " << local_mean << std::endl;
  MPI_Barrier(MPI_COMM_WORLD);

  if (rank == 0) std::cout << std::endl;

  // 3. Global reduction
  if (rank == 0) {
    std::cout << "3. Computing global statistics:" << std::endl;
  }

  double global_sum = 0.0;
  MPI_Reduce(&local_sum, &global_sum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

  if (rank == 0) {
    size_t global_elements = local_data.size() * size;
    double global_mean = global_sum / global_elements;
    std::cout << "   - Global sum: " << global_sum << std::endl;
    std::cout << "   - Global mean: " << global_mean << std::endl;
    std::cout << "   - Total elements: " << global_elements << std::endl << std::endl;
  }

  // 4. Parallel I/O example
  if (rank == 0) {
    std::cout << "4. Parallel I/O:" << std::endl;
  }

#if NDARRAY_HAVE_PNETCDF
  // Parallel NetCDF write
  try {
    local_data.write_pnetcdf("parallel_output.nc", "data", MPI_COMM_WORLD);
    if (rank == 0) {
      std::cout << "   - Successfully wrote parallel NetCDF file" << std::endl;
    }
  } catch (const std::exception& e) {
    if (rank == 0) {
      std::cerr << "   - Error with parallel NetCDF: " << e.what() << std::endl;
    }
  }
#else
  if (rank == 0) {
    std::cout << "   - Parallel NetCDF not enabled" << std::endl;
    std::cout << "   - (compile with -DNDARRAY_USE_PNETCDF=ON)" << std::endl;
  }
#endif

  // Alternative: Each rank writes its own file
  std::string filename = "rank_" + std::to_string(rank) + "_data.bin";
  try {
    local_data.to_binary_file(filename);
    std::cout << "   - Rank " << rank << " wrote " << filename << std::endl;
  } catch (const std::exception& e) {
    std::cerr << "   - Rank " << rank << " error: " << e.what() << std::endl;
  }

  MPI_Barrier(MPI_COMM_WORLD);
  if (rank == 0) std::cout << std::endl;

  // 5. Gather example (optional for small data)
  if (rank == 0) {
    std::cout << "5. Gathering data to root:" << std::endl;
  }

  // Send a single value from each rank to root
  double local_value = local_data[0];
  std::vector<double> gathered_values;

  if (rank == 0) {
    gathered_values.resize(size);
  }

  MPI_Gather(&local_value, 1, MPI_DOUBLE,
             gathered_values.data(), 1, MPI_DOUBLE,
             0, MPI_COMM_WORLD);

  if (rank == 0) {
    std::cout << "   - Gathered values from all ranks: ";
    for (int i = 0; i < size; i++) {
      std::cout << gathered_values[i];
      if (i < size - 1) std::cout << ", ";
    }
    std::cout << std::endl << std::endl;
  }

  // 6. Domain decomposition example
  if (rank == 0) {
    std::cout << "6. Domain decomposition example:" << std::endl;
    std::cout << "   - Global domain: 400 x 10" << std::endl;
    std::cout << "   - Local domain per rank: 100 x 10" << std::endl;
    std::cout << "   - Decomposition: 1D along first dimension" << std::endl;
  }

  // Calculate global indices for this rank
  size_t global_start = rank * local_size;
  size_t global_end = global_start + local_size;

  std::cout << "   - Rank " << rank << " handles global indices ["
            << global_start << ", " << global_end << ")" << std::endl;

  MPI_Barrier(MPI_COMM_WORLD);

  if (rank == 0) {
    std::cout << std::endl << "=== Example completed successfully ===" << std::endl;
    std::cout << "Output files:" << std::endl;
    for (int i = 0; i < size; i++) {
      std::cout << "  - rank_" << i << "_data.bin" << std::endl;
    }
#if NDARRAY_HAVE_PNETCDF
    std::cout << "  - parallel_output.nc" << std::endl;
#endif
  }

  MPI_Finalize();
  return 0;

#else
  std::cout << "=== ndarray MPI Parallel Example ===" << std::endl;
  std::cout << "ERROR: MPI support not enabled!" << std::endl;
  std::cout << "Please compile with -DNDARRAY_USE_MPI=ON" << std::endl;
  std::cout << "and run with: mpirun -np 4 ./parallel_mpi" << std::endl;
  return 1;
#endif
}
