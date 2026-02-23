/**
 * Parallel ADIOS2 tests with MPI
 *
 * Tests ADIOS2 parallel I/O functionality:
 * - Parallel write from multiple MPI ranks
 * - Parallel read with domain decomposition
 * - Collective I/O operations
 *
 * Requires: NDARRAY_HAVE_ADIOS2 && NDARRAY_HAVE_MPI
 * Run with: mpirun -np 4 ./test_adios2_parallel
 */

#include <ndarray/ndarray.hh>
#include <iostream>
#include <cassert>
#include <filesystem>
#include <chrono>
#include <thread>

#if NDARRAY_HAVE_ADIOS2 && NDARRAY_HAVE_MPI
#include <adios2.h>
#include <mpi.h>

#define TEST_ASSERT(condition, message) \
  do { \
    if (!(condition)) { \
      std::cerr << "[Rank " << rank << "] FAILED: " << message << std::endl; \
      std::cerr << "  at " << __FILE__ << ":" << __LINE__ << std::endl; \
      MPI_Abort(MPI_COMM_WORLD, 1); \
    } \
  } while (0)

#define TEST_SECTION(name) \
  if (rank == 0) std::cout << "  Testing: " << name << std::endl

// Helper to open ADIOS2 file with retries for slow CI filesystems
template<typename IOType>
adios2::Engine open_with_retry(IOType& io, const std::string& filename, adios2::Mode mode, int rank, int max_retries = 10) {
  for (int attempt = 0; attempt < max_retries; attempt++) {
    try {
      return io.Open(filename, mode);
    } catch (const std::exception& e) {
      if (attempt == max_retries - 1) {
        if (rank == 0) {
          std::cerr << "Failed to open " << filename << " after " << max_retries
                    << " attempts. Last error: " << e.what() << std::endl;
        }
        throw;  // Rethrow on final attempt
      }
      // Exponential backoff: 50ms, 100ms, 200ms, 400ms, ...
      int delay_ms = 50 * (1 << attempt);
      if (rank == 0) {
        std::cerr << "Attempt " << (attempt + 1) << " to open " << filename
                  << " failed, retrying in " << delay_ms << "ms..." << std::endl;
      }
      std::this_thread::sleep_for(std::chrono::milliseconds(delay_ms));
    } catch (...) {
      if (attempt == max_retries - 1) {
        if (rank == 0) {
          std::cerr << "Failed to open " << filename << " after " << max_retries
                    << " attempts with unknown exception" << std::endl;
        }
        throw;  // Rethrow on final attempt
      }
      int delay_ms = 50 * (1 << attempt);
      if (rank == 0) {
        std::cerr << "Attempt " << (attempt + 1) << " to open " << filename
                  << " failed (unknown exception), retrying in " << delay_ms << "ms..." << std::endl;
      }
      std::this_thread::sleep_for(std::chrono::milliseconds(delay_ms));
    }
  }
  throw std::runtime_error("Failed to open file after retries");
}

int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);

  int rank = 0, nprocs = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

  // Cleanup: Remove old ADIOS2 files BEFORE tests to avoid conflicts when test
  // is run multiple times with different rank counts (e.g., CI: mpirun -np 2, then -np 4)
  // BP4 format creates directories. Use error_code to avoid throwing if files don't exist.
  if (rank == 0) {
    std::error_code ec;
    std::filesystem::remove_all("test_parallel_write.bp", ec);
    std::filesystem::remove_all("test_parallel_timeseries.bp", ec);
  }
  MPI_Barrier(MPI_COMM_WORLD);  // Ensure cleanup completes before tests start

  if (rank == 0) {
    std::cout << "=== Running Parallel ADIOS2 Tests ===" << std::endl;
    std::cout << "Running with " << nprocs << " MPI ranks" << std::endl << std::endl;
  }

  // Test 1: Parallel write - each rank writes its portion
  {
    TEST_SECTION("Parallel write from multiple ranks");

    const size_t global_nx = 20;
    const size_t global_ny = 30;
    const size_t local_nx = global_nx / nprocs;  // Each rank gets a slice
    const size_t local_ny = global_ny;

    // Create local data
    ftk::ndarray<float> local_data;
    local_data.reshapef(local_nx, local_ny);

    for (size_t j = 0; j < local_ny; j++) {
      for (size_t i = 0; i < local_nx; i++) {
        // Value depends on global position
        size_t global_i = rank * local_nx + i;
        local_data.f(i, j) = global_i * 100.0f + j;
      }
    }

    // Parallel write - use nested scope to ensure complete cleanup
    {
      adios2::ADIOS adios(MPI_COMM_WORLD);
      adios2::IO io = adios.DeclareIO("ParallelWrite");
      io.SetEngine("BP4");

      adios2::Engine writer = io.Open("test_parallel_write.bp", adios2::Mode::Write);

      // Define variable with global dimensions
      size_t offset_x = rank * local_nx;
      auto var = io.DefineVariable<float>("data",
        {global_ny, global_nx},           // Global dimensions (C-order)
        {0, offset_x},                     // Offset for this rank
        {local_ny, local_nx});             // Local dimensions

      writer.BeginStep();
      writer.Put(var, local_data.data());
      writer.PerformPuts();  // Ensure deferred writes are completed
      writer.EndStep();
      writer.Close();
    }  // ADIOS, IO, and engine objects destroyed here

    // Barrier after ADIOS2 cleanup
    MPI_Barrier(MPI_COMM_WORLD);

    if (rank == 0) {
      std::cout << "    - Each rank wrote " << local_nx << "x" << local_ny
                << " (global: " << global_nx << "x" << global_ny << ")" << std::endl;
    }
  }

  // Barrier before read to ensure write is fully complete across all ranks
  MPI_Barrier(MPI_COMM_WORLD);

  // Test 2: Parallel read - each rank reads its portion back
  {
    TEST_SECTION("Parallel read with domain decomposition");

    const size_t global_nx = 20;
    const size_t global_ny = 30;
    const size_t local_nx = global_nx / nprocs;
    const size_t local_ny = global_ny;

    ftk::ndarray<float> loaded;
    loaded.reshapef(local_nx, local_ny);

    // Parallel read - use nested scope to ensure complete cleanup
    {
      adios2::ADIOS adios(MPI_COMM_WORLD);
      adios2::IO io = adios.DeclareIO("ParallelRead");

      adios2::Engine reader = open_with_retry(io, "test_parallel_write.bp", adios2::Mode::ReadRandomAccess, rank);

      auto var = io.InquireVariable<float>("data");
      TEST_ASSERT(var, "Variable should exist");

      // Each rank reads its portion
      size_t offset_x = rank * local_nx;
      var.SetSelection({{0, offset_x}, {local_ny, local_nx}});

      reader.Get(var, loaded.data());
      reader.PerformGets();
      reader.Close();
    }  // ADIOS, IO, and engine objects destroyed here

    // Barrier after ADIOS2 cleanup
    MPI_Barrier(MPI_COMM_WORLD);

    // Verify data
    for (size_t j = 0; j < local_ny; j++) {
      for (size_t i = 0; i < local_nx; i++) {
        size_t global_i = rank * local_nx + i;
        float expected = global_i * 100.0f + j;
        TEST_ASSERT(std::abs(loaded.f(i, j) - expected) < 1e-5f,
                    "Wrong data in parallel read");
      }
    }

    if (rank == 0) {
      std::cout << "    - Each rank read and verified its portion" << std::endl;
    }
  }

  MPI_Barrier(MPI_COMM_WORLD);  // Ensure all ranks finish read before next test

  // Test 3: Parallel write with multiple timesteps
  {
    TEST_SECTION("Parallel write with time series");

    const size_t global_nx = 16;
    const size_t global_ny = 20;
    const size_t local_nx = global_nx / nprocs;
    const size_t local_ny = global_ny;
    const int nsteps = 3;

    // Parallel write - use nested scope to ensure complete cleanup
    {
      adios2::ADIOS adios(MPI_COMM_WORLD);
      adios2::IO io = adios.DeclareIO("ParallelTimeSeries");
      io.SetEngine("BP4");

      adios2::Engine writer = io.Open("test_parallel_timeseries.bp", adios2::Mode::Write);

      size_t offset_x = rank * local_nx;
      auto var = io.DefineVariable<double>("field",
        {global_ny, global_nx},
        {0, offset_x},
        {local_ny, local_nx});

      for (int step = 0; step < nsteps; step++) {
        ftk::ndarray<double> data;
        data.reshapef(local_nx, local_ny);

        for (size_t j = 0; j < local_ny; j++) {
          for (size_t i = 0; i < local_nx; i++) {
            size_t global_i = rank * local_nx + i;
            data.f(i, j) = step * 1000.0 + global_i * 10.0 + j;
          }
        }

        writer.BeginStep();
        writer.Put(var, data.data());
        writer.PerformPuts();  // Ensure deferred writes are completed
        writer.EndStep();
      }

      writer.Close();
    }  // ADIOS, IO, and engine objects destroyed here

    // Barrier after ADIOS2 cleanup
    MPI_Barrier(MPI_COMM_WORLD);

    if (rank == 0) {
      std::cout << "    - Wrote " << nsteps << " timesteps in parallel" << std::endl;
    }
  }

  // Barrier before read to ensure write is fully complete across all ranks
  MPI_Barrier(MPI_COMM_WORLD);

  // Test 4: Parallel read of specific timestep
  {
    TEST_SECTION("Parallel read of specific timestep");

    const size_t global_nx = 16;
    const size_t global_ny = 20;
    const size_t local_nx = global_nx / nprocs;
    const size_t local_ny = global_ny;
    const int read_step = 1;  // Read timestep 1

    ftk::ndarray<double> loaded;
    loaded.reshapef(local_nx, local_ny);

    // Parallel read - use nested scope to ensure complete cleanup
    {
      adios2::ADIOS adios(MPI_COMM_WORLD);
      adios2::IO io = adios.DeclareIO("ParallelReadTimestep");

      // Use streaming Mode::Read for better CI compatibility with timesteps
      adios2::Engine reader = open_with_retry(io, "test_parallel_timeseries.bp", adios2::Mode::Read, rank);

      // Advance to desired timestep
      for (int step = 0; step <= read_step; step++) {
        adios2::StepStatus status = reader.BeginStep();
        TEST_ASSERT(status == adios2::StepStatus::OK, "Failed to begin step");

        if (step == read_step) {
          // Read the data at this timestep
          auto var = io.InquireVariable<double>("field");
          TEST_ASSERT(var, "Variable should exist");

          // Select spatial region for this rank
          size_t offset_x = rank * local_nx;
          var.SetSelection({{0, offset_x}, {local_ny, local_nx}});

          reader.Get(var, loaded.data());
          reader.PerformGets();
        }

        reader.EndStep();
      }

      reader.Close();
    }  // ADIOS, IO, and engine objects destroyed here

    // Barrier after ADIOS2 cleanup
    MPI_Barrier(MPI_COMM_WORLD);

    // Verify timestep 1 data
    for (size_t j = 0; j < local_ny; j++) {
      for (size_t i = 0; i < local_nx; i++) {
        size_t global_i = rank * local_nx + i;
        double expected = read_step * 1000.0 + global_i * 10.0 + j;
        TEST_ASSERT(std::abs(loaded.f(i, j) - expected) < 1e-10,
                    "Wrong data in timestep read");
      }
    }

    if (rank == 0) {
      std::cout << "    - Each rank read timestep " << read_step << " correctly" << std::endl;
    }
  }

  MPI_Barrier(MPI_COMM_WORLD);

  if (rank == 0) {
    std::cout << std::endl << "=== All Parallel ADIOS2 Tests Passed ===" << std::endl;
  }

  // Extra barrier to ensure all ADIOS2 cleanup is complete before finalize
  MPI_Barrier(MPI_COMM_WORLD);

  MPI_Finalize();
  return 0;
}

#else

int main(int argc, char** argv) {
  std::cout << "ADIOS2 and/or MPI support not available - tests skipped" << std::endl;
  std::cout << "Build with -DNDARRAY_USE_ADIOS2=TRUE -DNDARRAY_USE_MPI=TRUE to enable" << std::endl;
  return 0;
}

#endif
