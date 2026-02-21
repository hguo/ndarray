/**
 * ADIOS2 functionality tests for ndarray
 *
 * Tests the ndarray ADIOS2 (Adaptable I/O System) functionality:
 * - BP (Binary Pack) file I/O
 * - Single and multi-step (time-series) data
 * - Various data types
 * - MPI parallel I/O (when MPI is available)
 * - High-level and low-level ADIOS2 APIs
 *
 * ADIOS2 is a high-performance I/O library supporting:
 * - BP4/BP5 formats
 * - Streaming engines (SST, InSituMPI)
 * - Compression (zlib, bzip2, blosc, etc.)
 * - Parallel I/O with MPI
 */

#include <ndarray/ndarray.hh>
#include <iostream>
#include <cassert>
#include <cmath>
#include <fstream>

#if NDARRAY_HAVE_ADIOS2
#include <adios2.h>
#endif

#if NDARRAY_HAVE_MPI
#include <mpi.h>
#endif

#define TEST_ASSERT(condition, message) \
  do { \
    if (!(condition)) { \
      std::cerr << "FAILED: " << message << std::endl; \
      std::cerr << "  at " << __FILE__ << ":" << __LINE__ << std::endl; \
      return 1; \
    } \
  } while (0)

#define TEST_SECTION(name) \
  std::cout << "  Testing: " << name << std::endl

int main(int argc, char** argv) {
#if NDARRAY_HAVE_ADIOS2

#if NDARRAY_HAVE_MPI
  MPI_Init(&argc, &argv);
  int rank = 0, nprocs = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

  if (rank == 0) {
#endif
    std::cout << "=== Running ndarray ADIOS2 Tests ===" << std::endl;
#if NDARRAY_HAVE_MPI
    std::cout << "Running with " << nprocs << " MPI process(es)" << std::endl;
#else
    std::cout << "Running in serial mode (MPI not available)" << std::endl;
#endif
    std::cout << std::endl;
#if NDARRAY_HAVE_MPI
  }
#endif

  // Test 1: Write and read single-step BP file
  {
    TEST_SECTION("Write and read single-step BP file");

    // Write data (only rank 0 for serial-style test)
#if NDARRAY_HAVE_MPI
    if (rank == 0) {
#endif
    {
      ftk::ndarray<float> data;
      data.reshapef(10, 20);

      for (size_t j = 0; j < 20; j++) {
        for (size_t i = 0; i < 10; i++) {
          data.f(i, j) = i * 10.0f + j;
        }
      }

#if NDARRAY_HAVE_MPI
      // Use MPI_COMM_SELF for independent I/O to avoid collective operations
      adios2::ADIOS adios(MPI_COMM_SELF);
#else
      adios2::ADIOS adios;
#endif
      adios2::IO io = adios.DeclareIO("TestIO");
      io.SetEngine("BP4");

      adios2::Engine writer = io.Open("test_adios2_single.bp", adios2::Mode::Write);

      // Define variable
      adios2::Variable<float> var = io.DefineVariable<float>(
        "temperature",
        {20, 10},  // Global dimensions (Fortran order reversed)
        {0, 0},    // Offset
        {20, 10}   // Local dimensions
      );

      writer.BeginStep();
      writer.Put(var, data.data());
      writer.EndStep();
      writer.Close();

      std::cout << "    - Wrote BP file: 10 x 20 array" << std::endl;
    }
#if NDARRAY_HAVE_MPI
    }
    MPI_Barrier(MPI_COMM_SELF);  // Wait for rank 0 to finish writing

    // All ranks read (using their own ADIOS context)
    {
#endif
    // Read data using high-level API
    {
      ftk::ndarray<float> loaded = ftk::ndarray<float>::from_bp(
        "test_adios2_single.bp", "temperature", 0);

      TEST_ASSERT(loaded.dimf(0) == 10, "Wrong x dimension");
      TEST_ASSERT(loaded.dimf(1) == 20, "Wrong y dimension");
      TEST_ASSERT(std::abs(loaded.f(5, 10) - 60.0f) < 1e-5f, "Wrong data value");

      std::cout << "    - Read BP file using from_bp()" << std::endl;
      std::cout << "    - Data verification passed" << std::endl;
    }

    std::cout << "    PASSED" << std::endl;
  }

  // Test 2: Multi-step (time-series) data
  {
    TEST_SECTION("Multi-step time-series data");

    const size_t nx = 8, ny = 12;
    const size_t nsteps = 5;

    // Write time series
    {
#if NDARRAY_HAVE_MPI
      adios2::ADIOS adios(MPI_COMM_SELF);
#else
      adios2::ADIOS adios;
#endif
      adios2::IO io = adios.DeclareIO("TimeSeriesIO");
      adios2::Engine writer = io.Open("test_adios2_timeseries.bp", adios2::Mode::Write);

      adios2::Variable<double> var = io.DefineVariable<double>(
        "velocity",
        {ny, nx},
        {0, 0},
        {ny, nx}
      );

      for (size_t step = 0; step < nsteps; step++) {
        ftk::ndarray<double> data;
        data.reshapef(nx, ny);

        for (size_t j = 0; j < ny; j++) {
          for (size_t i = 0; i < nx; i++) {
            data.f(i, j) = step * 100.0 + j * 10.0 + i;
          }
        }

        writer.BeginStep();
        writer.Put(var, data.data());
        writer.EndStep();
      }

      writer.Close();
      std::cout << "    - Wrote " << nsteps << " timesteps" << std::endl;
    }

    // Read specific timesteps
    {
      // Read step 0
      ftk::ndarray<double> step0 = ftk::ndarray<double>::from_bp(
        "test_adios2_timeseries.bp", "velocity", 0);
      TEST_ASSERT(std::abs(step0.f(0, 0) - 0.0) < 1e-10, "Wrong step 0 value");

      // Read step 2
      ftk::ndarray<double> step2 = ftk::ndarray<double>::from_bp(
        "test_adios2_timeseries.bp", "velocity", 2);
      TEST_ASSERT(std::abs(step2.f(0, 0) - 200.0) < 1e-10, "Wrong step 2 value");

      // Read step 4
      ftk::ndarray<double> step4 = ftk::ndarray<double>::from_bp(
        "test_adios2_timeseries.bp", "velocity", 4);
      TEST_ASSERT(std::abs(step4.f(0, 0) - 400.0) < 1e-10, "Wrong step 4 value");

      std::cout << "    - Read individual timesteps successfully" << std::endl;
    }

    // Read all steps (use -2 for NDARRAY_ADIOS2_STEPS_ALL)
    {
      ftk::ndarray<double> all_steps = ftk::ndarray<double>::from_bp(
        "test_adios2_timeseries.bp", "velocity", -2);

      // When reading all steps, data should be 3D: [nsteps, ny, nx] in C-order
      TEST_ASSERT(all_steps.nd() == 3, "Should be 3D when reading all steps");
      // In Fortran order: [nx, ny, nsteps]
      TEST_ASSERT(all_steps.dimf(0) == nx, "Wrong x dimension");
      TEST_ASSERT(all_steps.dimf(1) == ny, "Wrong y dimension");
      TEST_ASSERT(all_steps.dimf(2) == nsteps, "Wrong time dimension");

      std::cout << "    - Read all steps: " << all_steps.shapef()[0] << " x "
                << all_steps.shapef()[1] << " x " << all_steps.shapef()[2] << std::endl;
    }

    std::cout << "    PASSED" << std::endl;
  }

  // Test 3: Different data types
  {
    TEST_SECTION("Multiple data types");

#if NDARRAY_HAVE_MPI
    adios2::ADIOS adios(MPI_COMM_SELF);
#else
    adios2::ADIOS adios;
#endif
    adios2::IO io = adios.DeclareIO("MultiTypeIO");
    adios2::Engine writer = io.Open("test_adios2_types.bp", adios2::Mode::Write);

    // Float array
    ftk::ndarray<float> float_data;
    float_data.reshapef(5, 5);
    for (size_t i = 0; i < float_data.size(); i++) {
      float_data[i] = i * 1.5f;
    }

    // Double array
    ftk::ndarray<double> double_data;
    double_data.reshapef(4, 6);
    for (size_t i = 0; i < double_data.size(); i++) {
      double_data[i] = i * 2.5;
    }

    // Integer array
    ftk::ndarray<int> int_data;
    int_data.reshapef(6, 4);
    for (size_t i = 0; i < int_data.size(); i++) {
      int_data[i] = i * 3;
    }

    // Define and write variables
    auto var_float = io.DefineVariable<float>("float_field", {5, 5}, {0, 0}, {5, 5});
    auto var_double = io.DefineVariable<double>("double_field", {6, 4}, {0, 0}, {6, 4});
    auto var_int = io.DefineVariable<int>("int_field", {4, 6}, {0, 0}, {4, 6});

    writer.BeginStep();
    writer.Put(var_float, float_data.data());
    writer.Put(var_double, double_data.data());
    writer.Put(var_int, int_data.data());
    writer.EndStep();
    writer.Close();

    // Read back
    ftk::ndarray<float> loaded_float = ftk::ndarray<float>::from_bp(
      "test_adios2_types.bp", "float_field", 0);
    ftk::ndarray<double> loaded_double = ftk::ndarray<double>::from_bp(
      "test_adios2_types.bp", "double_field", 0);
    ftk::ndarray<int> loaded_int = ftk::ndarray<int>::from_bp(
      "test_adios2_types.bp", "int_field", 0);

    TEST_ASSERT(std::abs(loaded_float[10] - 15.0f) < 1e-5f, "Wrong float value");
    TEST_ASSERT(std::abs(loaded_double[10] - 25.0) < 1e-10, "Wrong double value");
    TEST_ASSERT(loaded_int[10] == 30, "Wrong int value");

    std::cout << "    - Wrote and read float, double, and int arrays" << std::endl;
    std::cout << "    PASSED" << std::endl;
  }

  // Test 4: Low-level ADIOS2 API with read_bp
  {
    TEST_SECTION("Low-level ADIOS2 API");

    // Create test file
    {
#if NDARRAY_HAVE_MPI
      adios2::ADIOS adios(MPI_COMM_SELF);
#else
      adios2::ADIOS adios;
#endif
      adios2::IO io = adios.DeclareIO("LowLevelIO");
      adios2::Engine writer = io.Open("test_adios2_lowlevel.bp", adios2::Mode::Write);

      ftk::ndarray<float> data;
      data.reshapef(16, 16);
      for (size_t i = 0; i < data.size(); i++) {
        data[i] = std::sin(i * 0.1);
      }

      auto var = io.DefineVariable<float>("wave", {16, 16}, {0, 0}, {16, 16});
      writer.BeginStep();
      writer.Put(var, data.data());
      writer.EndStep();
      writer.Close();
    }

    // Read using low-level API
    {
      ftk::ndarray<float> loaded;

#if NDARRAY_HAVE_MPI
      adios2::ADIOS adios(MPI_COMM_SELF);
#else
      adios2::ADIOS adios;
#endif
      adios2::IO io = adios.DeclareIO("LowLevelReadIO");
      adios2::Engine reader = io.Open("test_adios2_lowlevel.bp", adios2::Mode::ReadRandomAccess);

      loaded.read_bp(io, reader, "wave", 0);
      reader.Close();

      TEST_ASSERT(loaded.dimf(0) == 16, "Wrong dimension from low-level API");
      TEST_ASSERT(loaded.dimf(1) == 16, "Wrong dimension from low-level API");
      TEST_ASSERT(std::abs(loaded.f(5, 5) - std::sin(85 * 0.1)) < 1e-5f, "Wrong value from low-level API");

      std::cout << "    - Low-level API read_bp() works correctly" << std::endl;
    }

    std::cout << "    PASSED" << std::endl;
  }

  // Test 5: 3D arrays
  {
    TEST_SECTION("3D array I/O");

    const size_t nx = 8, ny = 10, nz = 6;

    // Write 3D array
    {
      ftk::ndarray<double> data3d;
      data3d.reshapef(nx, ny, nz);

      for (size_t k = 0; k < nz; k++) {
        for (size_t j = 0; j < ny; j++) {
          for (size_t i = 0; i < nx; i++) {
            data3d.f(i, j, k) = k * 100.0 + j * 10.0 + i;
          }
        }
      }

#if NDARRAY_HAVE_MPI
      adios2::ADIOS adios(MPI_COMM_SELF);
#else
      adios2::ADIOS adios;
#endif
      adios2::IO io = adios.DeclareIO("3DIO");
      adios2::Engine writer = io.Open("test_adios2_3d.bp", adios2::Mode::Write);

      auto var = io.DefineVariable<double>("data3d", {nz, ny, nx}, {0, 0, 0}, {nz, ny, nx});

      writer.BeginStep();
      writer.Put(var, data3d.data());
      writer.EndStep();
      writer.Close();

      std::cout << "    - Wrote 3D array: " << nx << " x " << ny << " x " << nz << std::endl;
    }

    // Read back
    {
      ftk::ndarray<double> loaded = ftk::ndarray<double>::from_bp(
        "test_adios2_3d.bp", "data3d", 0);

      TEST_ASSERT(loaded.nd() == 3, "Should be 3D");
      TEST_ASSERT(loaded.dimf(0) == nx, "Wrong x dimension");
      TEST_ASSERT(loaded.dimf(1) == ny, "Wrong y dimension");
      TEST_ASSERT(loaded.dimf(2) == nz, "Wrong z dimension");
      TEST_ASSERT(std::abs(loaded.f(3, 5, 2) - 253.0) < 1e-10, "Wrong 3D value");

      std::cout << "    - Read 3D array successfully" << std::endl;
    }

    std::cout << "    PASSED" << std::endl;
  }

  // Test 6: Using read_bp with filename (convenience function)
  {
    TEST_SECTION("Convenience read_bp(filename, varname, step)");

    // Create test file
    {
#if NDARRAY_HAVE_MPI
      adios2::ADIOS adios(MPI_COMM_SELF);
#else
      adios2::ADIOS adios;
#endif
      adios2::IO io = adios.DeclareIO("ConvenienceIO");
      adios2::Engine writer = io.Open("test_adios2_convenience.bp", adios2::Mode::Write);

      ftk::ndarray<float> data;
      data.reshapef(20, 15);
      for (size_t i = 0; i < data.size(); i++) {
        data[i] = i * 0.5f;
      }

      auto var = io.DefineVariable<float>("field", {15, 20}, {0, 0}, {15, 20});
      writer.BeginStep();
      writer.Put(var, data.data());
      writer.EndStep();
      writer.Close();
    }

    // Read using convenience function
    {
      ftk::ndarray<float> loaded;
#if NDARRAY_HAVE_MPI
      loaded.read_bp("test_adios2_convenience.bp", "field", 0, MPI_COMM_SELF);
#else
      loaded.read_bp("test_adios2_convenience.bp", "field", 0);
#endif

      TEST_ASSERT(loaded.size() == 300, "Wrong size from convenience function");
      TEST_ASSERT(std::abs(loaded[100] - 50.0f) < 1e-5f, "Wrong value from convenience function");

      std::cout << "    - Convenience function read_bp() works" << std::endl;
    }

    std::cout << "    PASSED" << std::endl;
  }

#if NDARRAY_HAVE_MPI
  // Test 7: Parallel I/O with MPI
  if (nprocs > 1) {
    TEST_SECTION("Parallel I/O with MPI");

    const size_t global_nx = 100, global_ny = 80;
    size_t local_ny = global_ny / nprocs;
    size_t start_y = rank * local_ny;

    // Adjust for last rank
    if (rank == nprocs - 1) {
      local_ny = global_ny - start_y;
    }

    // Write parallel
    {
      ftk::ndarray<double> local_data;
      local_data.reshapef(global_nx, local_ny);

      for (size_t j = 0; j < local_ny; j++) {
        for (size_t i = 0; i < global_nx; i++) {
          local_data.f(i, j) = rank * 1000.0 + (start_y + j) * 10.0 + i;
        }
      }

      // Use MPI_COMM_WORLD for true parallel I/O
      adios2::ADIOS adios(MPI_COMM_WORLD);
      adios2::IO io = adios.DeclareIO("ParallelIO");
      adios2::Engine writer = io.Open("test_adios2_parallel.bp", adios2::Mode::Write);

      auto var = io.DefineVariable<double>(
        "parallel_data",
        {global_ny, global_nx},
        {start_y, 0},
        {local_ny, global_nx}
      );

      writer.BeginStep();
      writer.Put(var, local_data.data());
      writer.EndStep();
      writer.Close();

      if (rank == 0) {
        std::cout << "    - " << nprocs << " ranks wrote " << global_nx << " x " << global_ny << " array" << std::endl;
      }
    }

    MPI_Barrier(MPI_COMM_WORLD);

    // Read on rank 0
    if (rank == 0) {
      ftk::ndarray<double> full_data = ftk::ndarray<double>::from_bp(
        "test_adios2_parallel.bp", "parallel_data", 0, MPI_COMM_WORLD);

      TEST_ASSERT(full_data.dimf(0) == global_nx, "Wrong global x dimension");
      TEST_ASSERT(full_data.dimf(1) == global_ny, "Wrong global y dimension");

      std::cout << "    - Rank 0 read full array successfully" << std::endl;
      std::cout << "    PASSED" << std::endl;
    }
  } else {
    if (rank == 0) {
      std::cout << "  Testing: Parallel I/O with MPI" << std::endl;
      std::cout << "    - SKIPPED (requires multiple MPI ranks)" << std::endl;
    }
  }
#endif

  // Cleanup
  std::remove("test_adios2_single.bp");
  std::remove("test_adios2_timeseries.bp");
  std::remove("test_adios2_types.bp");
  std::remove("test_adios2_lowlevel.bp");
  std::remove("test_adios2_3d.bp");
  std::remove("test_adios2_convenience.bp");
#if NDARRAY_HAVE_MPI
  std::remove("test_adios2_parallel.bp");

  MPI_Barrier(MPI_COMM_WORLD);

  if (rank == 0) {
#endif
    std::cout << std::endl;
    std::cout << "=== All ADIOS2 Tests Passed ===" << std::endl;
#if NDARRAY_HAVE_MPI
  }

  MPI_Finalize();
#endif

  return 0;

#else
  std::cout << "=== ADIOS2 Tests ===" << std::endl;
  std::cout << "SKIPPED: ADIOS2 support not enabled" << std::endl;
  std::cout << "  Enable with: -DNDARRAY_USE_ADIOS2=ON" << std::endl;
  std::cout << "  Also requires ADIOS2 library installed" << std::endl;
  return 0;
#endif
}
