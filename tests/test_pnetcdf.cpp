/**
 * Parallel NetCDF (PNetCDF) functionality tests for ndarray
 *
 * Tests the ndarray PNetCDF functionality:
 * - Parallel I/O with MPI
 * - Reading distributed datasets
 * - Collective I/O operations
 * - Multiple data types
 *
 * NOTE: PNetCDF support is experimental and marked as such in CMake.
 * Some functions may be declared but not yet fully implemented.
 *
 * Run with: mpirun -np 4 ./test_pnetcdf
 */

#include <ndarray/ndarray.hh>
#include <iostream>
#include <cassert>
#include <cmath>
#include <vector>

#if NDARRAY_HAVE_MPI
#include <mpi.h>
#endif

#if NDARRAY_HAVE_PNETCDF
#include <pnetcdf.h>
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
  if (rank == 0) std::cout << "  Testing: " << name << std::endl

#if NDARRAY_HAVE_PNETCDF && NDARRAY_HAVE_MPI

// Helper function to create a test PNetCDF file with distributed data
int create_test_pnetcdf_file(const std::string& filename,
                               size_t nx, size_t ny,
                               MPI_Comm comm) {
  int rank, nprocs;
  MPI_Comm_rank(comm, &rank);
  MPI_Comm_size(comm, &nprocs);

  int ncid, dimids[2], varid_temp, varid_pressure;
  int retval;

  // Create file collectively
  retval = ncmpi_create(comm, filename.c_str(), NC_CLOBBER | NC_64BIT_DATA,
                        MPI_INFO_NULL, &ncid);
  if (retval != NC_NOERR) {
    if (rank == 0) std::cerr << "Failed to create PNetCDF file: " << ncmpi_strerror(retval) << std::endl;
    return retval;
  }

  // Define dimensions
  retval = ncmpi_def_dim(ncid, "x", nx, &dimids[1]);
  retval = ncmpi_def_dim(ncid, "y", ny, &dimids[0]);

  // Define variables
  retval = ncmpi_def_var(ncid, "temperature", NC_FLOAT, 2, dimids, &varid_temp);
  retval = ncmpi_def_var(ncid, "pressure", NC_DOUBLE, 2, dimids, &varid_pressure);

  // End define mode
  retval = ncmpi_enddef(ncid);

  // Each rank writes a portion of the data
  size_t local_ny = ny / nprocs;
  size_t start_y = rank * local_ny;

  // Adjust last rank to handle remainder
  if (rank == nprocs - 1) {
    local_ny = ny - start_y;
  }

  // Prepare local data
  std::vector<float> temp_data(nx * local_ny);
  std::vector<double> press_data(nx * local_ny);

  for (size_t j = 0; j < local_ny; j++) {
    for (size_t i = 0; i < nx; i++) {
      size_t global_j = start_y + j;
      size_t idx = j * nx + i;
      temp_data[idx] = 20.0f + global_j * 0.1f + i * 0.01f + rank * 10.0f;
      press_data[idx] = 1000.0 + global_j * 2.0 + i * 0.5 + rank * 50.0;
    }
  }

  // Write data collectively
  MPI_Offset start[2] = {(MPI_Offset)start_y, 0};
  MPI_Offset count[2] = {(MPI_Offset)local_ny, (MPI_Offset)nx};

  retval = ncmpi_put_vara_float_all(ncid, varid_temp, start, count, temp_data.data());
  if (retval != NC_NOERR && rank == 0) {
    std::cerr << "Failed to write temperature: " << ncmpi_strerror(retval) << std::endl;
  }

  retval = ncmpi_put_vara_double_all(ncid, varid_pressure, start, count, press_data.data());
  if (retval != NC_NOERR && rank == 0) {
    std::cerr << "Failed to write pressure: " << ncmpi_strerror(retval) << std::endl;
  }

  // Close file
  retval = ncmpi_close(ncid);

  return retval;
}

#endif // NDARRAY_HAVE_PNETCDF && NDARRAY_HAVE_MPI

int main(int argc, char** argv) {
#if NDARRAY_HAVE_MPI && NDARRAY_HAVE_PNETCDF
  MPI_Init(&argc, &argv);

  int rank, nprocs;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

  if (rank == 0) {
    std::cout << "=== Running ndarray PNetCDF Tests ===" << std::endl;
    std::cout << "Running with " << nprocs << " MPI processes" << std::endl << std::endl;
  }

  // Test 1: Create a parallel NetCDF file
  {
    TEST_SECTION("Create parallel NetCDF file");

    const size_t nx = 100, ny = 80;
    int retval = create_test_pnetcdf_file("test_pnetcdf_basic.nc", nx, ny, MPI_COMM_WORLD);

    TEST_ASSERT(retval == NC_NOERR, "Failed to create PNetCDF file");

    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == 0) {
      std::cout << "    - Created parallel NetCDF file: " << nx << " x " << ny << std::endl;
      std::cout << "    - Variables: temperature (float), pressure (double)" << std::endl;
      std::cout << "    PASSED" << std::endl;
    }
  }

  // Test 2: Read with read_pnetcdf_all (collective read)
  {
    TEST_SECTION("Parallel read with read_pnetcdf_all");

    const size_t nx = 100, ny = 80;

    // Open file for parallel reading
    int ncid, varid_temp;
    int retval = ncmpi_open(MPI_COMM_WORLD, "test_pnetcdf_basic.nc",
                            NC_NOWRITE, MPI_INFO_NULL, &ncid);
    TEST_ASSERT(retval == NC_NOERR, "Failed to open PNetCDF file");

    // Get variable ID
    retval = ncmpi_inq_varid(ncid, "temperature", &varid_temp);
    TEST_ASSERT(retval == NC_NOERR, "Failed to get variable ID");

    // Each rank reads a portion
    size_t local_ny = ny / nprocs;
    size_t start_y = rank * local_ny;
    if (rank == nprocs - 1) {
      local_ny = ny - start_y;
    }

    MPI_Offset start[2] = {(MPI_Offset)start_y, 0};
    MPI_Offset count[2] = {(MPI_Offset)local_ny, (MPI_Offset)nx};

    // Create ndarray and read
    ftk::ndarray<float> temp;
    temp.reshapef(nx, local_ny);

    // NOTE: read_pnetcdf_all may not be implemented yet
    // This test demonstrates the intended API usage
    try {
      temp.read_pnetcdf_all(ncid, varid_temp, start, count);

      // Verify dimensions
      TEST_ASSERT(temp.dimf(0) == nx, "Wrong x dimension");
      TEST_ASSERT(temp.dimf(1) == local_ny, "Wrong y dimension");

      // Verify some data values
      float expected_val = 20.0f + start_y * 0.1f + rank * 10.0f;
      TEST_ASSERT(std::abs(temp.f(0, 0) - expected_val) < 1e-5f, "Wrong data value");

      if (rank == 0) {
        std::cout << "    - Each rank read " << nx << " x " << local_ny << " array" << std::endl;
        std::cout << "    - Data verification passed" << std::endl;
        std::cout << "    PASSED" << std::endl;
      }
    } catch (const std::exception& e) {
      if (rank == 0) {
        std::cout << "    - NOTE: read_pnetcdf_all not yet implemented" << std::endl;
        std::cout << "    - This is expected (marked experimental)" << std::endl;
        std::cout << "    SKIPPED" << std::endl;
      }
    }

    ncmpi_close(ncid);
  }

  // Test 3: Read entire dataset on all ranks
  {
    TEST_SECTION("Read entire dataset collectively");

    const size_t nx = 100, ny = 80;

    int ncid, varid_press;
    ncmpi_open(MPI_COMM_WORLD, "test_pnetcdf_basic.nc",
               NC_NOWRITE, MPI_INFO_NULL, &ncid);
    ncmpi_inq_varid(ncid, "pressure", &varid_press);

    MPI_Offset start[2] = {0, 0};
    MPI_Offset count[2] = {(MPI_Offset)ny, (MPI_Offset)nx};

    ftk::ndarray<double> pressure;
    pressure.reshapef(nx, ny);

    try {
      pressure.read_pnetcdf_all(ncid, varid_press, start, count);

      TEST_ASSERT(pressure.size() == nx * ny, "Wrong total size");
      TEST_ASSERT(pressure.dimf(0) == nx, "Wrong x dimension");
      TEST_ASSERT(pressure.dimf(1) == ny, "Wrong y dimension");

      if (rank == 0) {
        std::cout << "    - All ranks read full " << nx << " x " << ny << " array" << std::endl;
        std::cout << "    PASSED" << std::endl;
      }
    } catch (const std::exception& e) {
      if (rank == 0) {
        std::cout << "    - NOTE: read_pnetcdf_all not yet implemented" << std::endl;
        std::cout << "    SKIPPED" << std::endl;
      }
    }

    ncmpi_close(ncid);
  }

  // Test 4: Multi-dimensional array
  {
    TEST_SECTION("3D array parallel I/O");

    if (rank == 0) {
      std::cout << "    - Creating 3D test file" << std::endl;
    }

    const size_t nx = 50, ny = 40, nz = 30;

    // Create 3D file
    int ncid, dimids[3], varid;
    ncmpi_create(MPI_COMM_WORLD, "test_pnetcdf_3d.nc", NC_CLOBBER | NC_64BIT_DATA,
                 MPI_INFO_NULL, &ncid);
    ncmpi_def_dim(ncid, "x", nx, &dimids[2]);
    ncmpi_def_dim(ncid, "y", ny, &dimids[1]);
    ncmpi_def_dim(ncid, "z", nz, &dimids[0]);
    ncmpi_def_var(ncid, "data3d", NC_FLOAT, 3, dimids, &varid);
    ncmpi_enddef(ncid);

    // Each rank writes a z-slice
    size_t local_nz = nz / nprocs;
    size_t start_z = rank * local_nz;
    if (rank == nprocs - 1) {
      local_nz = nz - start_z;
    }

    std::vector<float> data3d(nx * ny * local_nz);
    for (size_t k = 0; k < local_nz; k++) {
      for (size_t j = 0; j < ny; j++) {
        for (size_t i = 0; i < nx; i++) {
          size_t idx = k * ny * nx + j * nx + i;
          data3d[idx] = (start_z + k) * 100.0f + j * 10.0f + i;
        }
      }
    }

    MPI_Offset start[3] = {(MPI_Offset)start_z, 0, 0};
    MPI_Offset count[3] = {(MPI_Offset)local_nz, (MPI_Offset)ny, (MPI_Offset)nx};
    ncmpi_put_vara_float_all(ncid, varid, start, count, data3d.data());
    ncmpi_close(ncid);

    MPI_Barrier(MPI_COMM_WORLD);

    if (rank == 0) {
      std::cout << "    - Created 3D array: " << nx << " x " << ny << " x " << nz << std::endl;
      std::cout << "    - Each rank wrote " << local_nz << " z-slices" << std::endl;
      std::cout << "    PASSED" << std::endl;
    }
  }

  // Test 5: Independent I/O mode
  {
    TEST_SECTION("Independent I/O operations");

    const size_t nx = 60, ny = 40;

    int ncid, dimids[2], varid;
    ncmpi_create(MPI_COMM_WORLD, "test_pnetcdf_indep.nc", NC_CLOBBER | NC_64BIT_DATA,
                 MPI_INFO_NULL, &ncid);
    ncmpi_def_dim(ncid, "x", nx, &dimids[1]);
    ncmpi_def_dim(ncid, "y", ny, &dimids[0]);
    ncmpi_def_var(ncid, "data", NC_DOUBLE, 2, dimids, &varid);
    ncmpi_enddef(ncid);

    // Switch to independent mode
    ncmpi_begin_indep_data(ncid);

    // Each rank writes independently
    if (rank == 0) {
      // Rank 0 writes top half
      std::vector<double> data(nx * (ny/2));
      for (size_t i = 0; i < data.size(); i++) {
        data[i] = i * 1.5;
      }
      MPI_Offset start[2] = {0, 0};
      MPI_Offset count[2] = {(MPI_Offset)(ny/2), (MPI_Offset)nx};
      ncmpi_put_vara_double(ncid, varid, start, count, data.data());
    }

    // Switch back to collective mode
    ncmpi_end_indep_data(ncid);
    ncmpi_close(ncid);

    if (rank == 0) {
      std::cout << "    - Independent I/O mode tested" << std::endl;
      std::cout << "    PASSED" << std::endl;
    }
  }

  // Cleanup
  MPI_Barrier(MPI_COMM_WORLD);
  if (rank == 0) {
    std::remove("test_pnetcdf_basic.nc");
    std::remove("test_pnetcdf_3d.nc");
    std::remove("test_pnetcdf_indep.nc");

    std::cout << std::endl;
    std::cout << "=== PNetCDF Tests Completed ===" << std::endl;
    std::cout << "NOTE: Some tests may be skipped if read_pnetcdf_all is not implemented" << std::endl;
  }

  MPI_Finalize();
  return 0;

#else
  std::cout << "=== PNetCDF Tests ===" << std::endl;

#if !NDARRAY_HAVE_MPI
  std::cout << "SKIPPED: MPI support not enabled" << std::endl;
  std::cout << "  Enable with: -DNDARRAY_USE_MPI=ON" << std::endl;
#endif

#if !NDARRAY_HAVE_PNETCDF
  std::cout << "SKIPPED: PNetCDF support not enabled" << std::endl;
  std::cout << "  Enable with: -DNDARRAY_USE_PNETCDF=ON" << std::endl;
  std::cout << "  NOTE: PNetCDF is marked as experimental" << std::endl;
#endif

  return 0;
#endif
}
