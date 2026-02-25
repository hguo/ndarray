/**
 * @file test_transpose_distributed_gpu.cpp
 * @brief Tests for GPU-accelerated distributed transpose (MPI + CUDA)
 *
 * Tests transpose of arrays that are both:
 * - Distributed across MPI ranks
 * - Stored on GPU devices
 */

#include <ndarray/ndarray.hh>
#include <ndarray/transpose.hh>
#include <iostream>
#include <cmath>
#include <cassert>

#if NDARRAY_HAVE_MPI && NDARRAY_HAVE_CUDA

#include <mpi.h>
#include <cuda_runtime.h>

using namespace ftk;

#define TEST_ASSERT(cond, msg) \
  do { \
    if (!(cond)) { \
      std::cerr << "[Rank " << rank << " FAILED] " << msg \
                << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
      MPI_Abort(MPI_COMM_WORLD, 1); \
    } \
  } while(0)

#define TEST_SUCCESS(msg) \
  do { \
    if (rank == 0) std::cout << "[PASSED] " << msg << std::endl; \
  } while(0)

/**
 * Initialize distributed array on GPU with pattern
 */
template <typename T>
void init_distributed_gpu_array(ndarray<T>& arr) {
  const size_t nd = arr.nd();
  const lattice& local_core = arr.local_core();

  // Initialize on CPU first
  std::vector<size_t> idx(nd);
  for (size_t i = 0; i < arr.nelem(); i++) {
    size_t tmp = i;
    for (size_t d = 0; d < nd; d++) {
      idx[d] = tmp % arr.dimf(d);
      tmp /= arr.dimf(d);
    }

    auto global_idx = arr.local_to_global(idx);
    T value = 0;
    for (size_t d = 0; d < nd; d++) {
      value += global_idx[d];
    }
    arr[i] = value;
  }

  // Move to GPU
  arr.to_device(NDARRAY_DEVICE_CUDA);
}

/**
 * Verify GPU array has correct transposed pattern
 */
template <typename T>
bool verify_transposed_gpu(const ndarray<T>& arr, const std::vector<size_t>& axes) {
  // Move to CPU for verification
  ndarray<T> cpu_arr = arr;
  cpu_arr.to_host();

  const size_t nd = arr.nd();
  for (size_t i = 0; i < cpu_arr.nelem(); i++) {
    std::vector<size_t> transposed_local_idx(nd);
    size_t tmp = i;
    for (size_t d = 0; d < nd; d++) {
      transposed_local_idx[d] = tmp % cpu_arr.dimf(d);
      tmp /= cpu_arr.dimf(d);
    }

    auto transposed_global_idx = cpu_arr.local_to_global(transposed_local_idx);

    std::vector<size_t> original_global_idx(nd);
    for (size_t d = 0; d < nd; d++) {
      original_global_idx[axes[d]] = transposed_global_idx[d];
    }

    T expected = 0;
    for (size_t d = 0; d < nd; d++) {
      expected += original_global_idx[d];
    }

    if (std::abs(cpu_arr[i] - expected) > 1e-10) {
      return false;
    }
  }
  return true;
}

/**
 * Test 1: 2D GPU distributed transpose
 */
void test_2d_gpu_distributed(MPI_Comm comm, int rank, int nprocs) {
  if (nprocs < 2) {
    if (rank == 0) std::cout << "[SKIPPED] 2D GPU distributed (requires >=2 ranks)" << std::endl;
    return;
  }

  // Create 1000×800 array, decompose along dimension 0
  ndarray<double> arr;
  arr.decompose(comm, {1000, 800}, nprocs,
                {static_cast<size_t>(nprocs), 0}, {0, 0});

  init_distributed_gpu_array(arr);

  // Transpose on GPU
  auto transposed = ftk::transpose(arr, {1, 0});

  // Verify on GPU
  TEST_ASSERT(transposed.is_on_device(), "Result should be on GPU");
  TEST_ASSERT(transposed.is_distributed(), "Result should be distributed");
  TEST_ASSERT(transposed.global_lattice().size(0) == 800, "Dim 0 transposed");
  TEST_ASSERT(transposed.global_lattice().size(1) == 1000, "Dim 1 transposed");

  // Verify decomposition
  auto new_decomp = transposed.decomp_pattern();
  TEST_ASSERT(new_decomp[0] == 0, "New dim 0 not decomposed");
  TEST_ASSERT(new_decomp[1] == static_cast<size_t>(nprocs), "New dim 1 decomposed");

  // Verify correctness
  TEST_ASSERT(verify_transposed_gpu(transposed, {1, 0}), "Data correct");

  TEST_SUCCESS("2D GPU distributed transpose");
}

/**
 * Test 2: 3D GPU distributed transpose
 */
void test_3d_gpu_distributed(MPI_Comm comm, int rank, int nprocs) {
  if (nprocs < 2) {
    if (rank == 0) std::cout << "[SKIPPED] 3D GPU distributed" << std::endl;
    return;
  }

  // Create 100×80×60 array
  size_t nx = 100, ny = 80, nz = 60;
  std::vector<size_t> decomp = {static_cast<size_t>(nprocs), 0, 0};

  ndarray<double> arr;
  arr.decompose(comm, {nx, ny, nz}, nprocs, decomp, {0, 0, 0});

  init_distributed_gpu_array(arr);

  // Transpose: {1, 0, 2}
  auto transposed = ftk::transpose(arr, {1, 0, 2});

  // Verify
  TEST_ASSERT(transposed.is_on_device(), "On GPU");
  TEST_ASSERT(transposed.global_lattice().size(0) == ny, "Dim 0");
  TEST_ASSERT(transposed.global_lattice().size(1) == nx, "Dim 1");
  TEST_ASSERT(transposed.global_lattice().size(2) == nz, "Dim 2");

  TEST_ASSERT(verify_transposed_gpu(transposed, {1, 0, 2}), "Data correct");

  TEST_SUCCESS("3D GPU distributed transpose");
}

/**
 * Test 3: Vector field on GPU distributed
 */
void test_vector_field_gpu_distributed(MPI_Comm comm, int rank, int nprocs) {
  if (nprocs < 2) {
    if (rank == 0) std::cout << "[SKIPPED] Vector field GPU distributed" << std::endl;
    return;
  }

  // Velocity: [3, 200, 160]
  ndarray<float> velocity;
  velocity.decompose(comm, {3, 200, 160}, nprocs,
                    {0, static_cast<size_t>(nprocs), 0}, {0, 0, 0});
  velocity.set_multicomponents(1);

  init_distributed_gpu_array(velocity);

  // Transpose spatial dims
  auto transposed = ftk::transpose(velocity, {0, 2, 1});

  // Verify
  TEST_ASSERT(transposed.is_on_device(), "On GPU");
  TEST_ASSERT(transposed.multicomponents() == 1, "Metadata preserved");
  TEST_ASSERT(transposed.global_lattice().size(0) == 3, "Component dim");
  TEST_ASSERT(transposed.global_lattice().size(1) == 160, "Transposed");
  TEST_ASSERT(transposed.global_lattice().size(2) == 200, "Transposed");

  TEST_SUCCESS("Vector field GPU distributed transpose");
}

/**
 * Test 4: Performance comparison
 */
void test_performance_gpu_vs_cpu_distributed(MPI_Comm comm, int rank, int nprocs) {
  if (nprocs < 2) {
    if (rank == 0) std::cout << "[SKIPPED] Performance test" << std::endl;
    return;
  }

  const size_t m = 2048, n = 2048;

  if (rank == 0) {
    std::cout << "\n=== Performance: " << m << "×" << n
              << " distributed across " << nprocs << " ranks ===" << std::endl;
  }

  // CPU version
  ndarray<float> cpu_arr;
  cpu_arr.decompose(comm, {m, n}, nprocs,
                   {static_cast<size_t>(nprocs), 0}, {0, 0});

  for (size_t i = 0; i < cpu_arr.nelem(); i++) {
    cpu_arr[i] = static_cast<float>(i);
  }

  MPI_Barrier(comm);
  auto cpu_start = MPI_Wtime();
  auto cpu_result = ftk::transpose(cpu_arr, {1, 0});
  MPI_Barrier(comm);
  auto cpu_time = MPI_Wtime() - cpu_start;

  // GPU version
  ndarray<float> gpu_arr = cpu_arr;
  gpu_arr.to_device(NDARRAY_DEVICE_CUDA);

  MPI_Barrier(comm);
  cudaDeviceSynchronize();
  auto gpu_start = MPI_Wtime();
  auto gpu_result = ftk::transpose(gpu_arr, {1, 0});
  cudaDeviceSynchronize();
  MPI_Barrier(comm);
  auto gpu_time = MPI_Wtime() - gpu_start;

  if (rank == 0) {
    std::cout << "CPU distributed: " << cpu_time * 1000 << " ms" << std::endl;
    std::cout << "GPU distributed: " << gpu_time * 1000 << " ms" << std::endl;
    if (gpu_time < cpu_time) {
      std::cout << "GPU speedup: " << cpu_time / gpu_time << "×" << std::endl;
    }
  }

  TEST_SUCCESS("Performance comparison GPU vs CPU distributed");
}

/**
 * Test 5: Multiple GPUs (one per rank)
 */
void test_multi_gpu(MPI_Comm comm, int rank, int nprocs) {
  int device_count;
  cudaGetDeviceCount(&device_count);

  if (device_count < nprocs) {
    if (rank == 0) {
      std::cout << "[SKIPPED] Multi-GPU test (need " << nprocs
                << " GPUs, have " << device_count << ")" << std::endl;
    }
    return;
  }

  // Assign one GPU per rank
  int device_id = rank % device_count;
  cudaSetDevice(device_id);

  // Create distributed array
  ndarray<double> arr;
  arr.decompose(comm, {1000, 800}, nprocs,
                {static_cast<size_t>(nprocs), 0}, {0, 0});

  init_distributed_gpu_array(arr);

  // Verify each rank is using correct GPU
  TEST_ASSERT(arr.get_device_id() == device_id, "Correct GPU assigned");

  // Transpose
  auto transposed = ftk::transpose(arr, {1, 0});

  // Verify result still on correct GPU
  TEST_ASSERT(transposed.get_device_id() == device_id, "Result on same GPU");
  TEST_ASSERT(verify_transposed_gpu(transposed, {1, 0}), "Data correct");

  TEST_SUCCESS("Multi-GPU (one per rank)");
}

int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);

  int rank, nprocs;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

  // Check CUDA availability
  int device_count = 0;
  cudaError_t err = cudaGetDeviceCount(&device_count);

  if (err != cudaSuccess || device_count == 0) {
    if (rank == 0) {
      std::cerr << "No CUDA devices available. Skipping GPU+MPI tests." << std::endl;
    }
    MPI_Finalize();
    return 0;
  }

  // Set device (simple: rank % device_count)
  int device_id = rank % device_count;
  cudaSetDevice(device_id);

  if (rank == 0) {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device_id);
    std::cout << "\n========================================" << std::endl;
    std::cout << "GPU + MPI Distributed Transpose Tests" << std::endl;
    std::cout << "MPI Ranks: " << nprocs << std::endl;
    std::cout << "GPUs Available: " << device_count << std::endl;
    std::cout << "GPU: " << prop.name << std::endl;
    std::cout << "========================================\n" << std::endl;
  }

  try {
    test_2d_gpu_distributed(MPI_COMM_WORLD, rank, nprocs);
    test_3d_gpu_distributed(MPI_COMM_WORLD, rank, nprocs);
    test_vector_field_gpu_distributed(MPI_COMM_WORLD, rank, nprocs);
    test_performance_gpu_vs_cpu_distributed(MPI_COMM_WORLD, rank, nprocs);
    test_multi_gpu(MPI_COMM_WORLD, rank, nprocs);

    if (rank == 0) {
      std::cout << "\n========================================" << std::endl;
      std::cout << "All GPU+MPI tests passed!" << std::endl;
      std::cout << "========================================\n" << std::endl;
    }

  } catch (const std::exception& e) {
    std::cerr << "[Rank " << rank << "] Exception: " << e.what() << std::endl;
    MPI_Abort(MPI_COMM_WORLD, 1);
  }

  MPI_Finalize();
  return 0;
}

#else // !NDARRAY_HAVE_MPI || !NDARRAY_HAVE_CUDA

int main() {
  std::cout << "MPI and/or CUDA not available. Skipping GPU+MPI tests." << std::endl;
  return 0;
}

#endif
