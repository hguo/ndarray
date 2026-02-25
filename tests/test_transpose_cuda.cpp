/**
 * @file test_transpose_cuda.cpp
 * @brief Comprehensive tests for CUDA-accelerated transpose
 *
 * Tests GPU transpose operations including:
 * - 2D transpose (optimized kernel)
 * - N-D transpose (general kernel)
 * - Correctness verification against CPU
 * - Performance benchmarking
 * - Metadata preservation
 */

#include <ndarray/ndarray.hh>
#include <ndarray/transpose.hh>
#include <iostream>
#include <cmath>
#include <cassert>
#include <chrono>

#if NDARRAY_HAVE_CUDA
#include <cuda_runtime.h>

using namespace ftk;

// Test utilities
#define TEST_ASSERT(cond, msg) \
  do { \
    if (!(cond)) { \
      std::cerr << "[FAILED] " << msg << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
      std::exit(1); \
    } \
  } while(0)

#define TEST_SUCCESS(msg) \
  std::cout << "[PASSED] " << msg << std::endl

/**
 * @brief Check if two arrays are approximately equal
 */
template <typename T>
bool arrays_equal(const ndarray<T>& a, const ndarray<T>& b, T tolerance = 1e-6) {
  if (a.nelem() != b.nelem()) return false;

  for (size_t i = 0; i < a.nelem(); i++) {
    if (std::abs(a[i] - b[i]) > tolerance) {
      std::cerr << "Mismatch at index " << i << ": " << a[i] << " != " << b[i] << std::endl;
      return false;
    }
  }
  return true;
}

/**
 * Test 1: Basic 2D transpose on GPU
 */
void test_2d_transpose_basic() {
  const size_t rows = 1000;
  const size_t cols = 800;

  // Create array on CPU and initialize
  ndarray<float> arr_cpu;
  arr_cpu.reshapef(rows, cols);

  for (size_t i = 0; i < rows; i++) {
    for (size_t j = 0; j < cols; j++) {
      arr_cpu.f(i, j) = i * cols + j;
    }
  }

  // Move to GPU
  ndarray<float> arr_gpu = arr_cpu;
  arr_gpu.to_device(NDARRAY_DEVICE_CUDA);

  // Transpose on GPU
  auto transposed_gpu = ftk::transpose(arr_gpu);

  // Verify it's on GPU
  TEST_ASSERT(transposed_gpu.is_on_device(), "Result should be on GPU");
  TEST_ASSERT(transposed_gpu.get_device_type() == NDARRAY_DEVICE_CUDA, "Result should be on CUDA");

  // Verify dimensions
  TEST_ASSERT(transposed_gpu.dimf(0) == cols, "Transposed dim 0 should be cols");
  TEST_ASSERT(transposed_gpu.dimf(1) == rows, "Transposed dim 1 should be rows");

  // Move back to CPU for verification
  transposed_gpu.to_host();

  // Verify correctness
  for (size_t i = 0; i < cols; i++) {
    for (size_t j = 0; j < rows; j++) {
      float expected = j * cols + i;
      TEST_ASSERT(std::abs(transposed_gpu.f(i, j) - expected) < 1e-5,
                  "Transposed value should match expected");
    }
  }

  TEST_SUCCESS("2D transpose basic");
}

/**
 * Test 2: 2D transpose with explicit axes
 */
void test_2d_transpose_with_axes() {
  const size_t m = 512;
  const size_t n = 768;

  ndarray<double> arr;
  arr.reshapef(m, n);

  // Initialize with pattern
  for (size_t i = 0; i < m; i++) {
    for (size_t j = 0; j < n; j++) {
      arr.f(i, j) = static_cast<double>(i + j);
    }
  }

  // Move to GPU
  arr.to_device(NDARRAY_DEVICE_CUDA);

  // Transpose with explicit axes
  auto transposed = ftk::transpose(arr, {1, 0});

  // Verify on GPU
  TEST_ASSERT(transposed.is_on_device(), "Result on GPU");
  TEST_ASSERT(transposed.dimf(0) == n, "Dim 0 transposed");
  TEST_ASSERT(transposed.dimf(1) == m, "Dim 1 transposed");

  // Verify correctness on CPU
  transposed.to_host();
  arr.to_host();

  for (size_t i = 0; i < n; i++) {
    for (size_t j = 0; j < m; j++) {
      TEST_ASSERT(std::abs(transposed.f(i, j) - arr.f(j, i)) < 1e-10,
                  "Transposed elements should match");
    }
  }

  TEST_SUCCESS("2D transpose with axes");
}

/**
 * Test 3: 3D transpose on GPU
 */
void test_3d_transpose() {
  const size_t nx = 64, ny = 48, nz = 32;

  ndarray<float> arr;
  arr.reshapef(nx, ny, nz);

  // Initialize
  for (size_t i = 0; i < nx; i++) {
    for (size_t j = 0; j < ny; j++) {
      for (size_t k = 0; k < nz; k++) {
        arr.f(i, j, k) = i * 100 + j * 10 + k;
      }
    }
  }

  // CPU transpose for reference
  auto cpu_transposed = ftk::transpose(arr, {2, 0, 1});

  // GPU transpose
  arr.to_device(NDARRAY_DEVICE_CUDA);
  auto gpu_transposed = ftk::transpose(arr, {2, 0, 1});

  // Verify on GPU
  TEST_ASSERT(gpu_transposed.is_on_device(), "Result on GPU");
  TEST_ASSERT(gpu_transposed.dimf(0) == nz, "Dim 0 correct");
  TEST_ASSERT(gpu_transposed.dimf(1) == nx, "Dim 1 correct");
  TEST_ASSERT(gpu_transposed.dimf(2) == ny, "Dim 2 correct");

  // Compare with CPU result
  gpu_transposed.to_host();
  TEST_ASSERT(arrays_equal(cpu_transposed, gpu_transposed, 1e-5f),
              "GPU result should match CPU");

  TEST_SUCCESS("3D transpose");
}

/**
 * Test 4: 4D transpose (complex permutation)
 */
void test_4d_transpose() {
  const size_t n0 = 16, n1 = 20, n2 = 24, n3 = 12;

  ndarray<double> arr;
  arr.reshapef(n0, n1, n2, n3);

  // Initialize
  for (size_t i = 0; i < arr.nelem(); i++) {
    arr[i] = static_cast<double>(i);
  }

  // CPU transpose
  auto cpu_transposed = ftk::transpose(arr, {3, 1, 0, 2});

  // GPU transpose
  arr.to_device(NDARRAY_DEVICE_CUDA);
  auto gpu_transposed = ftk::transpose(arr, {3, 1, 0, 2});

  // Verify dimensions
  TEST_ASSERT(gpu_transposed.dimf(0) == n3, "Dim 0");
  TEST_ASSERT(gpu_transposed.dimf(1) == n1, "Dim 1");
  TEST_ASSERT(gpu_transposed.dimf(2) == n0, "Dim 2");
  TEST_ASSERT(gpu_transposed.dimf(3) == n2, "Dim 3");

  // Compare results
  gpu_transposed.to_host();
  TEST_ASSERT(arrays_equal(cpu_transposed, gpu_transposed, 1e-10),
              "GPU result matches CPU");

  TEST_SUCCESS("4D transpose");
}

/**
 * Test 5: Identity transpose (no-op)
 */
void test_identity_transpose() {
  ndarray<float> arr;
  arr.reshapef(100, 200);

  for (size_t i = 0; i < arr.nelem(); i++) {
    arr[i] = static_cast<float>(i);
  }

  ndarray<float> original = arr;
  arr.to_device(NDARRAY_DEVICE_CUDA);

  // Identity permutation
  auto transposed = ftk::transpose(arr, {0, 1});

  // Should still be on GPU
  TEST_ASSERT(transposed.is_on_device(), "Result on GPU");

  // Compare with original
  transposed.to_host();
  TEST_ASSERT(arrays_equal(original, transposed, 1e-6f),
              "Identity transpose preserves data");

  TEST_SUCCESS("Identity transpose");
}

/**
 * Test 6: Metadata preservation
 */
void test_metadata_preservation() {
  // Vector field
  ndarray<float> velocity;
  velocity.reshapef(3, 100, 200);
  velocity.set_multicomponents(1);

  for (size_t i = 0; i < velocity.nelem(); i++) {
    velocity[i] = static_cast<float>(i);
  }

  // Move to GPU and transpose
  velocity.to_device(NDARRAY_DEVICE_CUDA);
  auto transposed = ftk::transpose(velocity, {0, 2, 1});

  // Verify metadata preserved
  TEST_ASSERT(transposed.multicomponents() == 1, "Multicomponents preserved");

  // Time-varying field
  ndarray<double> timeseries;
  timeseries.reshapef(100, 200, 50);
  timeseries.set_has_time(true);

  timeseries.to_device(NDARRAY_DEVICE_CUDA);
  auto ts_transposed = ftk::transpose(timeseries, {1, 0, 2});

  TEST_ASSERT(ts_transposed.has_time(), "has_time preserved");

  TEST_SUCCESS("Metadata preservation");
}

/**
 * Test 7: Large array transpose (stress test)
 */
void test_large_transpose() {
  const size_t large_m = 4096;
  const size_t large_n = 4096;

  ndarray<float> arr;
  arr.reshapef(large_m, large_n);

  // Initialize with simple pattern
  for (size_t i = 0; i < arr.nelem(); i++) {
    arr[i] = static_cast<float>(i % 1000);
  }

  // Move to GPU
  arr.to_device(NDARRAY_DEVICE_CUDA);

  // Transpose
  auto transposed = ftk::transpose(arr);

  // Verify dimensions
  TEST_ASSERT(transposed.dimf(0) == large_n, "Large transpose dim 0");
  TEST_ASSERT(transposed.dimf(1) == large_m, "Large transpose dim 1");
  TEST_ASSERT(transposed.is_on_device(), "Large transpose on GPU");

  // Spot check a few values
  transposed.to_host();
  arr.to_host();

  for (size_t test_i = 0; test_i < 10; test_i++) {
    size_t i = (test_i * large_m) / 10;
    for (size_t test_j = 0; test_j < 10; test_j++) {
      size_t j = (test_j * large_n) / 10;
      TEST_ASSERT(std::abs(transposed.f(j, i) - arr.f(i, j)) < 1e-5f,
                  "Large transpose correctness");
    }
  }

  TEST_SUCCESS("Large array transpose");
}

/**
 * Test 8: Multiple transposes in sequence
 */
void test_multiple_transposes() {
  ndarray<double> arr;
  arr.reshapef(50, 60, 40);

  for (size_t i = 0; i < arr.nelem(); i++) {
    arr[i] = static_cast<double>(i);
  }

  ndarray<double> original = arr;
  arr.to_device(NDARRAY_DEVICE_CUDA);

  // T1: {1, 0, 2}
  auto t1 = ftk::transpose(arr, {1, 0, 2});
  TEST_ASSERT(t1.dimf(0) == 60 && t1.dimf(1) == 50 && t1.dimf(2) == 40, "T1 dims");

  // T2: {2, 1, 0}
  auto t2 = ftk::transpose(t1, {2, 1, 0});
  TEST_ASSERT(t2.dimf(0) == 40 && t2.dimf(1) == 50 && t2.dimf(2) == 60, "T2 dims");

  // T3: back to original {2, 1, 0}
  auto t3 = ftk::transpose(t2, {1, 2, 0});
  TEST_ASSERT(t3.dimf(0) == 50 && t3.dimf(1) == 60 && t3.dimf(2) == 40, "T3 dims");

  // Should match original
  t3.to_host();
  TEST_ASSERT(arrays_equal(original, t3, 1e-10), "Multiple transposes round-trip");

  TEST_SUCCESS("Multiple transposes");
}

/**
 * Test 9: Performance comparison CPU vs GPU
 */
void test_performance_comparison() {
  const size_t m = 4096;
  const size_t n = 4096;

  std::cout << "\n=== Performance Comparison (4096x4096) ===" << std::endl;

  ndarray<float> arr;
  arr.reshapef(m, n);

  for (size_t i = 0; i < arr.nelem(); i++) {
    arr[i] = static_cast<float>(i);
  }

  // CPU timing
  auto cpu_start = std::chrono::high_resolution_clock::now();
  auto cpu_result = ftk::transpose(arr);
  auto cpu_end = std::chrono::high_resolution_clock::now();
  auto cpu_time = std::chrono::duration_cast<std::chrono::milliseconds>(cpu_end - cpu_start).count();

  // GPU timing (including transfers)
  auto gpu_start = std::chrono::high_resolution_clock::now();
  arr.to_device(NDARRAY_DEVICE_CUDA);
  auto gpu_result = ftk::transpose(arr);
  gpu_result.to_host();
  auto gpu_end = std::chrono::high_resolution_clock::now();
  auto gpu_time = std::chrono::duration_cast<std::chrono::milliseconds>(gpu_end - gpu_start).count();

  // GPU kernel only timing
  arr.to_device(NDARRAY_DEVICE_CUDA);
  cudaDeviceSynchronize();
  auto kernel_start = std::chrono::high_resolution_clock::now();
  auto gpu_result2 = ftk::transpose(arr);
  cudaDeviceSynchronize();
  auto kernel_end = std::chrono::high_resolution_clock::now();
  auto kernel_time = std::chrono::duration_cast<std::chrono::microseconds>(kernel_end - kernel_start).count();

  std::cout << "CPU time: " << cpu_time << " ms" << std::endl;
  std::cout << "GPU time (with transfers): " << gpu_time << " ms" << std::endl;
  std::cout << "GPU kernel only: " << kernel_time / 1000.0 << " ms" << std::endl;

  if (kernel_time / 1000.0 < cpu_time) {
    std::cout << "Speedup (kernel only): " << cpu_time / (kernel_time / 1000.0) << "x" << std::endl;
  }

  // Verify correctness
  gpu_result.to_host();
  TEST_ASSERT(arrays_equal(cpu_result, gpu_result, 1e-5f),
              "GPU result matches CPU");

  TEST_SUCCESS("Performance comparison");
}

int main() {
  // Check CUDA availability
  int device_count = 0;
  cudaError_t err = cudaGetDeviceCount(&device_count);

  if (err != cudaSuccess || device_count == 0) {
    std::cerr << "No CUDA devices available. Skipping GPU tests." << std::endl;
    return 0;
  }

  // Print device info
  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, 0);
  std::cout << "\n========================================" << std::endl;
  std::cout << "CUDA Transpose Tests" << std::endl;
  std::cout << "GPU: " << prop.name << std::endl;
  std::cout << "Compute Capability: " << prop.major << "." << prop.minor << std::endl;
  std::cout << "========================================\n" << std::endl;

  try {
    test_2d_transpose_basic();
    test_2d_transpose_with_axes();
    test_3d_transpose();
    test_4d_transpose();
    test_identity_transpose();
    test_metadata_preservation();
    test_large_transpose();
    test_multiple_transposes();
    test_performance_comparison();

    std::cout << "\n========================================" << std::endl;
    std::cout << "All CUDA transpose tests passed!" << std::endl;
    std::cout << "========================================\n" << std::endl;

  } catch (const std::exception& e) {
    std::cerr << "Exception: " << e.what() << std::endl;
    return 1;
  }

  return 0;
}

#else // !NDARRAY_HAVE_CUDA

int main() {
  std::cout << "CUDA not available. Skipping GPU tests." << std::endl;
  return 0;
}

#endif // NDARRAY_HAVE_CUDA
