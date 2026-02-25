/**
 * Transpose Performance Benchmarks
 *
 * Measures transpose performance across different:
 * - Matrix sizes
 * - Data types
 * - Algorithms (blocked vs naive)
 * - In-place vs out-of-place
 *
 * Validates that blocked transpose provides speedup for large matrices.
 */

#include <ndarray/ndarray.hh>
#include <ndarray/transpose.hh>
#include <iostream>
#include <iomanip>
#include <chrono>
#include <vector>
#if NDARRAY_HAVE_MPI
#include <mpi.h>
#endif

using namespace std::chrono;

// Naive transpose implementation for comparison
template <typename T>
ftk::ndarray<T> transpose_naive(const ftk::ndarray<T>& input) {
  const size_t rows = input.dimf(0);
  const size_t cols = input.dimf(1);

  ftk::ndarray<T> output;
  output.reshapef(cols, rows);

  for (size_t i = 0; i < rows; i++) {
    for (size_t j = 0; j < cols; j++) {
      output.f(j, i) = input.f(i, j);
    }
  }

  return output;
}

// Benchmark helper
template <typename Func>
double benchmark_ms(Func&& func, int iterations = 10) {
  // Warm-up
  func();

  auto start = high_resolution_clock::now();
  for (int i = 0; i < iterations; i++) {
    func();
  }
  auto end = high_resolution_clock::now();

  auto duration = duration_cast<microseconds>(end - start);
  return duration.count() / 1000.0 / iterations;  // Return average in milliseconds
}

void print_table_header() {
  std::cout << std::setw(12) << "Size"
            << std::setw(15) << "Naive (ms)"
            << std::setw(15) << "Blocked (ms)"
            << std::setw(12) << "Speedup"
            << std::setw(15) << "Throughput"
            << std::endl;
  std::cout << std::string(69, '-') << std::endl;
}

void benchmark_2d_transpose() {
  std::cout << "\n=== 2D Matrix Transpose Benchmark ===" << std::endl;
  print_table_header();

  std::vector<size_t> sizes = {64, 128, 256, 512, 1024, 2048};

  for (auto size : sizes) {
    ftk::ndarray<double> A;
    A.reshapef(size, size);

    // Fill with test data
    for (size_t i = 0; i < A.size(); i++) {
      A[i] = static_cast<double>(i);
    }

    // Benchmark naive transpose
    double time_naive = benchmark_ms([&]() {
      auto At = transpose_naive(A);
    }, 3);

    // Benchmark blocked transpose
    double time_blocked = benchmark_ms([&]() {
      auto At = ftk::transpose(A);
    }, 3);

    double speedup = time_naive / time_blocked;

    // Calculate throughput (GB/s)
    // Each element is read once and written once
    double bytes = 2.0 * size * size * sizeof(double);
    double throughput = (bytes / 1e9) / (time_blocked / 1000.0);

    std::cout << std::setw(12) << (std::to_string(size) + "x" + std::to_string(size))
              << std::setw(15) << std::fixed << std::setprecision(3) << time_naive
              << std::setw(15) << time_blocked
              << std::setw(12) << std::setprecision(2) << speedup << "x"
              << std::setw(15) << std::setprecision(2) << throughput << " GB/s"
              << std::endl;
  }
}

void benchmark_inplace_transpose() {
  std::cout << "\n=== In-Place Square Transpose Benchmark ===" << std::endl;
  std::cout << std::setw(12) << "Size"
            << std::setw(15) << "Out-place (ms)"
            << std::setw(15) << "In-place (ms)"
            << std::setw(12) << "Speedup"
            << std::endl;
  std::cout << std::string(54, '-') << std::endl;

  std::vector<size_t> sizes = {128, 256, 512, 1024, 2048};

  for (auto size : sizes) {
    ftk::ndarray<double> A;
    A.reshapef(size, size);

    // Fill with test data
    for (size_t i = 0; i < A.size(); i++) {
      A[i] = static_cast<double>(i);
    }

    // Benchmark out-of-place
    double time_outplace = benchmark_ms([&]() {
      auto At = ftk::transpose(A);
    }, 3);

    // Benchmark in-place
    double time_inplace = benchmark_ms([&]() {
      auto A_copy = A;  // Need fresh copy each time
      ftk::transpose_inplace(A_copy);
    }, 3);

    double speedup = time_outplace / time_inplace;

    std::cout << std::setw(12) << (std::to_string(size) + "x" + std::to_string(size))
              << std::setw(15) << std::fixed << std::setprecision(3) << time_outplace
              << std::setw(15) << time_inplace
              << std::setw(12) << std::setprecision(2) << speedup << "x"
              << std::endl;
  }
}

void benchmark_rectangular_transpose() {
  std::cout << "\n=== Rectangular Matrix Transpose Benchmark ===" << std::endl;
  std::cout << std::setw(15) << "Size"
            << std::setw(15) << "Time (ms)"
            << std::setw(15) << "Throughput"
            << std::endl;
  std::cout << std::string(45, '-') << std::endl;

  std::vector<std::pair<size_t, size_t>> sizes = {
    {1024, 512},
    {2048, 1024},
    {512, 2048},
    {4096, 512},
    {512, 4096}
  };

  for (const auto& [rows, cols] : sizes) {
    ftk::ndarray<double> A;
    A.reshapef(rows, cols);

    // Fill with test data
    for (size_t i = 0; i < A.size(); i++) {
      A[i] = static_cast<double>(i);
    }

    double time = benchmark_ms([&]() {
      auto At = ftk::transpose(A);
    }, 3);

    // Calculate throughput
    double bytes = 2.0 * rows * cols * sizeof(double);
    double throughput = (bytes / 1e9) / (time / 1000.0);

    std::cout << std::setw(15) << (std::to_string(rows) + "x" + std::to_string(cols))
              << std::setw(15) << std::fixed << std::setprecision(3) << time
              << std::setw(15) << std::setprecision(2) << throughput << " GB/s"
              << std::endl;
  }
}

void benchmark_nd_transpose() {
  std::cout << "\n=== N-D Tensor Transpose Benchmark ===" << std::endl;
  std::cout << std::setw(20) << "Shape"
            << std::setw(15) << "Time (ms)"
            << std::setw(15) << "Elements/ms"
            << std::endl;
  std::cout << std::string(50, '-') << std::endl;

  // 3D tensors
  {
    ftk::ndarray<float> A;
    A.reshapef(64, 64, 64);
    for (size_t i = 0; i < A.size(); i++) {
      A[i] = static_cast<float>(i);
    }

    double time = benchmark_ms([&]() {
      auto At = ftk::transpose(A, {2, 0, 1});
    }, 3);

    std::cout << std::setw(20) << "64x64x64"
              << std::setw(15) << std::fixed << std::setprecision(3) << time
              << std::setw(15) << std::setprecision(0) << (A.size() / time / 1000.0) << " M"
              << std::endl;
  }

  // 4D tensors
  {
    ftk::ndarray<float> A;
    A.reshapef(16, 32, 32, 16);
    for (size_t i = 0; i < A.size(); i++) {
      A[i] = static_cast<float>(i);
    }

    double time = benchmark_ms([&]() {
      auto At = ftk::transpose(A, {3, 1, 0, 2});
    }, 3);

    std::cout << std::setw(20) << "16x32x32x16"
              << std::setw(15) << std::fixed << std::setprecision(3) << time
              << std::setw(15) << std::setprecision(0) << (A.size() / time / 1000.0) << " M"
              << std::endl;
  }

  // 5D tensors
  {
    ftk::ndarray<float> A;
    A.reshapef(8, 8, 16, 16, 8);
    for (size_t i = 0; i < A.size(); i++) {
      A[i] = static_cast<float>(i);
    }

    double time = benchmark_ms([&]() {
      auto At = ftk::transpose(A, {4, 2, 0, 3, 1});
    }, 3);

    std::cout << std::setw(20) << "8x8x16x16x8"
              << std::setw(15) << std::fixed << std::setprecision(3) << time
              << std::setw(15) << std::setprecision(0) << (A.size() / time / 1000.0) << " M"
              << std::endl;
  }
}

void benchmark_data_types() {
  std::cout << "\n=== Data Type Performance ===" << std::endl;
  std::cout << std::setw(15) << "Type"
            << std::setw(15) << "Size"
            << std::setw(15) << "Time (ms)"
            << std::setw(15) << "Throughput"
            << std::endl;
  std::cout << std::string(60, '-') << std::endl;

  const size_t size = 1024;

  // Float
  {
    ftk::ndarray<float> A;
    A.reshapef(size, size);
    for (size_t i = 0; i < A.size(); i++) {
      A[i] = static_cast<float>(i);
    }

    double time = benchmark_ms([&]() {
      auto At = ftk::transpose(A);
    }, 3);

    double bytes = 2.0 * size * size * sizeof(float);
    double throughput = (bytes / 1e9) / (time / 1000.0);

    std::cout << std::setw(15) << "float"
              << std::setw(15) << "1024x1024"
              << std::setw(15) << std::fixed << std::setprecision(3) << time
              << std::setw(15) << std::setprecision(2) << throughput << " GB/s"
              << std::endl;
  }

  // Double
  {
    ftk::ndarray<double> A;
    A.reshapef(size, size);
    for (size_t i = 0; i < A.size(); i++) {
      A[i] = static_cast<double>(i);
    }

    double time = benchmark_ms([&]() {
      auto At = ftk::transpose(A);
    }, 3);

    double bytes = 2.0 * size * size * sizeof(double);
    double throughput = (bytes / 1e9) / (time / 1000.0);

    std::cout << std::setw(15) << "double"
              << std::setw(15) << "1024x1024"
              << std::setw(15) << std::fixed << std::setprecision(3) << time
              << std::setw(15) << std::setprecision(2) << throughput << " GB/s"
              << std::endl;
  }

  // Int
  {
    ftk::ndarray<int> A;
    A.reshapef(size, size);
    for (size_t i = 0; i < A.size(); i++) {
      A[i] = static_cast<int>(i);
    }

    double time = benchmark_ms([&]() {
      auto At = ftk::transpose(A);
    }, 3);

    double bytes = 2.0 * size * size * sizeof(int);
    double throughput = (bytes / 1e9) / (time / 1000.0);

    std::cout << std::setw(15) << "int"
              << std::setw(15) << "1024x1024"
              << std::setw(15) << std::fixed << std::setprecision(3) << time
              << std::setw(15) << std::setprecision(2) << throughput << " GB/s"
              << std::endl;
  }
}

int main(int argc, char** argv) {
#if NDARRAY_HAVE_MPI
  MPI_Init(&argc, &argv);
#endif

  std::cout << "=== Transpose Performance Benchmarks ===" << std::endl;
  std::cout << "Note: Times are averaged over multiple iterations" << std::endl;

  benchmark_2d_transpose();
  benchmark_inplace_transpose();
  benchmark_rectangular_transpose();
  benchmark_nd_transpose();
  benchmark_data_types();

  std::cout << "\n=== Benchmarks Complete ===" << std::endl;

#if NDARRAY_HAVE_MPI
  MPI_Finalize();
#endif
  return 0;
}
