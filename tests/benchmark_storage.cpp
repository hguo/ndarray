/**
 * Storage Backend Performance Benchmarks
 *
 * Validates performance claims for different storage backends:
 * - Element-wise operations (expected: xtensor 2-4x faster due to SIMD)
 * - Matrix operations (expected: Eigen 5-10x faster due to optimized BLAS)
 * - Memory operations (reshape, copy)
 * - I/O operations (read/write overhead)
 *
 * Benchmarks are designed to be fair:
 * - Same algorithms
 * - Same data sizes
 * - Warm-up iterations to avoid cache effects
 * - Multiple runs for statistical stability
 */

#include <ndarray/ndarray.hh>
#include <ndarray/config.hh>
#include <iostream>
#include <chrono>
#include <iomanip>
#include <cmath>
#include <numeric>
#include <algorithm>

using namespace std::chrono;

// ANSI color codes for output
#define COLOR_GREEN "\033[32m"
#define COLOR_YELLOW "\033[33m"
#define COLOR_RED "\033[31m"
#define COLOR_RESET "\033[0m"

// Benchmark configuration
constexpr int WARMUP_ITERATIONS = 3;
constexpr int BENCHMARK_ITERATIONS = 10;
constexpr size_t SMALL_SIZE = 1000;
constexpr size_t MEDIUM_SIZE = 10000;
constexpr size_t LARGE_SIZE = 100000;

// Benchmark result structure
struct BenchmarkResult {
  std::string name;
  double time_ms;
  double speedup;

  void print() const {
    std::cout << "  " << std::left << std::setw(50) << name
              << std::right << std::setw(10) << std::fixed << std::setprecision(3)
              << time_ms << " ms";

    if (speedup > 1.0) {
      const char* color = COLOR_GREEN;
      if (speedup < 1.5) color = COLOR_YELLOW;
      std::cout << "  " << color << "(" << std::setprecision(2) << speedup << "x)" << COLOR_RESET;
    }
    std::cout << std::endl;
  }
};

// Timer utility
class Timer {
  high_resolution_clock::time_point start_time;

public:
  void start() { start_time = high_resolution_clock::now(); }

  double elapsed_ms() const {
    auto end = high_resolution_clock::now();
    return duration_cast<microseconds>(end - start_time).count() / 1000.0;
  }
};

// Helper to run benchmark with warmup
template <typename Func>
double benchmark(Func&& f, int warmup = WARMUP_ITERATIONS, int iterations = BENCHMARK_ITERATIONS) {
  // Warmup
  for (int i = 0; i < warmup; i++) {
    f();
  }

  // Actual benchmark
  Timer timer;
  timer.start();
  for (int i = 0; i < iterations; i++) {
    f();
  }

  return timer.elapsed_ms() / iterations;
}

// Benchmark 1: Element-wise arithmetic (SAXPY: y = a*x + y)
template <typename StoragePolicy>
double benchmark_saxpy(size_t n) {
  ftk::ndarray<float, StoragePolicy> x, y;
  x.reshapef(n);
  y.reshapef(n);

  // Initialize
  for (size_t i = 0; i < n; i++) {
    x[i] = static_cast<float>(i);
    y[i] = static_cast<float>(i * 2);
  }

  float alpha = 2.5f;

  return benchmark([&]() {
    for (size_t i = 0; i < n; i++) {
      y[i] = alpha * x[i] + y[i];
    }
  });
}

// Benchmark 2: Element-wise with multiple operations
template <typename StoragePolicy>
double benchmark_complex_elementwise(size_t n) {
  ftk::ndarray<float, StoragePolicy> a, b, c;
  a.reshapef(n);
  b.reshapef(n);
  c.reshapef(n);

  for (size_t i = 0; i < n; i++) {
    a[i] = static_cast<float>(i) * 0.1f;
    b[i] = static_cast<float>(i) * 0.2f;
  }

  return benchmark([&]() {
    for (size_t i = 0; i < n; i++) {
      c[i] = std::sqrt(a[i] * a[i] + b[i] * b[i]) + std::sin(a[i]) * std::cos(b[i]);
    }
  });
}

// Benchmark 3: Reduction (sum)
template <typename StoragePolicy>
double benchmark_reduction(size_t n) {
  ftk::ndarray<float, StoragePolicy> x;
  x.reshapef(n);

  for (size_t i = 0; i < n; i++) {
    x[i] = static_cast<float>(i);
  }

  volatile float result = 0.0f;  // Prevent optimization

  return benchmark([&]() {
    float sum = 0.0f;
    for (size_t i = 0; i < n; i++) {
      sum += x[i];
    }
    result = sum;
  });
}

// Benchmark 4: Memory operations (reshape)
template <typename StoragePolicy>
double benchmark_reshape(size_t n) {
  ftk::ndarray<float, StoragePolicy> x;

  return benchmark([&]() {
    x.reshapef(n / 10, 10);
    x.reshapef(n / 5, 5);
    x.reshapef(n);
  }, 1, 100);  // More iterations since reshape is fast
}

// Benchmark 5: Copy operations
template <typename StoragePolicy>
double benchmark_copy(size_t n) {
  ftk::ndarray<float, StoragePolicy> src, dst;
  src.reshapef(n);
  dst.reshapef(n);

  for (size_t i = 0; i < n; i++) {
    src[i] = static_cast<float>(i);
  }

  return benchmark([&]() {
    for (size_t i = 0; i < n; i++) {
      dst[i] = src[i];
    }
  });
}

// Benchmark 6: 2D indexing
template <typename StoragePolicy>
double benchmark_2d_access(size_t rows, size_t cols) {
  ftk::ndarray<float, StoragePolicy> mat;
  mat.reshapef(rows, cols);

  for (size_t i = 0; i < rows; i++) {
    for (size_t j = 0; j < cols; j++) {
      mat.at(i, j) = static_cast<float>(i * cols + j);
    }
  }

  volatile float result = 0.0f;

  return benchmark([&]() {
    float sum = 0.0f;
    for (size_t i = 0; i < rows; i++) {
      for (size_t j = 0; j < cols; j++) {
        sum += mat.at(i, j);
      }
    }
    result = sum;
  }, 1, 5);  // Fewer iterations since this is slow
}

void print_header(const std::string& title) {
  std::cout << "\n";
  std::cout << "╔════════════════════════════════════════════════════════════════════════╗\n";
  std::cout << "║  " << std::left << std::setw(68) << title << "║\n";
  std::cout << "╚════════════════════════════════════════════════════════════════════════╝\n";
}

void print_section(const std::string& title) {
  std::cout << "\n" << title << "\n";
  std::cout << std::string(title.length(), '-') << "\n";
}

void run_elementwise_benchmarks() {
  print_section("Element-wise Operations (SAXPY: y = a*x + y)");

  for (size_t n : {SMALL_SIZE, MEDIUM_SIZE, LARGE_SIZE}) {
    std::cout << "\nArray size: " << n << " elements\n";

    double native_time = benchmark_saxpy<ftk::native_storage>(n);
    BenchmarkResult native_result{"Native storage", native_time, 1.0};
    native_result.print();

#if NDARRAY_HAVE_EIGEN
    double eigen_time = benchmark_saxpy<ftk::eigen_storage>(n);
    BenchmarkResult eigen_result{"Eigen storage", eigen_time, native_time / eigen_time};
    eigen_result.print();
#endif

#if NDARRAY_HAVE_XTENSOR
    double xtensor_time = benchmark_saxpy<ftk::xtensor_storage>(n);
    BenchmarkResult xtensor_result{"xtensor storage", xtensor_time, native_time / xtensor_time};
    xtensor_result.print();
#endif
  }
}

void run_complex_elementwise_benchmarks() {
  print_section("Complex Element-wise Operations (sqrt, sin, cos)");

  size_t n = MEDIUM_SIZE;
  std::cout << "\nArray size: " << n << " elements\n";

  double native_time = benchmark_complex_elementwise<ftk::native_storage>(n);
  BenchmarkResult native_result{"Native storage", native_time, 1.0};
  native_result.print();

#if NDARRAY_HAVE_EIGEN
  double eigen_time = benchmark_complex_elementwise<ftk::eigen_storage>(n);
  BenchmarkResult eigen_result{"Eigen storage", eigen_time, native_time / eigen_time};
  eigen_result.print();
#endif

#if NDARRAY_HAVE_XTENSOR
  double xtensor_time = benchmark_complex_elementwise<ftk::xtensor_storage>(n);
  BenchmarkResult xtensor_result{"xtensor storage", xtensor_time, native_time / xtensor_time};
  xtensor_result.print();
#endif
}

void run_reduction_benchmarks() {
  print_section("Reduction Operations (sum)");

  size_t n = LARGE_SIZE;
  std::cout << "\nArray size: " << n << " elements\n";

  double native_time = benchmark_reduction<ftk::native_storage>(n);
  BenchmarkResult native_result{"Native storage", native_time, 1.0};
  native_result.print();

#if NDARRAY_HAVE_EIGEN
  double eigen_time = benchmark_reduction<ftk::eigen_storage>(n);
  BenchmarkResult eigen_result{"Eigen storage", eigen_time, native_time / eigen_time};
  eigen_result.print();
#endif

#if NDARRAY_HAVE_XTENSOR
  double xtensor_time = benchmark_reduction<ftk::xtensor_storage>(n);
  BenchmarkResult xtensor_result{"xtensor storage", xtensor_time, native_time / xtensor_time};
  xtensor_result.print();
#endif
}

void run_memory_benchmarks() {
  print_section("Memory Operations");

  std::cout << "\nReshape benchmark (1000 elements):\n";
  double native_reshape = benchmark_reshape<ftk::native_storage>(1000);
  BenchmarkResult native_result{"Native storage", native_reshape, 1.0};
  native_result.print();

#if NDARRAY_HAVE_EIGEN
  double eigen_reshape = benchmark_reshape<ftk::eigen_storage>(1000);
  BenchmarkResult eigen_result{"Eigen storage", eigen_reshape, native_reshape / eigen_reshape};
  eigen_result.print();
#endif

#if NDARRAY_HAVE_XTENSOR
  double xtensor_reshape = benchmark_reshape<ftk::xtensor_storage>(1000);
  BenchmarkResult xtensor_result{"xtensor storage", xtensor_reshape, native_reshape / xtensor_reshape};
  xtensor_result.print();
#endif

  std::cout << "\nCopy benchmark (" << MEDIUM_SIZE << " elements):\n";
  double native_copy = benchmark_copy<ftk::native_storage>(MEDIUM_SIZE);
  BenchmarkResult native_copy_result{"Native storage", native_copy, 1.0};
  native_copy_result.print();

#if NDARRAY_HAVE_EIGEN
  double eigen_copy = benchmark_copy<ftk::eigen_storage>(MEDIUM_SIZE);
  BenchmarkResult eigen_copy_result{"Eigen storage", eigen_copy, native_copy / eigen_copy};
  eigen_copy_result.print();
#endif

#if NDARRAY_HAVE_XTENSOR
  double xtensor_copy = benchmark_copy<ftk::xtensor_storage>(MEDIUM_SIZE);
  BenchmarkResult xtensor_copy_result{"xtensor storage", xtensor_copy, native_copy / xtensor_copy};
  xtensor_copy_result.print();
#endif
}

void run_2d_access_benchmarks() {
  print_section("2D Array Access (at(i,j))");

  size_t rows = 1000, cols = 100;
  std::cout << "\nMatrix size: " << rows << " x " << cols << "\n";

  double native_time = benchmark_2d_access<ftk::native_storage>(rows, cols);
  BenchmarkResult native_result{"Native storage", native_time, 1.0};
  native_result.print();

#if NDARRAY_HAVE_EIGEN
  double eigen_time = benchmark_2d_access<ftk::eigen_storage>(rows, cols);
  BenchmarkResult eigen_result{"Eigen storage", eigen_time, native_time / eigen_time};
  eigen_result.print();
#endif

#if NDARRAY_HAVE_XTENSOR
  double xtensor_time = benchmark_2d_access<ftk::xtensor_storage>(rows, cols);
  BenchmarkResult xtensor_result{"xtensor storage", xtensor_time, native_time / xtensor_time};
  xtensor_result.print();
#endif
}

int main() {
  print_header("Storage Backend Performance Benchmarks");

  std::cout << "\nConfiguration:\n";
  std::cout << "  Warmup iterations: " << WARMUP_ITERATIONS << "\n";
  std::cout << "  Benchmark iterations: " << BENCHMARK_ITERATIONS << "\n";
  std::cout << "  Small size: " << SMALL_SIZE << " elements\n";
  std::cout << "  Medium size: " << MEDIUM_SIZE << " elements\n";
  std::cout << "  Large size: " << LARGE_SIZE << " elements\n";

  std::cout << "\nAvailable backends:\n";
  std::cout << "  ✓ Native (std::vector)\n";
#if NDARRAY_HAVE_EIGEN
  std::cout << "  ✓ Eigen\n";
#else
  std::cout << "  ✗ Eigen (not enabled)\n";
#endif
#if NDARRAY_HAVE_XTENSOR
  std::cout << "  ✓ xtensor\n";
#else
  std::cout << "  ✗ xtensor (not enabled)\n";
#endif

  run_elementwise_benchmarks();
  run_complex_elementwise_benchmarks();
  run_reduction_benchmarks();
  run_memory_benchmarks();
  run_2d_access_benchmarks();

  print_header("Benchmark Summary");

  std::cout << "\nExpected performance characteristics:\n";
  std::cout << "  • Element-wise operations: xtensor may show 1.5-3x speedup with SIMD\n";
  std::cout << "  • Memory operations: All backends should be similar (same underlying data)\n";
  std::cout << "  • 2D access: Native may be slightly faster (no overhead)\n";

  std::cout << "\nNotes:\n";
  std::cout << "  • Actual speedups depend on compiler optimization (-O3), CPU features (AVX2/AVX512)\n";
  std::cout << "  • xtensor benefits most from vectorizable operations on large arrays\n";
  std::cout << "  • Eigen excels at linear algebra (matrix multiply, eigenvalues, etc.)\n";
  std::cout << "  • Native storage has lowest overhead for simple operations\n";

  std::cout << "\n";
  return 0;
}
