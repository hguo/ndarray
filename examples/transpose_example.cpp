/**
 * Transpose Example
 *
 * Demonstrates the transpose functionality in ndarray:
 * - 2D matrix transpose
 * - 3D tensor permutation
 * - In-place transpose
 * - Performance comparison
 */

#include <ndarray/ndarray.hh>
#include <ndarray/transpose.hh>
#include <iostream>
#include <iomanip>
#include <chrono>

using namespace std::chrono;

void print_2d_array(const ftk::ndarray<double>& arr, const std::string& name) {
  std::cout << name << " (shape: " << arr.dimf(0) << "×" << arr.dimf(1) << "):" << std::endl;
  for (size_t i = 0; i < std::min(arr.dimf(0), size_t(5)); i++) {
    std::cout << "  [";
    for (size_t j = 0; j < std::min(arr.dimf(1), size_t(5)); j++) {
      std::cout << std::setw(7) << std::setprecision(2) << arr.f(i, j);
      if (j < arr.dimf(1) - 1 && j < 4) std::cout << ", ";
    }
    if (arr.dimf(1) > 5) std::cout << ", ...";
    std::cout << "]" << std::endl;
  }
  if (arr.dimf(0) > 5) std::cout << "  ..." << std::endl;
  std::cout << std::endl;
}

void example_basic_2d_transpose() {
  std::cout << "=== Example 1: Basic 2D Transpose ===" << std::endl;

  // Create a 3×4 matrix
  ftk::ndarray<double> A;
  A.reshapef(3, 4);

  // Fill with values: A[i,j] = i*10 + j
  for (size_t i = 0; i < 3; i++) {
    for (size_t j = 0; j < 4; j++) {
      A.f(i, j) = i * 10.0 + j;
    }
  }

  print_2d_array(A, "Original matrix A");

  // Transpose
  auto At = ftk::transpose(A);

  print_2d_array(At, "Transposed matrix A^T");
}

void example_3d_permutation() {
  std::cout << "=== Example 2: 3D Tensor Permutation ===" << std::endl;

  // Create a 2×3×4 tensor
  ftk::ndarray<double> tensor;
  tensor.reshapef(2, 3, 4);

  // Fill with unique values
  for (size_t i = 0; i < 2; i++) {
    for (size_t j = 0; j < 3; j++) {
      for (size_t k = 0; k < 4; k++) {
        tensor.f(i, j, k) = i * 100.0 + j * 10.0 + k;
      }
    }
  }

  std::cout << "Original tensor shape: (" << tensor.dimf(0) << ", "
            << tensor.dimf(1) << ", " << tensor.dimf(2) << ")" << std::endl;
  std::cout << "Sample values:" << std::endl;
  std::cout << "  tensor[0,0,0] = " << tensor.f(0,0,0) << std::endl;
  std::cout << "  tensor[1,2,3] = " << tensor.f(1,2,3) << std::endl;

  // Permute dimensions: (0,1,2) -> (2,0,1)
  // Shape becomes (4,2,3)
  auto permuted = ftk::transpose(tensor, {2, 0, 1});

  std::cout << "\nPermuted tensor shape: (" << permuted.dimf(0) << ", "
            << permuted.dimf(1) << ", " << permuted.dimf(2) << ")" << std::endl;
  std::cout << "Sample values (after permutation):" << std::endl;
  std::cout << "  permuted[0,0,0] = " << permuted.f(0,0,0)
            << " (was tensor[0,0,0])" << std::endl;
  std::cout << "  permuted[3,1,2] = " << permuted.f(3,1,2)
            << " (was tensor[1,2,3])" << std::endl;
  std::cout << std::endl;
}

void example_inplace_transpose() {
  std::cout << "=== Example 3: In-Place Transpose ===" << std::endl;

  // Create a square matrix
  ftk::ndarray<double> A;
  A.reshapef(4, 4);

  // Fill with values
  for (size_t i = 0; i < 4; i++) {
    for (size_t j = 0; j < 4; j++) {
      A.f(i, j) = i * 10.0 + j;
    }
  }

  print_2d_array(A, "Original square matrix");

  // Transpose in-place (no extra memory)
  ftk::transpose_inplace(A);

  print_2d_array(A, "After in-place transpose");
}

void example_performance_comparison() {
  std::cout << "=== Example 4: Performance Comparison ===" << std::endl;

  std::vector<size_t> sizes = {100, 500, 1000, 2000};

  std::cout << std::setw(12) << "Size"
            << std::setw(18) << "Out-place (ms)"
            << std::setw(18) << "In-place (ms)"
            << std::setw(12) << "Speedup"
            << std::endl;
  std::cout << std::string(60, '-') << std::endl;

  for (auto size : sizes) {
    ftk::ndarray<double> A;
    A.reshapef(size, size);

    // Fill with test data
    for (size_t i = 0; i < A.size(); i++) {
      A[i] = static_cast<double>(i);
    }

    // Benchmark out-of-place
    auto start = high_resolution_clock::now();
    auto At = ftk::transpose(A);
    auto end = high_resolution_clock::now();
    double time_outplace = duration_cast<microseconds>(end - start).count() / 1000.0;

    // Benchmark in-place
    start = high_resolution_clock::now();
    ftk::transpose_inplace(A);
    end = high_resolution_clock::now();
    double time_inplace = duration_cast<microseconds>(end - start).count() / 1000.0;

    double speedup = time_outplace / time_inplace;

    std::cout << std::setw(12) << (std::to_string(size) + "×" + std::to_string(size))
              << std::setw(18) << std::fixed << std::setprecision(3) << time_outplace
              << std::setw(18) << time_inplace
              << std::setw(12) << std::setprecision(2) << speedup << "x"
              << std::endl;
  }
  std::cout << std::endl;
}

void example_batch_processing() {
  std::cout << "=== Example 5: Batch Processing ===" << std::endl;

  // Simulate a batch of images: [batch, height, width, channels]
  ftk::ndarray<float> batch;
  batch.reshapef(16, 64, 64, 3);  // 16 images, 64×64, RGB

  std::cout << "Input batch shape: [batch, height, width, channels]" << std::endl;
  std::cout << "  Shape: (" << batch.dimf(0) << ", " << batch.dimf(1) << ", "
            << batch.dimf(2) << ", " << batch.dimf(3) << ")" << std::endl;

  // Reorder to [batch, channels, height, width] for certain processing
  auto reordered = ftk::transpose(batch, {0, 3, 1, 2});

  std::cout << "\nReordered batch shape: [batch, channels, height, width]" << std::endl;
  std::cout << "  Shape: (" << reordered.dimf(0) << ", " << reordered.dimf(1) << ", "
            << reordered.dimf(2) << ", " << reordered.dimf(3) << ")" << std::endl;
  std::cout << std::endl;
}

int main() {
  std::cout << "\n";
  std::cout << "╔════════════════════════════════════════════════╗" << std::endl;
  std::cout << "║     ndarray Transpose Examples                ║" << std::endl;
  std::cout << "╚════════════════════════════════════════════════╝" << std::endl;
  std::cout << "\n";

  example_basic_2d_transpose();
  example_3d_permutation();
  example_inplace_transpose();
  example_performance_comparison();
  example_batch_processing();

  std::cout << "All examples completed successfully!" << std::endl;

  return 0;
}
