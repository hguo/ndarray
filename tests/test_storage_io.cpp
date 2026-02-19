#include <ndarray/ndarray.hh>
#include <iostream>
#if NDARRAY_HAVE_MPI
#include <mpi.h>
#endif

int main(int argc, char** argv) {
#if NDARRAY_HAVE_MPI
  MPI_Init(&argc, &argv);
#endif
  std::cout << "Testing I/O with different storage backends...\n\n";

  // Test 1: Native storage - write test data
  std::cout << "1. Writing test data with native storage...\n";
  ftk::ndarray<float> native_arr;
  native_arr.reshapef(10, 20);
  for (size_t i = 0; i < native_arr.size(); i++) {
    native_arr[i] = static_cast<float>(i) * 0.5f;
  }
  native_arr.to_binary_file("test_data.bin");
  std::cout << "   Written " << native_arr.size() << " elements\n";

#if NDARRAY_HAVE_EIGEN
  // Test 2: Eigen storage - read test data
  std::cout << "\n2. Reading test data with Eigen storage...\n";
  ftk::ndarray<float, ftk::eigen_storage> eigen_arr;
  eigen_arr.reshapef(10, 20);
  eigen_arr.read_binary_file("test_data.bin");

  std::cout << "   Read " << eigen_arr.size() << " elements\n";
  std::cout << "   First 5 values: ";
  for (size_t i = 0; i < 5; i++) {
    std::cout << eigen_arr[i] << " ";
  }
  std::cout << "\n";

  // Verify correctness
  bool correct = true;
  for (size_t i = 0; i < eigen_arr.size(); i++) {
    float expected = static_cast<float>(i) * 0.5f;
    if (std::abs(eigen_arr[i] - expected) > 1e-6f) {
      correct = false;
      std::cout << "   ERROR: Mismatch at index " << i
                << " (expected " << expected << ", got " << eigen_arr[i] << ")\n";
      break;
    }
  }
  if (correct) {
    std::cout << "   ✓ All values match!\n";
  }

  // Test 3: Write with Eigen, read with native
  std::cout << "\n3. Writing with Eigen storage...\n";
  ftk::ndarray<float, ftk::eigen_storage> eigen_arr2;
  eigen_arr2.reshapef(5, 5);
  for (size_t i = 0; i < eigen_arr2.size(); i++) {
    eigen_arr2[i] = static_cast<float>(i) * 2.0f;
  }
  eigen_arr2.to_binary_file("test_data2.bin");
  std::cout << "   Written " << eigen_arr2.size() << " elements\n";

  std::cout << "\n4. Reading with native storage...\n";
  ftk::ndarray<float> native_arr2;
  native_arr2.reshapef(5, 5);
  native_arr2.read_binary_file("test_data2.bin");

  std::cout << "   Read " << native_arr2.size() << " elements\n";
  std::cout << "   First 5 values: ";
  for (size_t i = 0; i < 5; i++) {
    std::cout << native_arr2[i] << " ";
  }
  std::cout << "\n";

  // Verify correctness
  correct = true;
  for (size_t i = 0; i < native_arr2.size(); i++) {
    float expected = static_cast<float>(i) * 2.0f;
    if (std::abs(native_arr2[i] - expected) > 1e-6f) {
      correct = false;
      std::cout << "   ERROR: Mismatch at index " << i
                << " (expected " << expected << ", got " << native_arr2[i] << ")\n";
      break;
    }
  }
  if (correct) {
    std::cout << "   ✓ All values match!\n";
  }

  std::cout << "\n✓ All I/O tests passed! Storage backends are fully backend-agnostic.\n";
#else
  std::cout << "\nEigen not available - skipping Eigen storage tests\n";
#endif

#if NDARRAY_HAVE_MPI
  MPI_Finalize();
#endif

  return 0;
}
