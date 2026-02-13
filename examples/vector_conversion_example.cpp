#include <ndarray/ndarray.hh>
#include <iostream>
#include <vector>

/**
 * Example: Converting between std::vector and ndarray
 *
 * This example demonstrates the easiest ways to convert
 * std::vector to ndarray, addressing student feedback.
 */

int main() {
  std::cout << "=== std::vector to ndarray Conversion Examples ===" << std::endl << std::endl;

  // Example 1: Simple 1D array from vector (Constructor)
  {
    std::cout << "Example 1: Constructor (1D array)" << std::endl;

    std::vector<double> vec = {1.0, 2.0, 3.0, 4.0, 5.0};
    ftk::ndarray<double> arr(vec);

    std::cout << "  Input vector size: " << vec.size() << std::endl;
    std::cout << "  Array dimensions: " << arr.nd() << "D" << std::endl;
    std::cout << "  Array size: " << arr.size() << std::endl;
    std::cout << "  First element: " << arr[0] << std::endl;
    std::cout << "  Last element: " << arr[4] << std::endl;
    std::cout << std::endl;
  }

  // Example 2: Static factory method (1D)
  {
    std::cout << "Example 2: Static factory method (1D)" << std::endl;

    std::vector<float> vec = {10.0f, 20.0f, 30.0f, 40.0f};
    auto arr = ftk::ndarray<float>::from_vector_data(vec);

    std::cout << "  Created array with " << arr.size() << " elements" << std::endl;
    std::cout << "  Values: ";
    for (size_t i = 0; i < arr.size(); i++) {
      std::cout << arr[i] << " ";
    }
    std::cout << std::endl << std::endl;
  }

  // Example 3: Multi-dimensional array (2D)
  {
    std::cout << "Example 3: Create 2D array from vector" << std::endl;

    std::vector<double> vec = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
    auto arr = ftk::ndarray<double>::from_vector_data(vec, {3, 4});

    std::cout << "  Input: 12 elements" << std::endl;
    std::cout << "  Output: " << arr.nd() << "D array" << std::endl;
    std::cout << "  Shape: " << arr.dimf(0) << " x " << arr.dimf(1) << std::endl;
    std::cout << "  Element [0,0] = " << arr.f(0, 0) << std::endl;
    std::cout << "  Element [2,3] = " << arr.f(2, 3) << std::endl;
    std::cout << std::endl;
  }

  // Example 4: Multi-dimensional array (3D)
  {
    std::cout << "Example 4: Create 3D array from vector" << std::endl;

    std::vector<int> vec(24);
    for (int i = 0; i < 24; i++) {
      vec[i] = i;
    }

    auto arr = ftk::ndarray<int>::from_vector_data(vec, {2, 3, 4});

    std::cout << "  Input: 24 elements" << std::endl;
    std::cout << "  Shape: " << arr.dimf(0) << " x "
              << arr.dimf(1) << " x " << arr.dimf(2) << std::endl;
    std::cout << "  Total elements: " << arr.size() << std::endl;
    std::cout << std::endl;
  }

  // Example 5: Old method (still works)
  {
    std::cout << "Example 5: Old method copy_vector()" << std::endl;

    std::vector<double> vec = {100.0, 200.0, 300.0};
    ftk::ndarray<double> arr;
    arr.copy_vector(vec);

    std::cout << "  Array size: " << arr.size() << std::endl;
    std::cout << "  Values: ";
    for (size_t i = 0; i < arr.size(); i++) {
      std::cout << arr[i] << " ";
    }
    std::cout << std::endl << std::endl;
  }

  // Example 6: Convert back to vector
  {
    std::cout << "Example 6: Convert ndarray back to vector" << std::endl;

    std::vector<double> original = {1.5, 2.5, 3.5, 4.5};
    ftk::ndarray<double> arr(original);

    const std::vector<double>& result = arr.std_vector();

    std::cout << "  Original vector: ";
    for (auto v : original) std::cout << v << " ";
    std::cout << std::endl;

    std::cout << "  Returned vector: ";
    for (auto v : result) std::cout << v << " ";
    std::cout << std::endl << std::endl;
  }

  // Example 7: Practical MOPS-like workflow
  {
    std::cout << "Example 7: Practical workflow (simulation data)" << std::endl;

    // Step 1: Collect data from simulation in vector
    std::vector<double> simulation_output;
    for (int t = 0; t < 100; t++) {
      simulation_output.push_back(t * 0.1);  // Simulated data
    }

    // Step 2: Convert to ndarray for analysis
    ftk::ndarray<double> data(simulation_output);

    // Step 3: Reshape to 2D grid (10x10)
    data.reshapef(10, 10);

    // Step 4: Process data
    for (size_t i = 0; i < data.size(); i++) {
      data[i] *= 2.0;  // Example: scale by 2
    }

    std::cout << "  Input: " << simulation_output.size() << " simulation values" << std::endl;
    std::cout << "  Reshaped to: " << data.dimf(0) << " x " << data.dimf(1) << std::endl;
    std::cout << "  Processed " << data.size() << " elements" << std::endl;
    std::cout << "  Sample value [5,5]: " << data.f(5, 5) << std::endl;
    std::cout << std::endl;
  }

  // Example 8: Different data types
  {
    std::cout << "Example 8: Different data types" << std::endl;

    std::vector<int> int_vec = {1, 2, 3};
    ftk::ndarray<int> int_arr(int_vec);

    std::vector<float> float_vec = {1.5f, 2.5f, 3.5f};
    ftk::ndarray<float> float_arr(float_vec);

    std::vector<uint8_t> byte_vec = {255, 128, 0};
    ftk::ndarray<uint8_t> byte_arr(byte_vec);

    std::cout << "  Int array: " << int_arr[0] << std::endl;
    std::cout << "  Float array: " << float_arr[0] << std::endl;
    std::cout << "  Byte array: " << (int)byte_arr[0] << std::endl;
    std::cout << std::endl;
  }

  std::cout << "=== Summary ===" << std::endl;
  std::cout << "Recommended methods:" << std::endl;
  std::cout << "  1. Constructor: ndarray<T> arr(vec)" << std::endl;
  std::cout << "  2. Factory 1D: ndarray<T>::from_vector_data(vec)" << std::endl;
  std::cout << "  3. Factory N-D: ndarray<T>::from_vector_data(vec, shape)" << std::endl;
  std::cout << "  4. To vector: arr.std_vector()" << std::endl;
  std::cout << std::endl;
  std::cout << "All methods are now simple and convenient!" << std::endl;

  return 0;
}
