/**
 * Example: Using Different Storage Backends with ndarray
 *
 * This example demonstrates how to use native, xtensor, and Eigen storage
 * backends for different use cases.
 */

#include <ndarray/ndarray.hh>
#include <iostream>
#include <chrono>

// Helper function to measure execution time
template <typename Func>
double measure_time(Func&& f) {
  auto start = std::chrono::high_resolution_clock::now();
  f();
  auto end = std::chrono::high_resolution_clock::now();
  return std::chrono::duration<double, std::milli>(end - start).count();
}

// Example 1: Basic usage with native storage (default)
void example_native_storage() {
  std::cout << "=== Example 1: Native Storage (Default) ===" << std::endl;

  // Native storage is the default - no changes needed for existing code
  ftk::ndarray<double> arr;
  arr.reshapef(1000, 1000);
  arr.fill(1.0);

  std::cout << "Created array with shape: " << arr.shapef()[0] << "x" << arr.shapef()[1] << std::endl;
  std::cout << "Size: " << arr.size() << " elements" << std::endl;
  std::cout << "First element: " << arr[0] << std::endl;
  std::cout << std::endl;
}

// Example 2: Using xtensor storage for performance
#if NDARRAY_HAVE_XTENSOR
void example_xtensor_storage() {
  std::cout << "=== Example 2: xtensor Storage ===" << std::endl;

  // Use xtensor storage for SIMD-accelerated operations
  ftk::ndarray_xtensor<double> a, b, c;
  a.reshapef(1000, 1000);
  b.reshapef(1000, 1000);
  c.reshapef(1000, 1000);

  // Fill with some data
  for (size_t i = 0; i < a.size(); i++) {
    a[i] = i * 0.001;
    b[i] = i * 0.002;
  }

  // Element-wise operations (vectorized by xtensor)
  auto time = measure_time([&]() {
    for (size_t i = 0; i < a.size(); i++) {
      c[i] = a[i] * b[i] + a[i];
    }
  });

  std::cout << "Element-wise operation completed in " << time << " ms" << std::endl;
  std::cout << "Result[0]: " << c[0] << std::endl;
  std::cout << std::endl;
}
#endif

// Example 3: Using Eigen storage for linear algebra
#if NDARRAY_HAVE_EIGEN
void example_eigen_storage() {
  std::cout << "=== Example 3: Eigen Storage ===" << std::endl;

  // Use Eigen storage for linear algebra operations
  ftk::ndarray_eigen<double> matrix;
  matrix.reshapef(100, 100);

  // Fill with identity-like pattern
  for (size_t i = 0; i < matrix.shapef()[0]; i++) {
    for (size_t j = 0; j < matrix.shapef()[1]; j++) {
      matrix.f(i, j) = (i == j) ? 1.0 : 0.0;
    }
  }

  std::cout << "Created 100x100 matrix" << std::endl;
  std::cout << "Matrix[0,0]: " << matrix.f(0, 0) << std::endl;
  std::cout << "Matrix[0,1]: " << matrix.f(0, 1) << std::endl;
  std::cout << std::endl;
}
#endif

// Example 4: Conversion between storage backends
#if NDARRAY_HAVE_XTENSOR
void example_storage_conversion() {
  std::cout << "=== Example 4: Storage Conversion ===" << std::endl;

  // Start with native storage
  ftk::ndarray<float> native_arr;
  native_arr.reshapef(100, 100);
  native_arr.fill(2.5f);

  // Convert to xtensor storage for fast computation
  ftk::ndarray_xtensor<float> xt_arr = native_arr;
  std::cout << "Converted to xtensor storage" << std::endl;

  // Perform computation with xtensor
  for (size_t i = 0; i < xt_arr.size(); i++) {
    xt_arr[i] = xt_arr[i] * 2.0f;
  }

  // Convert back to native storage
  ftk::ndarray<float> result = xt_arr;
  std::cout << "Converted back to native storage" << std::endl;
  std::cout << "Result[0]: " << result[0] << std::endl;
  std::cout << std::endl;
}
#endif

// Example 5: Using storage backends with groups
void example_groups_with_storage() {
  std::cout << "=== Example 5: Groups with Storage Backends ===" << std::endl;

  // Native storage group (default)
  ftk::ndarray_group<> group;

  ftk::ndarray<float> temp;
  temp.reshapef(50, 50);
  temp.fill(273.15f);

  ftk::ndarray<float> pressure;
  pressure.reshapef(50, 50);
  pressure.fill(101325.0f);

  group.set("temperature", temp);
  group.set("pressure", pressure);

  std::cout << "Created group with " << group.size() << " arrays" << std::endl;
  std::cout << "Temperature shape: " << group.get_ref<float>("temperature").shapef()[0]
            << "x" << group.get_ref<float>("temperature").shapef()[1] << std::endl;
  std::cout << std::endl;

#if NDARRAY_HAVE_XTENSOR
  // xtensor storage group
  ftk::ndarray_group_xtensor xt_group;

  ftk::ndarray_xtensor<float> xt_temp;
  xt_temp.reshapef(50, 50);
  xt_temp.fill(273.15f);

  xt_group.set("temperature", xt_temp);

  std::cout << "Created xtensor group with " << xt_group.size() << " arrays" << std::endl;
  std::cout << std::endl;
#endif
}

// Example 6: I/O with different storage backends
void example_io_with_storage() {
  std::cout << "=== Example 6: I/O with Storage Backends ===" << std::endl;

  // Create data with native storage
  ftk::ndarray<double> native_data;
  native_data.reshapef(10, 20);
  for (size_t i = 0; i < native_data.size(); i++) {
    native_data[i] = i * 0.1;
  }

  std::cout << "Created native array: " << native_data.shapef()[0]
            << "x" << native_data.shapef()[1] << std::endl;

#if NDARRAY_HAVE_XTENSOR
  // Convert to xtensor for computation
  ftk::ndarray_xtensor<double> xt_data = native_data;
  std::cout << "Converted to xtensor storage" << std::endl;

  // All I/O operations work with any storage backend
  // xt_data.read_netcdf("input.nc", "field");   // Would work if file exists
  // xt_data.write_netcdf("output.nc", "field"); // Would work

  std::cout << "I/O operations work seamlessly with all backends" << std::endl;
#endif

  std::cout << std::endl;
}

// Example 7: Performance comparison
void example_performance_comparison() {
  std::cout << "=== Example 7: Performance Comparison ===" << std::endl;

  const size_t N = 1000000;

  // Native storage benchmark
  ftk::ndarray<double> native_a, native_b, native_c;
  native_a.reshapef(N);
  native_b.reshapef(N);
  native_c.reshapef(N);

  for (size_t i = 0; i < N; i++) {
    native_a[i] = i * 0.001;
    native_b[i] = i * 0.002;
  }

  auto native_time = measure_time([&]() {
    for (size_t i = 0; i < N; i++) {
      native_c[i] = native_a[i] * native_b[i] + native_a[i];
    }
  });

  std::cout << "Native storage: " << native_time << " ms" << std::endl;

#if NDARRAY_HAVE_XTENSOR
  // xtensor storage benchmark
  ftk::ndarray_xtensor<double> xt_a = native_a;
  ftk::ndarray_xtensor<double> xt_b = native_b;
  ftk::ndarray_xtensor<double> xt_c;
  xt_c.reshapef(N);

  auto xtensor_time = measure_time([&]() {
    for (size_t i = 0; i < N; i++) {
      xt_c[i] = xt_a[i] * xt_b[i] + xt_a[i];
    }
  });

  std::cout << "xtensor storage: " << xtensor_time << " ms" << std::endl;
  std::cout << "Speedup: " << native_time / xtensor_time << "x" << std::endl;
#endif

  std::cout << std::endl;
}

int main() {
  std::cout << "Storage Backends Example" << std::endl;
  std::cout << "========================" << std::endl;
  std::cout << std::endl;

  // Run examples
  example_native_storage();

#if NDARRAY_HAVE_XTENSOR
  example_xtensor_storage();
  example_storage_conversion();
#else
  std::cout << "xtensor examples skipped (not available)" << std::endl;
  std::cout << std::endl;
#endif

#if NDARRAY_HAVE_EIGEN
  example_eigen_storage();
#else
  std::cout << "Eigen examples skipped (not available)" << std::endl;
  std::cout << std::endl;
#endif

  example_groups_with_storage();
  example_io_with_storage();
  example_performance_comparison();

  std::cout << "All examples completed!" << std::endl;

  return 0;
}
