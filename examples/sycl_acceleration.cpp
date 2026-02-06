#include <ndarray/ndarray.hh>
#include <iostream>
#include <chrono>

#if NDARRAY_HAVE_SYCL
#include <CL/sycl.hpp>
#endif

/**
 * SYCL acceleration example for ndarray
 *
 * This example demonstrates:
 * - Using SYCL for heterogeneous acceleration
 * - Offloading array operations to accelerators (GPU/CPU)
 * - Performance comparison between CPU and SYCL
 * - Cross-platform acceleration (Intel, AMD, NVIDIA)
 *
 * Compile with SYCL support: -DNDARRAY_USE_SYCL=ON
 *
 * Note: SYCL is a cross-platform abstraction layer that works with:
 * - Intel GPUs (via Intel DPC++)
 * - NVIDIA GPUs (via DPC++ or hipSYCL)
 * - AMD GPUs (via hipSYCL)
 * - CPUs (fallback on any platform)
 */

void cpu_vector_add(const ftk::ndarray<float>& a,
                    const ftk::ndarray<float>& b,
                    ftk::ndarray<float>& result) {
  for (size_t i = 0; i < a.size(); i++) {
    result[i] = a[i] + b[i];
  }
}

void cpu_vector_scale(ftk::ndarray<float>& arr, float scale) {
  for (size_t i = 0; i < arr.size(); i++) {
    arr[i] *= scale;
  }
}

#if NDARRAY_HAVE_SYCL
void sycl_vector_add(const ftk::ndarray<float>& a,
                     const ftk::ndarray<float>& b,
                     ftk::ndarray<float>& result) {
  sycl::queue q{sycl::default_selector{}};

  const size_t size = a.size();

  // Create SYCL buffers
  sycl::buffer<float, 1> buf_a(a.data(), sycl::range<1>(size));
  sycl::buffer<float, 1> buf_b(b.data(), sycl::range<1>(size));
  sycl::buffer<float, 1> buf_result(result.data(), sycl::range<1>(size));

  // Submit kernel
  q.submit([&](sycl::handler& h) {
    auto acc_a = buf_a.get_access<sycl::access::mode::read>(h);
    auto acc_b = buf_b.get_access<sycl::access::mode::read>(h);
    auto acc_result = buf_result.get_access<sycl::access::mode::write>(h);

    h.parallel_for(sycl::range<1>(size), [=](sycl::id<1> idx) {
      acc_result[idx] = acc_a[idx] + acc_b[idx];
    });
  });

  q.wait();
}

void sycl_vector_scale(ftk::ndarray<float>& arr, float scale) {
  sycl::queue q{sycl::default_selector{}};

  const size_t size = arr.size();

  sycl::buffer<float, 1> buf(arr.data(), sycl::range<1>(size));

  q.submit([&](sycl::handler& h) {
    auto acc = buf.get_access<sycl::access::mode::read_write>(h);

    h.parallel_for(sycl::range<1>(size), [=](sycl::id<1> idx) {
      acc[idx] *= scale;
    });
  });

  q.wait();
}

void print_device_info(sycl::queue& q) {
  auto device = q.get_device();
  auto platform = device.get_platform();

  std::cout << "SYCL Device Information:" << std::endl;
  std::cout << "  Platform: " << platform.get_info<sycl::info::platform::name>() << std::endl;
  std::cout << "  Device: " << device.get_info<sycl::info::device::name>() << std::endl;
  std::cout << "  Vendor: " << device.get_info<sycl::info::device::vendor>() << std::endl;
  std::cout << "  Max compute units: "
            << device.get_info<sycl::info::device::max_compute_units>() << std::endl;
  std::cout << "  Max work group size: "
            << device.get_info<sycl::info::device::max_work_group_size>() << std::endl;
  std::cout << "  Global memory: "
            << device.get_info<sycl::info::device::global_mem_size>() / (1024*1024)
            << " MB" << std::endl;
  std::cout << std::endl;
}
#endif

int main() {
  std::cout << "=== ndarray SYCL Acceleration Example ===" << std::endl << std::endl;

#if NDARRAY_HAVE_SYCL
  // Print device information
  sycl::queue q{sycl::default_selector{}};
  print_device_info(q);

  // Create test arrays
  const size_t N = 10000000; // 10 million elements
  std::cout << "Array size: " << N << " elements ("
            << (N * sizeof(float)) / (1024*1024) << " MB)" << std::endl << std::endl;

  ftk::ndarray<float> a, b, result_cpu, result_sycl;
  a.reshapef(N);
  b.reshapef(N);
  result_cpu.reshapef(N);
  result_sycl.reshapef(N);

  // Initialize arrays
  std::cout << "Initializing arrays..." << std::endl;
  for (size_t i = 0; i < N; i++) {
    a[i] = static_cast<float>(i) * 0.1f;
    b[i] = static_cast<float>(i) * 0.2f;
  }
  std::cout << std::endl;

  // CPU vector addition
  std::cout << "Running CPU vector addition..." << std::endl;
  auto start_cpu = std::chrono::high_resolution_clock::now();
  cpu_vector_add(a, b, result_cpu);
  auto end_cpu = std::chrono::high_resolution_clock::now();
  auto duration_cpu = std::chrono::duration_cast<std::chrono::milliseconds>(end_cpu - start_cpu);
  std::cout << "  CPU time: " << duration_cpu.count() << " ms" << std::endl << std::endl;

  // SYCL vector addition
  std::cout << "Running SYCL vector addition..." << std::endl;
  auto start_sycl = std::chrono::high_resolution_clock::now();
  sycl_vector_add(a, b, result_sycl);
  auto end_sycl = std::chrono::high_resolution_clock::now();
  auto duration_sycl = std::chrono::duration_cast<std::chrono::milliseconds>(end_sycl - start_sycl);
  std::cout << "  SYCL time: " << duration_sycl.count() << " ms" << std::endl << std::endl;

  // Verify results
  std::cout << "Verifying results..." << std::endl;
  bool match = true;
  for (size_t i = 0; i < N; i++) {
    if (std::abs(result_cpu[i] - result_sycl[i]) > 0.001f) {
      match = false;
      std::cout << "  Mismatch at index " << i << ": CPU=" << result_cpu[i]
                << ", SYCL=" << result_sycl[i] << std::endl;
      break;
    }
  }
  if (match) {
    std::cout << "  Results match! ✓" << std::endl;
  }
  std::cout << std::endl;

  // Performance comparison
  std::cout << "Performance Comparison:" << std::endl;
  std::cout << "  Speedup: " << static_cast<double>(duration_cpu.count()) / duration_sycl.count()
            << "x" << std::endl;
  std::cout << std::endl;

  // Vector scaling example
  std::cout << "Running vector scaling (multiply by 2.0)..." << std::endl;
  ftk::ndarray<float> scale_test;
  scale_test.reshapef(N);
  for (size_t i = 0; i < N; i++) {
    scale_test[i] = static_cast<float>(i);
  }

  auto start_scale = std::chrono::high_resolution_clock::now();
  sycl_vector_scale(scale_test, 2.0f);
  auto end_scale = std::chrono::high_resolution_clock::now();
  auto duration_scale = std::chrono::duration_cast<std::chrono::milliseconds>(end_scale - start_scale);

  std::cout << "  SYCL scaling time: " << duration_scale.count() << " ms" << std::endl;
  std::cout << "  First element: " << scale_test[0] << " (expected: 0)" << std::endl;
  std::cout << "  Element at 100: " << scale_test[100] << " (expected: 200)" << std::endl;
  std::cout << std::endl;

  std::cout << "=== Example completed successfully ===" << std::endl;
  std::cout << std::endl;
  std::cout << "Note: SYCL provides portable performance across different accelerators." << std::endl;
  std::cout << "      The same code works on Intel, AMD, and NVIDIA GPUs, as well as CPUs." << std::endl;

#else
  std::cout << "ERROR: SYCL support not enabled!" << std::endl;
  std::cout << "Please compile with -DNDARRAY_USE_SYCL=ON" << std::endl;
  std::cout << std::endl;
  std::cout << "To use SYCL, you need a SYCL implementation:" << std::endl;
  std::cout << "  - Intel DPC++ (for Intel GPUs and CPUs)" << std::endl;
  std::cout << "  - hipSYCL (for AMD and NVIDIA GPUs)" << std::endl;
  std::cout << "  - ComputeCpp (Codeplay)" << std::endl;
  return 1;
#endif

  return 0;
}
