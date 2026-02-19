/**
 * Storage Backend Stream Tests
 *
 * Tests stream functionality with different storage backends:
 * - Synthetic streams with native/Eigen/xtensor storage
 * - YAML configuration parsing with different backends
 * - Multi-timestep reads with different backends
 * - Verify data integrity across backends
 *
 * NOTE: Requires NDARRAY_HAVE_YAML to be enabled
 */

#include <ndarray/config.hh>
#include <iostream>

#if NDARRAY_HAVE_YAML

#include <ndarray/ndarray_group_stream.hh>
#include <cassert>
#include <cmath>
#include <fstream>
#if NDARRAY_HAVE_MPI
#include <mpi.h>
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
  std::cout << "  " << name << std::endl

// Helper to create test YAML files
void create_synthetic_yaml(const std::string& filename,
                           int nx, int ny, int timesteps,
                           const std::string& varname = "scalar",
                           const std::string& dtype = "float32") {
  std::ofstream f(filename);
  f << "stream:\n";
  f << "  name: test_synthetic\n";
  f << "  substreams:\n";
  f << "    - name: woven\n";
  f << "      format: synthetic\n";
  f << "      dimensions: [" << nx << ", " << ny << "]\n";
  f << "      timesteps: " << timesteps << "\n";
  f << "      vars:\n";
  f << "        - name: " << varname << "\n";
  f << "          dtype: " << dtype << "\n";
  f.close();
}

void create_multi_var_yaml(const std::string& filename) {
  std::ofstream f(filename);
  f << "stream:\n";
  f << "  name: test_multi\n";
  f << "  substreams:\n";
  f << "    - name: woven\n";
  f << "      format: synthetic\n";
  f << "      dimensions: [10, 15]\n";
  f << "      timesteps: 3\n";
  f << "      vars:\n";
  f << "        - name: temperature\n";
  f << "          dtype: float32\n";
  f << "        - name: pressure\n";
  f << "          dtype: float32\n";
  f.close();
}

int test_stream_native_storage() {
  std::cout << "\n=== Test 1: Stream with Native Storage ===" << std::endl;

  TEST_SECTION("Create synthetic stream YAML");
  create_synthetic_yaml("test_stream_native.yaml", 8, 12, 5, "data", "float32");

  TEST_SECTION("Parse YAML and create native storage stream");
  ftk::stream<ftk::native_storage> s;
  s.parse_yaml("test_stream_native.yaml");

  TEST_SECTION("Read timestep 0");
  auto g0 = s.read(0);
  TEST_ASSERT(g0 != nullptr, "Should get valid group for timestep 0");
  TEST_ASSERT(g0->size() > 0, "Group should have arrays");

  auto data0 = g0->get_ptr<float>("data");
  TEST_ASSERT(data0 != nullptr, "Should retrieve 'data' array");
  TEST_ASSERT(data0->size() == 8 * 12, "Array should have correct size");
  TEST_ASSERT(data0->shapef(0) == 8, "First dimension should be 8");
  TEST_ASSERT(data0->shapef(1) == 12, "Second dimension should be 12");

  TEST_SECTION("Read timestep 2");
  auto g2 = s.read(2);
  TEST_ASSERT(g2 != nullptr, "Should get valid group for timestep 2");

  auto data2 = g2->get_ptr<float>("data");
  TEST_ASSERT(data2 != nullptr, "Should retrieve 'data' array from timestep 2");
  TEST_ASSERT(data2->size() == 8 * 12, "Size should be consistent across timesteps");

  TEST_SECTION("Verify data varies across timesteps");
  bool data_differs = false;
  for (size_t i = 0; i < std::min(data0->size(), data2->size()); i++) {
    if (std::abs((*data0)[i] - (*data2)[i]) > 1e-6f) {
      data_differs = true;
      break;
    }
  }
  TEST_ASSERT(data_differs, "Data should differ across timesteps for synthetic stream");

  std::cout << "  ✓ All native storage stream tests passed" << std::endl;
  return 0;
}

#if NDARRAY_HAVE_EIGEN
int test_stream_eigen_storage() {
  std::cout << "\n=== Test 2: Stream with Eigen Storage ===" << std::endl;

  TEST_SECTION("Create synthetic stream YAML");
  create_synthetic_yaml("test_stream_eigen.yaml", 10, 16, 4, "field", "float32");

  TEST_SECTION("Parse YAML and create Eigen storage stream");
  ftk::stream<ftk::eigen_storage> s;
  s.parse_yaml("test_stream_eigen.yaml");

  TEST_SECTION("Read timestep 0 with Eigen backend");
  auto g0 = s.read(0);
  TEST_ASSERT(g0 != nullptr, "Should get valid group");

  auto field0 = g0->get_ptr<float>("field");
  TEST_ASSERT(field0 != nullptr, "Should retrieve 'field' array");
  TEST_ASSERT(field0->size() == 10 * 16, "Array should have correct size");

  // Verify it's actually using Eigen storage
  TEST_ASSERT(typeid(*field0) == typeid(ftk::ndarray<float, ftk::eigen_storage>),
              "Array should be using Eigen storage");

  TEST_SECTION("Read multiple timesteps");
  auto g1 = s.read(1);
  auto g3 = s.read(3);

  TEST_ASSERT(g1 != nullptr && g3 != nullptr, "Should read all timesteps");

  auto field1 = g1->get_ptr<float>("field");
  auto field3 = g3->get_ptr<float>("field");

  TEST_ASSERT(field1->size() == field3->size(), "Sizes should be consistent");

  std::cout << "  ✓ All Eigen storage stream tests passed" << std::endl;
  return 0;
}
#endif

#if NDARRAY_HAVE_XTENSOR
int test_stream_xtensor_storage() {
  std::cout << "\n=== Test 3: Stream with xtensor Storage ===" << std::endl;

  TEST_SECTION("Create synthetic stream YAML");
  create_synthetic_yaml("test_stream_xtensor.yaml", 12, 18, 3, "array", "float32");

  TEST_SECTION("Parse YAML and create xtensor storage stream");
  ftk::stream<ftk::xtensor_storage> s;
  s.parse_yaml("test_stream_xtensor.yaml");

  TEST_SECTION("Read timestep 0 with xtensor backend");
  auto g0 = s.read(0);
  TEST_ASSERT(g0 != nullptr, "Should get valid group");

  auto array0 = g0->get_ptr<float>("array");
  TEST_ASSERT(array0 != nullptr, "Should retrieve 'array' array");
  TEST_ASSERT(array0->size() == 12 * 18, "Array should have correct size");

  TEST_SECTION("Read timestep 2");
  auto g2 = s.read(2);
  auto array2 = g2->get_ptr<float>("array");
  TEST_ASSERT(array2 != nullptr, "Should retrieve array from timestep 2");

  std::cout << "  ✓ All xtensor storage stream tests passed" << std::endl;
  return 0;
}
#endif

int test_stream_multi_var() {
  std::cout << "\n=== Test 4: Multi-Variable Streams with Different Backends ===" << std::endl;

  TEST_SECTION("Create multi-variable YAML");
  create_multi_var_yaml("test_stream_multivar.yaml");

  TEST_SECTION("Read with native storage");
  ftk::stream<ftk::native_storage> s_native;
  s_native.parse_yaml("test_stream_multivar.yaml");
  auto g_native = s_native.read(1);

  TEST_ASSERT(g_native != nullptr, "Should get valid group");
  TEST_ASSERT(g_native->size() >= 2, "Should have at least 2 variables");

  auto temp_native = g_native->get_ptr<float>("temperature");
  auto pres_native = g_native->get_ptr<float>("pressure");

  TEST_ASSERT(temp_native != nullptr, "Should have temperature");
  TEST_ASSERT(pres_native != nullptr, "Should have pressure");
  TEST_ASSERT(temp_native->size() == 10 * 15, "Temperature should have correct size");
  TEST_ASSERT(pres_native->size() == 10 * 15, "Pressure should have correct size");

#if NDARRAY_HAVE_EIGEN
  TEST_SECTION("Read same stream with Eigen storage");
  ftk::stream<ftk::eigen_storage> s_eigen;
  s_eigen.parse_yaml("test_stream_multivar.yaml");
  auto g_eigen = s_eigen.read(1);

  TEST_ASSERT(g_eigen != nullptr, "Should get valid group with Eigen");

  auto temp_eigen = g_eigen->get_ptr<float>("temperature");
  auto pres_eigen = g_eigen->get_ptr<float>("pressure");

  TEST_ASSERT(temp_eigen != nullptr, "Should have temperature with Eigen");
  TEST_ASSERT(pres_eigen != nullptr, "Should have pressure with Eigen");
  TEST_ASSERT(temp_eigen->size() == temp_native->size(), "Sizes should match across backends");

  TEST_SECTION("Verify data consistency across backends");
  // Synthetic streams should produce deterministic data
  // So same timestep should have same data regardless of backend
  bool data_matches = true;
  for (size_t i = 0; i < std::min(temp_native->size(), temp_eigen->size()); i++) {
    if (std::abs((*temp_native)[i] - (*temp_eigen)[i]) > 1e-6f) {
      data_matches = false;
      std::cerr << "  Data mismatch at index " << i << ": "
                << (*temp_native)[i] << " vs " << (*temp_eigen)[i] << std::endl;
      break;
    }
  }
  TEST_ASSERT(data_matches, "Same timestep should produce same data across backends");
#endif

  std::cout << "  ✓ All multi-variable stream tests passed" << std::endl;
  return 0;
}

int test_stream_timestep_iteration() {
  std::cout << "\n=== Test 5: Timestep Iteration with Different Backends ===" << std::endl;

  TEST_SECTION("Create test stream");
  create_synthetic_yaml("test_stream_iteration.yaml", 6, 8, 10, "value", "float32");

  TEST_SECTION("Iterate through timesteps with native storage");
  ftk::stream<ftk::native_storage> s_native;
  s_native.parse_yaml("test_stream_iteration.yaml");

  std::vector<float> first_values_native;
  for (int t = 0; t < 10; t++) {
    auto g = s_native.read(t);
    TEST_ASSERT(g != nullptr, "Should read all timesteps");

    auto arr = g->get_ptr<float>("value");
    TEST_ASSERT(arr != nullptr, "Should have 'value' array");
    TEST_ASSERT(arr->size() == 6 * 8, "Size should be consistent");

    first_values_native.push_back((*arr)[0]);
  }

  TEST_ASSERT(first_values_native.size() == 10, "Should have collected 10 values");

#if NDARRAY_HAVE_EIGEN
  TEST_SECTION("Iterate through same timesteps with Eigen storage");
  ftk::stream<ftk::eigen_storage> s_eigen;
  s_eigen.parse_yaml("test_stream_iteration.yaml");

  std::vector<float> first_values_eigen;
  for (int t = 0; t < 10; t++) {
    auto g = s_eigen.read(t);
    auto arr = g->get_ptr<float>("value");
    first_values_eigen.push_back((*arr)[0]);
  }

  TEST_SECTION("Verify consistent iteration across backends");
  TEST_ASSERT(first_values_eigen.size() == first_values_native.size(),
              "Should have same number of timesteps");

  for (size_t t = 0; t < first_values_native.size(); t++) {
    TEST_ASSERT(std::abs(first_values_native[t] - first_values_eigen[t]) < 1e-6f,
                "Values should match across backends for same timestep");
  }
#endif

  std::cout << "  ✓ All timestep iteration tests passed" << std::endl;
  return 0;
}

int main(int argc, char** argv) {
#if NDARRAY_HAVE_MPI
  MPI_Init(&argc, &argv);
#endif
  std::cout << "\n";
  std::cout << "╔════════════════════════════════════════════════════════════╗" << std::endl;
  std::cout << "║     Storage Backend Stream Test Suite                     ║" << std::endl;
  std::cout << "╚════════════════════════════════════════════════════════════╝" << std::endl;

  int result = 0;

  // Test 1: Stream with native storage
  result |= test_stream_native_storage();

#if NDARRAY_HAVE_EIGEN
  // Test 2: Stream with Eigen storage
  result |= test_stream_eigen_storage();
#else
  std::cout << "\n⊘ Skipping Eigen storage stream tests (NDARRAY_HAVE_EIGEN not defined)" << std::endl;
#endif

#if NDARRAY_HAVE_XTENSOR
  // Test 3: Stream with xtensor storage
  result |= test_stream_xtensor_storage();
#else
  std::cout << "\n⊘ Skipping xtensor storage stream tests (NDARRAY_HAVE_XTENSOR not defined)" << std::endl;
#endif

  // Test 4: Multi-variable streams
  result |= test_stream_multi_var();

  // Test 5: Timestep iteration
  result |= test_stream_timestep_iteration();

  std::cout << "\n";
  std::cout << "╔════════════════════════════════════════════════════════════╗" << std::endl;
  if (result == 0) {
    std::cout << "║  ✓✓✓ ALL STREAM BACKEND TESTS PASSED ✓✓✓                 ║" << std::endl;
  } else {
    std::cout << "║  ✗✗✗ SOME TESTS FAILED ✗✗✗                               ║" << std::endl;
  }
  std::cout << "╚════════════════════════════════════════════════════════════╝" << std::endl;
  std::cout << "\n";

  // Cleanup test files
  std::remove("test_stream_native.yaml");
  std::remove("test_stream_eigen.yaml");
  std::remove("test_stream_xtensor.yaml");
  std::remove("test_stream_multivar.yaml");
  std::remove("test_stream_iteration.yaml");

#if NDARRAY_HAVE_MPI
  MPI_Finalize();
#endif

  return result;
}

#else // !NDARRAY_HAVE_YAML

int main(int argc, char** argv) {
#if NDARRAY_HAVE_MPI
  MPI_Init(&argc, &argv);
#endif
  std::cout << "\n";
  std::cout << "╔════════════════════════════════════════════════════════════╗" << std::endl;
  std::cout << "║     Storage Backend Stream Test Suite                     ║" << std::endl;
  std::cout << "╚════════════════════════════════════════════════════════════╝" << std::endl;
  std::cout << "\n";
  std::cout << "⊘ Stream tests require NDARRAY_HAVE_YAML to be enabled" << std::endl;
  std::cout << "⊘ Please rebuild with yaml-cpp support to run these tests" << std::endl;
  std::cout << "\n";
#if NDARRAY_HAVE_MPI
  MPI_Finalize();
#endif
  return 0;
}

#endif // NDARRAY_HAVE_YAML
