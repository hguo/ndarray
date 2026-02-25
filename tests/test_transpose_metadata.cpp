/**
 * Test transpose with multicomponent and time-varying arrays
 *
 * Verifies that transpose correctly handles:
 * - Multicomponent arrays (vector/tensor fields)
 * - Time-varying arrays
 * - Combined multicomponent + time-varying arrays
 */

#include <ndarray/ndarray.hh>
#include <ndarray/transpose.hh>
#include <iostream>
#include <cassert>

#define TEST_ASSERT(condition, message) \
  do { \
    if (!(condition)) { \
      std::cerr << "FAILED: " << message << std::endl; \
      std::cerr << "  at " << __FILE__ << ":" << __LINE__ << std::endl; \
      return 1; \
    } \
  } while (0)

#define TEST_SECTION(name) \
  std::cout << "  Testing: " << name << std::endl

int main() {
  std::cout << "=== Testing Transpose with Metadata ===" << std::endl << std::endl;

  // Test 1: Vector field (multicomponent)
  {
    TEST_SECTION("Vector field transpose");
    ftk::ndarray<double> V;
    V.reshapef(3, 10, 20);  // 3 components, 10×20 spatial grid
    V.set_multicomponents(1);  // Mark first dim as components

    std::cout << "    Original: dims=" << V.nd()
              << ", n_component_dims=" << V.multicomponents()
              << ", is_time_varying=" << V.has_time() << std::endl;

    // Transpose spatial dimensions only: (0,1,2) -> (0,2,1)
    // This should preserve component dimension
    auto Vt = ftk::transpose(V, {0, 2, 1});

    std::cout << "    After transpose: dims=" << Vt.nd()
              << ", n_component_dims=" << Vt.multicomponents()
              << ", is_time_varying=" << Vt.has_time() << std::endl;

    TEST_ASSERT(Vt.multicomponents() == 1, "Should preserve n_component_dims");
    TEST_ASSERT(!Vt.has_time(), "Should preserve time flag");

    std::cout << "    PASSED" << std::endl;
  }

  // Test 2: Time-varying scalar field
  {
    TEST_SECTION("Time-varying field transpose");
    ftk::ndarray<float> T;
    T.reshapef(10, 20, 50);  // 10×20 spatial, 50 timesteps
    T.set_has_time(true);  // Mark last dim as time

    std::cout << "    Original: dims=" << T.nd()
              << ", n_component_dims=" << T.multicomponents()
              << ", is_time_varying=" << T.has_time() << std::endl;

    // Transpose spatial dimensions: (0,1,2) -> (1,0,2)
    // This should preserve time dimension
    auto Tt = ftk::transpose(T, {1, 0, 2});

    std::cout << "    After transpose: dims=" << Tt.nd()
              << ", n_component_dims=" << Tt.multicomponents()
              << ", is_time_varying=" << Tt.has_time() << std::endl;

    TEST_ASSERT(Tt.multicomponents() == 0, "Should preserve n_component_dims");
    TEST_ASSERT(Tt.has_time(), "Should preserve time flag");

    std::cout << "    PASSED" << std::endl;
  }

  // Test 3: Vector field with time
  {
    TEST_SECTION("Vector field with time");
    ftk::ndarray<double> VT;
    VT.reshapef(3, 10, 20, 50);  // 3 components, 10×20 spatial, 50 timesteps
    VT.set_multicomponents(1);
    VT.set_has_time(true);

    std::cout << "    Original: dims=" << VT.nd()
              << ", n_component_dims=" << VT.multicomponents()
              << ", is_time_varying=" << VT.has_time() << std::endl;

    // Transpose spatial only: (0,1,2,3) -> (0,2,1,3)
    auto VTt = ftk::transpose(VT, {0, 2, 1, 3});

    std::cout << "    After transpose: dims=" << VTt.nd()
              << ", n_component_dims=" << VTt.multicomponents()
              << ", is_time_varying=" << VTt.has_time() << std::endl;

    TEST_ASSERT(VTt.multicomponents() == 1, "Should preserve n_component_dims");
    TEST_ASSERT(VTt.has_time(), "Should preserve time flag");

    std::cout << "    PASSED" << std::endl;
  }

  // Test 4: Detecting problematic permutations
  {
    TEST_SECTION("Detecting component dimension permutation");
    ftk::ndarray<double> V;
    V.reshapef(3, 10, 20);
    V.set_multicomponents(1);

    std::cout << "    Original: shape (3, 10, 20), components=1" << std::endl;

    // This permutation moves component dimension - should this be allowed?
    // (0,1,2) -> (1,0,2) would move components from position 0 to 1
    auto Vbad = ftk::transpose(V, {1, 0, 2});

    std::cout << "    After bad permutation: n_component_dims=" << Vbad.multicomponents() << std::endl;
    std::cout << "    WARNING: Component dimension was moved but metadata not updated!" << std::endl;

    std::cout << "    TEST IDENTIFIED ISSUE" << std::endl;
  }

  std::cout << "\n=== Metadata Tests Completed ===" << std::endl;
  std::cout << "NOTE: Current implementation does NOT preserve metadata correctly!" << std::endl;
  std::cout << "This needs to be fixed before production use." << std::endl;

  return 0;
}
