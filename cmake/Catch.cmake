# Catch2 integration for ndarray tests
# This file provides compatibility for Catch-based tests

# If Catch2 is not found, provide a stub for catch_discover_tests
if (NOT TARGET Catch2::Catch2)
  function(catch_discover_tests target)
    # Add test that just runs the executable
    add_test(NAME ${target} COMMAND ${target})
  endfunction()
endif()
