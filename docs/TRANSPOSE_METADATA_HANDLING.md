# Transpose Metadata Handling

## Overview

The `ndarray` library uses special metadata to track dimension semantics:

1. **`n_component_dims`**: Number of leading dimensions representing components (0=scalar, 1=vector, 2=tensor)
2. **`is_time_varying`**: Whether the last dimension represents time

The transpose operation must properly handle this metadata to preserve semantic correctness.

## Dimension Layout

ndarray assumes a specific dimension layout:
```
[component_dims..., spatial_dims..., time_dim]
 <-- n_component_dims --> <-- spatial --> <-- is_time_varying -->
```

### Examples

**Scalar field (static)**:
- Shape: `[100, 200, 50]` (3D spatial grid)
- `n_component_dims = 0`
- `is_time_varying = false`

**Vector field (static)**:
- Shape: `[3, 100, 200, 50]` (3 components, 3D spatial grid)
- `n_component_dims = 1`
- `is_time_varying = false`

**Scalar field (time-varying)**:
- Shape: `[100, 200, 50, 100]` (3D spatial grid, 100 timesteps)
- `n_component_dims = 0`
- `is_time_varying = true`

**Vector field (time-varying)**:
- Shape: `[3, 100, 200, 50, 100]` (3 components, 3D spatial, 100 timesteps)
- `n_component_dims = 1`
- `is_time_varying = true`

## Transpose Behavior

### Metadata Preservation

The transpose operation **copies metadata from input to output**:
```cpp
output.set_multicomponents(input.multicomponents());
output.set_has_time(input.has_time());
```

### Semantic Validation

The implementation checks if the permutation preserves dimension semantics:

**Safe Permutations** (no warning):
- Component dimensions stay in the first `n_component_dims` positions
- Time dimension (if present) stays at the end
- Only spatial dimensions are permuted

**Unsafe Permutations** (warning issued):
- Component dimensions are moved to non-component positions
- Time dimension is moved from the end
- Metadata becomes semantically incorrect

### Warning Messages

When an unsafe permutation is detected, a warning is printed:
```
[NDARRAY WARNING] transpose: permutation moves component or time dimensions.
  The resulting array may have incorrect metadata.
  Original: n_component_dims=1, has_time=true
  Consider manually adjusting metadata after transpose.
```

## Usage Guidelines

### ✅ Safe Usage Examples

#### Example 1: Transpose spatial dimensions only
```cpp
ftk::ndarray<double> V;
V.reshapef(3, 100, 200);  // 3 components, 100×200 spatial
V.set_multicomponents(1);

// Transpose spatial dimensions: (0,1,2) → (0,2,1)
// Component dimension (0) stays first ✓
auto Vt = ftk::transpose(V, {0, 2, 1});  // Shape: [3, 200, 100]
// Metadata: n_component_dims=1 (correct)
```

#### Example 2: Transpose spatial, preserve time
```cpp
ftk::ndarray<float> T;
T.reshapef(100, 200, 50);  // 100×200 spatial, 50 timesteps
T.set_has_time(true);

// Transpose spatial: (0,1,2) → (1,0,2)
// Time dimension (2) stays last ✓
auto Tt = ftk::transpose(T, {1, 0, 2});  // Shape: [200, 100, 50]
// Metadata: is_time_varying=true (correct)
```

#### Example 3: Vector field with time
```cpp
ftk::ndarray<double> VT;
VT.reshapef(3, 100, 200, 50);  // 3 components, 100×200 spatial, 50 time
VT.set_multicomponents(1);
VT.set_has_time(true);

// Transpose spatial only: (0,1,2,3) → (0,2,1,3)
// Components first, time last ✓
auto VTt = ftk::transpose(VT, {0, 2, 1, 3});  // Shape: [3, 200, 100, 50]
// Metadata: n_component_dims=1, is_time_varying=true (correct)
```

### ⚠️ Unsafe Usage Examples (with warnings)

#### Example 1: Moving component dimension
```cpp
ftk::ndarray<double> V;
V.reshapef(3, 100, 200);
V.set_multicomponents(1);

// BAD: Moves component dimension to position 1
auto Vbad = ftk::transpose(V, {1, 0, 2});  // Shape: [100, 3, 200]
// WARNING: n_component_dims=1 but first dimension is not components!
// Metadata is now semantically incorrect
```

**Fix**: Manually update metadata after transpose:
```cpp
auto Vbad = ftk::transpose(V, {1, 0, 2});
Vbad.set_multicomponents(0);  // No longer has component semantics in expected position
// Or restructure the array to maintain semantics
```

#### Example 2: Moving time dimension
```cpp
ftk::ndarray<float> T;
T.reshapef(100, 200, 50);
T.set_has_time(true);

// BAD: Moves time dimension from last position
auto Tbad = ftk::transpose(T, {2, 0, 1});  // Shape: [50, 100, 200]
// WARNING: is_time_varying=true but last dimension is not time!
// Metadata is incorrect
```

**Fix**:
```cpp
auto Tbad = ftk::transpose(T, {2, 0, 1});
Tbad.set_has_time(false);  // Time is no longer in last position
// Manually track that dimension 0 is now time
```

## Best Practices

### 1. Keep Special Dimensions Fixed

When transposing arrays with metadata, **only permute spatial dimensions**:
```cpp
// Good: Preserve component and time positions
ftk::transpose(array, {0, ..., spatial_perms..., nd-1});
//                    ^          spatial            ^
//                 components                      time
```

### 2. Clear Metadata for Complex Permutations

If you need to permute component or time dimensions, clear metadata first:
```cpp
array.set_multicomponents(0);  // Remove component semantics
array.set_has_time(false);     // Remove time semantics
auto transposed = ftk::transpose(array, arbitrary_axes);
// Manually reinterpret dimensions as needed
```

### 3. Document Dimension Semantics

When working with transposed arrays, document what each dimension represents:
```cpp
// After complex transpose
auto T = ftk::transpose(data, {2, 0, 3, 1});
// Resulting dims: [time, components, spatial_x, spatial_y]
// Note: metadata does not reflect this - handled manually
```

### 4. Use Helper Functions

Consider creating domain-specific transpose helpers:
```cpp
// Transpose only spatial dimensions of vector field
template <typename T>
ndarray<T> transpose_spatial_vector(const ndarray<T>& V,
                                   const std::vector<size_t>& spatial_axes) {
  // Build full axes: [0, spatial_axes+1, ...]
  size_t n_comp = V.multicomponents();
  std::vector<size_t> full_axes(V.nd());
  for (size_t i = 0; i < n_comp; i++)
    full_axes[i] = i;  // Keep component dims first
  for (size_t i = 0; i < spatial_axes.size(); i++)
    full_axes[n_comp + i] = spatial_axes[i] + n_comp;

  return ftk::transpose(V, full_axes);
}
```

## Implementation Details

### Metadata Copying

Metadata is always copied from input to output:
```cpp
output.set_multicomponents(input.multicomponents());
output.set_has_time(input.has_time());
```

### Semantic Validation

The function `preserves_dimension_semantics()` checks:
1. All axes < `n_component_dims` map to positions < `n_component_dims`
2. If `has_time`, the last axis maps to the last position

### In-Place Transpose

In-place transpose does **not** modify metadata, as it only works for square 2D matrices without special semantics.

## Testing

The test file `tests/test_transpose_metadata.cpp` verifies:
- Metadata preservation for safe permutations
- Warning generation for unsafe permutations
- Correct behavior for vector fields, time-varying fields, and combined cases

## Summary

| Scenario | Behavior | Metadata Correctness |
|----------|----------|---------------------|
| Transpose spatial dims only | ✅ No warning | ✅ Correct |
| Move component dimensions | ⚠️ Warning | ❌ Incorrect |
| Move time dimension | ⚠️ Warning | ❌ Incorrect |
| Complex permutation | ⚠️ Warning | ❌ Incorrect |

**Key Takeaway**: Transpose is safest when only permuting spatial dimensions, keeping component dimensions at the beginning and time at the end.

---

**Document Version**: 1.0
**Date**: 2026-02-25
**Related**: `TRANSPOSE_DESIGN.md`, `transpose.hh`
