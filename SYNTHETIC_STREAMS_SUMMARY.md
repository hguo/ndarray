# Synthetic Streams Summary for FTK and FTK2

This document summarizes all synthetic data streams that need to be supported in ndarray for FTK and FTK2.

## Currently Implemented in ndarray

Based on `include/ndarray/synthetic.hh`, the following synthetic data generators are available:

### 2D Scalar Fields (Time-Varying)

1. **`synthetic_woven_2D`** ✓
   - Formula: `f(x,y,t) = cos(x*cos(t) - y*sin(t)) * sin(x*sin(t) + y*cos(t))`
   - Rotating spiral woven pattern
   - Parameters: width, height, time, scaling_factor (default 15.0)

2. **`synthetic_woven_2Dt`** ✓
   - Time-series version of woven data
   - Parameters: width, height, timesteps, scaling_factor

3. **`synthetic_capped_woven_grad_2D`** ✓
   - Woven gradient with capping
   - Parameters: width, height, time, cap, scaling_factor

4. **`synthetic_woven_2D_part`** ✓
   - Woven data for a subset region (for distributed computing)
   - Parameters: extent, core, time, scaling_factor

5. **`synthetic_merger_2D`** ✓
   - Formula: `f(x,y,t) = max(e^(-x̂(t)²-ŷ(t)²), e^(-x̃(t)²-ŷ(t)²))`
   - Two rotating and translating maxima that merge and split
   - Parameters: width, height, time

6. **`synthetic_moving_extremum`** ✓ (N-dimensional)
   - Formula: `f(x,y,t) = (x-x₀)² + (y-y₀)²` (2D example)
   - Single moving minimum/maximum
   - Parameters: shape, x0[N], dir[N], time
   - Supports 2D and 3D

7. **`synthetic_volcano`** ✓
   - Volcano-like shape
   - Parameters: shape, x0, time, reverse flag

### 2D Vector Fields (Time-Varying)

8. **`synthetic_double_gyre`** ✓
   - Classic double-gyre flow for Lagrangian coherent structures
   - Parameters: width, height, time, zchannel flag, time_scale, perturbation

9. **`synthetic_time_varying_double_gyre`** ✓
   - Time series of double gyre
   - Parameters: width, height, timesteps, time_scale

### 3D Scalar Fields (Time-Varying)

10. **`synthetic_moving_ramp`** ✓
    - Formula: `f(x,y,z,t) = x - (x₀ + rate*t)`
    - Moving planar ramp
    - Parameters: shape, x0, rate, time

11. **`synthetic_moving_dual_ramp`** ✓
    - Formula: `f(x,y,z,t) = |x - x₀| - rate*t`
    - Two moving parallel planes
    - Parameters: shape, x0, rate, time, offset

### 3D Vector Fields (Time-Varying)

12. **`synthetic_abc_flow`** ✓
    - ABC (Arnold-Beltrami-Childress) flow
    - Parameters: width, height, depth, time, A, B, C coefficients

13. **`synthetic_tornado`** ✓
    - Tornado-like vortex flow
    - Parameters: xs, ys, zs, time

### Unstructured Mesh Variants

14. **`synthetic_woven_2D_unstructured`** ✓
    - Woven data on unstructured 2D mesh
    - Parameters: coords, time, scaling_factor

15. **`synthetic_double_gyre_unstructured`** ✓
    - Double gyre on unstructured mesh
    - Parameters: coords, time, time_scale, perturbation

16. **`synthetic_moving_extremum_unstructured`** ✓
    - Moving extremum on unstructured mesh
    - Parameters: coords, x0, dir, time, dimensionality

17. **`synthetic_moving_extremum_grad_unstructured`** ✓
    - Gradient of moving extremum on unstructured mesh
    - Parameters: coords, x0, dir, time, dimensionality

## Summary by Category

### Scalar Field Patterns (2D)
- ✓ **Woven** - Rotating spiral pattern
- ✓ **Capped Woven Gradient** - Gradient with capping
- ✓ **Merger** - Merging/splitting maxima
- ✓ **Moving Extremum 2D** - Single moving critical point
- ✓ **Volcano** - Volcano-like elevation

### Vector Field Patterns (2D)
- ✓ **Double Gyre** - Coherent structure flow

### Scalar Field Patterns (3D)
- ✓ **Moving Extremum 3D** - Single moving critical point
- ✓ **Moving Ramp** - Single moving plane
- ✓ **Moving Dual Ramp** - Two moving planes

### Vector Field Patterns (3D)
- ✓ **ABC Flow** - Chaotic flow
- ✓ **Tornado** - Vortex structure

## FTK ParaView Source Filters

The following ParaView source filters are implemented in FTK:

1. **ABCFlow3DSource** ✓ (maps to `synthetic_abc_flow`)
2. **CappedWovenGradient2DSource** ✓ (maps to `synthetic_capped_woven_grad_2D`)
3. **DoubleGyre2DSource** ✓ (maps to `synthetic_double_gyre`)
4. **Merger2DSource** ✓ (maps to `synthetic_merger_2D`)
5. **MovingDualRamp3DSource** ✓ (maps to `synthetic_moving_dual_ramp`)
6. **MovingExtremum2DSource** ✓ (maps to `synthetic_moving_extremum`)
7. **MovingExtremum3DSource** ✓ (maps to `synthetic_moving_extremum`)
8. **MovingRamp3DSource** ✓ (maps to `synthetic_moving_ramp`)
9. **SpiralWoven2DSource** ✓ (maps to `synthetic_woven_2D`)
10. **TornadoFlow3DSource** ✓ (maps to `synthetic_tornado`)

## FTK CLI Synthetic Options

From `src/cli/legacy/constants.hh`, the CLI supports:
- `--synthetic woven`
- `--synthetic double_gyre`
- `--synthetic merger`
- `--synthetic moving_extremum_2d`
- `--synthetic moving_extremum_3d`
- `--synthetic moving_ramp_3d`
- `--synthetic moving_dual_ramp_3d`
- `--synthetic volcano_2d`
- `--synthetic tornado`

## FTK2 Usage

FTK2 examples demonstrate usage of:
- `ftk::synthetic_woven_2Dt<double>()` - Used in critical_point_2d.cpp
- `ftk::synthetic_merger_2D<double>()` - Used in levelset_2d.cpp
- Custom sphere intersection data - Used in fiber_3d.cpp

## Recommendations

All major synthetic streams are **already implemented** in ndarray. The implementation is complete and covers:

### ✓ Complete Coverage
1. All scalar field patterns (woven, merger, moving extremum, volcano, ramps)
2. All vector field patterns (double gyre, ABC flow, tornado)
3. Both structured and unstructured mesh variants
4. 2D, 3D, and time-varying versions

### Additional Features to Consider

1. **Perturbation/Noise Injection** ✓
   - Already supported via `perturbation` parameter in some functions
   - Could be standardized across all synthetic functions

2. **Parameter Configuration**
   - All functions support customizable parameters
   - JSON-based configuration already supported in stream interface

3. **Performance Optimizations**
   - Consider GPU-accelerated versions for large-scale data generation
   - Parallel generation for distributed computing scenarios

## Conclusion

The ndarray library already has **comprehensive support** for all synthetic streams used by FTK and FTK2. No additional synthetic stream types need to be implemented. The existing implementation covers:

- ✅ All 10 ParaView source filters
- ✅ All CLI synthetic options
- ✅ All FTK2 example requirements
- ✅ Both structured and unstructured variants
- ✅ 2D, 3D, and time-varying versions

**No action needed** - the synthetic stream support is already complete!
