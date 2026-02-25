# Synthetic YAML Stream Examples for FTK/FTK2

This directory contains comprehensive YAML configuration examples for all synthetic data generators available in the ndarray library. These synthetic streams are designed for testing and developing FTK and FTK2 feature tracking algorithms.

## Overview

Synthetic data streams provide controlled, reproducible test cases for algorithm development and validation. Each YAML file configures one or more synthetic data generators with specific parameters for different analysis scenarios.

## Quick Start

```bash
# Example: Use a synthetic stream with your FTK application
./my_ftk_app --input yaml_exampls/woven.yaml

# For distributed/parallel execution
mpirun -np 4 ./my_ftk_app --input yaml_exampls/distributed_merger.yaml
```

## 2D Scalar Field Streams

### `woven.yaml`
**Purpose**: Basic 2D rotating spiral woven pattern
**Use Cases**: Critical point tracking, gradient analysis
**Dimensions**: 32×32, 100 timesteps
**Features**: Simple test case, fast generation

### `merger_2d.yaml`
**Purpose**: Two merging/splitting Gaussian maxima
**Use Cases**: Topological event detection, critical point tracking
**Dimensions**: 128×128, 100 timesteps
**Features**: Merge and split events, rotation and translation

### `moving_extremum_2d.yaml`
**Purpose**: Single moving critical point (configurable as min or max)
**Use Cases**: Critical point trajectory tracking
**Dimensions**: 64×64, 100 timesteps
**Features**: Predictable motion, configurable direction and speed

### `volcano_2d.yaml`
**Purpose**: Static volcano-like elevation pattern
**Use Cases**: Ridge line detection, topographic analysis
**Dimensions**: 128×128, static (1 timestep)
**Features**: Radial symmetry, single peak

### `capped_woven_gradient_2d.yaml`
**Purpose**: Gradient of woven pattern with magnitude capping
**Use Cases**: Vector field analysis, bounded gradient testing
**Dimensions**: 128×128, 50 timesteps
**Features**: Controlled gradient magnitudes

## 2D Vector Field Streams

### `double_gyre_2d.yaml`
**Purpose**: Classic double-gyre flow pattern
**Use Cases**: Lagrangian coherent structure (LCS) detection, vortex analysis
**Dimensions**: 256×128, 50 timesteps
**Features**: Transport barriers, periodic flow

### `time_varying_double_gyre.yaml`
**Purpose**: Extended double gyre with detailed temporal resolution
**Use Cases**: Detailed LCS analysis, particle tracking studies
**Dimensions**: 512×256, 200 timesteps
**Features**: High temporal resolution, configurable parameters (A, omega, epsilon)

## 3D Scalar Field Streams

### `moving_extremum_3d.yaml`
**Purpose**: Single moving critical point in 3D space
**Use Cases**: 3D critical point tracking, visualization testing
**Dimensions**: 32×32×32, 50 timesteps
**Features**: Predictable 3D motion

### `moving_ramp_3d.yaml`
**Purpose**: Single moving planar surface
**Use Cases**: Isosurface tracking, plane detection
**Dimensions**: 64×64×64, 80 timesteps
**Features**: Simple planar geometry, uniform motion

### `moving_dual_ramp_3d.yaml`
**Purpose**: Two parallel moving planes
**Use Cases**: Multi-surface tracking, collision detection
**Dimensions**: 64×64×64, 60 timesteps
**Features**: Two surfaces, configurable separation

## 3D Vector Field Streams

### `abc_flow_3d.yaml`
**Purpose**: Arnold-Beltrami-Childress chaotic flow
**Use Cases**: Chaotic flow analysis, vortex detection
**Dimensions**: 64×64×64, 25 timesteps
**Features**: Chaotic trajectories, complex vortex structures

### `tornado_3d.yaml`
**Purpose**: Tornado-like vortex with funnel shape
**Use Cases**: Vortex core line detection, funnel visualization
**Dimensions**: 48×48×48, 40 timesteps
**Features**: Realistic vortex structure, rotating flow

## Distributed/Parallel Streams

### `distributed_merger.yaml`
**Purpose**: Domain-decomposed merger pattern
**Use Cases**: Parallel critical point tracking
**Dimensions**: 512×512, 100 timesteps
**Features**: Auto-decomposition, ghost layers for gradients

### `distributed_abc_flow.yaml`
**Purpose**: Domain-decomposed ABC flow
**Use Cases**: Large-scale parallel vortex analysis
**Dimensions**: 128×128×128, 50 timesteps
**Features**: 2×2×2 decomposition, 3D ghost layers

### `distributed_tornado.yaml`
**Purpose**: Domain-decomposed tornado vortex
**Use Cases**: Parallel vortex core detection
**Dimensions**: 96×96×96, 80 timesteps
**Features**: Hybrid decomposition, large ghost layers

### `distributed_synthetic_stream.yaml` (in `examples/`)
**Purpose**: Multiple distributed synthetic patterns
**Use Cases**: Learning distributed configuration
**Features**: Multiple distribution examples, detailed comments

## Multi-Field Streams

### `multi_synthetic_features.yaml`
**Purpose**: Multiple synthetic fields combined
**Use Cases**: Multi-field correlation, feature correspondence
**Contains**: Woven, merger, double gyre, and moving extremum
**Features**: Different field types in one stream

### `complex_3d_multi_field.yaml`
**Purpose**: Multiple 3D fields for comprehensive analysis
**Use Cases**: 3D multi-field tracking, correlation studies
**Contains**: Moving extremum, ABC flow, moving ramp
**Features**: Scalar, vector, and levelset fields

## Performance and Testing Streams

### `small_test_synthetic.yaml`
**Purpose**: Minimal configuration for quick testing
**Dimensions**: 16×16 and 32×16
**Use Cases**: Rapid prototyping, debugging, unit tests
**Features**: Fast generation, multiple small fields

### `high_resolution_woven.yaml`
**Purpose**: Large-scale woven pattern
**Dimensions**: 1024×1024, 100 timesteps
**Use Cases**: Performance testing, memory efficiency validation
**Features**: High resolution, long time series

### `multi_resolution_woven.yaml`
**Purpose**: Same pattern at multiple resolutions
**Dimensions**: 32×32, 128×128, 512×512
**Use Cases**: Multi-scale analysis, adaptive refinement testing
**Features**: Three resolution levels

### `benchmark_synthetic.yaml`
**Purpose**: Standard configuration for benchmarking
**Contains**: 2D/3D scalar and vector fields
**Use Cases**: Performance comparison, optimization validation
**Features**: Balanced dimensions for consistent timing

## YAML Configuration Reference

### Basic Structure
```yaml
stream:
  name: stream_name
  substreams:
    - name: substream_name
      format: synthetic
      dimensions: [width, height, depth]  # 2D or 3D
      timesteps: 100
      vars:
        - name: variable_name
          dtype: float32  # or float64
```

### Common Parameters

#### Woven Patterns
- `scaling_factor`: Controls pattern frequency (default: 15.0)
- `delta`: Time step size (default: 0.1)
- `cap`: Gradient magnitude cap (capped_woven_gradient only)

#### Double Gyre
- `time_scale`: Time scale factor (default: 0.1)
- `perturbation`: Perturbation amplitude (default: 0.25)
- `A`, `omega`, `epsilon`: Flow parameters

#### Moving Extremum
- `x0`: Initial position [x, y] or [x, y, z]
- `dir`: Movement direction [dx, dy] or [dx, dy, dz]
- `sign`: 1 for minimum, -1 for maximum

#### ABC Flow
- `A`, `B`, `C`: Flow coefficients (default: √3, √2, 1)

#### Moving Ramp/Dual Ramp
- `x0`: Initial plane position
- `rate`: Movement speed
- `offset`: Separation distance (dual ramp only)

#### Tornado
- `time_param`: Time parameter for rotation (default: 0.0)

#### Volcano
- `x0`: Peak location [x, y]
- `radius`: Base radius
- `reverse`: false for peak, true for crater

### Distributed Parameters
```yaml
vars:
  - name: variable_name
    dtype: float32
    distribution: distributed  # or replicated
    decomposition:
      pattern: [nx, ny, nz]  # 0 = auto-decompose
      ghost: [gx, gy, gz]    # ghost layer widths
```

## Usage Tips

1. **Start Small**: Use `small_test_synthetic.yaml` for initial algorithm development
2. **Scale Up**: Progress to medium-sized streams (64-128 grid points) for testing
3. **Validate with Known Features**: Use `moving_extremum_2d.yaml` for predictable features
4. **Test Topology**: Use `merger_2d.yaml` for merge/split events
5. **Parallel Development**: Use distributed versions with appropriate ghost layers
6. **Benchmark Consistently**: Use `benchmark_synthetic.yaml` for performance comparisons

## Relationship to FTK

These YAML streams correspond to FTK's ParaView source filters and CLI synthetic options:

| YAML Stream | FTK ParaView Filter | FTK CLI Option |
|-------------|---------------------|----------------|
| `woven.yaml` | SpiralWoven2DSource | `--synthetic woven` |
| `merger_2d.yaml` | Merger2DSource | `--synthetic merger` |
| `double_gyre_2d.yaml` | DoubleGyre2DSource | `--synthetic double_gyre` |
| `moving_extremum_2d.yaml` | MovingExtremum2DSource | `--synthetic moving_extremum_2d` |
| `moving_extremum_3d.yaml` | MovingExtremum3DSource | `--synthetic moving_extremum_3d` |
| `abc_flow_3d.yaml` | ABCFlow3DSource | N/A |
| `tornado_3d.yaml` | TornadoFlow3DSource | `--synthetic tornado` |
| `moving_ramp_3d.yaml` | MovingRamp3DSource | `--synthetic moving_ramp_3d` |
| `moving_dual_ramp_3d.yaml` | MovingDualRamp3DSource | `--synthetic moving_dual_ramp_3d` |
| `volcano_2d.yaml` | N/A | `--synthetic volcano_2d` |
| `capped_woven_gradient_2d.yaml` | CappedWovenGradient2DSource | N/A |

## Contributing

When adding new synthetic YAML streams:
1. Follow the naming convention: `<pattern>_<dimension>.yaml`
2. Include descriptive comments at the top of the file
3. Document the use case and key parameters
4. Update this README with the new entry
5. Test with at least one FTK algorithm

## References

- FTK Documentation: https://github.com/hguo/ftk
- ndarray Synthetic Header: `include/ndarray/synthetic.hh`
- Distributed Stream Example: `examples/distributed_synthetic_stream.yaml`
- Stream Summary: `docs/SYNTHETIC_STREAMS_SUMMARY.md`
