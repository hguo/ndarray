# Synthetic YAML Streams Overview

## Visual Organization

```
yaml_exampls/
│
├─── 📚 Documentation
│    ├── SYNTHETIC_STREAMS_README.md    (Comprehensive guide)
│    ├── INDEX.md                       (Quick reference)
│    └── SYNTHETIC_STREAMS_OVERVIEW.md  (This file)
│
├─── 🔬 2D Scalar Field Streams
│    ├── woven.yaml                     (Basic woven pattern)
│    ├── merger_2d.yaml                 (Merging maxima)
│    ├── moving_extremum_2d.yaml        (Moving critical point)
│    ├── volcano_2d.yaml                (Static volcano shape)
│    ├── capped_woven_gradient_2d.yaml  (Bounded gradient)
│    └── high_resolution_woven.yaml     (Performance testing)
│
├─── 🌊 2D Vector Field Streams
│    ├── double_gyre_2d.yaml            (LCS detection)
│    └── time_varying_double_gyre.yaml  (Fine temporal resolution)
│
├─── 📦 3D Scalar Field Streams
│    ├── moving_extremum_3d.yaml        (3D critical point)
│    ├── moving_ramp_3d.yaml            (Single plane)
│    └── moving_dual_ramp_3d.yaml       (Dual planes)
│
├─── 🌪️  3D Vector Field Streams
│    ├── abc_flow_3d.yaml               (Chaotic flow)
│    └── tornado_3d.yaml                (Vortex structure)
│
├─── ⚡ Distributed/Parallel Streams
│    ├── distributed_merger.yaml         (Parallel critical point tracking)
│    ├── distributed_abc_flow.yaml       (Parallel vortex analysis)
│    └── distributed_tornado.yaml        (Parallel vortex detection)
│
├─── 🎯 Multi-Field Streams
│    ├── multi_synthetic_features.yaml   (Combined 2D fields)
│    ├── complex_3d_multi_field.yaml     (Combined 3D fields)
│    └── multi_resolution_woven.yaml     (Multi-scale analysis)
│
└─── 🔧 Testing & Benchmarking
     ├── small_test_synthetic.yaml       (Quick testing)
     └── benchmark_synthetic.yaml        (Standard benchmarks)
```

## Synthetic Data Generator Coverage Map

| Generator Function | 2D YAML | 3D YAML | Distributed | Multi-Field | Test/Bench |
|-------------------|---------|---------|-------------|-------------|------------|
| `synthetic_woven_2D` | ✅ woven.yaml | - | - | ✅ multi_synthetic | ✅ small_test |
| `synthetic_woven_2Dt` | ✅ high_res | - | - | - | ✅ benchmark |
| `synthetic_capped_woven_grad_2D` | ✅ capped | - | - | - | - |
| `synthetic_merger_2D` | ✅ merger_2d | - | ✅ dist_merger | ✅ multi_synthetic | - |
| `synthetic_moving_extremum` | ✅ moving_2d | ✅ moving_3d | - | ✅ multi_synthetic + complex_3d | - |
| `synthetic_volcano` | ✅ volcano_2d | - | - | - | - |
| `synthetic_double_gyre` | ✅ double_gyre | - | - | ✅ multi_synthetic | ✅ benchmark |
| `synthetic_time_varying_double_gyre` | ✅ time_varying | - | - | - | - |
| `synthetic_abc_flow` | - | ✅ abc_flow_3d | ✅ dist_abc | ✅ complex_3d | ✅ benchmark |
| `synthetic_tornado` | - | ✅ tornado_3d | ✅ dist_tornado | - | - |
| `synthetic_moving_ramp` | - | ✅ moving_ramp | - | ✅ complex_3d | - |
| `synthetic_moving_dual_ramp` | - | ✅ dual_ramp | - | - | - |

**Legend**: ✅ = Covered, - = Not applicable for this dimension

## Feature Analysis Matrix

### By Scientific Application

| Application Area | Recommended Streams | Algorithm Types |
|-----------------|--------------------|-----------------|
| **Critical Point Tracking** | merger_2d, moving_extremum_2d/3d | Maxima, minima, saddles |
| **Topological Analysis** | merger_2d, volcano_2d | Birth/death events, persistence |
| **Vortex Detection** | double_gyre_2d, abc_flow_3d, tornado_3d | Vortex cores, swirling strength |
| **Lagrangian Coherent Structures** | double_gyre_2d, time_varying_double_gyre | FTLE, ridges, transport barriers |
| **Isosurface Tracking** | moving_ramp_3d, moving_dual_ramp_3d | Levelset evolution, interface tracking |
| **Gradient-Based Features** | capped_woven_gradient_2d | Ridge/valley lines, gradient thresholds |
| **Multi-Field Correlation** | multi_synthetic_features, complex_3d_multi_field | Feature correspondence |

### By Computational Requirements

| Category | Memory | Compute | I/O | Best For |
|----------|--------|---------|-----|----------|
| **Small Test** | Low (< 10 MB) | Seconds | Minimal | Debugging, unit tests |
| **Medium Resolution** | Medium (10-100 MB) | Minutes | Moderate | Algorithm development |
| **High Resolution** | High (> 1 GB) | Hours | Heavy | Production, validation |
| **Distributed** | Scaled | Parallel | Heavy | Large-scale, HPC |

### Dimension & Complexity Breakdown

```
Complexity Scale: ⭐ (Simple) → ⭐⭐⭐⭐⭐ (Complex)

2D Scalar Fields:
  woven.yaml                    ⭐⭐     32×32×100
  moving_extremum_2d.yaml       ⭐⭐     64×64×100
  merger_2d.yaml                ⭐⭐⭐   128×128×100
  volcano_2d.yaml               ⭐      128×128×1 (static)
  capped_woven_gradient_2d.yaml ⭐⭐⭐   128×128×50
  high_resolution_woven.yaml    ⭐⭐⭐⭐  1024×1024×100

2D Vector Fields:
  double_gyre_2d.yaml           ⭐⭐⭐   256×128×50
  time_varying_double_gyre.yaml ⭐⭐⭐⭐  512×256×200

3D Scalar Fields:
  moving_extremum_3d.yaml       ⭐⭐⭐   32×32×32×50
  moving_ramp_3d.yaml           ⭐⭐⭐   64×64×64×80
  moving_dual_ramp_3d.yaml      ⭐⭐⭐⭐  64×64×64×60

3D Vector Fields:
  abc_flow_3d.yaml              ⭐⭐⭐⭐  64×64×64×25
  tornado_3d.yaml               ⭐⭐⭐⭐  48×48×48×40

Distributed:
  distributed_merger.yaml       ⭐⭐⭐⭐⭐ 512×512×100
  distributed_abc_flow.yaml     ⭐⭐⭐⭐⭐ 128³×50
  distributed_tornado.yaml      ⭐⭐⭐⭐⭐ 96³×80

Multi-Field:
  multi_synthetic_features.yaml ⭐⭐⭐   4 fields
  complex_3d_multi_field.yaml   ⭐⭐⭐⭐⭐ 3 fields (3D)
  multi_resolution_woven.yaml   ⭐⭐⭐⭐  3 scales
```

## Workflow Recommendations

### 1. Algorithm Development Workflow
```
Step 1: Start with small_test_synthetic.yaml
        ↓
Step 2: Test core logic with moving_extremum_2d.yaml (predictable features)
        ↓
Step 3: Validate topology with merger_2d.yaml (merge/split events)
        ↓
Step 4: Test on vector fields with double_gyre_2d.yaml
        ↓
Step 5: Scale to 3D with moving_extremum_3d.yaml or abc_flow_3d.yaml
        ↓
Step 6: Benchmark with benchmark_synthetic.yaml
```

### 2. Parallel Algorithm Development
```
Step 1: Verify serial correctness with medium-sized streams
        ↓
Step 2: Test on distributed_merger.yaml (2D, simpler)
        ↓
Step 3: Validate 3D parallelization with distributed_abc_flow.yaml
        ↓
Step 4: Stress test with large distributed configurations
```

### 3. Multi-Field Algorithm Development
```
Step 1: Single field testing (any appropriate stream)
        ↓
Step 2: Two-field correlation with multi_synthetic_features.yaml
        ↓
Step 3: Complex 3D multi-field with complex_3d_multi_field.yaml
```

## Parameter Tuning Guide

### Resolution Selection
- **Prototyping**: 16-32 grid points per dimension
- **Development**: 64-128 grid points per dimension
- **Validation**: 256-512 grid points per dimension
- **Production**: 512+ grid points per dimension

### Timestep Selection
- **Static/Steady**: 1-5 timesteps
- **Basic temporal**: 10-50 timesteps
- **Fine temporal**: 100-200 timesteps
- **Very fine**: 500+ timesteps (for detailed particle tracking)

### Ghost Layer Selection (Distributed)
- **Finite difference (1st order)**: 1 layer
- **Finite difference (2nd order)**: 2 layers
- **Higher-order stencils**: 3-4 layers
- **Feature detection**: 2-3 layers recommended

## Integration with FTK/FTK2

### FTK CLI Integration
```bash
# Critical point tracking
ftk -f woven -w 128 -h 128 --nsteps 100 --feature critical_point_tracking_2d

# Using YAML instead
ftk --stream-config yaml_exampls/merger_2d.yaml --feature critical_point_tracking_2d

# Parallel execution
mpirun -np 4 ftk --stream-config yaml_exampls/distributed_merger.yaml --feature critical_point_tracking_2d
```

### FTK2 API Integration
```cpp
// C++ example
#include <ndarray/ndarray_group_stream.hh>

// Load synthetic stream from YAML
auto stream = ftk::ndarray_group_stream::from_yaml("yaml_exampls/merger_2d.yaml");

// Access timesteps
for (int t = 0; t < stream.n_timesteps(); t++) {
    auto data = stream.get_timestep(t);
    // Process data...
}
```

### Python Integration
```python
# Python example (if Python bindings available)
import ndarray

# Load stream
stream = ndarray.load_yaml_stream("yaml_exampls/double_gyre_2d.yaml")

# Iterate through timesteps
for t, data in enumerate(stream):
    # Process velocity field
    velocity = data['velocity']
    # Compute FTLE or other features...
```

## Best Practices

### ✅ Do's
- Start with small test configurations for rapid iteration
- Use predictable patterns (moving_extremum) to validate correctness
- Test topology handling with merger patterns
- Benchmark consistently with benchmark_synthetic.yaml
- Use distributed versions for scalability testing
- Document parameter choices in your analysis code

### ❌ Don'ts
- Don't start with high-resolution streams during debugging
- Don't skip intermediate validation steps
- Don't use distributed configurations before serial validation
- Don't forget ghost layers for distributed stencil operations
- Don't mix very different grid sizes in multi-field analysis without proper interpolation

## Future Extensions

### Potential New Streams
- **Turbulence Models**: Homogeneous isotropic turbulence, turbulent channel flow
- **Reaction-Diffusion**: Gray-Scott, Turing patterns
- **Fluid Instabilities**: Rayleigh-Taylor, Kelvin-Helmholtz
- **Planetary Flows**: Hadley cell circulation, polar vortices
- **Medical Imaging**: Synthetic CT/MRI phantoms
- **Material Science**: Phase field evolution, grain growth

### Advanced Features
- **Adaptive Mesh Refinement**: Multi-level grids with refinement
- **Unstructured Meshes**: Tetrahedral/hexahedral mesh variants
- **Time-Dependent Parameters**: Varying flow parameters over time
- **Stochastic Variations**: Adding controlled noise/perturbations
- **Composite Patterns**: Superposition of multiple generators

## Getting Help

- **Documentation**: See `SYNTHETIC_STREAMS_README.md` for detailed parameter descriptions
- **Examples**: Check `../examples/` directory for usage examples
- **Source Code**: Review `../include/ndarray/synthetic.hh` for implementation details
- **Issues**: Report problems at the project's issue tracker

## Summary Statistics

| Category | Count | Total Data Size (est.) |
|----------|-------|----------------------|
| 2D Scalar Streams | 6 | ~500 MB - 5 GB |
| 2D Vector Streams | 2 | ~1-8 GB |
| 3D Scalar Streams | 3 | ~500 MB - 2 GB |
| 3D Vector Streams | 2 | ~1-2 GB |
| Distributed Streams | 3 | ~10-50 GB |
| Multi-Field Streams | 3 | ~1-5 GB |
| Test/Benchmark | 2 | ~10 MB - 10 GB |
| **Total** | **21 synthetic** | **~15-80 GB** |

Note: Data sizes are estimates and depend on dtype (float32 vs float64) and actual timesteps generated.

---

**Last Updated**: 2024-02-25
**Version**: 1.0
**Maintainer**: ndarray/FTK development team
