# Synthetic YAML Streams Index

This index provides a quick reference to all YAML stream configurations available in this directory.

## File Count
- **Total YAML files**: 31 (12 original + 19 new synthetic streams)
- **Documentation files**: 2 (SYNTHETIC_STREAMS_README.md, INDEX.md)

## Complete File List

### Original Examples (Real Data)
1. `5jets.yaml` - Five jets dataset
2. `cm1.yaml` - CM1 atmospheric model data
3. `combustion.yaml` - Combustion simulation data
4. `cylinder2D.yaml` - 2D cylinder flow simulation
5. `cylinder2D-writer.yaml` - 2D cylinder with writer configuration
6. `island_coalescence.yaml` - Island coalescence simulation
7. `mpas.yaml` - MPAS climate model data
8. `mpas1.yaml` - MPAS variant configuration
9. `mpas_with_patterns.yaml` - MPAS with pattern matching
10. `supernova.yaml` - Supernova simulation data
11. `xgc_adios2.yaml` - XGC fusion simulation (ADIOS2 format)
12. `xgc_h5.yaml` - XGC fusion simulation (HDF5 format)

### Original Synthetic Examples
13. `woven.yaml` - Basic woven pattern
14. `woven-writer.yaml` - Woven pattern with writer

### New 2D Scalar Synthetic Streams
15. `merger_2d.yaml` - Merging/splitting Gaussian maxima
16. `moving_extremum_2d.yaml` - Single moving critical point (2D)
17. `volcano_2d.yaml` - Volcano-like elevation pattern
18. `capped_woven_gradient_2d.yaml` - Woven gradient with capping
19. `high_resolution_woven.yaml` - High-res woven for performance testing

### New 2D Vector Synthetic Streams
20. `double_gyre_2d.yaml` - Classic double-gyre flow
21. `time_varying_double_gyre.yaml` - Extended double gyre with fine time resolution

### New 3D Scalar Synthetic Streams
22. `moving_extremum_3d.yaml` - Single moving critical point (3D)
23. `moving_ramp_3d.yaml` - Single moving planar surface
24. `moving_dual_ramp_3d.yaml` - Two parallel moving planes

### New 3D Vector Synthetic Streams
25. `abc_flow_3d.yaml` - Arnold-Beltrami-Childress chaotic flow
26. `tornado_3d.yaml` - Tornado vortex structure

### New Distributed Synthetic Streams
27. `distributed_merger.yaml` - Domain-decomposed merger pattern
28. `distributed_abc_flow.yaml` - Domain-decomposed ABC flow
29. `distributed_tornado.yaml` - Domain-decomposed tornado

### New Multi-Field Synthetic Streams
30. `multi_synthetic_features.yaml` - Multiple 2D synthetic fields combined
31. `complex_3d_multi_field.yaml` - Multiple 3D synthetic fields combined
32. `multi_resolution_woven.yaml` - Woven pattern at three resolutions

### New Testing/Benchmark Streams
33. `small_test_synthetic.yaml` - Minimal configuration for quick testing
34. `benchmark_synthetic.yaml` - Standard benchmark configuration

## Quick Selection Guide

### By Dimension
- **2D Scalar**: woven, merger_2d, moving_extremum_2d, volcano_2d, capped_woven_gradient_2d
- **2D Vector**: double_gyre_2d, time_varying_double_gyre
- **3D Scalar**: moving_extremum_3d, moving_ramp_3d, moving_dual_ramp_3d
- **3D Vector**: abc_flow_3d, tornado_3d

### By Use Case
- **Quick Testing**: small_test_synthetic.yaml
- **Critical Point Tracking**: merger_2d.yaml, moving_extremum_2d.yaml, moving_extremum_3d.yaml
- **Vortex Detection**: double_gyre_2d.yaml, abc_flow_3d.yaml, tornado_3d.yaml
- **Isosurface Tracking**: moving_ramp_3d.yaml, moving_dual_ramp_3d.yaml
- **Parallel Processing**: distributed_merger.yaml, distributed_abc_flow.yaml, distributed_tornado.yaml
- **Multi-Field Analysis**: multi_synthetic_features.yaml, complex_3d_multi_field.yaml
- **Performance Testing**: high_resolution_woven.yaml, benchmark_synthetic.yaml
- **Multi-Scale**: multi_resolution_woven.yaml

### By Complexity
- **Beginner**: woven.yaml, small_test_synthetic.yaml, moving_extremum_2d.yaml
- **Intermediate**: merger_2d.yaml, double_gyre_2d.yaml, volcano_2d.yaml
- **Advanced**: abc_flow_3d.yaml, tornado_3d.yaml, complex_3d_multi_field.yaml
- **Expert**: distributed_abc_flow.yaml, time_varying_double_gyre.yaml, benchmark_synthetic.yaml

### By Grid Size
- **Small (≤32)**: small_test_synthetic.yaml, woven.yaml, moving_extremum_3d.yaml
- **Medium (32-128)**: merger_2d.yaml, double_gyre_2d.yaml, moving_extremum_2d.yaml, abc_flow_3d.yaml
- **Large (128-512)**: high_resolution_woven.yaml, time_varying_double_gyre.yaml, multi_resolution_woven.yaml
- **Very Large (≥512)**: distributed_merger.yaml, distributed_abc_flow.yaml

## New Additions Summary (2024)

### What Was Added
This update added **19 new synthetic YAML stream configurations** to provide comprehensive coverage of all FTK/FTK2 synthetic data generators:

1. Individual configurations for each synthetic data type
2. Distributed/parallel versions for scalable testing
3. Multi-field configurations for correlation analysis
4. Multi-resolution configurations for adaptive methods
5. Benchmark configurations for performance testing
6. Small test configurations for rapid development

### Coverage
- ✅ All 17 synthetic generators from `include/ndarray/synthetic.hh`
- ✅ All 10 FTK ParaView source filters
- ✅ All FTK CLI synthetic options
- ✅ Distributed computing variants
- ✅ Multi-field combinations
- ✅ Testing and benchmarking configurations

### Documentation
- `SYNTHETIC_STREAMS_README.md` - Comprehensive guide to all synthetic streams
- `INDEX.md` - This file, quick reference index
- In-file comments - Each YAML file includes descriptive comments

## See Also
- `SYNTHETIC_STREAMS_README.md` - Detailed documentation for each stream
- `../examples/distributed_synthetic_stream.yaml` - Distributed stream examples with detailed comments
- `../docs/SYNTHETIC_STREAMS_SUMMARY.md` - Summary of synthetic data generators in ndarray
- `../include/ndarray/synthetic.hh` - Source code for synthetic data generators

## Quick Start Examples

```bash
# Test critical point tracking with merger pattern
./ftk_app --input yaml_exampls/merger_2d.yaml

# Test vortex detection with double gyre
./ftk_app --input yaml_exampls/double_gyre_2d.yaml

# Quick algorithm debugging
./ftk_app --input yaml_exampls/small_test_synthetic.yaml

# Parallel processing test
mpirun -np 4 ./ftk_app --input yaml_exampls/distributed_merger.yaml

# Benchmark performance
./ftk_app --input yaml_exampls/benchmark_synthetic.yaml
```
