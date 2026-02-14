# HDF5 Multiple Timesteps Per File

## Feature Overview

The HDF5 stream now supports reading multiple timesteps (datasets) from a single HDF5 file using the `timesteps_per_file` parameter and h5_name format patterns.

## Use Case

This is useful when your simulation outputs multiple timesteps to a single HDF5 file with different dataset names, such as:
- `data_t0`, `data_t1`, `data_t2` for timesteps 0, 1, 2

## Configuration

### YAML Configuration

```yaml
stream:
  name: my_hdf5_stream
  substreams:
    - name: h5_data
      format: h5
      filenames:
        - output_file0.h5
        - output_file1.h5
      timesteps_per_file: 3  # Each file contains 3 timesteps
      vars:
        - name: pressure
          h5_name: data_t%d    # Pattern: %d will be replaced with 0, 1, 2
          dtype: float64
```

### How It Works

1. **Total timesteps calculation**: `total_timesteps = number_of_files × timesteps_per_file`
   - In the example above: 2 files × 3 timesteps = 6 total timesteps

2. **Timestep to file mapping**: When reading timestep `i`:
   - `file_index = i / timesteps_per_file`
   - `local_timestep = i % timesteps_per_file`

3. **Dataset name formatting**:
   - If `h5_name` contains `%d`, it's formatted with the local timestep index
   - Example: `h5_name: data_t%d` becomes `data_t0`, `data_t1`, `data_t2`

## Example

Given the configuration above:

| Global Timestep | File | Local Timestep | Dataset Name |
|-----------------|------|----------------|--------------|
| 0 | output_file0.h5 | 0 | data_t0 |
| 1 | output_file0.h5 | 1 | data_t1 |
| 2 | output_file0.h5 | 2 | data_t2 |
| 3 | output_file1.h5 | 0 | data_t0 |
| 4 | output_file1.h5 | 1 | data_t1 |
| 5 | output_file1.h5 | 2 | data_t2 |

## Code Example

```cpp
#include <ndarray/ndarray_group_stream.hh>

// Parse YAML configuration
ftk::stream s;
s.parse_yaml("config.yaml");

// Total timesteps: 6
int total = s.total_timesteps();

// Read specific timesteps
auto data0 = s.read(0);  // Reads data_t0 from output_file0.h5
auto data1 = s.read(1);  // Reads data_t1 from output_file0.h5
auto data3 = s.read(3);  // Reads data_t0 from output_file1.h5

// Access the variable
auto pressure = data0->get_arr<double>("pressure");
```

## Backward Compatibility

If `timesteps_per_file` is not specified, it defaults to 1, maintaining backward compatibility:
- Each file is treated as one timestep
- No formatting is applied to h5_name

## Implementation Details

- **Location**: `include/ndarray/ndarray_stream_hdf5.hh`
- **Test**: `tests/test_ndarray_stream.cpp` Test 13
- **Added in**: 2026-02-14

## Format Specifiers

Currently supported format specifiers in `h5_name`:
- `%d`: Integer timestep index (0, 1, 2, ...)

Future enhancements could support:
- `%03d`: Zero-padded integers (000, 001, 002, ...)
- `%s`: String replacements
- Multiple format specifiers

## Limitations

- Only one `%d` format specifier per h5_name is currently supported
- All files must have the same number of timesteps per file
- Dataset names must follow a predictable pattern

## Related

- See `tests/test_ndarray_stream.cpp` for a complete working example
- See NetCDF stream for similar timestep handling
