# HPC Site-Specific Configuration Scripts

This directory contains pre-configured build scripts for common HPC centers.

## Usage

1. Choose the script for your HPC center
2. Source it to load modules and set cmake flags
3. Build as normal

```bash
# Example: Building on NERSC Cori
cd ndarray
source modules/nersc-cori.sh
mkdir build && cd build
cmake ..
make -j8
```

## Available Configurations

| Script | System | Description |
|--------|--------|-------------|
| `template.sh` | Template | Template for creating site-specific configs |
| `minimal.sh` | Any system | Minimal build, no optional deps |
| `standard.sh` | Generic HPC | Standard HPC with MPI + NetCDF |

Add site-specific configurations as needed for your HPC center.

## Contributing

If you successfully build on a system not listed here, please contribute:

1. Create a new script based on `template.sh`
2. Test it builds successfully
3. Submit a pull request

Include in your PR:
- System name and architecture
- Module versions used
- Any special considerations
