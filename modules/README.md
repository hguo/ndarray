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

| Script | System | Tested | Description |
|--------|--------|--------|-------------|
| `nersc-cori.sh` | NERSC Cori | ✅ | Cray XC40, Intel Xeon |
| `nersc-perlmutter.sh` | NERSC Perlmutter | ⚠️ | HPE Cray EX, AMD EPYC + NVIDIA A100 |
| `tacc-stampede2.sh` | TACC Stampede2 | ✅ | Intel Xeon Phi KNL |
| `ornl-summit.sh` | ORNL Summit | ⚠️ | IBM Power9 + NVIDIA V100 |
| `minimal.sh` | Any system | ✅ | Minimal build, no optional deps |
| `standard.sh` | Generic HPC | ✅ | Standard HPC with MPI + NetCDF |

✅ = Tested and verified
⚠️ = Configuration exists but needs testing

## Contributing

If you successfully build on a system not listed here, please contribute:

1. Create a new script based on `template.sh`
2. Test it builds successfully
3. Submit a pull request

Include in your PR:
- System name and architecture
- Module versions used
- Any special considerations
