# Wilkins Quick Start Guide

This guide uses your specific installations to set up Wilkins with ndarray.

## Your System Setup

- **HDF5**: `~/local/hdf5-1.14.6-mpich-5.0.0` (parallel with MPICH)
- **MPI**: `/Users/guo.2154/local/mpich-5.0.0`
- **Python**: `/Users/guo.2154/local/Python-3.10.6-openssl-1.1.1q`
- **CMake**: `/Users/guo.2154/local/cmake-4.2.3`

## Quick Start (3 Steps)

### 1. Verify Your Environment

```bash
./verify_wilkins_deps.sh
```

This checks if your base dependencies (HDF5, MPI, Python) are accessible and shows what needs to be installed.

### 2. Install Wilkins and Dependencies

```bash
./install_wilkins.sh
```

This will:
- Install **Henson** to `~/local/henson`
- Install **LowFive** to `~/local/lowfive`
- Install **Wilkins** to `~/local/wilkins`

Installation takes about 10-20 minutes depending on your system.

### 3. Build ndarray with Wilkins Support

```bash
./build_with_wilkins.sh
```

This creates a `build_wilkins` directory with ndarray built with Henson/Wilkins support.

## Environment Variables

Add these to your `~/.bashrc` or `~/.zshrc`:

```bash
# Wilkins environment
export PATH=/Users/guo.2154/local/mpich-5.0.0/bin:$PATH
export PATH=$HOME/local/hdf5-1.14.6-mpich-5.0.0/bin:$PATH
export PATH=/Users/guo.2154/local/Python-3.10.6-openssl-1.1.1q/bin:$PATH
export PATH=$HOME/local/wilkins/bin:$PATH

export LD_LIBRARY_PATH=/Users/guo.2154/local/mpich-5.0.0/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$HOME/local/hdf5-1.14.6-mpich-5.0.0/lib:$LD_LIBRARY_PATH

export HENSON=$HOME/local/henson
export HDF5_ROOT=$HOME/local/hdf5-1.14.6-mpich-5.0.0

# For running Wilkins workflows
export HDF5_VOL_CONNECTOR="lowfive under_vol=0;under_info={};"
export HDF5_PLUGIN_PATH=$HOME/local/lowfive/lib
```

Then:
```bash
source ~/.bashrc  # or ~/.zshrc
```

## Testing

Test Wilkins examples:
```bash
cd ~/local/wilkins/examples/lowfive/cycle
./run_cycle.sh
```

Test ndarray:
```bash
cd build_wilkins
make test
```

## What is Wilkins?

**Wilkins** is an in-situ workflow system that allows multiple tasks to:
- Communicate through **HDF5** (both in-memory and on-disk)
- Run concurrently with **MPI**
- Exchange data without file I/O overhead

### Key Components

1. **Henson** - Execution model (manages task launching)
2. **LowFive** - Data transport (HDF5-based in-memory communication)
3. **Wilkins** - Workflow orchestration (task graph management)

### Example Workflow

```yaml
# workflow.yaml
tasks:
  - name: simulation
    exec: ./sim_executable
    ranks: 4

  - name: analysis
    exec: ./analysis_executable
    ranks: 2

connections:
  - from: simulation
    to: analysis
    dataset: /simulation/data
```

Run with:
```bash
mpirun -n 6 python wilkins-master.py workflow.yaml
```

## Using ndarray in Wilkins Workflows

Your ndarray library in `build_wilkins/lib/libndarray.so` is now:
- ✓ Linked with Henson
- ✓ Position-independent code enabled
- ✓ Built as a shared library
- ✓ Ready for Wilkins workflows

Write your simulation/analysis code using ndarray, compile as executables linked against `libndarray.so`, and orchestrate them with Wilkins.

## Directory Structure After Installation

```
~/local/
├── henson/              # Henson installation
│   ├── include/
│   └── lib/
│       ├── libhenson.a
│       └── libhenson-pmpi-static.a
├── lowfive/            # LowFive installation
│   ├── include/
│   └── lib/
│       ├── liblowfive.dylib
│       └── liblowfive-dist.a
└── wilkins/            # Wilkins installation
    ├── bin/
    │   └── wilkins-master.py
    └── examples/

workspace/projects/ndarray/
├── build_wilkins/      # ndarray built with Wilkins support
│   └── lib/
│       └── libndarray.so
└── wilkins_build/      # Temporary build files (created during install)
```

## More Information

- Full setup guide: `docs/WILKINS_SETUP.md`
- Wilkins repository: https://github.com/orcunyildiz/wilkins
- LowFive repository: https://github.com/diatomic/LowFive
- Henson repository: https://github.com/henson-insitu/henson

## Troubleshooting

**Problem**: CMake can't find HDF5/MPI/Python

**Solution**: Make sure the paths are correct in the scripts. They're hardcoded to your specific installations.

**Problem**: Library not found at runtime

**Solution**: Check that LD_LIBRARY_PATH includes all library directories (see Environment Variables section).

**Problem**: Wilkins examples fail

**Solution**: Ensure HDF5_VOL_CONNECTOR and HDF5_PLUGIN_PATH are set before running.
