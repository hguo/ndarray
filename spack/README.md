# Spack Package for ndarray

This directory contains the Spack package recipe for ndarray.

## Quick Start

### Option 1: Install from Spack Repository (After Submission)

```bash
# Once the package is merged into Spack mainline
spack install ndarray

# With specific variants
spack install ndarray+mpi+netcdf+hdf5
```

### Option 2: Install from Local Repository (Development)

```bash
# Add this repository to Spack
spack repo add /path/to/ndarray/spack

# Install
spack install ndarray
```

### Option 3: Use Package.py Directly

```bash
# Copy package.py to your Spack repo
cp spack/package.py $(spack location -p spack)/var/spack/repos/builtin/packages/ndarray/package.py

# Install
spack install ndarray
```

## Common Configurations

### Minimal Build (Workstation)
```bash
spack install ndarray~mpi~netcdf~hdf5~yaml
```

### Standard HPC Build
```bash
spack install ndarray+mpi+netcdf+hdf5+yaml
```

### Advanced I/O (Parallel)
```bash
spack install ndarray+mpi+netcdf+hdf5+pnetcdf+adios2
```

### Visualization Pipeline
```bash
spack install ndarray+mpi+netcdf+vtk
```

### Full Featured
```bash
spack install ndarray+mpi+netcdf+hdf5+pnetcdf+adios2+vtk+yaml+openmp
```

### GPU Enabled (CUDA)
```bash
spack install ndarray+mpi+netcdf+cuda
```

## Available Variants

| Variant | Default | Description |
|---------|---------|-------------|
| `+shared` | ON | Build shared libraries |
| `+tests` | OFF | Build test suite |
| `+examples` | OFF | Build examples |
| `+netcdf` | ON | NetCDF file I/O |
| `+hdf5` | ON | HDF5 file I/O |
| `+adios2` | OFF | ADIOS2 in-situ I/O |
| `+vtk` | OFF | VTK visualization export |
| `+mpi` | ON | MPI parallel computing |
| `+openmp` | OFF | OpenMP thread parallelism |
| `+pnetcdf` | OFF | Parallel NetCDF (requires +mpi) |
| `+cuda` | OFF | NVIDIA GPU acceleration |
| `+sycl` | OFF | Intel/AMD GPU acceleration |
| `+yaml` | ON | YAML stream configuration |
| `+png` | OFF | PNG image I/O |

## Dependency Resolution

Spack automatically resolves all dependencies and ensures compatibility:

```bash
# Check what will be installed
spack spec ndarray+mpi+netcdf+hdf5

# View dependency tree
spack graph ndarray+mpi+netcdf+hdf5
```

## Using Installed Package

```bash
# Load the package and all dependencies
spack load ndarray

# Or use in your environment
spack env create myproject
spack env activate myproject
spack add ndarray+mpi+netcdf
spack install
```

## Compiler-Specific Builds

```bash
# Build with specific compiler
spack install ndarray %gcc@11.2.0

# Build with Intel compiler
spack install ndarray %intel@2021.4.0

# Build with Cray compiler
spack install ndarray %cray@14.0.0
```

## Site-Specific Configurations

For your HPC center, create a `packages.yaml` configuration:

```yaml
# ~/.spack/packages.yaml
packages:
  ndarray:
    variants: +mpi+netcdf+hdf5+yaml

  # Use system MPI
  mpi:
    buildable: false
    externals:
    - spec: "openmpi@4.1.1"
      prefix: /usr/local/openmpi/4.1.1

  # Use system NetCDF
  netcdf-c:
    buildable: false
    externals:
    - spec: "netcdf-c@4.8.1+mpi"
      prefix: /usr/local/netcdf/4.8.1
```

## Integration with Environments

Example environment for FTK workflow:

```yaml
# spack.yaml
spack:
  specs:
  - ndarray+mpi+netcdf+hdf5+adios2+vtk
  - ftk  # Assuming FTK also has Spack package
  - paraview  # For visualization

  view: true
  concretizer:
    unify: true
```

```bash
spack env create ftk-workflow spack.yaml
spack env activate ftk-workflow
spack install
```

## Troubleshooting

### Build Fails

```bash
# View build log
spack install --verbose ndarray

# Keep build directory for debugging
spack install --keep-stage ndarray
cd $(spack location -b ndarray)
```

### Dependency Conflicts

```bash
# Use specific versions
spack install ndarray ^netcdf-c@4.8.1 ^hdf5@1.12.1

# Force specific dependency
spack install ndarray ^mpich
```

### Module Files

```bash
# Generate module files
spack module tcl refresh

# Load via modules
module load ndarray
```

## Submitting to Spack Mainline

Once tested, submit package to Spack repository:

1. Fork https://github.com/spack/spack
2. Copy `package.py` to `var/spack/repos/builtin/packages/ndarray/`
3. Test the package
4. Submit pull request

```bash
# Test checklist
spack style --fix  # Format check
spack lint ndarray  # Package lint
spack install --test=root ndarray  # Build and test
spack uninstall ndarray  # Clean install test
```

## Benefits of Spack

1. **Automatic dependency resolution**: No manual cmake flags
2. **Multiple versions/variants**: Can install different configurations side-by-side
3. **Reproducible builds**: Exact specification of entire software stack
4. **Site configuration**: Reuse system-installed libraries
5. **Module generation**: Automatic environment modules
6. **Compiler selection**: Easy to build with different compilers
7. **Environment management**: Isolated environments for different projects

## Examples

### Development Workflow
```bash
# Create development environment
spack env create ndarray-dev
spack env activate ndarray-dev
spack add ndarray+tests+examples@main
spack install

# After code changes
spack cd ndarray
cd build
make
```

### Production Deployment
```bash
# Install specific version
spack install ndarray@0.0.1+mpi+netcdf+hdf5

# Generate module file
spack module tcl refresh ndarray

# Users load via modules
module load ndarray/0.0.1-gcc-11.2.0-mpi
```

### CI/CD Integration
```yaml
# .github/workflows/spack.yml
- name: Install via Spack
  run: |
    git clone https://github.com/spack/spack.git
    . spack/share/spack/setup-env.sh
    spack install ndarray+tests
    spack load ndarray
    ctest
```
