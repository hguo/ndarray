# ndarray Documentation

**Version**: 0.0.6

👉 **[Start Here: Getting Started Guide](GETTING_STARTED.md)** - Complete tutorial for new users
📚 **[Documentation Index](INDEX.md)** - Browse all documentation by topic

---

## Quick Links

### For New Users
- **[Getting Started](GETTING_STARTED.md)** - 15-minute tutorial with examples
- **[Installation Guide](GETTING_STARTED.md#installation)** - Build from source
- **[Your First Program](GETTING_STARTED.md#your-first-program)** - Hello World

### Core Concepts
- **[Array Indexing](ARRAY_INDEXING.md)** - Fortran vs C order
- **[Dimension Ordering](DIMENSION_ORDERING.md)** - Understanding conventions
- **[Storage Backends](STORAGE_BACKENDS.md)** - Native, xtensor, Eigen

### I/O Formats
- **[Parallel HDF5](PARALLEL_HDF5.md)** - MPI-parallel HDF5 I/O
- **[GPU Support](GPU_SUPPORT.md)** - CUDA, HIP, SYCL
- **[ADIOS2](archive/ADIOS2_TESTS.md)** - High-performance I/O

### Parallel Computing
- **[Distributed Arrays](DISTRIBUTED_NDARRAY.md)** - MPI domain decomposition
- **[Multi-component Arrays](MULTICOMPONENT_ARRAYS.md)** - Vector fields

---

## Documentation Organization

```
docs/
├── GETTING_STARTED.md       ⭐ Start here!
├── INDEX.md                  📚 Complete documentation index
│
├── Core Concepts
│   ├── ARRAY_INDEXING.md
│   ├── DIMENSION_ORDERING.md
│   ├── MULTICOMPONENT_ARRAYS.md
│   └── STORAGE_BACKENDS.md
│
├── I/O & Formats
│   ├── PARALLEL_HDF5.md
│   ├── GPU_SUPPORT.md
│   ├── IO_BACKEND_AGNOSTIC.md
│   └── PNG_SUPPORT.md
│
├── Parallel Computing
│   ├── DISTRIBUTED_NDARRAY.md
│   ├── DISTRIBUTED_INDEXING_CLARIFICATION.md
│   └── MULTICOMPONENT_ARRAYS_DISTRIBUTED.md
│
├── Advanced
│   ├── ERROR_HANDLING.md
│   ├── EXCEPTION_HANDLING.md
│   └── ZERO_COPY_OPTIMIZATION.md
│
├── progress/                 📊 Development progress
│   ├── CRITICAL_ANALYSIS.md
│   └── IMPROVEMENTS_SUMMARY_2026-02-20.md
│
└── archive/                  🗄️  Older/internal docs
    ├── ADIOS2_TESTS.md
    ├── VTK_TESTS.md
    └── ...
```

---

## Getting Help

- **New users**: Start with [GETTING_STARTED.md](GETTING_STARTED.md)
- **Specific topics**: Check [INDEX.md](INDEX.md)
- **Examples**: See `../tests/` directory
- **Issues**: https://github.com/hguo/ndarray/issues

---

## Contributing to Documentation

Documentation improvements are welcome! Please:
1. Follow Markdown best practices
2. Include code examples that compile
3. Update INDEX.md when adding new docs
4. Test examples before submitting

See [CONTRIBUTING.md](../CONTRIBUTING.md) for details.
