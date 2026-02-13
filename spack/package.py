# Copyright 2013-2024 Lawrence Livermore National Security, LLC and other
# Spack Project Developers. See the top-level COPYRIGHT file for details.
#
# SPDX-License-Identifier: (Apache-2.0 OR MIT)

from spack.package import *


class Ndarray(CMakePackage):
    """Multidimensional array library for scientific computing and data analysis.

    ndarray provides C++ templates for n-dimensional arrays with support for
    various I/O formats (NetCDF, HDF5, ADIOS2, VTK), parallel computing (MPI,
    OpenMP), and GPU acceleration (CUDA, SYCL). Designed for HPC applications
    and topological data analysis workflows.
    """

    homepage = "https://github.com/hguo/ndarray"
    url = "https://github.com/hguo/ndarray/archive/v0.0.1.tar.gz"
    git = "https://github.com/hguo/ndarray.git"

    maintainers("hguo")

    license("MIT")

    version("main", branch="main")
    version("0.0.1", sha256="<TODO: add sha256 checksum>")

    # Core variants
    variant("shared", default=True, description="Build shared libraries")
    variant("tests", default=False, description="Build tests")
    variant("examples", default=False, description="Build examples")

    # I/O format support
    variant("netcdf", default=True, description="Enable NetCDF support")
    variant("hdf5", default=True, description="Enable HDF5 support")
    variant("adios2", default=False, description="Enable ADIOS2 support")
    variant("vtk", default=False, description="Enable VTK support")

    # Parallel computing
    variant("mpi", default=True, description="Enable MPI support")
    variant("openmp", default=False, description="Enable OpenMP support")
    variant("pnetcdf", default=False, description="Enable parallel NetCDF support")

    # GPU acceleration
    variant("cuda", default=False, description="Enable CUDA support")
    variant("sycl", default=False, description="Enable SYCL support")

    # Utilities
    variant("yaml", default=True, description="Enable YAML stream support")
    variant("png", default=False, description="Enable PNG image I/O")

    # Dependencies
    depends_on("cmake@3.10:", type="build")
    depends_on("yaml-cpp", when="+yaml")

    # I/O libraries
    depends_on("netcdf-c@4.6.0:", when="+netcdf")
    depends_on("netcdf-cxx4", when="+netcdf")
    depends_on("hdf5@1.10.0:", when="+hdf5")
    depends_on("adios2@2.7.0:", when="+adios2")
    depends_on("vtk@8.0:", when="+vtk")

    # Parallel I/O
    depends_on("mpi", when="+mpi")
    depends_on("parallel-netcdf", when="+pnetcdf")
    depends_on("hdf5+mpi", when="+hdf5+mpi")
    depends_on("netcdf-c+mpi", when="+netcdf+mpi")
    depends_on("adios2+mpi", when="+adios2+mpi")

    # GPU support
    depends_on("cuda@10.0:", when="+cuda")
    # SYCL support requires specific compilers

    # Utilities
    depends_on("libpng", when="+png")

    # Conflicts
    conflicts("+pnetcdf", when="~mpi", msg="Parallel NetCDF requires MPI")
    conflicts("+sycl", when="+cuda", msg="Cannot enable both CUDA and SYCL")

    def cmake_args(self):
        args = []

        # Build options
        args.append(self.define_from_variant("BUILD_SHARED_LIBS", "shared"))
        args.append(self.define_from_variant("NDARRAY_BUILD_TESTS", "tests"))
        args.append(self.define_from_variant("NDARRAY_BUILD_EXAMPLES", "examples"))

        # Feature flags - use AUTO for optional dependencies
        args.append(self.define("NDARRAY_USE_YAML",
                               "TRUE" if "+yaml" in self.spec else "FALSE"))
        args.append(self.define("NDARRAY_USE_NETCDF",
                               "TRUE" if "+netcdf" in self.spec else "FALSE"))
        args.append(self.define("NDARRAY_USE_HDF5",
                               "TRUE" if "+hdf5" in self.spec else "FALSE"))
        args.append(self.define("NDARRAY_USE_ADIOS2",
                               "TRUE" if "+adios2" in self.spec else "FALSE"))
        args.append(self.define("NDARRAY_USE_VTK",
                               "TRUE" if "+vtk" in self.spec else "FALSE"))
        args.append(self.define("NDARRAY_USE_MPI",
                               "TRUE" if "+mpi" in self.spec else "FALSE"))
        args.append(self.define("NDARRAY_USE_OPENMP",
                               "TRUE" if "+openmp" in self.spec else "FALSE"))
        args.append(self.define("NDARRAY_USE_PNETCDF",
                               "TRUE" if "+pnetcdf" in self.spec else "FALSE"))
        args.append(self.define("NDARRAY_USE_CUDA",
                               "TRUE" if "+cuda" in self.spec else "FALSE"))
        args.append(self.define("NDARRAY_USE_SYCL",
                               "TRUE" if "+sycl" in self.spec else "FALSE"))
        args.append(self.define("NDARRAY_USE_PNG",
                               "TRUE" if "+png" in self.spec else "FALSE"))

        # MPI compiler wrappers
        if "+mpi" in self.spec:
            args.append(self.define("CMAKE_CXX_COMPILER", self.spec["mpi"].mpicxx))
            args.append(self.define("CMAKE_C_COMPILER", self.spec["mpi"].mpicc))

        return args
