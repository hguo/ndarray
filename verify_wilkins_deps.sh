#!/bin/bash

# Wilkins Dependencies Verification Script
# Checks your specific installations for Wilkins support

echo "========================================="
echo "Wilkins Dependencies Verification Script"
echo "========================================="
echo ""

# Your specific installation paths
MPI_ROOT=/Users/guo.2154/local/mpich-5.0.0
HDF5_ROOT=$HOME/local/hdf5-1.14.6-mpich-5.0.0
PYTHON_ROOT=/Users/guo.2154/local/Python-3.10.6-openssl-1.1.1q
CMAKE_ROOT=/Users/guo.2154/local/cmake-4.2.3
HENSON_ROOT=$HOME/local/henson
LOWFIVE_ROOT=$HOME/local/lowfive
WILKINS_ROOT=$HOME/local/wilkins

# Check HDF5
echo "1. Checking HDF5 (1.14.6 with MPICH)..."
if [ -f "$HDF5_ROOT/bin/h5cc" ]; then
    echo "   ✓ HDF5 found at $HDF5_ROOT"
    $HDF5_ROOT/bin/h5cc -showconfig 2>/dev/null | grep "Version" | head -1 || echo "   Version: 1.14.6"
    if $HDF5_ROOT/bin/h5cc -showconfig 2>/dev/null | grep -q "Parallel HDF5: yes"; then
        echo "   ✓ Parallel HDF5 enabled"
    else
        echo "   ℹ Parallel status unknown (but should be enabled with MPICH)"
    fi
else
    echo "   ✗ HDF5 not found at $HDF5_ROOT"
fi
echo ""

# Check MPI
echo "2. Checking MPI (MPICH 5.0.0)..."
if [ -f "$MPI_ROOT/bin/mpicxx" ]; then
    echo "   ✓ mpicxx found at $MPI_ROOT/bin/mpicxx"
    $MPI_ROOT/bin/mpicxx --version 2>/dev/null | head -1 || echo "   Version: 5.0.0"
else
    echo "   ✗ mpicxx not found at $MPI_ROOT/bin/"
fi
if [ -f "$MPI_ROOT/bin/mpicc" ]; then
    echo "   ✓ mpicc found at $MPI_ROOT/bin/mpicc"
else
    echo "   ✗ mpicc not found at $MPI_ROOT/bin/"
fi
echo ""

# Check Python
echo "3. Checking Python (3.10.6)..."
if [ -f "$PYTHON_ROOT/bin/python3" ]; then
    echo "   ✓ python3 found at $PYTHON_ROOT/bin/python3"
    $PYTHON_ROOT/bin/python3 --version 2>/dev/null || echo "   Version: 3.10.6"
else
    echo "   ✗ python3 not found at $PYTHON_ROOT/bin/"
fi
echo ""

# Check CMake
echo "4. Checking CMake (4.2.3)..."
if [ -f "$CMAKE_ROOT/bin/cmake" ]; then
    echo "   ✓ cmake found at $CMAKE_ROOT/bin/cmake"
    $CMAKE_ROOT/bin/cmake --version | head -1
else
    echo "   ✗ cmake not found at $CMAKE_ROOT/bin/"
fi
echo ""

# Check Henson
echo "5. Checking Henson..."
if [ -d "$HENSON_ROOT" ]; then
    echo "   ✓ Henson directory exists: $HENSON_ROOT"
    if [ -f "$HENSON_ROOT/lib/libhenson.a" ]; then
        echo "   ✓ libhenson.a found"
    else
        echo "   ✗ libhenson.a not found at $HENSON_ROOT/lib/"
    fi
    if [ -f "$HENSON_ROOT/lib/libhenson-pmpi-static.a" ] || [ -f "$HENSON_ROOT/lib/libhenson-pmpi.so" ]; then
        echo "   ✓ libhenson-pmpi library found"
    else
        echo "   ✗ libhenson-pmpi library not found"
    fi
else
    echo "   ✗ Henson not installed at $HENSON_ROOT"
    echo "   ℹ Run ./install_wilkins.sh to install"
fi

if [ -n "$HENSON" ]; then
    echo "   ✓ HENSON environment variable set: $HENSON"
else
    echo "   ℹ HENSON environment variable not set"
fi
echo ""

# Check LowFive
echo "6. Checking LowFive..."
if [ -d "$LOWFIVE_ROOT" ]; then
    echo "   ✓ LowFive directory exists: $LOWFIVE_ROOT"
    if [ -f "$LOWFIVE_ROOT/lib/liblowfive.dylib" ] || [ -f "$LOWFIVE_ROOT/lib/liblowfive.so" ]; then
        echo "   ✓ LowFive libraries found"
    else
        echo "   ✗ LowFive libraries not found at $LOWFIVE_ROOT/lib/"
    fi
else
    echo "   ✗ LowFive not installed at $LOWFIVE_ROOT"
    echo "   ℹ Run ./install_wilkins.sh to install"
fi
echo ""

# Check Wilkins
echo "7. Checking Wilkins..."
if [ -d "$WILKINS_ROOT" ]; then
    echo "   ✓ Wilkins directory exists: $WILKINS_ROOT"
    if [ -f "$WILKINS_ROOT/bin/wilkins-master.py" ]; then
        echo "   ✓ wilkins-master.py found"
    else
        echo "   ✗ wilkins-master.py not found at $WILKINS_ROOT/bin/"
    fi
else
    echo "   ✗ Wilkins not installed at $WILKINS_ROOT"
    echo "   ℹ Run ./install_wilkins.sh to install"
fi

if [ -n "$HDF5_VOL_CONNECTOR" ]; then
    echo "   ✓ HDF5_VOL_CONNECTOR set"
else
    echo "   ℹ HDF5_VOL_CONNECTOR not set (needed for runtime)"
fi

if [ -n "$HDF5_PLUGIN_PATH" ]; then
    echo "   ✓ HDF5_PLUGIN_PATH set: $HDF5_PLUGIN_PATH"
else
    echo "   ℹ HDF5_PLUGIN_PATH not set (needed for runtime)"
fi
echo ""

# Check ndarray configuration
echo "8. Checking ndarray configuration..."
if [ -f "build_wilkins/CMakeCache.txt" ]; then
    echo "   Build: build_wilkins"
    HENSON_ENABLED=$(grep "NDARRAY_USE_HENSON:STRING" build_wilkins/CMakeCache.txt 2>/dev/null | cut -d= -f2)
    HDF5_ENABLED=$(grep "NDARRAY_HAVE_HDF5" build_wilkins/CMakeCache.txt 2>/dev/null | cut -d= -f2 | head -1)
    MPI_ENABLED=$(grep "NDARRAY_HAVE_MPI" build_wilkins/CMakeCache.txt 2>/dev/null | cut -d= -f2 | head -1)

    echo "   HDF5: ${HDF5_ENABLED:-unknown}"
    echo "   Henson: ${HENSON_ENABLED:-unknown}"
    echo "   MPI: ${MPI_ENABLED:-unknown}"
else
    echo "   ✗ build_wilkins not configured yet"
    echo "   ℹ Run ./build_with_wilkins.sh after installing dependencies"
fi
echo ""

# Summary
echo "========================================="
echo "Summary"
echo "========================================="
echo ""

# Count what's installed
INSTALLED=0
NEEDED=3

[ -f "$HENSON_ROOT/lib/libhenson.a" ] && ((INSTALLED++))
[ -f "$LOWFIVE_ROOT/lib/liblowfive.dylib" ] || [ -f "$LOWFIVE_ROOT/lib/liblowfive.so" ] && ((INSTALLED++))
[ -f "$WILKINS_ROOT/bin/wilkins-master.py" ] && ((INSTALLED++))

if [ $INSTALLED -eq 0 ]; then
    echo "To install all Wilkins dependencies:"
    echo "  ./install_wilkins.sh"
    echo ""
elif [ $INSTALLED -lt $NEEDED ]; then
    echo "Some dependencies are missing. Run:"
    echo "  ./install_wilkins.sh"
    echo ""
else
    echo "✓ All Wilkins dependencies installed!"
    echo ""
    if [ ! -f "build_wilkins/CMakeCache.txt" ]; then
        echo "Next step - build ndarray with Wilkins support:"
        echo "  ./build_with_wilkins.sh"
    else
        echo "✓ ndarray built with Wilkins support in build_wilkins/"
    fi
    echo ""
fi

echo "For more details, see: docs/WILKINS_SETUP.md"
echo ""
