# Contributing to ndarray

Thank you for your interest in contributing to ndarray! This document provides guidelines and instructions for contributing to the project.

> **Note on AI-Assisted Development**: This project includes significant portions of code, documentation, examples, and tests that were generated or enhanced with AI assistance (starting 2026). Contributors should be aware that while the AI-generated code is functional, it should be reviewed carefully. We welcome contributions that improve, refactor, or validate the existing codebase.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [How to Contribute](#how-to-contribute)
- [Coding Standards](#coding-standards)
- [Testing Guidelines](#testing-guidelines)
- [Documentation](#documentation)
- [Pull Request Process](#pull-request-process)
- [Release Process](#release-process)

## Code of Conduct

This project follows a simple code of conduct:
- Be respectful and constructive in all interactions
- Focus on what is best for the community
- Show empathy towards other community members

## Getting Started

1. Fork the repository on GitHub
2. Clone your fork locally:
   ```bash
   git clone https://github.com/YOUR_USERNAME/ndarray.git
   cd ndarray
   ```
3. Add the upstream repository:
   ```bash
   git remote add upstream https://github.com/hguo/ndarray.git
   ```
4. Create a branch for your work:
   ```bash
   git checkout -b feature/your-feature-name
   ```

## Development Setup

### Prerequisites

- C++17 compatible compiler (GCC 7+, Clang 5+, MSVC 2017+)
- CMake 3.10 or higher
- yaml-cpp
- Optional: NetCDF, HDF5, ADIOS2, MPI, VTK (depending on features you're working on)

### Building the Project

```bash
mkdir build && cd build
cmake .. -DNDARRAY_BUILD_TESTS=ON
make
```

### Running Tests

```bash
cd build
ctest --output-on-failure
```

## How to Contribute

### Reporting Bugs

When reporting bugs, please include:
- A clear, descriptive title
- Detailed steps to reproduce the issue
- Expected behavior vs actual behavior
- Your environment (OS, compiler version, CMake version)
- Minimal code example demonstrating the issue
- Any relevant error messages or stack traces

### Suggesting Enhancements

Enhancement suggestions are tracked as GitHub issues. When creating an enhancement suggestion:
- Use a clear, descriptive title
- Provide a detailed description of the proposed feature
- Explain why this enhancement would be useful
- Include code examples if applicable

### Contributing Code

We welcome code contributions! Here are the types of contributions we're looking for:

1. **Bug fixes** - Fix reported or discovered bugs
2. **New features** - Implement new functionality
3. **Performance improvements** - Optimize existing code
4. **Documentation** - Improve or add documentation
5. **Tests** - Add or improve test coverage
6. **Examples** - Create new example programs

## Coding Standards

### C++ Style Guidelines

- Follow C++17 standard practices
- Use meaningful variable and function names
- Keep functions focused and concise
- Add comments for complex logic

### Header Files

- Use include guards or `#pragma once`
- Keep headers self-contained (include all dependencies)
- Document public APIs with comments
- Use templates appropriately for generic programming

### Example Code Style

```cpp
namespace ftk {

template <typename T>
class ndarray {
public:
  // Constructor with clear documentation
  ndarray(const std::vector<size_t>& dims) {
    reshapef(dims);
  }

  // Public methods with descriptive names
  void reshapef(const std::vector<size_t>& dims_);

  // Accessor methods
  size_t size() const { return p.size(); }
  const T* data() const { return p.data(); }

private:
  std::vector<T> p;
  std::vector<size_t> dims;
};

} // namespace ftk
```

### CMake Style

- Use lowercase for function names
- Use clear variable names with NDARRAY_ prefix
- Comment complex CMake logic
- Follow existing patterns in CMakeLists.txt

## Testing Guidelines

### Writing Tests

Tests should be:
- **Isolated** - Each test should be independent
- **Repeatable** - Same input should produce same output
- **Fast** - Tests should run quickly
- **Clear** - Test names should describe what they test

### Test Structure

Create test files in the `tests/` directory:

```cpp
#include <ndarray/ndarray.hh>
#include <catch2/catch.hpp>

TEST_CASE("ndarray basic operations", "[ndarray][core]") {
  ftk::ndarray<double> arr;

  SECTION("reshaping") {
    arr.reshapef(10, 20, 30);
    REQUIRE(arr.nd() == 3);
    REQUIRE(arr.size() == 6000);
  }

  SECTION("filling") {
    arr.reshapef(100);
    arr.fill(3.14);
    REQUIRE(arr[0] == 3.14);
  }
}
```

### Running Specific Tests

```bash
ctest -R ndarray_core  # Run specific test
ctest -V                # Verbose output
```

## Documentation

### Code Documentation

- Document all public APIs
- Use clear, concise comments
- Include usage examples in documentation
- Document parameters and return values

### README and Guides

- Keep documentation up-to-date with code changes
- Use clear, simple language
- Include working code examples
- Update CHANGELOG.md for notable changes

## Pull Request Process

1. **Before submitting**:
   - Ensure your code follows the coding standards
   - Add or update tests as needed
   - Update documentation if applicable
   - Run all tests and ensure they pass
   - Update CHANGELOG.md under [Unreleased] section

2. **Creating the PR**:
   - Use a clear, descriptive title
   - Reference any related issues
   - Describe what changes you made and why
   - Include any relevant context or screenshots

3. **PR Template**:
   ```markdown
   ## Description
   Brief description of changes

   ## Type of Change
   - [ ] Bug fix
   - [ ] New feature
   - [ ] Performance improvement
   - [ ] Documentation update
   - [ ] Other (describe)

   ## Testing
   - [ ] All existing tests pass
   - [ ] New tests added (if applicable)
   - [ ] Manual testing performed

   ## Related Issues
   Fixes #123
   ```

4. **Review process**:
   - Respond to reviewer comments
   - Make requested changes
   - Keep the PR focused and atomic
   - Squash commits if requested

5. **After merge**:
   - Delete your feature branch
   - Sync your fork with upstream

## Release Process

Releases follow [Semantic Versioning](https://semver.org/):

- **MAJOR** (1.0.0): Incompatible API changes
- **MINOR** (0.1.0): New functionality, backward compatible
- **PATCH** (0.0.1): Bug fixes, backward compatible

### Creating a Release

1. Update version in `CMakeLists.txt`
2. Update CHANGELOG.md with release date
3. Create a git tag:
   ```bash
   git tag -a v1.0.0 -m "Release version 1.0.0"
   git push origin v1.0.0
   ```
4. Create GitHub release from tag

## Questions or Need Help?

- **Questions**: Open a [GitHub Discussion](https://github.com/hguo/ndarray/discussions)
- **Bugs**: Create an [Issue](https://github.com/hguo/ndarray/issues)
- **Urgent issues**: Contact the maintainers

## License

By contributing to ndarray, you agree that your contributions will be licensed under the MIT License.

---

Thank you for contributing to ndarray! Your efforts help make scientific computing in C++ better for everyone.
