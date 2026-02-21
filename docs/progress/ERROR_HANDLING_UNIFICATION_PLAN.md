# Error Handling Unification Plan

**Date**: 2026-02-20
**Status**: IN PROGRESS (Phase 1 Complete, Phase 5 Complete)
**Priority**: HIGH (Critical API consistency issue)

## Problem Statement

The library has **inconsistent error handling** across different I/O backends:

```cpp
// NetCDF - throws exceptions ✅ Good
if (status != NC_NOERR) throw netcdf_error(...);

// HDF5 - returns bool ❌ Inconsistent
if (dtype < 0) { warn(...); return false; }

// VTK - calls fatal() ❌ Kills process!
if (!reader) fatal(ERR_VTK_FILE_CANNOT_OPEN);

// ADIOS2 - returns bool ❌ Inconsistent
if (empty()) { warn(...); return false; }
```

**Impact**: Users cannot write safe error handling code because behavior is unpredictable.

## Survey Results

From `include/ndarray/ndarray.hh`:
- **49 fatal() calls** - Process termination (unrecoverable)
- **14 warn() calls** - Non-fatal warnings
- **19 return false** - Silent failure (requires checking)

### Categories of Errors

1. **I/O Errors** (file not found, read/write failure)
   - Current: Mixed (bool, fatal, exceptions)
   - Target: Throw `std::runtime_error` or custom exception

2. **Invalid Arguments** (bad dimensions, null pointers)
   - Current: fatal() kills process
   - Target: Throw `std::invalid_argument`

3. **Logic Errors** (array not distributed, wrong mode)
   - Current: fatal() or return false
   - Target: Throw `std::logic_error`

4. **Unsupported Operations** (feature not compiled in)
   - Current: fatal()
   - Target: Throw `std::runtime_error` with clear message

## Target Design

### Exception Hierarchy

```cpp
// Existing in include/ndarray/error.hh:
std::runtime_error
  └─ file_error
       ├─ netcdf_error
       ├─ hdf5_error (NEW)
       ├─ adios_error (NEW)
       └─ vtk_error (NEW)

std::invalid_argument (use directly)
std::logic_error (use directly)
std::out_of_range (use directly)
```

### API Changes

**Function signatures remain the same**, but:
- `bool read_h5_did()` → `void read_h5_did()` (throws on error)
- `bool to_h5()` → `void to_h5()` (throws on error)
- Remove all `return false` from I/O functions
- Replace `fatal()` with appropriate exceptions

## Implementation Plan

### Phase 1: HDF5 Backend ✅ COMPLETED (2026-02-20)

**Files Modified**:
- `include/ndarray/ndarray.hh` - Function implementations
- `include/ndarray/ndarray_base.hh` - Function declaration and call site

**Functions Fixed** (2 functions):
1. ✅ `read_h5_did()` - Line 1789
   - Changed signature: `bool read_h5_did(hid_t did)` → `void read_h5_did(hid_t did)`
   - Replaced `fatal("unsupported h5 extent type")` with `throw hdf5_error(ERR_HDF5_IO, "Unsupported HDF5 extent type")`
   - Updated declaration in ndarray.hh line 415
   - Updated virtual declaration in ndarray_base.hh line 340

2. ✅ `to_h5()` - Line 1818
   - Changed signature: `bool to_h5(...)` → `void to_h5(...)`
   - Replaced `warn(ERR_HDF5_UNSUPPORTED_TYPE); return false;` with `throw hdf5_error(ERR_HDF5_UNSUPPORTED_TYPE, "Unsupported data type for HDF5 output")`
   - Replaced `if (file_id < 0) return false;` with exception throw
   - Replaced `return dataset_id >= 0;` with exception throw on error
   - Updated declaration in ndarray.hh line 416

**Call Sites Updated**:
- `ndarray_base::read_h5(hid_t fid, const std::string& name)` - Line 598-606
  - Removed `bool succ = read_h5_did(did); return succ;`
  - Now calls `read_h5_did(did);` directly and throws on errors
  - Maintains bool return type for backward compatibility (always returns true or throws)

**Verification**:
- ✅ Compiled successfully in build_phdf5
- ✅ All distributed tests passed (27/27 in build_mpi)
- ✅ Parallel HDF5 tests passed (test_hdf5_auto: distributed read/write working)
- ✅ No test breakage - existing tests don't check return values

### Phase 2: VTK Backend (1 day)

**Files**: `include/ndarray/ndarray.hh`, `include/ndarray/ndarray_vtk.hh`

**Functions to fix** (~10 functions):
- All VTK read/write functions
- Replace `fatal(ERR_VTK_*)` with `throw vtk_error(...)`

**Example**:
```cpp
// BEFORE:
if (!reader) fatal(ERR_VTK_FILE_CANNOT_OPEN);

// AFTER:
if (!reader) throw vtk_error("Cannot open VTK file: " + filename);
```

### Phase 3: ADIOS2 Backend (1 day)

**Files**: `include/ndarray/ndarray.hh`

**Functions to fix**:
- `read_bp()` variants
- `write_bp()` variants
- Replace `return false` with exceptions

### Phase 4: Core Array Functions (1 day)

**Files**: `include/ndarray/ndarray_base.hh`, `include/ndarray/ndarray.hh`

**Functions to fix**:
- `reshape()` - Line 1244-1246
- `add()` - Line 1290
- Other fatal() calls for invalid arguments

**Changes**:
```cpp
// BEFORE:
if (dims.size() == 0) fatal(ERR_NDARRAY_RESHAPE_EMPTY);

// AFTER:
if (dims.size() == 0) throw std::invalid_argument("Cannot reshape to empty dimensions");
```

### Phase 5: Add Custom Exception Types ✅ COMPLETED (2026-02-20)

**File Modified**: `include/ndarray/error.hh`

**Changes Made**:
1. ✅ Added `hdf5_error` exception class (lines 290-296)
   ```cpp
   class hdf5_error : public file_error {
   public:
     explicit hdf5_error(int code, const std::string& msg = "")
       : file_error(code, msg) {}
     explicit hdf5_error(const std::string& msg)
       : file_error(msg) {}
   };
   ```

2. ✅ Updated `fatal()` function to throw `hdf5_error` for HDF5 error codes (lines 375-381)
   - Added range check: `if (err >= ERR_HDF5_NOT_PARALLEL && err <= ERR_HDF5_UNSUPPORTED_TYPE)`
   - Routes HDF5 errors to `hdf5_error` instead of generic `netcdf_error`

**Note**: `vtk_error` and `adios2_error` already existed in the codebase.

### Phase 6: Update Tests (1-2 days)

**Files**: All test files

**Changes**:
- Update tests that check return values
- Add try-catch blocks
- Verify error messages

**Example**:
```cpp
// BEFORE:
if (!array.read_h5(file, var)) {
  std::cerr << "Read failed" << std::endl;
}

// AFTER:
try {
  array.read_h5(file, var);
} catch (const hdf5_error& e) {
  std::cerr << "Read failed: " << e.what() << std::endl;
}
```

### Phase 7: Documentation (2-3 hours)

**Files**:
- `docs/EXCEPTION_HANDLING.md` (update)
- `README.md` (add section)
- Doxygen comments in headers

**Add**:
- List of all exception types
- When each is thrown
- Examples of proper error handling

## Breaking Changes

### API Changes

**Breaking**:
1. ❌ `bool read_h5_did()` → `void read_h5_did()`
2. ❌ `bool to_h5()` → `void to_h5()`
3. ❌ Functions that returned bool now throw

**Non-breaking**:
- All `*_auto()` functions already throw exceptions
- NetCDF functions already throw exceptions
- Serialization functions keep current behavior

### Migration Guide

```cpp
// OLD CODE (v1.x):
if (!array.read_h5("data.h5", "var")) {
  handle_error();
}

// NEW CODE (v2.0):
try {
  array.read_h5("data.h5", "var");
} catch (const hdf5_error& e) {
  handle_error(e.what());
}
```

## Rollout Strategy

### Option A: Immediate (Recommended)

**Pros**: Clean API, no technical debt
**Cons**: Breaking change

**Plan**:
1. Implement all changes
2. Update all tests
3. Bump version to v2.0
4. Document migration path

### Option B: Gradual (Safer)

**Pros**: Backward compatible
**Cons**: Maintains inconsistency

**Plan**:
1. Add new throwing versions with different names
2. Deprecate old versions
3. Remove in v3.0

**Example**:
```cpp
[[deprecated]] bool read_h5(...);  // Old, returns bool
void read_h5_or_throw(...);        // New, throws
```

## Validation Checklist

### Phase 1 - HDF5 Backend
- [x] HDF5 `return false` replaced with exceptions (2 functions)
- [x] HDF5 `fatal()` replaced with exceptions (1 call)
- [x] HDF5 `warn()` replaced with exceptions (1 call)
- [x] `hdf5_error` exception type added
- [x] Existing tests verified (no breakage)
- [x] Build verified in multiple configurations

### Remaining Work
- [ ] VTK `fatal()` replaced (~10 functions)
- [ ] ADIOS2 `return false` replaced with exceptions
- [ ] Core array `fatal()` replaced (reshape, add, etc.)
- [ ] Tests updated for new error handling patterns
- [ ] Documentation updated
- [ ] CI passes with new tests
- [ ] Migration guide written

## Timeline

**Total**: 5-7 days

- Day 1: HDF5 backend + exception types
- Day 2: VTK backend
- Day 3: ADIOS2 backend + core functions
- Day 4-5: Update all tests
- Day 6: Documentation
- Day 7: Review and polish

## Risks

1. **Test breakage**: Many tests may need updates
   - Mitigation: Update tests incrementally per backend

2. **User code breakage**: Existing users will need to adapt
   - Mitigation: Clear migration guide, version bump to v2.0

3. **Incomplete conversion**: Miss some error paths
   - Mitigation: Comprehensive grep for all patterns

## Success Criteria

✅ **Zero** fatal() calls in I/O code
✅ **Zero** bool returns from I/O functions
✅ **All** errors throw typed exceptions
✅ **All** tests pass
✅ **Clear** documentation on error handling

## Progress Summary

### Completed
1. ✅ Review plan with maintainer
2. ✅ Implement Phase 1 (HDF5 backend - `read_h5_did()` and `to_h5()`)
3. ✅ Implement Phase 5 (Add `hdf5_error` exception type)
4. ✅ Verify all existing tests still pass
5. ✅ Build verification in multiple configurations (build_mpi, build_phdf5)

### Next Steps
1. ⏸️ Implement Phase 2 (VTK backend - replace `fatal()` with `vtk_error`)
2. ⏸️ Implement Phase 3 (ADIOS2 backend - replace `return false` with exceptions)
3. ⏸️ Implement Phase 4 (Core functions - reshape, add, etc.)
4. ⏸️ Update Phase 6 (Update tests that check return values)
5. ⏸️ Update Phase 7 (Documentation)
6. ⏸️ Create feature branch and commit changes
7. ⏸️ Final review and merge

---

**Author**: Claude Sonnet 4.5
**Reviewed**: [Pending]
**Approved**: [Pending]
