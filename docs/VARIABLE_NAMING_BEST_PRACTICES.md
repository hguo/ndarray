# Variable Naming Best Practices

## Three-Layer Strategy

ndarray provides three mechanisms to handle variable name variations, used in this order:

```yaml
vars:
  - name: temperature
    possible_names:    # Layer 1: Exact matching (fast, explicit)
      - temperature
      - temp
    name_patterns:     # Layer 2: Pattern matching (flexible)
      - "time*_avg_temperature"
      - "*temperature*"
    # Layer 3: Smart suggestions (automatic, debug only)
    # Activated only on failure, provides helpful error messages
```

### Execution Order

```
1. Try each name in possible_names (O(n) lookup)
   ✓ Fast
   ✓ Explicit
   ✓ User controls exact matching

2. If not found, try name_patterns (O(n*m) pattern matching)
   ✓ Flexible
   ✓ Handles future naming schemes
   ⚠ Slower than exact match

3. If still not found AND required:
   ✓ List available variables
   ✓ Suggest similar names
   ✓ Help user fix YAML config
```

## When to Use Each Mechanism

### Use `possible_names` When:

✅ **You know the exact variable names**
```yaml
# MPAS monthly vs yearly output
vars:
  - name: normalVelocity
    possible_names:
      - normalVelocity
      - timeMonthly_avg_normalVelocity
      - timeYearly_avg_normalVelocity
```

✅ **You want strict control (avoid accidental matches)**
```yaml
# Only match these exact names, nothing else
vars:
  - name: temperature
    possible_names:
      - temperature
      - temperatureNew
    # Won't match temperatureOld (deprecated)
    # Won't match temperatureAnomaly (different variable)
```

✅ **Performance matters (reading many timesteps)**
```yaml
# Exact match is O(1) NetCDF lookup
# Pattern match requires listing all variables first
vars:
  - name: velocity
    possible_names:
      - normalVelocity  # Fast!
```

✅ **Documentation purpose**
```yaml
# YAML is self-documenting
vars:
  - name: ssh
    possible_names:
      - ssh                    # Standard MPAS output
      - sea_surface_height     # Analysis output
      - surfaceElevation       # Legacy format
    # Anyone reading this knows what data sources are supported
```

### Use `name_patterns` When:

✅ **Future-proofing against new naming schemes**
```yaml
vars:
  - name: temperature
    possible_names:
      - temperature  # Known variants
    name_patterns:
      - "time*_avg_temperature"  # Catches timeDaily/Weekly/Monthly/Yearly
```

✅ **Working with unknown data sources**
```yaml
# You receive MPAS data but don't know which variant
vars:
  - name: salinity
    name_patterns:
      - "salinity"          # Exact
      - "*salinity*"        # Any variant
```

✅ **Handling systematic naming conventions**
```yaml
# CESM adds prefixes systematically
vars:
  - name: temperature
    name_patterns:
      - "*TEMP*"    # Matches TEMP, POT_TEMP, TEMP_CUR
```

⚠️ **Caution: Patterns can match unintended variables**
```yaml
# BAD: Too broad
name_patterns:
  - "*temp*"  # Matches: temperature, temperatureAnomaly, temperatureGradient, tempData, etc.

# GOOD: More specific
name_patterns:
  - "*temperature"  # Matches: temperature, time*_temperature, but not temperatureGradient
```

### Rely on Smart Suggestions When:

✅ **Debugging YAML configs**
- Error message tells you what's available
- Suggests closest matches
- You update `possible_names` based on suggestion

✅ **Discovering new data formats**
- First run shows all available variables
- You add them to `possible_names`

❌ **Don't rely on for production**
- Smart suggestions are for debugging only
- Always use explicit `possible_names` or `name_patterns`

## Recommended Patterns

### Pattern 1: Strict Matching (Most Common)

```yaml
# For well-known datasets where names are documented
vars:
  - name: normalVelocity
    possible_names:
      - normalVelocity
      - timeMonthly_avg_normalVelocity
      - timeYearly_avg_normalVelocity
    # No patterns = no surprises
```

**Pros:**
- Fast
- Explicit
- No accidental matches
- Self-documenting

**Cons:**
- Must update YAML when new naming schemes appear

### Pattern 2: Hybrid (Recommended for MPAS)

```yaml
# Try known names first, fall back to patterns
vars:
  - name: temperature
    possible_names:
      - temperature              # Standard (try first)
      - temp                     # Abbreviation
      - timeMonthly_avg_temperature  # Common variant
    name_patterns:
      - "time*_avg_temperature"  # Catch new time averages
      - "*temperature"           # Last resort suffix match
```

**Pros:**
- Fast path for known names
- Flexible for new variants
- Future-proof

**Cons:**
- Slightly more complex config
- Pattern matching slower (only on fallback)

### Pattern 3: Pattern-First (For Unknown Data)

```yaml
# When exploring new datasets
vars:
  - name: velocity
    name_patterns:
      - "velocity"
      - "*velocity*"
      - "*vel*"
    optional: true  # Don't fail if not found
```

**Pros:**
- Works with unknown data sources
- Good for exploratory analysis

**Cons:**
- May match unintended variables
- Slower
- Should convert to explicit `possible_names` after discovery

### Pattern 4: Strict + Smart Suggestions

```yaml
# For production: strict matching + helpful errors
vars:
  - name: temperature
    possible_names:
      - temperature
    # No patterns = if not found, smart suggestion will help
```

**On success:**
- Fast exact match

**On failure:**
```
Variable not found.
  Tried: 'temperature'
  Available: 'timeMonthly_avg_temperature', ...
  Did you mean: 'timeMonthly_avg_temperature'?
  Add it to possible_names in your YAML.
```

## Anti-Patterns (Don't Do This)

### ❌ Anti-Pattern 1: Too Broad Patterns

```yaml
# BAD: Matches everything
vars:
  - name: temp
    name_patterns:
      - "*temp*"  # Matches: temp, temperature, temperatureAnomaly, tempData, attempt, etc.
```

**Fix:**
```yaml
# GOOD: Specific patterns
vars:
  - name: temperature
    name_patterns:
      - "*temperature"  # Only suffix matches
```

### ❌ Anti-Pattern 2: Patterns Before Exact Names

```yaml
# BAD: Slow!
vars:
  - name: velocity
    name_patterns:
      - "*velocity*"  # Tried first (slow)
    possible_names:
      - normalVelocity  # Should be tried first!
```

**Fix:** Order doesn't matter in YAML, code always tries `possible_names` first.

### ❌ Anti-Pattern 3: No Explicit Names for Common Cases

```yaml
# BAD: Always uses patterns (slow)
vars:
  - name: temperature
    name_patterns:
      - "temperature"
      - "*temperature*"
```

**Fix:**
```yaml
# GOOD: Exact name as possible_names (fast path)
vars:
  - name: temperature
    possible_names:
      - temperature  # Fast O(1) lookup
    name_patterns:
      - "*temperature*"  # Fallback
```

### ❌ Anti-Pattern 4: Relying on Smart Matching Only

```yaml
# BAD: No possible_names, no patterns
vars:
  - name: temperature  # Hopes smart matching will find it
```

**Problem:**
- Smart matching is for error messages only
- Won't actually read the variable
- Will fail

**Fix:**
```yaml
# GOOD: Always specify possible_names or patterns
vars:
  - name: temperature
    possible_names:
      - temperature
```

## Real-World Examples

### MPAS-Ocean Production Config

```yaml
stream:
  name: mpas_production
  substreams:
    - name: data
      format: netcdf
      filenames: "output.*.nc"
      vars:
        # High-priority variables: strict matching
        - name: normalVelocity
          possible_names:
            - normalVelocity
            - timeMonthly_avg_normalVelocity
          # No patterns: we control the data, know exact names

        # Research variables: flexible
        - name: temperature
          possible_names:
            - temperature
            - temperatureNew
          name_patterns:
            - "time*_avg_temperature"  # For future time aggregations
          optional: false

        # Optional analysis variables: very flexible
        - name: eddyKineticEnergy
          name_patterns:
            - "*eddyKineticEnergy*"
            - "*EKE*"
          optional: true  # May not exist in all runs
```

### Multi-Model Support

```yaml
# Supporting both MPAS and CESM/POP data
vars:
  - name: temperature
    possible_names:
      # MPAS names
      - temperature
      - timeMonthly_avg_temperature
      # CESM/POP names
      - TEMP
      - TEMP_CUR
      - POT_TEMP
    name_patterns:
      - "*temperature*"  # Catch any other variant
      - "*TEMP*"         # CESM-style
```

## Migration Strategy

### Phase 1: Discovery (Initial Setup)

```yaml
# Start with broad patterns to discover what's in files
vars:
  - name: velocity
    name_patterns:
      - "*velocity*"
```

### Phase 2: Document (After Testing)

```yaml
# Replace patterns with explicit names found during testing
vars:
  - name: velocity
    possible_names:
      - normalVelocity                # Found in standard output
      - timeMonthly_avg_normalVelocity # Found in monthly files
    name_patterns:
      - "time*_avg_normalVelocity"  # Keep as safety net
```

### Phase 3: Production (Optimize)

```yaml
# Remove patterns if all data sources are known
vars:
  - name: velocity
    possible_names:
      - normalVelocity
      - timeMonthly_avg_normalVelocity
      - timeYearly_avg_normalVelocity
    # No patterns = fastest, most reliable
```

## Performance Considerations

### Exact Match (possible_names)
```
Cost: O(n) where n = number of names in possible_names
Typical: ~3-5 names = ~3-5 NetCDF lookups
Fast: Yes
```

### Pattern Match (name_patterns)
```
Cost: O(n*m*k) where:
  n = number of patterns
  m = number of variables in file
  k = pattern matching cost
Typical: 2 patterns × 100 variables × 10 chars = 2000 operations
Slow: Relatively
```

### Impact on Timestep Loop

```python
# Reading 1000 timesteps with patterns
for t in range(1000):
    g = stream.read(t)  # Each read does pattern matching
    # 1000 × pattern_match_cost = significant overhead
```

**Recommendation:** Use `possible_names` for variables read in hot loops.

## Summary

| Mechanism | When to Use | Performance | Flexibility | Reliability |
|-----------|-------------|-------------|-------------|-------------|
| `possible_names` | Known variants | ⚡ Fast | Medium | ⭐ High |
| `name_patterns` | Unknown variants | 🐢 Slower | ⭐ High | Medium |
| Smart suggestions | Debugging | N/A | N/A | Helps fix config |

**Default Strategy:**
1. List known names in `possible_names` (fast, explicit)
2. Add `name_patterns` for future-proofing (fallback)
3. Use smart suggestions to discover new names → add to YAML

**Golden Rule:**
> `possible_names` defines intent, `name_patterns` provides flexibility, smart suggestions help debugging.

All three work together, not against each other.
