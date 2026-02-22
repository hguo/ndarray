# Scientific Data Variable Naming Conventions

## The Problem

Scientific datasets often use inconsistent variable names across different output types, formats, processing stages, and modeling systems. This creates significant challenges for analysis tools that need to work across diverse data sources.

### Common Naming Inconsistencies

**Example 1: MPAS-Ocean temporal variations**
```bash
# Instantaneous output
normalVelocity
vertVelocityTop
temperature

# Monthly average output
timeMonthly_avg_normalVelocity
timeMonthly_avg_vertVelocityTop
timeMonthly_avg_temperature

# Yearly average output
timeYearly_avg_normalVelocity
...
```

**Example 2: Multi-model comparisons**
- Climate Model A: `temp`, `u_wind`, `v_wind`
- Climate Model B: `temperature`, `u_velocity`, `v_velocity`
- Observations: `T`, `U`, `V`

**Example 3: Processing pipeline variations**
- Raw data: `SST_raw`
- Quality controlled: `SST_qc`
- Interpolated: `SST_interp`
- Final product: `SST`

**Impact:**
- Code breaks when switching between data sources
- Need different configurations for each naming convention
- Manual maintenance of variable name lists
- Hard to write generic analysis tools
- Limits code reusability across projects

## Current Solution (Partial)

ndarray supports `possible_names` in YAML:

```yaml
vars:
  - name: normalVelocity
    possible_names:
      - normalVelocity
      - timeMonthly_avg_normalVelocity
      - timeYearly_avg_normalVelocity
```

**Problems with current approach:**
1. ❌ Must list ALL possible names upfront
2. ❌ No runtime detection
3. ❌ Breaks when new naming conventions appear
4. ❌ Duplicates configuration across projects
5. ❌ No fuzzy matching (typos, case sensitivity)

## Better Solutions

### Solution 1: Regex Pattern Matching

```yaml
vars:
  - name: normalVelocity
    name_patterns:
      - "normalVelocity"
      - "time.*_avg_normalVelocity"  # Matches any time average
      - ".*[Nn]ormal[Vv]elocity.*"   # Fuzzy match
```

**Implementation:**
```cpp
// In variable struct
std::vector<std::string> name_patterns;  // Regex patterns
std::regex compiled_pattern;

// In read function
for (const auto& pattern : var.name_patterns) {
    std::regex re(pattern);
    for (const auto& nc_var : available_vars) {
        if (std::regex_match(nc_var, re)) {
            actual_varname = nc_var;
            break;
        }
    }
}
```

### Solution 2: Variable Alias Database

Create a standard alias file shared across projects:

```yaml
# mpas_ocean_aliases.yaml
aliases:
  normalVelocity:
    - normalVelocity
    - timeMonthly_avg_normalVelocity
    - timeYearly_avg_normalVelocity
    - timeDaily_avg_normalVelocity

  temperature:
    - temperature
    - timeMonthly_avg_temperature
    - temperatureNew
    - temp  # Common abbreviation

  salinity:
    - salinity
    - timeMonthly_avg_salinity
    - salt
```

**Usage:**
```yaml
stream:
  alias_file: mpas_ocean_aliases.yaml  # Load once
  substreams:
    - format: netcdf
      vars:
        - name: normalVelocity  # Automatically resolves aliases
```

### Solution 3: Smart Auto-Detection

```cpp
// Automatic variable name resolution with scoring

struct VariableNameMatcher {
    // Score how well a NetCDF variable matches the requested name
    static int score_match(const std::string& requested,
                          const std::string& available) {
        int score = 0;

        // Exact match
        if (requested == available) return 1000;

        // Case-insensitive match
        if (strcasecmp(requested, available) == 0) return 900;

        // Contains requested name
        if (available.find(requested) != std::string::npos) return 800;

        // Remove common prefixes/suffixes and match
        auto cleaned = remove_time_prefixes(available);
        if (cleaned == requested) return 700;

        // Levenshtein distance (typo tolerance)
        int distance = levenshtein(requested, available);
        if (distance <= 2) return 500 - distance * 100;

        return 0;
    }

    static std::string find_best_match(
        const std::string& requested,
        const std::vector<std::string>& available) {

        int best_score = 0;
        std::string best_match;

        for (const auto& var : available) {
            int s = score_match(requested, var);
            if (s > best_score) {
                best_score = s;
                best_match = var;
            }
        }

        if (best_score > 500) {  // Confidence threshold
            return best_match;
        }

        throw std::runtime_error(
            "Cannot find variable '" + requested + "'. " +
            "Available: " + join(available, ", ")
        );
    }
};
```

### Solution 4: Variable Introspection API

```cpp
// Query what variables are actually available at runtime

class StreamIntrospector {
public:
    // List all available variables in the dataset
    std::vector<std::string> list_variables() const;

    // Find variables matching a pattern
    std::vector<std::string> find_variables(const std::string& pattern) const;

    // Get variable metadata
    VariableInfo get_info(const std::string& name) const;

    // Suggest corrections for typos
    std::vector<std::string> suggest_names(const std::string& name) const;
};

// Usage
ftk::stream s;
s.parse_yaml("mpas.yaml");

// Discover what's actually in the file
auto vars = s.list_variables();
std::cout << "Available variables: ";
for (const auto& v : vars) {
    std::cout << v << ", ";
}

// Find time-averaged velocity variables
auto velocity_vars = s.find_variables(".*[Vv]elocity.*");

// Get suggestions
try {
    auto data = s.read_variable("normalVeolcity");  // Typo!
} catch (const std::exception& e) {
    auto suggestions = s.suggest_names("normalVeolcity");
    std::cerr << "Did you mean: " << suggestions[0] << "?" << std::endl;
}
```

## Comparison: Other Libraries

### xarray (Python)
```python
import xarray as xr

ds = xr.open_dataset('mpas_output.nc')

# List all variables
print(ds.data_vars)

# Fuzzy search
velocity_vars = [v for v in ds.data_vars if 'velocity' in v.lower()]

# Access with any name
temp = ds['temperature'] if 'temperature' in ds else ds['temp']
```

### NCO (NetCDF Operators)
```bash
# List variables
ncks -m output.nc | grep "name ="

# Rename variables in-place
ncrename -v timeMonthly_avg_temperature,temperature output.nc

# Extract variables with pattern
ncks -v ".*velocity.*" output.nc subset.nc
```

## Recommended Implementation Plan

### Phase 1: Improve Error Messages (Easy, High Impact)

```cpp
// Current: Cryptic error
// [NDARRAY FATAL] NetCDF variable not found

// Better: Helpful error with suggestions
void variable::read_netcdf(...) {
    bool found = false;
    for (const auto& varname : possible_names) {
        if (nc_inq_varid(ncid, varname.c_str(), &varid) == NC_NOERR) {
            found = true;
            break;
        }
    }

    if (!found) {
        // List what's actually available
        int nvars;
        nc_inq_nvars(ncid, &nvars);
        std::vector<std::string> available;
        for (int i = 0; i < nvars; i++) {
            char name[NC_MAX_NAME+1];
            nc_inq_varname(ncid, i, name);
            available.push_back(name);
        }

        std::stringstream ss;
        ss << "Variable not found. Tried: ";
        for (const auto& n : possible_names) ss << n << ", ";
        ss << "\nAvailable variables: ";
        for (const auto& n : available) ss << n << ", ";

        // Suggest close matches
        auto suggestions = find_similar(possible_names[0], available);
        if (!suggestions.empty()) {
            ss << "\nDid you mean: " << suggestions[0] << "?";
        }

        throw std::runtime_error(ss.str());
    }
}
```

### Phase 2: Add Regex Support (Medium Effort)

```yaml
vars:
  - name: normalVelocity
    name_pattern: "time.*_avg_normalVelocity|normalVelocity"
```

### Phase 3: Variable Introspection (Medium Effort)

```cpp
// Add to stream class
std::vector<std::string> stream::list_variables(int timestep) const;
VariableInfo stream::inspect_variable(const std::string& name) const;
```

### Phase 4: Alias Database (Low Priority)

Community-maintained alias file for common datasets.

## Workaround for Current Code

For immediate use with MOPS:

```cpp
// Helper function to try multiple names
template<typename T>
bool try_read_variable(
    ftk::ndarray_group* g,
    const std::vector<std::string>& possible_names,
    ftk::ndarray<T>& output)
{
    for (const auto& name : possible_names) {
        if (g->has(name)) {
            output = g->get_ref<T>(name);
            return true;
        }
    }
    return false;
}

// Usage in MOPS
std::vector<std::string> velocity_names = {
    "normalVelocity",
    "timeMonthly_avg_normalVelocity",
    "timeYearly_avg_normalVelocity"
};

ftk::ndarray<double> velocity;
if (!try_read_variable(g, velocity_names, velocity)) {
    std::cerr << "Could not find velocity in any known name" << std::endl;
}
```

## Real-World Examples

### MPAS-Ocean Common Name Variations

```yaml
# Create comprehensive alias list
temperature:
  - temperature
  - temperatureNew
  - timeMonthly_avg_temperature
  - timeYearly_avg_temperature
  - temp
  - T

salinity:
  - salinity
  - timeMonthly_avg_salinity
  - timeYearly_avg_salinity
  - salt
  - S

layerThickness:
  - layerThickness
  - timeMonthly_avg_layerThickness
  - h

normalVelocity:
  - normalVelocity
  - timeMonthly_avg_normalVelocity
  - uNormal
  - velocityNormal

ssh:
  - ssh
  - sea_surface_height
  - eta
  - zeta
  - surfaceElevation
```

### CESM/POP Ocean Model

```yaml
# Different model, different conventions
temperature:
  - TEMP
  - TEMP_CUR
  - POT_TEMP

salinity:
  - SALT
  - SALT_CUR

velocity:
  - UVEL
  - VVEL
  - WVEL
```

## Best Practice Recommendations

1. **Always use `possible_names`** in YAML configs
2. **List most common name first** (faster lookup)
3. **Document naming conventions** in README
4. **Create shared alias files** for your lab/project
5. **Use introspection tools** (ncdump, ncks) to verify names
6. **Test with multiple dataset types** before deploying

## Future: CF Conventions Standard Names

Long-term solution is to use CF standard names:

```netcdf
float temperature(time, nCells, nVertLevels);
  temperature:standard_name = "sea_water_temperature";
  temperature:long_name = "Potential Temperature";
```

Then search by `standard_name` instead of variable name:

```cpp
auto temp = find_by_standard_name("sea_water_temperature");
// Works regardless of actual variable name
```

But MPAS-Ocean files don't consistently use CF conventions yet.
