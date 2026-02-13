#ifndef _NDARRAY_VARIABLE_NAME_UTILS_HH
#define _NDARRAY_VARIABLE_NAME_UTILS_HH

#include <string>
#include <vector>
#include <algorithm>
#include <sstream>

#if NDARRAY_HAVE_NETCDF
#include <netcdf.h>
#endif

namespace ftk {

// Levenshtein distance for fuzzy string matching
inline int levenshtein_distance(const std::string& s1, const std::string& s2) {
    const size_t len1 = s1.size(), len2 = s2.size();
    std::vector<std::vector<int>> d(len1 + 1, std::vector<int>(len2 + 1));

    d[0][0] = 0;
    for(int i = 1; i <= len1; ++i) d[i][0] = i;
    for(int i = 1; i <= len2; ++i) d[0][i] = i;

    for(int i = 1; i <= len1; ++i) {
        for(int j = 1; j <= len2; ++j) {
            d[i][j] = std::min({
                d[i-1][j] + 1,
                d[i][j-1] + 1,
                d[i-1][j-1] + (s1[i-1] == s2[j-1] ? 0 : 1)
            });
        }
    }
    return d[len1][len2];
}

// Case-insensitive string comparison
inline bool iequals(const std::string& a, const std::string& b) {
    return std::equal(a.begin(), a.end(), b.begin(), b.end(),
        [](char a, char b) { return tolower(a) == tolower(b); });
}

// Find similar variable names using fuzzy matching
inline std::vector<std::string> find_similar_names(
    const std::string& query,
    const std::vector<std::string>& candidates,
    int max_suggestions = 3)
{
    std::vector<std::pair<int, std::string>> scored;

    for (const auto& candidate : candidates) {
        int score = 0;

        // Exact match (shouldn't happen if we got here)
        if (query == candidate) {
            score = 1000;
        }
        // Case-insensitive match
        else if (iequals(query, candidate)) {
            score = 900;
        }
        // Substring match
        else if (candidate.find(query) != std::string::npos) {
            score = 800;
        }
        // Query is substring of candidate (case-insensitive)
        else {
            std::string query_lower = query;
            std::string cand_lower = candidate;
            std::transform(query_lower.begin(), query_lower.end(), query_lower.begin(), ::tolower);
            std::transform(cand_lower.begin(), cand_lower.end(), cand_lower.begin(), ::tolower);

            if (cand_lower.find(query_lower) != std::string::npos) {
                score = 700;
            }
            // Levenshtein distance (typo tolerance)
            else {
                int distance = levenshtein_distance(query, candidate);
                if (distance <= 3) {
                    score = 500 - distance * 100;
                }
            }
        }

        if (score > 0) {
            scored.push_back({score, candidate});
        }
    }

    // Sort by score descending
    std::sort(scored.begin(), scored.end(),
        [](const auto& a, const auto& b) { return a.first > b.first; });

    // Return top suggestions
    std::vector<std::string> suggestions;
    for (int i = 0; i < std::min(max_suggestions, (int)scored.size()); i++) {
        suggestions.push_back(scored[i].second);
    }

    return suggestions;
}

#if NDARRAY_HAVE_NETCDF
// List all variables in a NetCDF file
inline std::vector<std::string> list_netcdf_variables(int ncid) {
    std::vector<std::string> vars;

    int nvars;
    if (nc_inq_nvars(ncid, &nvars) != NC_NOERR) {
        return vars;
    }

    for (int i = 0; i < nvars; i++) {
        char name[NC_MAX_NAME + 1];
        if (nc_inq_varname(ncid, i, name) == NC_NOERR) {
            vars.push_back(name);
        }
    }

    return vars;
}

// Create helpful error message when variable not found
inline std::string create_variable_not_found_message(
    const std::vector<std::string>& tried_names,
    int ncid)
{
    std::stringstream ss;
    ss << "Variable not found.\n";

    ss << "  Tried names: ";
    for (size_t i = 0; i < tried_names.size(); i++) {
        ss << "'" << tried_names[i] << "'";
        if (i < tried_names.size() - 1) ss << ", ";
    }
    ss << "\n";

    // List available variables
    auto available = list_netcdf_variables(ncid);
    if (!available.empty()) {
        ss << "  Available variables (" << available.size() << "): ";
        for (size_t i = 0; i < std::min<size_t>(10, available.size()); i++) {
            ss << "'" << available[i] << "'";
            if (i < std::min<size_t>(10, available.size()) - 1) ss << ", ";
        }
        if (available.size() > 10) {
            ss << " ... (+" << (available.size() - 10) << " more)";
        }
        ss << "\n";

        // Suggest similar names
        auto suggestions = find_similar_names(tried_names[0], available, 3);
        if (!suggestions.empty()) {
            ss << "  Did you mean: ";
            for (size_t i = 0; i < suggestions.size(); i++) {
                ss << "'" << suggestions[i] << "'";
                if (i < suggestions.size() - 1) ss << ", ";
            }
            ss << "?\n";
        }
    }

    return ss.str();
}
#endif

// Helper to join strings
inline std::string join(const std::vector<std::string>& vec, const std::string& delim) {
    std::stringstream ss;
    for (size_t i = 0; i < vec.size(); i++) {
        ss << vec[i];
        if (i < vec.size() - 1) ss << delim;
    }
    return ss.str();
}

// Check if variable name matches a simple pattern (supports * wildcard)
inline bool matches_pattern(const std::string& name, const std::string& pattern) {
    // Simple wildcard matching (* means any characters)
    size_t star_pos = pattern.find('*');

    if (star_pos == std::string::npos) {
        // No wildcard, exact match
        return name == pattern;
    }

    std::string prefix = pattern.substr(0, star_pos);
    std::string suffix = pattern.substr(star_pos + 1);

    if (name.length() < prefix.length() + suffix.length()) {
        return false;
    }

    bool prefix_match = name.substr(0, prefix.length()) == prefix;
    bool suffix_match = name.substr(name.length() - suffix.length()) == suffix;

    return prefix_match && suffix_match;
}

} // namespace ftk

#endif
