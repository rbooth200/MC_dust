#ifndef _HEADERS_UTILS_H
#define _HEADERS_UTILS_H

#include <string>
#include <vector>

inline std::vector<std::string> split(const std::string& str,
                                      const std::string& delim) {
    std::vector<std::string> tokens;
    std::size_t prev = 0, pos = 0;
    do {
        pos = str.find(delim, prev);
        if (pos == std::string::npos) pos = str.length();

        std::string token = str.substr(prev, pos - prev);
        if (!token.empty()) tokens.push_back(token);

        prev = pos + delim.length();
    } while (pos < str.length() && prev < str.length());
    return tokens;
}

#endif  //_HEADERS_UTILS_H