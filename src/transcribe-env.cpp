// transcribe-env.cpp - see transcribe-env.h.

#include "transcribe-env.h"

#include <cstdlib>
#include <string>

#ifdef _WIN32
#    ifndef WIN32_LEAN_AND_MEAN
#        define WIN32_LEAN_AND_MEAN
#    endif
#    ifndef NOMINMAX
#        define NOMINMAX
#    endif
#    include <windows.h>
#endif

namespace transcribe::env {

const char * str(const char * name) {
    const char * v = std::getenv(name);
    if (v == nullptr || v[0] == '\0') {
        return nullptr;
    }
    return v;
}

std::string utf8(const char * name) {
#ifdef _WIN32
    const int wn = MultiByteToWideChar(CP_UTF8, 0, name, -1, nullptr, 0);
    if (wn <= 0) {
        return {};
    }
    std::wstring wname(static_cast<size_t>(wn), L'\0');
    MultiByteToWideChar(CP_UTF8, 0, name, -1, wname.data(), wn);
    const DWORD len = GetEnvironmentVariableW(wname.c_str(), nullptr, 0);
    if (len == 0) {
        return {};
    }
    std::wstring wval(static_cast<size_t>(len), L'\0');
    const DWORD  got = GetEnvironmentVariableW(wname.c_str(), wval.data(), len);
    wval.resize(got);
    const int n =
        WideCharToMultiByte(CP_UTF8, 0, wval.data(), static_cast<int>(wval.size()), nullptr, 0, nullptr, nullptr);
    std::string out(static_cast<size_t>(n > 0 ? n : 0), '\0');
    if (n > 0) {
        WideCharToMultiByte(CP_UTF8, 0, wval.data(), static_cast<int>(wval.size()), out.data(), n, nullptr, nullptr);
    }
    return out;
#else
    const char * v = std::getenv(name);
    return v != nullptr ? std::string(v) : std::string();
#endif
}

bool flag(const char * name) {
    const char * v = str(name);
    return v != nullptr && v[0] != '0';
}

}  // namespace transcribe::env
