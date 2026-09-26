// transcribe-env.h - tiny helpers for reading environment variables with one
// consistent parse convention across the codebase.
//
// INTERNAL, C++17. Not part of the public ABI.
//
// A "flag" env var is considered ON when it is set, non-empty, and its first
// character is not '0' (so FOO=1 / FOO=on / FOO=yes are ON; FOO=0 and FOO=
// and unset are OFF). This matches the dominant pre-existing convention
// (TRANSCRIBE_PERF_DEBUG) and is the single source of truth for boolean env
// toggles — do not hand-roll getenv() comparisons elsewhere.

#pragma once

#include <string>

namespace transcribe::env {

// True iff `name` is set, non-empty, and its first character is not '0'.
bool flag(const char * name);

// The value of `name`, or nullptr if it is unset or empty. The returned
// pointer is owned by the environment (do not free); valid until the next
// setenv/putenv.
const char * str(const char * name);

// The value of `name` as UTF-8, or "" if unset. On Windows the C runtime's
// getenv() returns the ANSI code page (accented text arrives as invalid
// UTF-8), so this reads the UTF-16 environment and converts. Use it for any
// variable that can carry natural-language text.
std::string utf8(const char * name);

}  // namespace transcribe::env
