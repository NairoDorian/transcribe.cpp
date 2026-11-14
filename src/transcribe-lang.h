// transcribe-lang.h - language-code matching shared by every family.
//
// Callers spell languages many ways ("fr", "FR", "fr_FR", "fr-CA", "iw"),
// and models advertise their own codes ("fr", "fr-FR", "<fr-FR>" tags, "he"
// vs "iw"). This header answers one question: which of a model's codes does
// a caller's code name? It is header-only so architecture plugins
// (TRANSCRIBE_ARCH_DL) use it without a new link dependency.
//
// Matching rules, applied in order, stopping at the first that matches:
//   1. exact after normalization: ASCII lower-case, trimmed, '_' -> '-'
//      ("EN_us" names "en-US");
//   2. the request has no subtag: every model code with the same primary
//      subtag ("fr" names "fr-FR" and "fr-CA"; "en" names "en");
//   3. the request has a subtag the model lacks: the model's bare primary
//      code if it has one ("en-AU" names "en"), else every model code with
//      that primary ("ar-EG" names "ar-AR", "zh-TW" names "zh-CN").
// Primary subtags are compared after the alias table below, which folds the
// pairs models genuinely disagree on (Whisper's "iw"/"jw", NeMo's "nb").
// Regional requests never cross to a different primary language.
#pragma once

#include <cctype>
#include <string>
#include <string_view>
#include <vector>

namespace transcribe::lang {

// ASCII lower-case, surrounding whitespace removed, '_' -> '-'.
inline std::string normalize(std::string_view code) {
    size_t b = 0, e = code.size();
    while (b < e && std::isspace(static_cast<unsigned char>(code[b]))) {
        ++b;
    }
    while (e > b && std::isspace(static_cast<unsigned char>(code[e - 1]))) {
        --e;
    }
    std::string out;
    out.reserve(e - b);
    for (size_t i = b; i < e; ++i) {
        const char c = code[i];
        out.push_back(c == '_' ? '-' : static_cast<char>(std::tolower(static_cast<unsigned char>(c))));
    }
    return out;
}

// Canonical primary subtag: folds legacy / macro-language spellings so both
// sides of a comparison agree.
inline std::string canonical_primary(std::string_view primary) {
    struct Alias {
        const char * from;
        const char * to;
    };

    static constexpr Alias k_aliases[] = {
        { "iw",  "he" }, // Hebrew (legacy ISO 639-1, used by Whisper)
        { "in",  "id" }, // Indonesian (legacy)
        { "ji",  "yi" }, // Yiddish (legacy)
        { "jw",  "jv" }, // Javanese (Whisper's spelling)
        { "no",  "nb" }, // Norwegian macro-language -> Bokmal
        { "fil", "tl" }, // Filipino -> Tagalog
        { "zho", "zh" },
        { "eng", "en" },
    };
    for (const Alias & a : k_aliases) {
        if (primary == a.from) {
            return a.to;
        }
    }
    return std::string(primary);
}

// Primary subtag of a normalized code ("pt-br" -> "pt").
inline std::string_view primary_of(std::string_view normalized) {
    const size_t dash = normalized.find('-');
    return dash == std::string_view::npos ? normalized : normalized.substr(0, dash);
}

inline bool has_subtag(std::string_view normalized) {
    return normalized.find('-') != std::string_view::npos;
}

// Indices into `model_codes` that `requested` names (see the rules above).
// Empty when the request names none of them, including an empty request.
inline std::vector<int> resolve_all(std::string_view requested, const std::vector<std::string> & model_codes) {
    std::vector<int>  out;
    const std::string req = normalize(requested);
    if (req.empty()) {
        return out;
    }
    const std::string req_primary = canonical_primary(primary_of(req));
    // Rule 1: exact, with the primary subtag canonicalized on both sides.
    auto              canonical   = [](const std::string & n) {
        const std::string_view p = primary_of(n);
        return canonical_primary(p) + std::string(n.substr(p.size()));
    };
    const std::string        req_canon = canonical(req);
    std::vector<std::string> norm(model_codes.size());
    for (size_t i = 0; i < model_codes.size(); ++i) {
        norm[i] = normalize(model_codes[i]);
        if (canonical(norm[i]) == req_canon) {
            out.push_back(static_cast<int>(i));
        }
    }
    if (!out.empty()) {
        return out;
    }
    // Same primary language, any subtag.
    std::vector<int> same_primary;
    std::vector<int> bare_primary;
    for (size_t i = 0; i < norm.size(); ++i) {
        if (canonical_primary(primary_of(norm[i])) != req_primary) {
            continue;
        }
        same_primary.push_back(static_cast<int>(i));
        if (!has_subtag(norm[i])) {
            bare_primary.push_back(static_cast<int>(i));
        }
    }
    if (!has_subtag(req)) {
        return same_primary;                                    // rule 2
    }
    return bare_primary.empty() ? same_primary : bare_primary;  // rule 3
}

}  // namespace transcribe::lang
