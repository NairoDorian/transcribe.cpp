// tekken_pretok_unit.cpp - pure-host test of the Tekken (Mistral /
// Voxtral) pretokenizer split, no model required.
//
// Expected splits are regex.findall() with the tekken.json pattern,
// except the ASCII-only case cases marked below.

#include "transcribe-unicode.h"

#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

namespace {

int g_failures = 0;

std::string byte_encode(const std::string & raw) {
    std::string out;
    for (char c : raw) {
        out += transcribe::unicode::byte_to_unicode(static_cast<uint8_t>(c));
    }
    return out;
}

// u8 literals become char8_t in C++20; the tokenizer consumes UTF-8 bytes.
template <typename Char> std::string utf8_bytes(const Char * text) {
    return reinterpret_cast<const char *>(text);
}

struct Case {
    std::string              text;
    std::vector<std::string> expected;  // raw UTF-8 pretokens
};

void check_case(const Case & c) {
    const std::vector<std::string> got = transcribe::unicode::pretokenize_tekken(c.text);
    bool                           ok  = got.size() == c.expected.size();
    for (size_t i = 0; ok && i < got.size(); ++i) {
        ok = got[i] == byte_encode(c.expected[i]);
    }
    if (!ok) {
        std::fprintf(stderr, "FAIL pretokenize_tekken(\"%s\"): got %zu pieces, expected %zu\n", c.text.c_str(),
                     got.size(), c.expected.size());
        ++g_failures;
    }
}

}  // namespace

int main() {
    const std::vector<Case> cases = {
        // Case split: lower->Upper boundary splits, UPPER+lower stays one word.
        { utf8_bytes(u8"iPhone McDonald HTTPServer"),
         { utf8_bytes(u8"i"), utf8_bytes(u8"Phone"), utf8_bytes(u8" Mc"), utf8_bytes(u8"Donald"),
            utf8_bytes(u8" HTTPServer") }                                                                                                                               },
        { utf8_bytes(u8"ABC aBC AbC"),
         { utf8_bytes(u8"ABC"), utf8_bytes(u8" a"), utf8_bytes(u8"BC"), utf8_bytes(u8" Ab"), utf8_bytes(u8"C") }                                                        },
        // No contraction alternative: the apostrophe prefixes the letters.
        { utf8_bytes(u8"don't I'M"),
         { utf8_bytes(u8"don"), utf8_bytes(u8"'t"), utf8_bytes(u8" I"), utf8_bytes(u8"'M") }                                                                            },
        // Symbol runs swallow a trailing [\r\n/]*.
        { utf8_bytes(u8"!\n/x"),                                                                     { utf8_bytes(u8"!\n/"), utf8_bytes(u8"x") }                        },
        { utf8_bytes(u8"a /\r\n/ b"),                                                                { utf8_bytes(u8"a"), utf8_bytes(u8" /\r\n/"), utf8_bytes(u8" b") } },
        // ASCII-only case: non-ASCII letters sit in both classes, so the
        // regex split before U+00C4 does not happen.
        { utf8_bytes(u8"\u00d6l\u00c4nderung \u01c5emal"),
         { utf8_bytes(u8"\u00d6l\u00c4nderung"), utf8_bytes(u8" \u01c5emal") }                                                                                          },
        // Lm (U+02B0) sits in both classes.
        { utf8_bytes(u8"a\u02b0B"),                                                                  { utf8_bytes(u8"a\u02b0"), utf8_bytes(u8"B") }                     },
        // Combining marks join letter runs, or stand alone after a prefix.
        { utf8_bytes(u8"e\u0301cole \u0301 !\u0301a"),
         { utf8_bytes(u8"e\u0301cole"), utf8_bytes(u8" \u0301"), utf8_bytes(u8" !\u0301"), utf8_bytes(u8"a") }                                                          },
        // Devanagari: vowel signs / virama are \p{M}, so words stay whole.
        { utf8_bytes(u8"\u0928\u092e\u0938\u094d\u0924\u0947 \u0926\u0941\u0928\u093f\u092f\u093e"),
         { utf8_bytes(u8"\u0928\u092e\u0938\u094d\u0924\u0947"),
            utf8_bytes(u8" \u0926\u0941\u0928\u093f\u092f\u093e") }                                                                                                     },
        // Single-codepoint \p{N}, including non-ASCII digits / fractions.
        { utf8_bytes(u8"12 \u0663\u00bd"),
         { utf8_bytes(u8"1"), utf8_bytes(u8"2"), utf8_bytes(u8" "), utf8_bytes(u8"\u0663"), utf8_bytes(u8"\u00bd") }                                                    },
        // Whitespace alternatives.
        { utf8_bytes(u8"  x  \n\n  y   "),
         { utf8_bytes(u8" "), utf8_bytes(u8" x"), utf8_bytes(u8"  \n\n"), utf8_bytes(u8" "), utf8_bytes(u8" y"),
            utf8_bytes(u8"   ") }                                                                                                                                       },
        // Supplementary-plane letters (binary-search path of the flags table).
        { utf8_bytes(u8"\U00010400\U00010428 \U0001D400"),
         { utf8_bytes(u8"\U00010400\U00010428"), utf8_bytes(u8" \U0001D400") }                                                                                          },
        { utf8_bytes(u8""),                                                                          {}                                                                 },
    };
    for (const Case & c : cases) {
        check_case(c);
    }

    if (g_failures > 0) {
        std::fprintf(stderr, "tekken_pretok_unit: %d failures\n", g_failures);
        return EXIT_FAILURE;
    }
    std::fprintf(stdout, "tekken_pretok_unit: ok\n");
    return EXIT_SUCCESS;
}
