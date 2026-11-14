// Unit tests for transcribe-lang.h: how a caller's language code names a
// model's codes (the language allowlist and, later, hint resolution).
#include "transcribe-lang.h"

#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

namespace {

int g_failures = 0;

#define CHECK(cond)                                                              \
    do {                                                                         \
        if (!(cond)) {                                                           \
            std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); \
            ++g_failures;                                                        \
        }                                                                        \
    } while (0)

using Codes = std::vector<std::string>;

std::vector<int> r(const char * req, const Codes & codes) {
    return transcribe::lang::resolve_all(req, codes);
}

void test_normalize() {
    CHECK(transcribe::lang::normalize("  EN_us ") == "en-us");
    CHECK(transcribe::lang::normalize("zh-CN") == "zh-cn");
    CHECK(transcribe::lang::normalize("") == "");
}

// Nemotron-style regional tags: a bare code allows every region.
void test_regional_tags() {
    const Codes tags = { "en-US", "en-GB", "fr-FR", "fr-CA", "pt-BR", "pt-PT", "zh-CN", "ar-AR", "nb-NO", "nn-NO" };
    CHECK((r("fr", tags) == std::vector<int>{ 2, 3 }));
    CHECK((r("FR", tags) == std::vector<int>{ 2, 3 }));
    CHECK((r("fr-CA", tags) == std::vector<int>{ 3 }));
    CHECK((r("fr_ca", tags) == std::vector<int>{ 3 }));
    CHECK((r("pt-BR", tags) == std::vector<int>{ 4 }));
    // A region the model lacks falls back to the primary language.
    CHECK((r("ar-EG", tags) == std::vector<int>{ 7 }));
    CHECK((r("zh-TW", tags) == std::vector<int>{ 6 }));
    CHECK((r("zh", tags) == std::vector<int>{ 6 }));
    // Norwegian macro-language folds to Bokmal, never Nynorsk.
    CHECK((r("no", tags) == std::vector<int>{ 8 }));
    CHECK((r("nb", tags) == std::vector<int>{ 8 }));
    // Unknown languages name nothing; never cross to another primary.
    CHECK(r("de", tags).empty());
    CHECK(r("", tags).empty());
    CHECK(r("e", tags).empty());
}

// Whisper-style bare codes, including its legacy spellings.
void test_bare_codes_and_aliases() {
    const Codes whisper = { "en", "fr", "iw", "jw", "zh", "tl" };
    CHECK((r("en-US", whisper) == std::vector<int>{ 0 }));
    CHECK((r("en", whisper) == std::vector<int>{ 0 }));
    CHECK((r("he", whisper) == std::vector<int>{ 2 }));
    CHECK((r("he-IL", whisper) == std::vector<int>{ 2 }));
    CHECK((r("jv", whisper) == std::vector<int>{ 3 }));
    CHECK((r("fil", whisper) == std::vector<int>{ 5 }));
    CHECK((r("zh-CN", whisper) == std::vector<int>{ 4 }));
    // Prefer the bare code when the model has both forms.
    const Codes mixed = { "en-US", "en", "en-GB" };
    CHECK((r("en", mixed) == std::vector<int>{ 1 }));
    CHECK((r("en-AU", mixed) == std::vector<int>{ 1 }));
    CHECK((r("en-gb", mixed) == std::vector<int>{ 2 }));
}

}  // namespace

int main() {
    test_normalize();
    test_regional_tags();
    test_bare_codes_and_aliases();
    if (g_failures == 0) {
        std::printf("language_match_unit: OK\n");
    }
    return g_failures == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
