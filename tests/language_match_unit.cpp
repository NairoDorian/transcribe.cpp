// Unit tests for transcribe-lang.h: how a caller's language code names a
// model's codes (the language allowlist and, later, hint resolution), and for
// transcribe-lang-alphabet.h: which token texts a list of languages may write.
#include "transcribe-lang-alphabet.h"
#include "transcribe-lang.h"

#include <cstdio>
#include <cstdlib>
#include <string>
#include <tuple>
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

using transcribe::lang::AlphabetFilter;
using transcribe::lang::build_alphabet_filter;

AlphabetFilter alpha(const Codes & codes, bool & usable) {
    return build_alphabet_filter(codes, usable);
}

// The union of the listed languages' alphabets: a fr+en speaker may write
// either language's letters anywhere, never Cyrillic or a Spanish ñ.
void test_alphabet_union() {
    bool                 usable = false;
    const AlphabetFilter fr_en  = alpha({ "fr", "en" }, usable);
    CHECK(usable);
    CHECK(fr_en.allows("hello world"));
    CHECK(
        fr_en.allows("\xE2\x96\x81"
                     "bonjour"));                       // SentencePiece word marker is neutral
    CHECK(fr_en.allows("d\xC3\xA9j\xC3\xA0"));          // "déjà"
    CHECK(fr_en.allows("\xC5\x93uvre"));                // "œuvre"
    CHECK(fr_en.allows("2024, 50% \xE2\x80\x94 ok?"));  // digits, punctuation, em dash
    CHECK(!fr_en.allows("\xD0\xBF\xD1\x80\xD0\xB8"));   // "при" (Cyrillic)
    CHECK(!fr_en.allows("a\xC3\xB1o"));                 // "año"
    CHECK(
        !fr_en.allows("stra\xC3\x9F"
                      "e"));               // "straße"
    CHECK(!fr_en.allows("\xE4\xB8\xAD"));  // "中"

    const AlphabetFilter ru_en = alpha({ "ru", "en-US" }, usable);
    CHECK(usable);
    CHECK(ru_en.allows("\xD0\xBF\xD1\x80\xD0\xB8"));
    CHECK(ru_en.allows("hello"));
    CHECK(!ru_en.allows("\xCE\xB1"));  // Greek alpha

    const AlphabetFilter zh_en = alpha({ "zh", "en" }, usable);
    CHECK(zh_en.allows("\xE4\xB8\xAD\xE6\x96\x87"));  // "中文"
    CHECK(zh_en.allows("\xEF\xBC\x8C"));              // full-width comma
    CHECK(!zh_en.allows("\xE3\x81\x82"));             // hiragana "あ"
    CHECK(!zh_en.allows("\xEA\xB0\x80"));             // hangul "가"
}

// Byte-level BPE splits a character across tokens: a cut sequence is judged
// by the script its lead bytes fix, and a cut Latin letter passes (the next
// token decides which letter it is).
void test_alphabet_partial_utf8() {
    bool                 usable = false;
    const AlphabetFilter fr_en  = alpha({ "fr", "en" }, usable);
    CHECK(!fr_en.allows("\xD0"));      // Cyrillic lead byte
    CHECK(!fr_en.allows("\xE4\xB8"));  // Han, two of three bytes
    CHECK(fr_en.allows("\xC3"));       // some Latin-1 letter
    CHECK(fr_en.allows("\xA9"));       // a stray continuation byte
}

// Cyrillic neighbours: Ukrainian and Russian share most letters but not all.
void test_alphabet_cyrillic_letters() {
    bool                 usable = false;
    const AlphabetFilter uk_en  = alpha({ "uk", "en" }, usable);
    CHECK(usable);
    CHECK(uk_en.allows("\xD1\x97\xD0\xB6\xD0\xB0\xD0\xBA"));  // "їжак"
    CHECK(!uk_en.allows("\xD0\xB2\xD1\x8B"));                 // "вы" (ы is Russian)
    CHECK(!uk_en.allows("\xD1\x91\xD0\xB6"));                 // "ёж"
    CHECK(uk_en.allows("\xD0"));                              // a cut Cyrillic letter passes
    const AlphabetFilter ru = alpha({ "ru" }, usable);
    CHECK(ru.allows("\xD0\xB2\xD1\x8B"));
    CHECK(!ru.allows("\xD1\x97"));                               // "ї"
    const AlphabetFilter ru_uk = alpha({ "ru", "uk" }, usable);  // the union writes both
    CHECK(ru_uk.allows("\xD0\xB2\xD1\x8B") && ru_uk.allows("\xD1\x97"));
    const AlphabetFilter kk = alpha({ "kk" }, usable);           // no letter list: any Cyrillic
    CHECK(kk.allows("\xD2\x9B"));                                // "қ"

    // A language the model cannot name constrains only its script.
    const AlphabetFilter uk_coarse = build_alphabet_filter({ "uk", "en" }, usable, { "uk" });
    CHECK(usable);
    CHECK(uk_coarse.allows("\xD0\xB2\xD1\x8B"));  // "вы" passes: Cyrillic is open
    CHECK(!uk_coarse.allows("a\xC3\xB1o"));       // "año": en's Latin letters still bind
    CHECK(!uk_coarse.allows("\xCE\xB1"));         // Greek still blocked
    const AlphabetFilter ca_coarse = build_alphabet_filter({ "ca", "fr" }, usable, { "ca" });
    CHECK(ca_coarse.allows("a\xC3\xB1o"));        // an unknown Latin language opens Latin letters
    CHECK(!ca_coarse.allows("\xD0\xB2"));         // but not another script
}

// transcribe_run_params::allowlist_letters: zero (the init default) keeps the
// letter restriction; ANY turns it off.
void test_letters_toggle() {
    transcribe_run_params p{};
    CHECK(transcribe::lang::letters_restricted(&p));
    CHECK(transcribe::lang::letters_restricted(nullptr));
    p.allowlist_letters = TRANSCRIBE_ALLOWLIST_LETTERS_ANY;
    CHECK(!transcribe::lang::letters_restricted(&p));
}

// Fail open: a language with no alphabet entry would have its letters banned.
void test_alphabet_unknown_language() {
    bool usable = true;
    std::ignore = alpha({ "fr", "xx" }, usable);
    CHECK(!usable);
    std::ignore = alpha({}, usable);
    CHECK(!usable);
    std::ignore = alpha({ "pt-BR" }, usable);  // regions write like the language
    CHECK(usable);
}

}  // namespace

int main() {
    test_normalize();
    test_regional_tags();
    test_bare_codes_and_aliases();
    test_alphabet_union();
    test_alphabet_partial_utf8();
    test_alphabet_cyrillic_letters();
    test_alphabet_unknown_language();
    test_letters_toggle();
    if (g_failures == 0) {
        std::printf("language_match_unit: OK\n");
    }
    return g_failures == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
