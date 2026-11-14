// transcribe-lang-alphabet.h - "only write in the allowed languages' alphabets".
//
// The language allowlist (transcribe_run_params::allowed_languages) has an
// exact mechanism on families that expose a language decision (Nemotron's
// locale tags, Whisper's language tokens, Qwen3-ASR's "language X" prefix).
// Families whose language is implicit (Parakeet TDT v3 / ultra / redux,
// Granite) have no such hook; what they do have is a token-level decoder. This
// header turns an allowlist into an alphabet filter that a decoder applies to
// its argmax: a token may only be emitted when every letter it contains
// belongs to a script, or a Latin letter, of an allowed language.
//
// What it constrains: script confusions (Latin vs Cyrillic vs Greek vs CJK,
// e.g. French speech decoded as Russian) and language-specific Latin letters
// (no "ñ", "ß", "ã" or "ł" when only French and English are allowed). What it
// cannot: two languages sharing one alphabet. ASCII, digits, punctuation,
// symbols and spaces are always allowed, so names, numbers and code survive.
//
// Header-only so architecture plugins (TRANSCRIBE_ARCH_DL) need no new link
// dependency, like transcribe-lang.h.
#pragma once

#include "transcribe-lang.h"

#include <algorithm>
#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

namespace transcribe::lang {

enum : uint16_t {
    k_script_latin      = 1u << 0,
    k_script_cyrillic   = 1u << 1,
    k_script_greek      = 1u << 2,
    k_script_han        = 1u << 3,
    k_script_kana       = 1u << 4,
    k_script_hangul     = 1u << 5,
    k_script_arabic     = 1u << 6,
    k_script_hebrew     = 1u << 7,
    k_script_devanagari = 1u << 8,
    k_script_thai       = 1u << 9,
    k_script_other      = 1u << 10,  // every other alphabetic script
};

// Script of a code point when it is a letter outside ASCII; 0 for everything
// that is always allowed (ASCII, digits, punctuation, symbols, spaces,
// combining marks, CJK punctuation, emoji).
inline uint16_t letter_script(uint32_t cp) {
    if (cp < 0x80) {
        return 0;
    }
    if ((cp >= 0x00C0 && cp <= 0x024F && cp != 0x00D7 && cp != 0x00F7) || (cp >= 0x1E00 && cp <= 0x1EFF)) {
        return k_script_latin;
    }
    if ((cp >= 0x0370 && cp <= 0x03FF) || (cp >= 0x1F00 && cp <= 0x1FFF)) {
        return k_script_greek;
    }
    if ((cp >= 0x0400 && cp <= 0x052F) || (cp >= 0x2DE0 && cp <= 0x2DFF) || (cp >= 0xA640 && cp <= 0xA69F)) {
        return k_script_cyrillic;
    }
    if (cp >= 0x0590 && cp <= 0x05FF) {
        return k_script_hebrew;
    }
    if ((cp >= 0x0600 && cp <= 0x06FF) || (cp >= 0x0750 && cp <= 0x077F) || (cp >= 0x08A0 && cp <= 0x08FF) ||
        (cp >= 0xFB50 && cp <= 0xFDFF) || (cp >= 0xFE70 && cp <= 0xFEFF)) {
        return k_script_arabic;
    }
    if (cp >= 0x0900 && cp <= 0x097F) {
        return k_script_devanagari;
    }
    if (cp >= 0x0E00 && cp <= 0x0E7F) {
        return k_script_thai;
    }
    if ((cp >= 0x3040 && cp <= 0x30FF) || (cp >= 0x31F0 && cp <= 0x31FF) || (cp >= 0xFF66 && cp <= 0xFF9F)) {
        return k_script_kana;
    }
    if ((cp >= 0x1100 && cp <= 0x11FF) || (cp >= 0x3130 && cp <= 0x318F) || (cp >= 0xAC00 && cp <= 0xD7AF)) {
        return k_script_hangul;
    }
    if ((cp >= 0x3400 && cp <= 0x4DBF) || (cp >= 0x4E00 && cp <= 0x9FFF) || (cp >= 0xF900 && cp <= 0xFAFF) ||
        (cp >= 0x20000 && cp <= 0x2FA1F)) {
        return k_script_han;
    }
    if ((cp >= 0x0530 && cp <= 0x058F) || (cp >= 0x0980 && cp <= 0x0DFF) || (cp >= 0x10A0 && cp <= 0x10FF) ||
        (cp >= 0x1200 && cp <= 0x137F)) {
        return k_script_other;  // Armenian, Indic beyond Devanagari, Georgian, Ethiopic
    }
    return 0;
}

struct LanguageAlphabet {
    const char * code;         // primary subtag
    uint16_t     scripts;      // non-Latin scripts the language writes in
    const char * latin_extra;  // non-ASCII Latin letters it uses (UTF-8, both cases)
    bool         any_latin;    // too many Latin diacritics to list (Vietnamese)
    // Cyrillic letters it uses (UTF-8, both cases); null = any Cyrillic. Listed
    // where neighbours are confused: Ukrainian writes і ї є ґ and never ы э ъ ё,
    // Russian the reverse, so "uk" alone keeps a Ukrainian speaker out of
    // Russian spelling.
    const char * cyrillic = nullptr;
};

// The languages the shipped multilingual models can detect, and then some.
// A Latin-script language lists the letters beyond ASCII its orthography uses
// (plus the loanword letters English text routinely carries).
inline const std::vector<LanguageAlphabet> & alphabets() {
    static const std::vector<LanguageAlphabet> k = {
        { "en", 0, "éèêëàâçïîôöüÉÈÊËÀÂÇÏÎÔÖÜ", false },
        { "fr", 0, "àâæçéèêëîïôœùûüÿÀÂÆÇÉÈÊËÎÏÔŒÙÛÜŸ", false },
        { "de", 0, "äöüßẞÄÖÜéÉ", false },
        { "es", 0, "áéíóúüñÁÉÍÓÚÜÑ", false },
        { "it", 0, "àèéìíîòóùúÀÈÉÌÍÎÒÓÙÚ", false },
        { "pt", 0, "áâãàçéêíóôõúüÁÂÃÀÇÉÊÍÓÔÕÚÜ", false },
        { "nl", 0, "áéíóúàèëïöüÁÉÍÓÚÀÈËÏÖÜĳĲ", false },
        { "ca", 0, "àçèéíïòóúüÀÇÈÉÍÏÒÓÚÜ", false },
        { "gl", 0, "áéíóúñüÁÉÍÓÚÑÜ", false },
        { "eu", 0, "ñÑ", false },
        { "sv", 0, "åäöéÅÄÖÉ", false },
        { "da", 0, "æøåéÆØÅÉ", false },
        { "nb", 0, "æøåéèêóòôÆØÅÉÈÊÓÒÔ", false },
        { "nn", 0, "æøåéèêóòôÆØÅÉÈÊÓÒÔ", false },
        { "fi", 0, "äöåšžÄÖÅŠŽ", false },
        { "et", 0, "äöõüšžÄÖÕÜŠŽ", false },
        { "is", 0, "áðéíóúýþæöÁÐÉÍÓÚÝÞÆÖ", false },
        { "pl", 0, "ąćęłńóśźżĄĆĘŁŃÓŚŹŻ", false },
        { "cs", 0, "áčďéěíňóřšťúůýžÁČĎÉĚÍŇÓŘŠŤÚŮÝŽ", false },
        { "sk", 0, "áäčďéíĺľňóôŕšťúýžÁÄČĎÉÍĹĽŇÓÔŔŠŤÚÝŽ", false },
        { "sl", 0, "čšžćđČŠŽĆĐ", false },
        { "hr", 0, "čćđšžČĆĐŠŽ", false },
        { "bs", 0, "čćđšžČĆĐŠŽ", false },
        { "hu", 0, "áéíóöőúüűÁÉÍÓÖŐÚÜŰ", false },
        { "ro", 0, "ăâîșțşţĂÂÎȘȚŞŢ", false },
        { "tr", 0, "çğıİöşüâîûÇĞÖŞÜÂÎÛ", false },
        { "az", 0, "çəğıİöşüÇƏĞÖŞÜ", false },
        { "lt", 0, "ąčęėįšųūžĄČĘĖĮŠŲŪŽ", false },
        { "lv", 0, "āčēģīķļņšūžĀČĒĢĪĶĻŅŠŪŽ", false },
        { "mt", 0, "ċġħżàèìòùĊĠĦŻÀÈÌÒÙ", false },
        { "ga", 0, "áéíóúÁÉÍÓÚ", false },
        { "cy", 0, "âêîôûŵŷäëïöüáéíóúàèìòùÂÊÎÔÛŴŶÄËÏÖÜÁÉÍÓÚÀÈÌÒÙ", false },
        { "sq", 0, "çëÇË", false },
        { "af", 0, "êëîïôûéÊËÎÏÔÛÉ", false },
        { "tl", 0, "ñÑ", false },
        { "id", 0, "éÉ", false },
        { "ms", 0, "éÉ", false },
        { "sw", 0, "", false },
        { "vi", 0, "", true },
        { "ru", k_script_cyrillic, "", false, "абвгдеёжзийклмнопрстуфхцчшщъыьэюяАБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯ" },
        { "uk", k_script_cyrillic, "", false, "абвгґдеєжзиіїйклмнопрстуфхцчшщьюяАБВГҐДЕЄЖЗИІЇЙКЛМНОПРСТУФХЦЧШЩЬЮЯ" },
        { "be", k_script_cyrillic, "", false, "абвгдеёжзійклмнопрстуўфхцчшыьэюяАБВГДЕЁЖЗІЙКЛМНОПРСТУЎФХЦЧШЫЬЭЮЯ" },
        { "bg", k_script_cyrillic, "", false, "абвгдежзийклмнопрстуфхцчшщъьюяАБВГДЕЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЬЮЯ" },
        { "mk", k_script_cyrillic, "", false, "абвгдѓежзѕијклљмнњопрстќуфхцчџшАБВГДЃЕЖЗЅИЈКЛЉМНЊОПРСТЌУФХЦЧЏШ" },
        { "sr", k_script_cyrillic, "čćđšžČĆĐŠŽ", false,
         "абвгдђежзијклљмнњопрстћуфхцчџшАБВГДЂЕЖЗИЈКЛЉМНЊОПРСТЋУФХЦЧЏШ" },
        { "kk", k_script_cyrillic, "", false },
        { "ky", k_script_cyrillic, "", false },
        { "mn", k_script_cyrillic, "", false },
        { "tg", k_script_cyrillic, "", false },
        { "tt", k_script_cyrillic, "", false },
        { "el", k_script_greek, "", false },
        { "zh", k_script_han, "", false },
        { "yue", k_script_han, "", false },
        { "ja", static_cast<uint16_t>(k_script_han | k_script_kana), "", false },
        { "ko", static_cast<uint16_t>(k_script_hangul | k_script_han), "", false },
        { "ar", k_script_arabic, "", false },
        { "fa", k_script_arabic, "", false },
        { "ur", k_script_arabic, "", false },
        { "ps", k_script_arabic, "", false },
        { "he", k_script_hebrew, "", false },
        { "yi", k_script_hebrew, "", false },
        { "hi", k_script_devanagari, "", false },
        { "mr", k_script_devanagari, "", false },
        { "ne", k_script_devanagari, "", false },
        { "th", k_script_thai, "", false },
        { "bn", k_script_other, "", false },
        { "ta", k_script_other, "", false },
        { "te", k_script_other, "", false },
        { "ka", k_script_other, "", false },
        { "hy", k_script_other, "", false },
    };
    return k;
}

// Decode one UTF-8 code point at s[i]. Returns the byte length consumed (>= 1)
// and sets `cp`; `complete` is false when the sequence runs past the end (the
// missing continuation bytes are taken as 0x80, the smallest).
inline size_t utf8_next(std::string_view s, size_t i, uint32_t & cp, bool & complete) {
    const auto b0 = static_cast<unsigned char>(s[i]);
    size_t     len;
    if (b0 < 0x80) {
        cp       = b0;
        complete = true;
        return 1;
    }
    if ((b0 & 0xE0) == 0xC0) {
        len = 2;
        cp  = b0 & 0x1Fu;
    } else if ((b0 & 0xF0) == 0xE0) {
        len = 3;
        cp  = b0 & 0x0Fu;
    } else if ((b0 & 0xF8) == 0xF0) {
        len = 4;
        cp  = b0 & 0x07u;
    } else {
        cp       = 0;  // a stray continuation byte: neutral
        complete = true;
        return 1;
    }
    complete = true;
    for (size_t k = 1; k < len; ++k) {
        if (i + k >= s.size() || (static_cast<unsigned char>(s[i + k]) & 0xC0) != 0x80) {
            // Cut short: the missing continuation bits are zeros, which gives
            // the smallest code point the sequence can still become.
            complete = false;
            cp <<= 6 * (len - k);
            return k;
        }
        cp = (cp << 6) | (static_cast<unsigned char>(s[i + k]) & 0x3Fu);
    }
    return len;
}

// Alphabet filter for an allowlist.
struct AlphabetFilter {
    uint16_t              scripts   = 0;
    bool                  any_latin = false;
    std::vector<uint32_t> latin_extra;  // sorted
    bool                  any_cyrillic = false;
    std::vector<uint32_t> cyrillic;     // sorted; consulted unless any_cyrillic

    // True when every letter of `bytes` (a token's decoded bytes) may be
    // written. A sequence cut at the token's end (byte-level BPE splits
    // characters across tokens) is judged by its script when the leading
    // bytes determine it, and passes when it is a Latin letter (its exact
    // letter is not known yet; the next token completes it).
    bool allows(std::string_view bytes) const {
        for (size_t i = 0; i < bytes.size();) {
            uint32_t     cp       = 0;
            bool         complete = true;
            const size_t n        = utf8_next(bytes, i, cp, complete);
            i += n;
            const uint16_t s = letter_script(cp);
            if (s == 0) {
                continue;
            }
            if (s == k_script_latin) {
                if (!complete || any_latin || std::binary_search(latin_extra.begin(), latin_extra.end(), cp)) {
                    continue;
                }
                return false;
            }
            if ((scripts & s) == 0) {
                return false;
            }
            if (s == k_script_cyrillic && complete && !any_cyrillic &&
                !std::binary_search(cyrillic.begin(), cyrillic.end(), cp)) {
                return false;  // a letter of another Cyrillic language
            }
        }
        return true;
    }
};

// Build the filter for the languages a request resolved to (the union of
// their alphabets: a speaker who switches between them mid-sentence may write
// any of them at any point). `usable` is false when the list is empty or any
// language in it has no alphabet entry: the filter would then ban letters that
// language writes, so the caller must not apply it (fail open).
//
// `script_only` lists the codes (spelled as in `codes`) of languages the model
// cannot name: they contribute their script, never a letter list. A model that
// does not know Ukrainian writes it, if at all, through Russian; banning "ы"
// then pushes it into misspelled Russian (measured on R2T2: a sentence it half
// got right in auto mode came out wholly Russian), so its Cyrillic stays open.
inline AlphabetFilter build_alphabet_filter(const std::vector<std::string> & codes,
                                            bool &                           usable,
                                            const std::vector<std::string> & script_only = {}) {
    AlphabetFilter f;
    usable = !codes.empty();
    std::vector<std::string> known;
    for (const auto & a : alphabets()) {
        known.emplace_back(a.code);
    }
    for (const std::string & code : codes) {
        // Primary subtag only: an alphabet is a property of the language,
        // not of the region ("pt-BR" writes like "pt").
        const std::string      norm = canonical_primary(primary_of(normalize(code)));
        const std::vector<int> hits = resolve_all(norm, known);
        if (hits.empty()) {
            usable = false;
            continue;
        }
        const bool coarse = std::find(script_only.begin(), script_only.end(), code) != script_only.end();
        for (const int k : hits) {
            const LanguageAlphabet & a = alphabets()[static_cast<size_t>(k)];
            f.scripts |= a.scripts;
            if (coarse) {
                // Its script, any letter of it: Latin when it writes Latin
                // (no other script, or Latin letters beside it).
                f.any_cyrillic = f.any_cyrillic || (a.scripts & k_script_cyrillic) != 0;
                f.any_latin    = f.any_latin || a.scripts == 0 || a.any_latin || a.latin_extra[0] != '\0';
                continue;
            }
            if ((a.scripts & k_script_cyrillic) != 0) {
                if (a.cyrillic == nullptr) {
                    f.any_cyrillic = true;
                } else {
                    const std::string_view letters(a.cyrillic);
                    for (size_t j = 0; j < letters.size();) {
                        uint32_t cp       = 0;
                        bool     complete = true;
                        j += utf8_next(letters, j, cp, complete);
                        f.cyrillic.push_back(cp);
                    }
                }
            }
            f.any_latin = f.any_latin || a.any_latin;
            const std::string_view extra(a.latin_extra);
            for (size_t j = 0; j < extra.size();) {
                uint32_t cp       = 0;
                bool     complete = true;
                j += utf8_next(extra, j, cp, complete);
                f.latin_extra.push_back(cp);
            }
        }
    }
    std::sort(f.cyrillic.begin(), f.cyrillic.end());
    f.cyrillic.erase(std::unique(f.cyrillic.begin(), f.cyrillic.end()), f.cyrillic.end());
    std::sort(f.latin_extra.begin(), f.latin_extra.end());
    f.latin_extra.erase(std::unique(f.latin_extra.begin(), f.latin_extra.end()), f.latin_extra.end());
    return f;
}

}  // namespace transcribe::lang
