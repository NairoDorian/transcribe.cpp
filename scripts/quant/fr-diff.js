// Compare each French arm's transcript against the Q8_0 reference.
//
// Three levels, because they mean different things:
//   raw    -- byte-identical? catches punctuation the model chose differently
//   strict -- identical after lowercasing, stripping punctuation and collapsing
//             whitespace? Still distinguishes voila from voilà.
//   loose  -- strict, plus diacritics folded? This is the one that answers the
//             German question: a language switch or a dropped word shows up
//             here, punctuation and accents do not.
//
// The loose level is what matters. Punctuation drift is real but benign, and
// accent loss (voilà -> voila) is orthographic -- the same class as the
// kanji/kana variation seen between arms on Japanese, where 五十円 -> 50円 and
// いけない -> 行けない are the same words written differently. Neither is the
// failure mode the ladder is being tested for; a switch to English is.
//
// Also flags a switch to English by counting English function words in the
// hypothesis, since that is precisely how D5/E/L/N failed on German.
//
// Usage: node fr-diff.js <val-fr-dir> <arm> [arm...]
const fs = require("fs");
const path = require("path");

const dir = process.argv[2];
const arms = process.argv.slice(3);
const CLIPS = ["fr1", "fr2", "fr3", "fr4", "fr5"];

// Words that are French-only or English-only, chosen to be unambiguous in
// short speech: "the/of/and/is/it" essentially never appear in French output,
// and "le/la/les/des/et/est/une" never in English output.
const EN = new Set("the of and is it to you that this with for are was on in a an".split(" "));
const FR = new Set("le la les des et est une un que qui dans pour avec sur pas je il elle nous vous ce cette".split(" "));

const strip = (s) =>
  s.toLowerCase()
    .normalize("NFC")
    // Keep letters (incl. accents) and digits; everything else is a separator.
    // Apostrophes become separators so "j'aimerais" -> "j aimerais" on both
    // sides; the comparison is between arms, not against a reference text.
    .replace(/[^\p{L}\p{N}]+/gu, " ")
    .trim();

// Diacritics folded: NFD splits "à" into "a" + a combining mark, and the mark
// is not \p{L}, so the same cleanup drops it.
const fold = (s) => strip(s.normalize("NFD").replace(/\p{Mn}+/gu, ""));

const read = (p) => { try { return fs.readFileSync(p, "utf8").trim(); } catch { return null; } };

const ref = {};
for (const c of CLIPS) ref[c] = read(path.join(dir, `q8-${c}.txt`));

const pad = (s, n) => (s + " ".repeat(n)).slice(0, n);
console.log(pad("arm", 7) + pad("clip", 6) + pad("raw", 9) + pad("strict", 9) + pad("loose", 9) + pad("en", 5) + "note");

let anyContentDrift = false;

for (const arm of arms) {
  for (const c of CLIPS) {
    const hyp = read(path.join(dir, `${arm}-${c}.txt`));
    const r = ref[c];
    if (hyp === null || r === null) {
      console.log(pad(arm, 7) + pad(c, 6) + pad("-", 9) + pad("-", 9) + pad("-", 9) + pad("-", 5) + "missing");
      continue;
    }
    const sameRaw = hyp === r;
    const wS = strip(r).split(" ").filter(Boolean);
    const hS = strip(hyp).split(" ").filter(Boolean);
    const sameStrict = wS.join(" ") === hS.join(" ");
    const wL = fold(r).split(" ").filter(Boolean);
    const hL = fold(hyp).split(" ").filter(Boolean);
    const sameLoose = wL.join(" ") === hL.join(" ");

    const enHits = hL.filter((w) => EN.has(w)).length;
    const frHits = hL.filter((w) => FR.has(w)).length;
    // Only meaningful when the clip actually has words to judge.
    const looksEnglish = hL.length >= 4 && enHits > frHits;

    let note = "";
    if (looksEnglish) { note = "ENGLISH?"; anyContentDrift = true; }
    else if (!sameLoose) {
      // Show what moved: words in the hypothesis that the reference lacks and
      // vice versa, so a real substitution is visible rather than implied.
      const only = (a, b) => { const s = new Set(b); return a.filter((w) => !s.has(w)); };
      const extra = only(hL, wL), missing = only(wL, hL);
      note = `+[${extra.join(",")}] -[${missing.join(",")}]`;
      anyContentDrift = true;
    } else if (!sameStrict) {
      note = "diacritics only";
    } else if (!sameRaw) {
      note = "punctuation only";
    }

    console.log(
      pad(arm, 7) + pad(c, 6) +
      pad(sameRaw ? "same" : "diff", 9) +
      pad(sameStrict ? "same" : "diff", 9) +
      pad(sameLoose ? "same" : "DIFF", 9) +
      pad(`${enHits}/${frHits}`, 5) + note,
    );
  }
}

console.log();
console.log(anyContentDrift
  ? "VERDICT: at least one arm differs in words, not just orthography -- read the notes above"
  : "VERDICT: every arm matches Q8_0 on French words; differences are punctuation/accents only");
