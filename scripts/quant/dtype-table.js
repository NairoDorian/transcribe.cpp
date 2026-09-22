// Print the dtype of one representative tensor per block, for N packages.
//
// Why not gguf-types.js: that one diffs two packages and prints "<a> -> <b>" group
// headers with family lines under them. Reading a single block's dtype out of it
// means associating a family line with the header above it, which is exactly the
// awk step that silently printed "?" for every arm last time. This prints the
// answer directly, and asserts the block is uniform (all 28 layers agree) so a
// stray override cannot hide behind a representative tensor.
const { execFileSync } = require("child_process");
const fs = require("fs");

const GGML = {
  0: "F32", 1: "F16", 2: "Q4_0", 3: "Q4_1", 6: "Q5_0", 7: "Q5_1", 8: "Q8_0",
  9: "Q8_1", 10: "Q2_K", 11: "Q3_K", 12: "Q4_K", 13: "Q5_K", 14: "Q6_K",
  15: "Q8_K", 16: "IQ2_XXS", 17: "IQ2_XS", 18: "IQ3_XXS", 19: "IQ1_S",
  20: "IQ4_NL", 21: "IQ3_S", 22: "IQ2_S", 23: "IQ4_XS", 24: "I8",
  25: "I16", 26: "I32", 27: "I64", 28: "F64", 30: "BF16",
};

const LIMIT = 256 * 1024 * 1024;
function parse(path) {
  const fd = fs.openSync(path, "r");
  const buf = Buffer.alloc(LIMIT);
  const read = fs.readSync(fd, buf, 0, LIMIT, 0);
  fs.closeSync(fd);
  const file = buf.subarray(0, read);
  let p = 4;
  const u32 = () => { const v = file.readUInt32LE(p); p += 4; return v; };
  const u64 = () => { const v = Number(file.readBigUInt64LE(p)); p += 8; return v; };
  const str = () => { const n = u64(); const s = file.toString("utf8", p, p + n); p += n; return s; };
  u32();
  const tensorCount = u64(), kvCount = u64();
  const skip = (t) => {
    if (t === 9) { const et = u32(), n = u64(); for (let i = 0; i < n; i++) skip(et); return; }
    if (t === 8) { str(); return; }
    p += ({ 0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8 }[t] ?? 0);
  };
  for (let i = 0; i < kvCount; i++) { str(); const t = u32(); skip(t); }
  const rows = [];
  for (let i = 0; i < tensorCount && p < file.length; i++) {
    const name = str(); const nd = u32();
    for (let d = 0; d < nd; d++) u64();
    const type = u32(); u64();
    rows.push({ name, type: GGML[type] ?? `?${type}` });
  }
  return rows;
}

// One block = every tensor matching the pattern. All must agree.
const BLOCKS = [
  ["attn",  /^thinker\.model\.layers\.\d+\.self_attn\.\w+_proj\.weight$/],
  ["gate/up", /^thinker\.model\.layers\.\d+\.mlp\.(gate|up)_proj\.weight$/],
  ["down",  /^thinker\.model\.layers\.\d+\.mlp\.down_proj\.weight$/],
  ["embed", /^thinker\.model\.embed_tokens\.weight$/],
  ["tower", /^thinker\.audio_tower\./],
];

const arms = process.argv.slice(2);
const table = [];
for (const spec of arms) {
  const [tag, ...rest] = spec.split("=");
  const path = rest.join("=");
  if (!fs.existsSync(path)) { table.push([tag, "MISSING", "", "", "", ""]); continue; }
  const rows = parse(path);
  const cells = [];
  for (const [, re] of BLOCKS) {
    const hit = rows.filter((r) => re.test(r.name));
    if (!hit.length) { cells.push("-"); continue; }
    const types = [...new Set(hit.map((r) => r.type))];
    cells.push(types.length === 1 ? types[0] : `MIXED{${types.sort().join(",")}}`);
  }
  table.push([tag, ...cells]);
}

const heads = ["arm", ...BLOCKS.map(([n]) => n)];
const w = heads.map((h, i) => Math.max(h.length, ...table.map((r) => String(r[i] ?? "").length)));
console.log(heads.map((h, i) => h.padEnd(w[i])).join("  "));
console.log(w.map((x) => "-".repeat(x)).join("  "));
for (const r of table) console.log(r.map((c, i) => String(c ?? "").padEnd(w[i])).join("  "));
