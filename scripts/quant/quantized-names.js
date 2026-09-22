// Print the names of tensors in one GGUF whose dtype is quantized (Q4_K/Q8_0).
//
// The converter's --keep-type has no mid-string wildcard -- a pattern is either
// an exact name or a trailing-* prefix -- so protecting one half of the model
// means listing its matmuls by exact name. This dumps that list from a package
// the converter itself already got right.
const fs = require("fs");

const file = process.argv[2];
const scope = process.argv[3] || "";

const fd = fs.openSync(file, "r");
const buf = Buffer.alloc(256 * 1024 * 1024);
const read = fs.readSync(fd, buf, 0, buf.length, 0);
fs.closeSync(fd);
const f = buf.subarray(0, read);
let q = 4;
const u32 = () => { const v = f.readUInt32LE(q); q += 4; return v; };
const u64 = () => { const v = Number(f.readBigUInt64LE(q)); q += 8; return v; };
const str = () => { const n = u64(); const s = f.toString("utf8", q, q + n); q += n; return s; };

u32();
const tensorCount = u64();
const kvCount = u64();
const skip = (t) => {
  if (t === 9) { const e = u32(), n = u64(); for (let i = 0; i < n; i++) skip(e); return; }
  if (t === 8) { str(); return; }
  const s = { 0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8 };
  q += s[t] ?? 0;
};
for (let i = 0; i < kvCount; i++) { str(); const t = u32(); skip(t); }

const out = [];
for (let i = 0; i < tensorCount; i++) {
  const name = str();
  const nd = u32();
  for (let d = 0; d < nd; d++) u64();
  const type = u32();
  u64();
  if (type !== 12 && type !== 8) continue;   // only the quantized ones
  if (scope && !name.startsWith(scope)) continue;
  out.push(name);
}
console.log(out.join("\n"));
