"""thrush-ternary-v2 weight unpacking shared across scripts.

`moondream/parakeet-redux` stores its quantized linear weights in the
`thrush-ternary-v2` format declared by the checkpoint's `ternary.json`:

    <module>.qweight  uint8   [out_features, ceil(in_features / 5)]
    <module>.scales   float16 [out_features, ceil(in_features / 128)]

Each byte of a row packs five ternary codes (values {0, 1, 2}), least
significant digit first: element `i` of a row is base-3 digit `i % 5`
of byte `i // 5`; padding digits at the tail of a row are zero. The
dequantized weight is

    w[r, c] = scales[r, c // group_size] * (code[r, c] - 1)

so codes map to {-1, 0, +1} scaled by a per-128-element group scale.

First consumer: the Stage-2 HF reference dumper
(`scripts/dump_reference_parakeet_hf.py`), which must unpack redux
weights into the dense float32 ParakeetForTDT parameters before it can
dump reference tensors. The Stage-3 converter imports the same function
so the pack format has exactly one implementation.
"""

from __future__ import annotations

import numpy as np

#: Base of the ternary digit encoding (codes in {0, 1, 2}).
TERTIARY_BASE = 3
#: Codes packed per qweight byte (5 trits fit into a byte: 3^5 = 243 <= 255).
ELEMENTS_PER_BYTE = 5
#: Weight quantization group size (per ternary.json `quant.group_size`).
DEFAULT_GROUP_SIZE = 128
#: 3^0 .. 3^4, least significant digit first (element i = digit i%5).
_DIGIT_POWERS = np.array([1, 3, 9, 27, 81], dtype=np.uint32)


def unpack_thrush_ternary(
    qweight: np.ndarray,
    scales: np.ndarray,
    *,
    in_features: int,
    group_size: int = DEFAULT_GROUP_SIZE,
) -> np.ndarray:
    """Dequantize one thrush-ternary-v2 matrix into dense float32.

    Arguments:
      qweight     uint8-like `[out_features, ceil(in_features / 5)]` packed codes.
      scales      float16-like `[out_features, ceil(in_features / group_size)]`
                  per-group scales.
      in_features The logical row width. Needed because rows are padded to a
                  whole byte and scales to a whole group; both paddings are
                  sliced off before returning.
      group_size  Scale group width; must match the checkpoint's ternary.json.

    Returns a float32 `[out_features, in_features]` array.

    Raises ValueError when the packed shapes disagree with `in_features` /
    `group_size` — a silent shape mismatch here would corrupt every weight,
    so the invariants are checked, not assumed.
    """
    qweight = np.asarray(qweight)
    scales = np.asarray(scales)
    if qweight.ndim != 2:
        raise ValueError(f"qweight must be 2-D, got shape {qweight.shape}")
    if scales.ndim != 2:
        raise ValueError(f"scales must be 2-D, got shape {scales.shape}")
    if group_size <= 0:
        raise ValueError(f"group_size must be positive, got {group_size}")

    out_features, n_bytes = qweight.shape
    if scales.shape[0] != out_features:
        raise ValueError(
            f"scales has {scales.shape[0]} rows, qweight has {out_features}"
        )
    expected_bytes = -(-in_features // ELEMENTS_PER_BYTE)  # ceil
    if n_bytes != expected_bytes:
        raise ValueError(
            f"qweight row is {n_bytes} bytes, expected {expected_bytes} "
            f"for in_features={in_features}"
        )
    expected_groups = -(-in_features // group_size)  # ceil
    if scales.shape[1] != expected_groups:
        raise ValueError(
            f"scales has {scales.shape[1]} groups, expected {expected_groups} "
            f"for in_features={in_features} group_size={group_size}"
        )

    # codes[r, b, d] = floor(byte[r, b] / 3^d) % 3  ->  element r, b*5+d.
    raw = qweight.astype(np.uint32, copy=False)[:, :, np.newaxis]
    codes = (raw // _DIGIT_POWERS) % TERTIARY_BASE
    codes = codes.reshape(out_features, -1)[:, :in_features].astype(np.float32)

    # scales[r, c // group_size], expanded per element then sliced to width
    # (covers both the byte padding and the group padding).
    scale_cols = np.repeat(
        scales.astype(np.float32, copy=False), group_size, axis=1
    )[:, :in_features]
    return scale_cols * (codes - 1.0)


def ternary_weight_zero_fraction(weight: np.ndarray) -> float:
    """Fraction of exactly-zero elements in an unpacked weight.

    Mirrors `quantized_modules[].zero_fraction` in ternary.json (code == 1
    <=> weight == 0). Used as a cheap post-unpack sanity check: a wrong
    byte offset or digit stride changes this count, and the checkpoint
    ships the expected value.
    """
    weight = np.asarray(weight)
    if weight.size == 0:
        return 0.0
    return float(np.count_nonzero(weight == 0) / weight.size)


# ---------------------------------------------------------------------------
# thrush codes -> GGML_TYPE_TQ1_G128 (patches/ggml/0003), lossless
# ---------------------------------------------------------------------------

#: ggml type id of the downstream group-128 ternary type (ggml.h).
GGML_TYPE_TQ1_G128 = 96
#: Weights per TQ1_G128 block (two 128-weight scale groups).
TQ1_G128_BLOCK = 256
#: Bytes per TQ1_G128 block: qs[48] + qh[4] + 2 x fp16 scale (1.75 bpw).
TQ1_G128_BYTES = 56


def unpack_thrush_codes(qweight: np.ndarray, *, in_features: int) -> np.ndarray:
    """thrush-ternary-v2 packed rows -> raw codes {0, 1, 2}, uint8 [O, in_features]."""
    qweight = np.asarray(qweight)
    raw = qweight.astype(np.uint32, copy=False)[:, :, np.newaxis]
    codes = (raw // _DIGIT_POWERS) % TERTIARY_BASE
    return codes.reshape(qweight.shape[0], -1)[:, :in_features].astype(np.uint8)


def _pack_trits(c: np.ndarray, n_trits: int) -> np.ndarray:
    """c[..., n, m] trits (n most significant first) -> ceil(v*256/243) bytes [..., m]."""
    v = np.zeros(c.shape[:-2] + c.shape[-1:], dtype=np.uint32)
    for n in range(n_trits):
        v = v * 3 + c[..., n, :]
    if n_trits == 4:
        v = v * 3  # fifth, least significant trit is padding 0
    return ((v * 256 + 242) // 243).astype(np.uint8)


def pack_tq1_g128(codes: np.ndarray, scales: np.ndarray) -> np.ndarray:
    """Pack ternary codes and per-128 scales into GGML_TYPE_TQ1_G128 rows.

    codes   uint8 [O, I], values 0, 1, 2 (weight -1, 0, +1); I % 256 == 0.
    scales  float16 [O, I / 128], copied bit-for-bit.

    Returns uint8 [O, I / 256 * 56]: the exact byte image ggml's
    ggml_tq1_g128_pack_codes() writes (layout documented on block_tq1_g128 in
    ggml-common.h: per group, 16 bytes x 5 trits for elements 0..79, 8 bytes x
    5 trits for 80..119, 2 bytes x 4 trits for 120..127).
    """
    codes = np.asarray(codes, dtype=np.uint8)
    scales = np.asarray(scales, dtype=np.float16)
    rows, cols = codes.shape
    if cols % TQ1_G128_BLOCK:
        raise ValueError(f"row length {cols} is not a multiple of {TQ1_G128_BLOCK}")
    if scales.shape != (rows, cols // 128):
        raise ValueError(f"scales shape {scales.shape}, expected {(rows, cols // 128)}")
    if codes.max(initial=0) > 2:
        raise ValueError("codes must be in {0, 1, 2}")
    nb = cols // TQ1_G128_BLOCK
    g = codes.reshape(rows, nb, 2, 128)
    a = _pack_trits(g[..., 0:80].reshape(rows, nb, 2, 5, 16), 5)     # [R, nb, 2, 16]
    b = _pack_trits(g[..., 80:120].reshape(rows, nb, 2, 5, 8), 5)    # [R, nb, 2, 8]
    h = _pack_trits(g[..., 120:128].reshape(rows, nb, 2, 4, 2), 4)   # [R, nb, 2, 2]
    qs = np.concatenate([a, b], axis=-1).reshape(rows, nb, 48)       # group-major
    qh = h.reshape(rows, nb, 4)
    d = scales.reshape(rows, nb, 2).view(np.uint8).reshape(rows, nb, 4)
    return np.concatenate([qs, qh, d], axis=-1).reshape(rows, nb * TQ1_G128_BYTES)


def unpack_tq1_g128(packed: np.ndarray, *, in_features: int) -> np.ndarray:
    """Inverse of pack_tq1_g128 (decoding like ggml's dequantize_row_tq1_g128)."""
    packed = np.asarray(packed, dtype=np.uint8)
    rows = packed.shape[0]
    nb = in_features // TQ1_G128_BLOCK
    blk = packed.reshape(rows, nb, TQ1_G128_BYTES)
    qs = blk[..., 0:48].reshape(rows, nb, 2, 24)
    qh = blk[..., 48:52].reshape(rows, nb, 2, 2)
    d = blk[..., 52:56].copy().view(np.float16).reshape(rows, nb, 2).astype(np.float32)

    def trits(byte: np.ndarray, n_trits: int) -> np.ndarray:
        out = [((((byte.astype(np.uint16) * (3 ** n)) & 0xFF) * 3) >> 8) for n in range(n_trits)]
        return np.stack(out, axis=-2)  # [..., n, m]

    a = trits(qs[..., 0:16], 5).reshape(rows, nb, 2, 80)
    b = trits(qs[..., 16:24], 5).reshape(rows, nb, 2, 40)
    h = trits(qh, 4).reshape(rows, nb, 2, 8)
    codes = np.concatenate([a, b, h], axis=-1).astype(np.float32)   # [R, nb, 2, 128]
    w = (codes - 1.0) * d[..., np.newaxis]
    return w.reshape(rows, in_features)


def register_gguf_tq1_g128():
    """Teach gguf-py about GGML_TYPE_TQ1_G128 and return its enum member.

    gguf-py's GGMLQuantizationType has no member for the downstream id 96, so
    GGUFReader rejects files containing it and GGUFWriter cannot name it.
    This injects a real member (name "TQ1_G128", value 96) and its block
    geometry into gguf-py, idempotently. Call it before reading or writing a
    GGUF that carries ternary tensors.
    """
    import gguf

    T = gguf.GGMLQuantizationType
    m = T._value2member_map_.get(GGML_TYPE_TQ1_G128)
    if m is None:
        m = int.__new__(T, GGML_TYPE_TQ1_G128)
        m._name_ = "TQ1_G128"
        m._value_ = GGML_TYPE_TQ1_G128
        T._value2member_map_[GGML_TYPE_TQ1_G128] = m
        T._member_map_["TQ1_G128"] = m
    gguf.GGML_QUANT_SIZES[m] = (TQ1_G128_BLOCK, TQ1_G128_BYTES)
    return m


#: Back-compat name used by the converter.
gguf_tq1_g128_dtype = register_gguf_tq1_g128
