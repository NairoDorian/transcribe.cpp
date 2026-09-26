#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "numpy>=1.24",
# ]
# ///
"""Pin the thrush-ternary-v2 unpack math against hand-computed cases.

`scripts/lib/ternary.py` is the single implementation of the pack format
declared by `moondream/parakeet-redux`'s `ternary.json`. It is consumed by
the Stage-2 HF reference dumper (redux weights must be unpacked before any
tensor can be dumped) and, later, the Stage-3 converter — both of which
would produce silently wrong models if the digit order, scale broadcast, or
padding slice were off.

The expectations below are computed by hand from the ternary.json spec
("element i of a row is base-3 digit i%5 of byte i//5, least significant
digit first; pad digits 0" + `w = scale * (code - 1)`), not by re-running
the implementation, plus a round-trip test against an independent
element-wise *encoder* (encode known codes into bytes the slow way, decode
with the vectorized unpack, compare).

Run standalone (exit-code driven):  uv run scripts/lib/test_ternary.py
Or under pytest:                    pytest scripts/lib/test_ternary.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from lib.ternary import (  # noqa: E402
    DEFAULT_GROUP_SIZE,
    TQ1_G128_BYTES,
    pack_tq1_g128,
    unpack_thrush_codes,
    unpack_thrush_ternary,
    unpack_tq1_g128,
    ternary_weight_zero_fraction,
)

_TESTS: list = []


def _test(fn):
    _TESTS.append(fn)
    return fn


def _encode_row(values: np.ndarray, n_bytes: int) -> np.ndarray:
    """Independent reference encoder: pack codes into bytes, one row.

    values: per-element codes (length <= n_bytes * 5), values in {0, 1, 2}.
    byte[i // 5] += value[i] * 3 ** (i % 5)   (least significant digit first)
    """
    row = np.zeros(n_bytes, dtype=np.uint8)
    for i, v in enumerate(values.tolist()):
        row[i // 5] = np.uint8(int(row[i // 5]) + int(v) * (3 ** (i % 5)))
    return row


@_test
def test_single_byte_hand_computed():
    # byte 196 = 2*81 + 1*27 + 0*9 + 2*3 + 1*1 -> codes [1, 2, 0, 1, 2]
    qweight = np.array([[196]], dtype=np.uint8)
    scales = np.array([[0.5]], dtype=np.float16)
    w = unpack_thrush_ternary(qweight, scales, in_features=5)
    expected = np.array([[0.0, 0.5, -0.5, 0.0, 0.5]], dtype=np.float32)
    np.testing.assert_allclose(w, expected, rtol=0, atol=0)
    assert w.dtype == np.float32
    assert w.shape == (1, 5)


@_test
def test_grouped_scales_and_byte_padding():
    # in_features=9 -> 2 bytes (byte 2 padded past element 6), group_size=4
    # -> 3 scale groups. Hand-decoded below from the spec.
    #   byte 196 -> [1, 2, 0, 1, 2]; byte 1 -> [1, 0, 0, 0, 0]
    #   codes[:9] = [1, 2, 0, 1, 2, 1, 0, 0, 0]
    #   scale cols  = [1,1,1,1, 2,2,2,2, 3]  (group = index // 4)
    #   w           = [0, 1, -1, 0, 2, 0, -2, -2, -3]
    qweight = np.array([[196, 1]], dtype=np.uint8)
    scales = np.array([[1.0, 2.0, 3.0]], dtype=np.float16)
    w = unpack_thrush_ternary(qweight, scales, in_features=9, group_size=4)
    expected = np.array([[0.0, 1.0, -1.0, 0.0, 2.0, 0.0, -2.0, -2.0, -3.0]],
                        dtype=np.float32)
    np.testing.assert_allclose(w, expected, rtol=0, atol=0)
    assert w.shape == (1, 9)


@_test
def test_multi_row_round_trip_vs_elementwise_encoder():
    rng = np.random.default_rng(20260926)
    out_features, in_features = 3, 1000  # 200 bytes, 8 groups of 128
    n_bytes = -(-in_features // 5)
    n_groups = -(-in_features // DEFAULT_GROUP_SIZE)
    codes = rng.integers(0, 3, size=(out_features, in_features), dtype=np.int64)
    qweight = np.stack(
        [_encode_row(row, n_bytes) for row in codes]
    )
    scales = (rng.random((out_features, n_groups)) * 4 - 2).astype(np.float16)

    w = unpack_thrush_ternary(qweight, scales, in_features=in_features)
    assert w.shape == (out_features, in_features)

    # Independent element-wise expectation.
    for r in range(out_features):
        scale_cols = np.repeat(scales[r].astype(np.float64),
                               DEFAULT_GROUP_SIZE)[:in_features]
        expected = scale_cols * (codes[r].astype(np.float64) - 1.0)
        np.testing.assert_allclose(
            w[r].astype(np.float64), expected, rtol=1e-6, atol=1e-6,
            err_msg=f"row {r} disagrees with element-wise decode",
        )


@_test
def test_default_group_size_matches_ternary_json():
    # ternary.json quant.group_size == 128 for parakeet-redux.
    assert DEFAULT_GROUP_SIZE == 128


@_test
def test_zero_fraction_helper():
    qweight = np.array([[196]], dtype=np.uint8)  # codes [1, 2, 0, 1, 2]
    scales = np.array([[0.5]], dtype=np.float16)
    w = unpack_thrush_ternary(qweight, scales, in_features=5)
    assert ternary_weight_zero_fraction(w) == 0.4  # codes 1 at elements 0, 3


@_test
def test_shape_mismatch_raises():
    scales_ok = np.zeros((1, 1), dtype=np.float16)
    qweight_ok = np.zeros((1, 1), dtype=np.uint8)  # 5 elements

    # Wrong byte count for in_features (needs 2 bytes, has 1).
    try:
        unpack_thrush_ternary(qweight_ok, scales_ok, in_features=9)
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for wrong byte count")

    # Wrong scale-group count for in_features/group_size (needs 2, has 1).
    try:
        unpack_thrush_ternary(
            np.zeros((1, 10), dtype=np.uint8),  # 50 elements
            scales_ok,
            in_features=50,
            group_size=32,
        )
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for wrong group count")

    # Row-count mismatch between qweight and scales.
    try:
        unpack_thrush_ternary(
            np.zeros((2, 1), dtype=np.uint8),
            np.zeros((1, 1), dtype=np.float16),
            in_features=5,
        )
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for row mismatch")


@_test
def test_tq1_g128_round_trip_exact():
    # Random codes and scales survive pack -> unpack bit-exactly.
    rng = np.random.default_rng(7)
    codes = rng.integers(0, 3, size=(5, 1024), dtype=np.uint8)
    scales = rng.uniform(1e-4, 0.5, size=(5, 8)).astype(np.float16)
    packed = pack_tq1_g128(codes, scales)
    assert packed.shape == (5, 4 * TQ1_G128_BYTES), packed.shape
    w = unpack_tq1_g128(packed, in_features=1024)
    want = (codes.astype(np.float32) - 1.0) * np.repeat(scales.astype(np.float32), 128, axis=1)
    assert np.array_equal(w, want), "TQ1_G128 round trip is not exact"


@_test
def test_tq1_g128_extreme_bytes():
    # All-2 and all-0 groups hit the top/bottom of the ceil(v*256/243) map.
    for c in (0, 1, 2):
        codes = np.full((1, 256), c, dtype=np.uint8)
        scales = np.ones((1, 2), dtype=np.float16)
        w = unpack_tq1_g128(pack_tq1_g128(codes, scales), in_features=256)
        assert np.all(w == c - 1), f"constant code {c} decodes to {np.unique(w)}"


@_test
def test_tq1_g128_matches_thrush_unpack():
    # thrush-ternary-v2 -> codes -> TQ1_G128 dequantizes to the same weights
    # unpack_thrush_ternary produces directly (the dense reference path).
    rng = np.random.default_rng(11)
    rows, cols = 3, 1024
    codes = rng.integers(0, 3, size=(rows, cols), dtype=np.uint8)
    q = np.zeros((rows, -(-cols // 5)), dtype=np.uint8)
    for r in range(rows):
        q[r] = _encode_row(codes[r], q.shape[1])
    scales = rng.uniform(1e-3, 0.2, size=(rows, cols // 128)).astype(np.float16)
    dense = unpack_thrush_ternary(q, scales, in_features=cols)
    assert np.array_equal(unpack_thrush_codes(q, in_features=cols), codes)
    via = unpack_tq1_g128(pack_tq1_g128(unpack_thrush_codes(q, in_features=cols), scales), in_features=cols)
    assert np.array_equal(via, dense), "TQ1_G128 disagrees with thrush unpack"


def main() -> int:
    failures = 0
    for t in _TESTS:
        try:
            t()
        except AssertionError as e:
            failures += 1
            print(f"FAIL {t.__name__}: {e}")
        except ValueError as e:
            failures += 1
            print(f"FAIL {t.__name__}: unexpected ValueError: {e}")
        else:
            print(f"ok   {t.__name__}")
    print(f"\n{len(_TESTS) - failures}/{len(_TESTS)} checks passed.")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
