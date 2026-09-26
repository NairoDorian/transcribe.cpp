#!/usr/bin/env python3
"""Dump Parakeet reference tensors from HuggingFace transformers.

Stage-2 oracle dumper for the moondream parakeet-ultra / parakeet-redux
port (docs/porting/parakeet-ultra-redux-master-plan.md). It writes the
same tensor names as scripts/dump_reference_parakeet_nemo.py so
compare_tensors.py and tests/tolerances/parakeet.json work unchanged;
only the producing framework differs (transformers ParakeetForTDT on
moondream checkpoints instead of NeMo on nvidia checkpoints).

Framework notes:

  * mel comes from transformers' ParakeetFeatureExtractor and is dumped
    straight from the model input — there is no preprocessor module to
    hook. The stage label matches the NeMo dumper so coverage tooling
    sees one (name, stage) pair per tensor across frameworks.
  * weights are loaded manually rather than via from_pretrained so a
    single code path serves both the plain-F16 ultra checkpoint and
    redux's thrush-ternary-v2 packed tensors (scripts/lib/ternary.py),
    with a strict missing/unexpected key check against the constructed
    model as the guard.
  * attention is forced to eager: the rel-position attention is computed
    plainly here, matching the reduction order of the C++ implementation
    (sdpa would be numerically near-identical but not the same summation
    order, so it would only add noise to the tolerance compare).
  * dec.joint.0 is log-softmaxed over the full joint width (vocab +
    duration logits), matching the C++ joint dump (decoder.cpp
    joint_step, the log_softmax block gated by debug::enabled) and NeMo's
    joint_after_projection representation.
  * the transcript comes from a manual TDT greedy loop over the HF
    modules: model.generate() cannot run on these checkpoints (neither
    config.json nor generation_config.json defines a decoder start id,
    and the call raises ValueError). The loop feeds the predictor ZEROS
    at step 0 — the same start state as the C++/NeMo decoder and the
    dec.embed.0 dump — and applies the exact C++/NeMo advance rule
    (decoder.cpp TDT emit block). Probe-verified on the oracle clip:
    zero-start and blank-start, and both the C++/NeMo advance rule and
    the alternate HF-style rule, all produce identical transcripts.

Usage (validate.py cmd_ref invokes these; see reference.entrypoint):

    uv run --project scripts/envs/parakeet scripts/dump_reference_parakeet_hf.py \
        encoder --model moondream/parakeet-ultra --audio samples/jfk.wav \
        --out build/validate/parakeet/<variant>/jfk/ref --torch-threads 1 --language en
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np

# Load scripts/lib modules by absolute path: when this runs inside the
# NeMo-bearing parakeet reference env, nemo_toolkit ships a top-level
# `scripts/` package that shadows the repo's scripts/ on sys.path.
# Loading by file path bypasses the shadow entirely.
import importlib.util as _importlib_util


def _load_by_path(path: Path, name: str):
    spec = _importlib_util.spec_from_file_location(name, path)
    module = _importlib_util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_LIB_DIR = Path(__file__).resolve().parent / "lib"
_ref_dump = _load_by_path(_LIB_DIR / "ref_dump.py", "transcribe_ref_dump")
write_tensor = _ref_dump.write_tensor
write_transcript = _ref_dump.write_transcript
_ternary = _load_by_path(_LIB_DIR / "ternary.py", "transcribe_ternary")
unpack_thrush_ternary = _ternary.unpack_thrush_ternary
ternary_weight_zero_fraction = _ternary.ternary_weight_zero_fraction


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def resolve_path(raw: str | os.PathLike[str]) -> Path:
    return Path(raw).expanduser().resolve()


def configure_torch(args: argparse.Namespace) -> None:
    import torch

    torch.manual_seed(0)
    if args.torch_threads > 0:
        torch.set_num_threads(args.torch_threads)
        torch.set_num_interop_threads(1)
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except TypeError:
        torch.use_deterministic_algorithms(True)


def to_np(t) -> np.ndarray:
    """Convert a torch Tensor to contiguous fp32 numpy, squeezing
    leading size-1 dims to match C++ dump conventions."""
    import torch

    if isinstance(t, torch.Tensor):
        a = t.detach().to(dtype=torch.float32, device="cpu").numpy()
    else:
        a = np.asarray(t, dtype=np.float32)
    while a.ndim > 1 and a.shape[0] == 1:
        a = a[0]
    return np.ascontiguousarray(a, dtype=np.float32)


def load_audio(audio_path: Path) -> tuple[np.ndarray, int]:
    import soundfile as sf

    pcm, sr = sf.read(str(audio_path), dtype="float32", always_2d=False)
    if pcm.ndim > 1:
        pcm = pcm.mean(axis=1)
    return np.ascontiguousarray(pcm, dtype=np.float32), int(sr)


def resolve_hf_file(model: str, filename: str, revision: str | None) -> Path:
    """Local path to `filename` inside `model` (dir) or the HF hub (repo id)."""
    local = Path(model).expanduser()
    if (local / filename).exists():
        return local / filename
    from huggingface_hub import hf_hub_download

    return Path(hf_hub_download(model, filename, revision=revision))


# ---------------------------------------------------------------------------
# Model / frontend loading
# ---------------------------------------------------------------------------

def _check_zero_fracs(args: argparse.Namespace, zero_fracs: dict[str, float]) -> None:
    """Cross-check unpacked weights against ternary.json's zero_fraction.

    The checkpoint ships the exact fraction of zero elements per quantized
    module (code == 1 <=> weight == 0). The count is a permutation-
    independent invariant of the packed rows, so a wrong byte offset, digit
    stride, or scale broadcast shifts it: agreement is cheap evidence the
    pack format was decoded the way the checkpoint encoder wrote it.
    """
    import json

    try:
        path = resolve_hf_file(args.model, "ternary.json", args.revision)
        spec = json.loads(path.read_text(encoding="utf-8"))
        expected = {
            m["name"]: float(m["zero_fraction"])
            for m in spec.get("quantized_modules", [])
            if "zero_fraction" in m
        }
    except Exception as e:  # no ternary.json / hub hiccup: check is best-effort
        print(f"zero_fraction cross-check skipped ({type(e).__name__}: {e})")
        return
    if not expected:
        print("zero_fraction cross-check skipped (ternary.json has no entries)")
        return

    mismatches = []
    for base, actual in zero_fracs.items():
        want = expected.get(base)
        if want is None:
            mismatches.append((base, actual, None))
        elif abs(actual - want) > 1e-6:
            mismatches.append((base, actual, want))
    unchecked = sorted(set(expected) - set(zero_fracs))
    if mismatches or unchecked:
        raise SystemExit(
            "error: ternary unpack does not agree with ternary.json:\n"
            f"  mismatched/unknown ({len(mismatches)}): {mismatches[:5]}\n"
            f"  in json but not unpacked ({len(unchecked)}): {unchecked[:5]}"
        )
    print(
        f"zero_fraction cross-check OK against ternary.json "
        f"({len(zero_fracs)} modules)"
    )


def load_model(args: argparse.Namespace):
    """Construct ParakeetForTDT from config, then load safetensors weights.

    Returns (model, info) where info carries provenance fields for the
    dump source dicts. The manual load (rather than from_pretrained) is
    what lets one path handle both plain checkpoints and redux's
    thrush-ternary-v2 packed tensors; strict key reconciliation against
    the constructed model is the guard against packing/name drift.
    """
    import torch
    from transformers import AutoConfig, ParakeetForTDT

    config = AutoConfig.from_pretrained(args.model, revision=args.revision)
    model_type = getattr(config, "model_type", None)
    if model_type != "parakeet_tdt":
        raise SystemExit(
            f"error: expected a parakeet_tdt checkpoint, got model_type="
            f"{model_type!r} from {args.model!r}; this dumper implements the "
            "TDT path only (moondream/parakeet-ultra, moondream/parakeet-redux)"
        )

    # Explicit eager attention (see module docstring).
    config._attn_implementation = "eager"
    config.encoder_config._attn_implementation = "eager"

    model = ParakeetForTDT(config)

    ckpt = resolve_hf_file(args.model, "model.safetensors", args.revision)
    print(f"Loading weights from: {ckpt}")
    from safetensors.torch import load_file

    raw = load_file(str(ckpt), device="cpu")

    target = model.state_dict()
    state: dict[str, torch.Tensor] = {}
    zero_fracs: dict[str, float] = {}
    n_ternary = 0
    for key, value in raw.items():
        if key.startswith("vad_head."):
            # Moondream-photon-only VAD head; not part of ParakeetForTDT.
            continue
        if key.endswith(".qweight"):
            base = key[: -len(".qweight")]
            weight_key = base + ".weight"
            if weight_key not in target:
                raise SystemExit(
                    f"error: ternary tensor {key!r} has no dense counterpart "
                    f"{weight_key!r} in ParakeetForTDT"
                )
            dense_shape = tuple(target[weight_key].shape)
            in_features = dense_shape[1]
            scales = raw.get(base + ".scales")
            if scales is None:
                raise SystemExit(f"error: {key!r} has no sibling .scales tensor")
            weight = unpack_thrush_ternary(
                value.numpy(), scales.numpy(), in_features=in_features
            )
            zero_fracs[base] = ternary_weight_zero_fraction(weight)
            # Pointwise convs are Conv1d [O, I, 1] (ternary.json as_conv1d);
            # the packed rows only carry [O, I].
            if int(np.prod(dense_shape)) != weight.size:
                raise SystemExit(
                    f"error: unpacked {key!r} has shape {weight.shape}, "
                    f"dense parameter is {dense_shape}"
                )
            state[weight_key] = torch.from_numpy(
                np.ascontiguousarray(weight.reshape(dense_shape))
            )
            n_ternary += 1
        elif key.endswith(".scales"):
            continue  # consumed together with its .qweight sibling above
        else:
            state[key] = value

    if n_ternary:
        _check_zero_fracs(args, zero_fracs)
        print(f"unpacked {n_ternary} thrush-ternary-v2 tensors")

    missing = sorted(k for k in target if k not in state)
    unexpected = sorted(k for k in state if k not in target)
    if missing or unexpected:
        raise SystemExit(
            "error: checkpoint does not match ParakeetForTDT(config):\n"
            f"  missing ({len(missing)}): {missing[:8]}\n"
            f"  unexpected ({len(unexpected)}): {unexpected[:8]}"
        )
    model.load_state_dict(state, strict=True)  # F16 -> F32 via copy_
    model.eval()

    # The reference is single-precision end to end; a leftover F16/BF16
    # parameter would silently change every downstream dump.
    bad = [n for n, p in model.named_parameters() if p.dtype != torch.float32]
    if bad:
        raise SystemExit(f"error: non-float32 parameters after load: {bad[:8]}")

    info = {
        "weights_format": "thrush-ternary-v2" if n_ternary else "safetensors",
        "n_ternary": n_ternary,
    }
    return model, info


def build_feature_extractor(model):
    from transformers import ParakeetFeatureExtractor

    fe = ParakeetFeatureExtractor(
        feature_size=model.config.encoder_config.num_mel_bins
    )
    print(
        f"frontend: feature_size={fe.feature_size} sampling_rate={fe.sampling_rate} "
        f"n_fft={fe.n_fft} hop={fe.hop_length} win={fe.win_length} "
        f"preemph={fe.preemphasis}"
    )
    return fe


def load_tokenizer(args: argparse.Namespace):
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(args.model, revision=args.revision)


def featurize(fe, pcm: np.ndarray, sr: int):
    """Run the HF feature extractor; returns (1, T, n_mels) float32."""
    if sr != 16000:
        raise SystemExit(f"error: audio sample rate is {sr}, expected 16000")
    feats = fe(
        np.ascontiguousarray(pcm, dtype=np.float32),
        sampling_rate=sr,
        return_tensors="pt",
    )["input_features"]
    return feats


def make_source(
    *,
    args: argparse.Namespace,
    audio_path: Path,
    n_samples: int,
    sample_rate: int,
    arch: str,
    weights_format: str,
) -> dict[str, Any]:
    import torch
    import transformers

    source: dict[str, Any] = {
        "kind": "parakeet-hf",
        "model": args.model,
        "model_dtype": "f32",
        "device": "cpu",
        "torch_threads": args.torch_threads,
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "audio": audio_path.name,
        "n_samples": int(n_samples),
        "sample_rate": int(sample_rate),
        "arch": arch,
        "weights_format": weights_format,
    }
    if getattr(args, "revision", None):
        source["revision"] = args.revision
    return source


def select_blocks(model, requested: list[int] | None) -> list[int]:
    """Resolve which encoder block indices to dump.

    Default {0, n/2, n-1} mirrors the C++ spot-check set (model.cpp
    dumps block 0, mid_block_idx = n/2, last_block_idx = n-1; 0/12/23 on
    the 24-layer checkpoints) so the default ref set lines up with the
    default C++ dump set. Extra blocks only appear on one side when a
    bisect flag was used (TRANSCRIBE_DUMP_ALL_BLOCKS vs --blocks), and
    compare_tensors ignores tensors absent from both sides.
    """
    n = len(model.encoder.layers)
    if n == 0:
        return []
    if requested:
        return sorted({i for i in requested if 0 <= i < n})
    if n == 1:
        return [0]
    return sorted({0, n // 2, n - 1})


# ---------------------------------------------------------------------------
# Hook-based intermediate capture
# ---------------------------------------------------------------------------

def _make_sub_block_forward(layer, idx: int, intermediates: dict[str, Any]):
    """Replicate ParakeetEncoderBlock.forward with mid-pass residual taps.

    A torch forward hook only sees a module's return value; the C++
    observer points (after FF1 / attn / conv / FF2, before the final
    norm_out) sit inside the block body, so the forward is wrapped here —
    the same technique scripts/dump_reference_parakeet_nemo.py uses for
    the NeMo block. Body must stay line-for-line equivalent to
    transformers' ParakeetEncoderBlock.forward (0.5 FF scaling, pre-norm
    sublayers, final norm_out).
    """

    def wrapped_forward(
        hidden_states,
        attention_mask=None,
        position_embeddings=None,
        **kwargs,
    ):
        residual = hidden_states
        out = layer.feed_forward1(layer.norm_feed_forward1(hidden_states))
        hidden_states = residual + 0.5 * out  # conformer FF scaling factor
        intermediates[f"enc.block.{idx}.ff1"] = hidden_states.detach().clone()

        attn_out, _ = layer.self_attn(
            hidden_states=layer.norm_self_att(hidden_states),
            attention_mask=attention_mask,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        hidden_states = hidden_states + attn_out
        intermediates[f"enc.block.{idx}.attn"] = hidden_states.detach().clone()

        conv_out = layer.conv(
            layer.norm_conv(hidden_states), attention_mask=attention_mask
        )
        hidden_states = hidden_states + conv_out
        intermediates[f"enc.block.{idx}.conv"] = hidden_states.detach().clone()

        ff2 = layer.feed_forward2(layer.norm_feed_forward2(hidden_states))
        hidden_states = hidden_states + 0.5 * ff2
        intermediates[f"enc.block.{idx}.ff2"] = hidden_states.detach().clone()

        return layer.norm_out(hidden_states)

    return wrapped_forward


def capture_intermediates(
    model, block_indices: list[int], sub_block_indices: list[int] | None = None
):
    """Register forward hooks on key encoder sub-modules.

    For block indices in `sub_block_indices`, additionally wrap the
    block's forward to tap the running residual at the C++ observer
    points (saved under `enc.block.<i>.{ff1,attn,conv,ff2}`).

    Returns (intermediates, hook_handles, restore_callbacks).
    """
    intermediates: dict[str, Any] = {}
    hooks = []
    restore_cbs: list[Any] = []

    def _hook(name, extract_idx=0):
        def fn(_module, _input, output):
            if isinstance(output, tuple):
                intermediates[name] = output[extract_idx].detach().clone()
            else:
                intermediates[name] = output.detach().clone()

        return fn

    enc = model.encoder
    hooks.append(enc.subsampling.register_forward_hook(_hook("enc.pre_encode.out")))
    hooks.append(enc.encode_positions.register_forward_hook(_hook("enc.pos_emb")))
    for i in block_indices:
        if i < len(enc.layers):
            hooks.append(
                enc.layers[i].register_forward_hook(_hook(f"enc.block.{i}.out"))
            )

    if sub_block_indices:
        for idx in sub_block_indices:
            if idx >= len(enc.layers):
                continue
            layer = enc.layers[idx]
            original_forward = layer.forward
            layer.forward = _make_sub_block_forward(layer, idx, intermediates)
            restore_cbs.append((layer, original_forward))

    return intermediates, hooks, restore_cbs


def run_encoder(model, feats):
    import torch

    with torch.inference_mode():
        out = model.encoder(input_features=feats, attention_mask=None)
    return out.last_hidden_state


# ---------------------------------------------------------------------------
# Greedy TDT decode (transcript path)
# ---------------------------------------------------------------------------

def greedy_tdt(
    model,
    enc_proj,
    *,
    blank: int,
    vocab: int,
    durations: list[int],
    max_symbols: int,
) -> list[int]:
    """Manual TDT greedy decode over the HF embedding/LSTM/joint modules.

    Start state: predictor fed ZEROS at step 0 (C++/NeMo convention, the
    same state dec.embed.0 dumps; a blank-token embedding start produces
    the same transcript on the oracle clip but is NOT what the C++ does).

    Advance rule (verbatim from src/arch/parakeet/decoder.cpp TDT emit
    block, which documents itself as matching the NeMo reference):

        step += duration
        new_symbols += 1
        if duration != 0:
            new_symbols = 0
        elif max_symbols and new_symbols >= max_symbols:
            step += 1; new_symbols = 0
        elif is_blank and max_symbols:
            step += 1; new_symbols = 0   # fast-forward identical no-ops

    Non-blank dur-0 tokens therefore emit without advancing (TDT allows
    several tokens per frame), capped at max_symbols per frame; a
    blank dur-0 advances one frame immediately. On the oracle clip this
    is transcript-identical to the HF generation mixin's rule (verified
    by probe); the C++/NeMo rule is used so the reference transcript can
    never disagree with the C++ loop for loop-structure reasons alone.
    """
    import torch

    with torch.inference_mode():
        t_enc = enc_proj.shape[1]
        hidden = None
        cell = None
        dec_out = None
        dirty = True
        last_token = -1
        step = 0
        new_symbols = 0
        tokens: list[int] = []
        max_iters = 16 * t_enc + 1024
        it = 0
        while step < t_enc and it < max_iters:
            it += 1
            if dirty:
                if last_token < 0:
                    x = torch.zeros(1, 1, model.config.decoder_hidden_size)
                else:
                    x = model.decoder.embedding(torch.tensor([[last_token]]))
                lstm_out, (hidden, cell) = model.decoder.lstm(
                    x, (hidden, cell) if hidden is not None else None
                )
                dec_out = model.decoder.decoder_projector(lstm_out)[:, -1, :]
                dirty = False

            logits = model.joint(
                encoder_hidden_states=enc_proj[:, step, None, :],
                decoder_hidden_states=dec_out[:, None, :],
            ).squeeze(1)
            lp = torch.log_softmax(logits, dim=-1)[0]
            token = int(lp[:vocab].argmax())
            duration = durations[int(lp[vocab:].argmax())]

            is_blank = token == blank
            if not is_blank:
                tokens.append(token)
                last_token = token
                dirty = True

            step += duration
            new_symbols += 1
            if duration != 0:
                new_symbols = 0
            elif max_symbols > 0 and new_symbols >= max_symbols:
                step += 1
                new_symbols = 0
            elif is_blank and max_symbols > 0:
                step += 1
                new_symbols = 0

        if it >= max_iters:
            print(
                f"warning: greedy decode stopped at max_iters={max_iters} "
                f"(step={step}/{t_enc}, tokens={len(tokens)})",
                file=sys.stderr,
            )
        return tokens


# ---------------------------------------------------------------------------
# Subcommands
# ---------------------------------------------------------------------------

def cmd_encoder(args: argparse.Namespace) -> int:
    """Dump encoder intermediates: mel, pre_encode, pos_emb, per-block, final."""
    configure_torch(args)

    model, info = load_model(args)
    audio_path = resolve_path(args.audio)
    out_dir = resolve_path(args.out)
    pcm, sr = load_audio(audio_path)
    if sr != 16000:
        print(f"error: audio sample rate is {sr}, expected 16000", file=sys.stderr)
        return 1

    n_layers = len(model.encoder.layers)
    print(f"audio: {audio_path.name} samples={pcm.size} sr={sr} arch=tdt")
    print(f"encoder: {n_layers} layers  weights={info['weights_format']}")

    source = make_source(
        args=args,
        audio_path=audio_path,
        n_samples=pcm.size,
        sample_rate=sr,
        arch="tdt",
        weights_format=info["weights_format"],
    )

    def dump(name: str, t, stage: str) -> None:
        a = to_np(t)
        print(
            f"  {name}: shape={a.shape} min={a.min():.4e} max={a.max():.4e} "
            f"mean={a.mean():.6e}"
        )
        write_tensor(name, a, stage, source, out_dir=out_dir)

    fe = build_feature_extractor(model)
    feats = featurize(fe, pcm, sr)
    print(f"features: {tuple(feats.shape)}")

    block_indices = select_blocks(model, args.blocks)
    sub_blocks = sorted(set(args.sub_blocks or []))
    print(f"dumping blocks: {block_indices}  sub-blocks: {sub_blocks}")
    intermediates, hooks, restore_cbs = capture_intermediates(
        model, block_indices, sub_block_indices=sub_blocks
    )
    try:
        last_hidden = run_encoder(model, feats)
    finally:
        for h in hooks:
            h.remove()
        for layer, original_forward in restore_cbs:
            layer.forward = original_forward

    dump("enc.mel.in", feats[0].transpose(0, 1), "frontend.mel.norm")
    dump("enc.pre_encode.out", intermediates["enc.pre_encode.out"], "encoder.pre_encode")
    dump("enc.pos_emb", intermediates["enc.pos_emb"], "encoder.pos_emb")

    for i in block_indices:
        key = f"enc.block.{i}.out"
        if key in intermediates:
            dump(key, intermediates[key], f"encoder.block{i}.out")

    # Sub-block intermediates (C++ observer points), opt-in only: the C++
    # emits them under TRANSCRIBE_DUMP_SUB_BLOCKS, and the tolerance file
    # has no entries for them, so default-off keeps both sides aligned.
    for i in sub_blocks:
        for tag in ("ff1", "attn", "conv", "ff2"):
            key = f"enc.block.{i}.{tag}"
            if key in intermediates:
                dump(key, intermediates[key], f"encoder.block{i}.{tag}")

    dump("enc.final", last_hidden, "encoder.final")
    return 0


def cmd_decode(args: argparse.Namespace) -> int:
    """Dump encoder output + decoder first step + joint + greedy transcript.

    Per-tensor (mirrors the C++ debug block in the TDT decode loop):
      dec.enc_out       encoder output (T_enc, d_model)
      dec.embed.0       predictor input at step 0 (zeros)
      dec.lstm.<l>.h.0  per-layer LSTM hidden state after one step
      dec.lstm.<l>.c.0  per-layer LSTM cell state after one step
      dec.joint.0       log-softmax joint output for (encoder frame 0,
                        predictor start state)
    """
    configure_torch(args)
    import torch

    model, info = load_model(args)
    tokenizer = load_tokenizer(args)
    audio_path = resolve_path(args.audio)
    out_dir = resolve_path(args.out)
    pcm, sr = load_audio(audio_path)
    if sr != 16000:
        print(f"error: audio sample rate is {sr}, expected 16000", file=sys.stderr)
        return 1

    print(f"audio: {audio_path.name} samples={pcm.size} sr={sr} arch=tdt")
    print(f"weights={info['weights_format']}")

    source = make_source(
        args=args,
        audio_path=audio_path,
        n_samples=pcm.size,
        sample_rate=sr,
        arch="tdt",
        weights_format=info["weights_format"],
    )

    def dump(name: str, t, stage: str) -> None:
        a = to_np(t)
        print(
            f"  {name}: shape={a.shape} min={a.min():.4e} max={a.max():.4e} "
            f"mean={a.mean():.6e}"
        )
        write_tensor(name, a, stage, source, out_dir=out_dir)

    fe = build_feature_extractor(model)
    feats = featurize(fe, pcm, sr)
    print(f"features: {tuple(feats.shape)}")

    cfg = model.config
    pred_hidden = cfg.decoder_hidden_size
    vocab = cfg.vocab_size
    blank = int(cfg.blank_token_id)
    durations = [int(d) for d in cfg.durations]
    max_symbols = int(cfg.max_symbols_per_step)
    n_layers = int(cfg.num_decoder_layers)
    print(
        f"decoder: hidden={pred_hidden} layers={n_layers} vocab={vocab} "
        f"blank={blank} durations={durations} max_symbols={max_symbols}"
    )

    with torch.inference_mode():
        last_hidden = run_encoder(model, feats)
        dump("dec.enc_out", last_hidden, "decoder.enc_out")

        # First-step predictor: input zeros -> LSTM from zero state.
        dump("dec.embed.0", np.zeros((pred_hidden,), dtype=np.float32), "decoder.embed")
        x0 = torch.zeros(1, 1, pred_hidden)
        lstm_out, (h_all, c_all) = model.decoder.lstm(x0, None)
        for layer in range(h_all.shape[0]):
            dump(f"dec.lstm.{layer}.h.0", h_all[layer], "decoder.lstm")
            dump(f"dec.lstm.{layer}.c.0", c_all[layer], "decoder.lstm")

        dec_out = model.decoder.decoder_projector(lstm_out)[:, -1, :]  # (1, H)
        enc_proj = model.encoder_projector(last_hidden)  # (1, T, H)
        logits = model.joint(
            encoder_hidden_states=enc_proj[:, 0:1, :],
            decoder_hidden_states=dec_out[:, None, :],
        ).squeeze(1)  # (1, vocab + n_durations)
        joint_lp = torch.log_softmax(logits, dim=-1)[0]
        dump("dec.joint.0", joint_lp, "decoder.joint")

    if not args.skip_transcript:
        print("\n  Running greedy TDT transcription...")
        tokens = greedy_tdt(
            model,
            enc_proj,
            blank=blank,
            vocab=vocab,
            durations=durations,
            max_symbols=max_symbols,
        )
        text = tokenizer.decode(tokens, skip_special_tokens=True)
        print(f"  Transcription: {text}")
        write_transcript(out_dir, text, source=source, tokens=tokens)

    return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def add_common_args(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--model",
        required=True,
        help="HF repo id (e.g. moondream/parakeet-ultra) or local model directory",
    )
    p.add_argument("--audio", required=True, help="16 kHz mono wav file")
    p.add_argument("--out", required=True, help="Output directory")
    p.add_argument(
        "--torch-threads",
        type=int,
        default=1,
        help="Torch intra-op threads for deterministic dumps (default: 1)",
    )
    p.add_argument(
        "--language",
        default="en",
        help=(
            "Target language tag; accepted for validate.py parity. The "
            "moondream TDT checkpoints take no language prompt (no prompt "
            "dictionary in config.json), so this is unused."
        ),
    )
    p.add_argument(
        "--revision",
        default=None,
        help=(
            "HF revision to pin for hub downloads. Ignored when --model "
            "resolves to a local directory."
        ),
    )


def main() -> int:
    p = argparse.ArgumentParser(
        description=(
            "Dump Parakeet reference tensors from HuggingFace transformers "
            "(moondream/parakeet-*)."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    ep = sub.add_parser("encoder", help="Dump encoder intermediates")
    add_common_args(ep)
    ep.add_argument(
        "--blocks",
        type=int,
        nargs="*",
        default=None,
        help=(
            "Block indices to dump. Default: {0, n_layers/2, n_layers-1} "
            "based on actual depth."
        ),
    )
    ep.add_argument(
        "--sub-blocks",
        type=int,
        nargs="*",
        default=None,
        help=(
            "Block indices for which to ALSO dump sub-block intermediates "
            "(after FF1, attn, conv, FF2). Matches the C++ BlockObserver "
            "tags (TRANSCRIBE_DUMP_SUB_BLOCKS). Default: empty."
        ),
    )
    ep.set_defaults(func=cmd_encoder)

    dp = sub.add_parser(
        "decode",
        help="Dump encoder + decoder first step + joint + transcript",
    )
    add_common_args(dp)
    dp.add_argument(
        "--skip-transcript",
        action="store_true",
        help="Only dump tensors; do not run greedy transcription",
    )
    dp.set_defaults(func=cmd_decode)

    args = p.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
