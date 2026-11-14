# Nemotron-3-Diarization port plan

Status: draft plan v3, 2026-10-04 (final revision: checkpoint/config intel
extracted from audio.cpp's own converter, our validate/DER harness read
end-to-end, every stage now carries exact commands and file anchors). Not
started. No gates cleared.

Goal: port `nvidia/Nemotron-3-Diarization` into this repo as a second
diarization-only family, so our speaker-attribution surface stops being capped
at the 4-speaker `diar_streaming_sortformer_4spk-v2.1` we ship today.

## Why now

- Our shipped diarizer is `nvidia/diar_streaming_sortformer_4spk-v2.1`
  (`docs/porting/families/sortformer.md`, ship gate passed 2026-07-22, AMI DER
  14.59% F32 / 14.23% F16 / 14.73% Q8_0). It is capped at **4 speakers**, and
  the family doc records `>4 speakers | OUT OF SCOPE - architectural 4-speaker
  cap` (`src/arch/sortformer/weights.h:27`: `max_speakers = 4; // hard
  architectural cap`). Every Sortformer variant degrades past 4 speakers.
- Nemotron-3-Diarization (released 2026-09-23) is Sortformer's direct
  successor from the same upstream team: same arrival-order output contract,
  **8 speakers**, streaming + offline, one checkpoint for every latency
  profile. It beats the exact baseline we ported:

  | Dataset | Baseline 4spk-v2.1 (30.4 s) | Nemotron-3 (30.4 s) | Nemotron-3 (0.32 s) |
  |---|---:|---:|---:|
  | DIHARD III full DER | 19.09 | **12.73** | **13.55** |
  | CALLHOME-Part2 DER | 10.32 | **9.10** | **11.32** |
  | NOTSOFAR1 SC DER | 30.49 | **11.00** | **14.53** |
  | AliMeeting Far DER | 13.69 | **10.47** | **11.60** |

  It is also faster (RTFx 1340 eager / 4385 compiled offline vs 874 / 1468).
- Independent evidence agrees: r/speechtech and r/LocalLLaMA threads from this
  week ("I compared diarization models on 15 clinical conversations: Nemotron 3,
  Pyannote, Sortformer, VibeVoice and Muse") and the Soniqo 132-recording
  multilingual set (Nemotron 3 MLX 10.98% / CoreML 11.20% vs Sortformer
  streaming 13.81% English, 22.32% Arabic). Clinical/diary audio is exactly the
  multi-party, overlapping, >4-speaker case our current cap fails.

## Checkpoint and config intel (settled by reading audio.cpp's converter)

Source `audio.cpp/tests/nemotron_3_diar/convert_gguf.py` reads the `.nemo`
directly, so these are file facts, not guesses:

- HF repo `nvidia/Nemotron-3-Diarization`, revision pinned by audio.cpp =
  **`723e19c601d99b7e58fba6a14e32153e0afe48d9`**; `.nemo` sha256
  `867c53f552998f772e5b5e5c082962ae85ee7ca5669c2bc17d7f615133d4e96d`
  (198,676,480 B). License **openmdw-1.1**, text at
  <https://openmdw.ai/license/1-1/> - reviewer's link, Stage 1 human field.
- `.nemo` layout: tar containing `model_config.yaml` + `model_weights.ckpt`
  (a torch checkpoint, `torch.load(..., weights_only=False)`). The state_dict
  also carries `preprocessor.featurizer.fb` (squeeze to `preprocessor.fb`) and
  `preprocessor.featurizer.window` (audio.cpp drops it; our `MelConfig` can
  consume a shipped window, `src/transcribe-mel.h:98-100`, but Sortformer's
  converter recomputes it - match whatever the Sortformer converter does, and
  note it in the manifest).
- **Tensor names are the NeMo state_dict names verbatim** (the converter copies
  every key unchanged), so the Stage-3 tensor table below is confirmed rather
  than inferred.
- `model_config.yaml` keys our converter must read (all referenced by
  audio.cpp's `model_config()`):

  | Block | Keys |
  |---|---|
  | top | `sample_rate=16000`, `max_num_of_spks=8`, `high_resolution`, `output_subsampling_factor` |
  | `preprocessor` | `features`, `sample_rate`, `n_fft`, `window_size`, `window_stride`, `window`, `dither`, `normalize`, `preemph` (absent -> NeMo default 0.97) |
  | `encoder` | `feat_in`, `d_model=512`, `ff_expansion`, `n_heads=8`, `n_layers=31`, `subsampling="feature_stacking"`, `subsampling_factor=8`, `self_attention_model="rope"`, `pre_block_norm`, `qkv_bias`, `qk_norm`, `rope_base` (default 10000), `rotary_fraction` (default 1.0) |
  | `sortformer_modules` | `num_spks=8`, `tf_d_model=192`, **and the whole streaming block**: `spkcache_len`, `fifo_len`, `chunk_len`, `chunk_left_context`, `chunk_right_context`, `spkcache_update_period`, `spkcache_sil_frames_per_spk`, `pred_score_threshold`, `scores_boost_latest`, `sil_threshold`, `strong_boost_rate`, `weak_boost_rate`, `min_pos_scores_rate`, `max_index` |

  Caveats found in that converter: it **hardcodes** `layer_norm_eps = 1.0e-5`
  and `head.upsample_kernel_size = 3` instead of reading them, and defaults
  `rope_base`/`rotary_fraction`. Stage 1 must check the yaml for those four
  keys and read them if present - do not inherit the hardcodes silently.
  Note `qkv_bias` / `qk_norm` / `pre_block_norm` are *flags*: audio.cpp builds
  the layer as bias-less QKV, no QK-norm, pre-block norm on. Confirm the flag
  values at Stage 1; if `qk_norm=true` our graph changes shape.
- Reference dtype is **open until measured**: HF `model.safetensors` is
  396,954,592 B (~100M x F32) while audio.cpp's own artifacts are named
  `...-bf16.gguf`, so the `.nemo` state_dict is probably BF16. Stage 1
  measures the state_dict dtype; Stage 2 records the dtype NeMo's forward
  actually runs in; those two together pick the reference dtype (repo
  convention allows F32 / F16 / BF16 per the intake, AGENTS.md).
- NeMo revisions: audio.cpp validated against **NVIDIA-NeMo/Speech `cf724ac`**
  using `SortformerEncLabelModel.forward_streaming`, FP32, CPU
  (`docs/models/nemotron_3_diar.md`, Validation section), and against NVIDIA's
  NeMo-Speech.cpp `8c15060` (max |Δ| 1.1e-1, 2 flips - a port, not an oracle).
  Our `scripts/envs/sortformer/pyproject.toml` pins NeMo `@6967f48fda2a`
  (shared with parakeet; declared in
  `tests/golden/sortformer/...manifest.json -> reference.revision`) and that
  commit predates this checkpoint. Stage 2 decides: try `6967f48fda2a` first,
  fall back to `cf724ac`, and record whichever loads in the Nemotron golden
  manifest. Either way the pin lives only in the **new** env - existing envs
  stay untouched (env dep bumps are their own pass, AGENTS.md).

## What we already have (reuse verdict, file by file)

| Piece | Where | Verdict for Nemotron |
|---|---|---|
| AOSC speaker cache + FIFO host state machine | `src/arch/sortformer/stream.cpp:270-406` `compress_spkcache`, `:410-468` `streaming_update_sync` | **Verified equivalent** to the reference `AoscState::compress/update` (`audio.cpp .../streaming.cpp:287-439`): same log-odds score, `pred_score_threshold` clamp, `-inf` disable, `scores_boost_latest`, strong/weak top-k with `scale*log(0.5)` deltas, ATen-style tie-break, `max_index` sentinel, `+inf` silence pad rows, flat-index selection, lazy `spkcache_preds` seeding, identical `pop = max(period, C - fifo_len + F)` formula. **One delta**: silence rows. We fill `st.mean_sil_emb` accumulated online by `get_silence_profile` (`stream.cpp:142-180`); the reference fills the checkpoint tensor `sortformer_modules.learnable_sil_emb` (`assets.cpp:80-81`, `streaming.cpp:424-427`). Extract a shared `src/arch/diar/aosc.*` taking `const float *silence_emb`, or copy with the online profile compiled out. Prefer extract: one call site each. |
| AOSC parameters | `src/arch/sortformer/sortformer.h:81-109` `SortformerStreamParams` | Already carries every knob the yaml ships (14 non-cadence fields). Our converter only writes 4 (`scripts/convert-sortformer.py:253-256`); Nemotron ships **all** as `stt.nemotron_diar.stream.*` so the GGUF is self-describing and profiles are data, not code. `sil_threshold` ships in the yaml but audio.cpp's AOSC never reads it - find out at Stage 2 what NeMo uses it for (silence profile? vestigial?) before dropping the field. |
| probs -> speaker segments | `src/arch/sortformer/model.cpp:678-701` `probs_to_speaker_segments` | `n_spk`- and `ms_per_frame`-generic; ABI (`t0_ms`/`t1_ms`/`speaker_id`) has no 4-speaker or 80 ms assumption. Gaps vs reference `Session::decode_turns` (`session.cpp:337-377`): threshold literal `0.5f` at `model.cpp:1010`, no `min_frames`/`pad_frames`, no per-turn mean confidence (`row.p` stays NaN), `>` where NeMo use `>=`. Generalize (params struct), do not fork. |
| Mel frontend | `src/transcribe-mel.h` / `transcribe-mel.cpp`, wired at `model.cpp:558-575` | Reusable unchanged: `MelConfig::pre_emphasis` (`transcribe-mel.h:41`), `normalize`, `nemo_seq_len_ceil=true` (`model.cpp:573`), KV `stt.frontend.pre_emphasis` read at `sortformer/weights.cpp:98` / written at `convert-sortformer.py:221`, shipped window support at `transcribe-mel.h:98-100`. The reference `NemoMelFrontend` is the same NeMo preprocessor. Do **not** copy its streaming mel machinery: `StreamScheduler::available_mel_frames` uses `(samples - n_fft/2)/hop` (`streaming.cpp:170-176`) and `finalize()` pads the tail with a geometric pre-emphasis decay (`streaming.cpp:253-270`) - both exist only because it computes mel incrementally. We compute the whole buffer once, already validated (`enc.mel.in` parity 1.6e-4 for Sortformer). |
| Diarizer-only family pattern, capability table, scope policy | `docs/porting/families/sortformer.md`, `src/arch/sortformer/capabilities.cpp:13` `apply_family_invariants` | Direct precedent; copy the OUT OF SCOPE rows and the invariant shape. |
| Converter scaffolding | `scripts/convert-sortformer.py` (loads via NeMo itself, `:171`), `scripts/lib/gguf_common.py`, `scripts/envs/sortformer` | Clone for Stage 3. The `enc.*` Conformer / NEST tables do **not** apply. |
| Oracle + DER harness | `scripts/diar/run_reference_sortformer_nemo.py` (has `PRESETS` + `_apply_preset` writing `m.sortformer_modules.*`, `:35-56`), `run_cpp_sortformer.py` (drives transcribe-cli with `TRANSCRIBE_SORTFORMER_STREAM_PRESET` + `TRANSCRIBE_DUMP_DIR`, then **NeMo's own `ts_vad_post_processing`** so post-processing is bit-identical), `score_der.py`, `postprocessing/diar_streaming_sortformer_4spk-v2_{dihard3-dev,callhome-part1}.yaml`, `ingest_ami.py` + `fetch_ami_forced_alignment.py` | Extend for Stage 2/7. The yaml field names in `_apply_preset` are exactly the yaml keys above, so the preset plumbing may carry over nearly as-is. AMI manifests are **not** in a fresh checkout: Stage 2/7 precondition is `uv run scripts/diar/ingest_ami.py --config ihm --split test` then `scripts/diar/fetch_ami_forced_alignment.py`. |
| Validation driver | `scripts/validate.py` (`ref`/`cpp`/`mel`/`compare`/`all`, `--family/--variant/--backend/--report`), env hooks `VALIDATE_SORTFORMER_PRESET`, `TRANSCRIBE_SORTFORMER_STREAM_PRESET`, `TRANSCRIBE_SORTFORMER_OFFLINE_DUMP` (`validate.py:412-421, 468-471`) | Nemotron needs its own three hooks at the same sites. Also `validate.py:384-386` only forwards `source_model.hf_revision` via `--revision` for a hard-coded family list - add `nemotron_diar` and make the new dumper accept `--revision`, or the Stage-1 pin is decorative. |
| Run-ext + kind registration | `src/arch/sortformer/model.cpp:1092-1156`, `docs/extension-kinds.md` | Registry's next slot is empty. Reserve FourCC **`NMDZ` = `0x5A444D4E`**, RUN slot, symbol `TRANSCRIBE_EXT_KIND_NEMOTRON_DIAR_PROFILE`, same PR as `include/transcribe/nemotron_diar.h`. |
| Family build registration | `CMakeLists.txt:55` (family list), `:196-200` (`_srcs_*`), `:398` (`TRANSCRIBE_ENABLE_ARCH_*`), `src/transcribe-arch.cpp:145` pattern; shared-source precedent `CMakeLists.txt:341-364` | Mechanical (Stage 4). |
| Catalog capability shape | `catalog/diar_streaming_sortformer_4spk-v2.1.json` (`diarize.max_speakers: 4`) | New entry with `max_speakers: 8`. |

**Fork vs. reuse decision: new family, not a Sortformer variant.** Our
Sortformer transformer hard-rejects `pre_ln=true`
(`src/arch/sortformer/weights.cpp:114-115`) and caps `max_speakers` at 4
(`sortformer/weights.h:27`). Nemotron is pre-LN + RoPE + 8 columns + 10 ms
output with a Linear pre-encode instead of the parakeet Conformer our
`conf::`/`pk::` path builds. Bending one family into two architectures behind
`tf_pre_ln`-style branches would touch a shipped, gated family for no shared
code - the only shared code is the AOSC host state (extract it) and the
frontend/segment helpers (already family-agnostic).

## Reference architecture, as built by audio.cpp

Config asserts at load (`assets.cpp:46-47`): `num_speakers == 8`,
`high_resolution == true`, `encoder.subsampling_factor == head.upsample_factor`
(both 8). Tensor names are NeMo state_dict keys.

| Stage | Reference tensors (exact names) | Op order (as built) |
|---|---|---|
| Frontend | `preprocessor.fb` `[128, 257]` | NeMo mel, then feature stacking x8 -> `[enc_frames, 1024]` (host-side) |
| Pre-encode | `encoder.pre_encode.proj` `[1024, 512]`, **no bias** | single Linear (`encoder.cpp:115-147`) |
| Embed norm | `encoder.embed_norm` | LayerNorm **before** layer 0 (`encoder.cpp:186-188`), gated on `pre_block_norm` |
| 31 layers | `encoder.layers.{i}.norm1`, `.attn.w_qkv` `[1536,512]` **no bias**, `.attn.out_proj` `[512,512]`+bias, `.norm2`, `.ffn.net.0`+bias, `.ffn.net.3`+bias | **pre-LN**: norm1 -> qkv -> split heads -> RoPE(cos/sin) -> non-causal attention (additive mask `-10000`) -> out_proj -> add; norm2 -> Linear -> **exact GELU (erf)** -> Linear -> add (`encoder.cpp:71-92`) |
| Final norm | `encoder.final_norm` | LayerNorm after the stack (`encoder.cpp:194-196`) |
| Proj + mask | `sortformer_modules.encoder_proj` `[192, 512]`+bias | Linear, then `mul` by `frame_mask` so padded frames are zero *before* the conv (`encoder.cpp:197-202`) |
| Upsample | `sortformer_modules.subpixel_upsample.weight` `[1536,192,3]`, `...subpixel_upsample.bias` `[1536]` | Conv1d k=3, stride 1, pad 1 -> transpose -> reshape `[T*8, 192]` = channel-to-time interleave (PixelShuffle-1D: sub-frame s takes output channels `s*192..`) (`encoder.cpp:203-219`) |
| Head | `sortformer_modules.first_hidden_to_hidden` `[192,192]`+bias, `sortformer_modules.single_hidden_to_spks` `[8,192]`+bias | ReLU -> Linear -> ReLU -> Linear -> **sigmoid** (`encoder.cpp:220-226`) |
| Silence rows | `sortformer_modules.learnable_sil_emb` `[512]` | checkpoint vector used verbatim for disabled AOSC rows (`assets.cpp:80-81`, `streaming.cpp:424-427`) |

Geometry: encoder frame = 80 ms (subsampling 8), output = 10 ms
(`upsample_factor` 8; turn spans use `kOutputHopSamples = 160`), AOSC consumes
80 ms-averaged probs while the emitted timeline is 10 ms
(`session.cpp:300-334`). Preferred chunk = `chunk_len * 1280` samples
(`session.cpp:511`). Load validates `spkcache_len >= 16`
(`streaming.cpp:101-105`).

Official profiles (`streaming.cpp:49-107`; buffer latency =
`chunk_len + chunk_right_context`, all lengths in **80 ms encoder frames**):

| Profile | Buffer latency | spkcache | fifo | chunk | right ctx | update period |
|---|---:|---:|---:|---:|---:|---:|
| `very_high` (default) | 30.4 s | 264 | 40 | 340 | 40 | 300 |
| `low` | 1.04 s | 264 | 264 | 9 | 4 | 222 |
| `very_low` | 0.64 s | 264 | 264 | 6 | 2 | 222 |
| `ultra_low` | 0.32 s | 264 | 264 | 3 | 1 | 222 |
| `asr_la13` / `la6` / `la3` / `la0` | 1.12 / 0.56 / 0.32 / 0.08 s | 264 | 264 | 14 / 7 / 4 / 1 | 0 | 222 |
| `custom` | per-field options | | | | | |

Max packed window = `spkcache + fifo + chunk` = 644 encoder frames (51.5 s) at
`very_high`, 537 at `low` - this drives arena sizing in Stage 4. The profiles
are visibly Sortformer's presets evolved (`very_high` reuses chunk 340 / right
40 / fifo 40 / period 300 from `run_reference_sortformer_nemo.py:35-39`, with
spkcache 188 -> 264), which is why the preset plumbing is likely to carry over.

## What is genuinely new in our C++

- **Pre-LN + RoPE block.** `tf_block` (`model.cpp:277-318`) is post-LN, no
  positional encoding, separate biased Q/K/V, **ReLU** FFN, and NeMo's
  `1/sqrt(sqrt(head_dim))` pre-scale. Nemotron needs pre-LN, fused bias-less
  QKV, split-half RoPE, **exact GELU**, standard `1/sqrt(head_dim)` scale. Write
  a new block; do not generalize `tf_block` (scale/activation/order differ in
  four places). Stage 2 must pin the scale convention by dumping an
  attention-score tensor - trust neither implementation.
- **RoPE position base.** The reference builds cos/sin tables indexed by
  `t = 0..T-1` of the *packed window* (`encoder.cpp:179-184, 239-244,
  248-270`), i.e. positions reset at every `spkcache|fifo|chunk` window and are
  **not** absolute stream time. Highest-risk silent-parity trap; confirm with a
  per-layer hidden dumped at two different window offsets.
- **Pre-encode stacking + Linear.** No dw-striding subsample: host stacks 8
  mel frames, then bias-less `1024 -> 512` Linear. The whole
  `conf::build_pre_encode` / `sf_pre_view` / `sf_conv_policy` path
  (`model.cpp:328-454`) is unused.
- **Masks as graph inputs.** `attention_mask` `[B,1,T,T]` with `-10000` on
  `key >= length`, and `frame_mask` `[B,T,1]` multiplied after `encoder_proj`
  (`encoder.cpp:272-292, 197-202`). Our sortformer masks nothing ("all frames
  valid in sync mode", `model.cpp:514-515`) - Nemotron needs both because the
  packed window is zero-padded to profile capacity (`session.cpp:279-285`).
- **Conv upsample head** between `encoder_proj` and the ReLU
  (`model.cpp:516` has proj -> ReLU directly).
- **Arena.** Our per-chunk compute context is `32 MB` (`model.cpp:869`) with
  `ggml_backend_sched_new(..., graph_size=8192)` (`model.cpp:709`). A
  644-frame window's intermediates alone run several hundred MB (31 layers x
  [qkv, attn out, ffn of `intermediate_size`, residuals]); audio.cpp defaults
  to a **1 GB** arena (`session.cpp:21-22`, overridable
  `nemotron_3_diar.graph_arena_mb`). Budget 512 MB - 1 GB, or adopt
  `ggml_gallocr` sizing.
- **Decode options.** `min_frames` / `pad_frames` (10 ms units), mean
  confidence, `>=` threshold, and the streaming emit rule: only *closed* turns,
  deduped by `(start, end, speaker)` (`session.cpp:545-566`).
- **Attention lowering.** The reference exposes `auto|flash|eager`
  (`docs/models/nemotron_3_diar.md` session options); we already have the
  equivalent env plumbing via `transcribe::flash::apply_env_overrides`
  (`model.cpp:491-493`). No new surface; validation runs CPU/eager as usual.

## Scope decision (must be settled in Stage 1)

The family doc's diarization scope policy admits a diarizer only if it
(a) feeds an in-repo transcription pipeline, or (b) reuses an encoder family we
already maintain. Nemotron's 31-layer RoPE encoder is new, so today it
satisfies neither clause. Options, in preference order:

1. **Wire it as the parakeet multitalker diarizer.** `spk_supervision='diar'`
   currently embeds Sortformer (`src/arch/parakeet/model.cpp:609-676`,
   `stt.parakeet.diarizer.*` KVs). Nemotron's `asr_la*` profiles exist precisely
   for ASR lookahead alignment (the reference documents them against NeMo's
   multitalker pipeline; `asr_la0` = one 80 ms frame per step), and it exports
   a 10 ms speaker-probability timeline. An alternate, opt-in provider
   satisfies clause (a).
2. **Generalize clause (a)** to "diarizer that improves an in-repo
   speaker-attribution consumer" (CLI `--diarize`, `transcribe_speaker_segment`
   ABI) - a policy edit in the family doc, needs explicit user sign-off.
3. Ship it as a standalone diarization family anyway - not permitted by the
   current policy; do not do this silently.

Recommendation: plan for (1), run Stage 1 intake now, let the scope sign-off
gate block Stage 4. The parakeet bundle consumes a `T x 4` Sortformer tensor in
its layer-0 speaker kernel - 8-column 10 ms interop is a *consumer-side* change
and stays OUT OF SCOPE in the capability table with that reason.

## Stages

Mapping onto `.agents/skills/porting-*`. Independent, in order, one gate each.

### Stage 1 - intake
- `uv run scripts/intake.py inspect --repo nvidia/Nemotron-3-Diarization
  --family nemotron_diar --variant nemotron-3-diarization --out
  reports/porting/nemotron_diar/nemotron-3-diarization/intake.json`
  (pin `hf_revision` `723e19c601d99b7e58fba6a14e32153e0afe48d9` unless a newer
  revision is deliberate and re-verified).
- Carry forward from `reports/porting/sortformer/diar_streaming_sortformer_4spk-v2.1/intake.json`:
  `reference_framework`, `architecture_pattern = encoder-diarizer`,
  `known_risks`, and the frontend block (should match Sortformer's exactly -
  verify, don't assume; Sortformer had to resolve `preemph` at Stage 2, see
  its `intake.json:216,223`).
- **One measurement script in the new env** (uv-run, no bare python): open the
  `.nemo` (tarfile + `torch.load`, mirroring
  `audio.cpp .../convert_gguf.py:34-66`), then dump:
  1. `model_config.yaml` verbatim -> resolves every hparam above, plus the
     presence of `layer_norm_eps` / `upsample_kernel_size` / `rope_base` /
     `rotary_fraction` that audio.cpp hardcodes or defaults.
  2. `state_dict` key list + per-key dtype histogram -> confirms the tensor
     table is verbatim and settles **reference dtype** (F32 staging vs native
     BF16).
  3. Flag values: `encoder.qkv_bias`, `encoder.qk_norm`,
     `encoder.pre_block_norm`, `encoder.ff_expansion`.
- Human fields: acceptance dataset (AMI + FA RTTMs + DIHARD-3 post-processing,
  our Sortformer acceptance set; LibriSpeech exemption unchanged), license
  review of openmdw-1.1 (link above), scope decision, FourCC `NMDZ`, family
  key `nemotron_diar`.
- Draft `docs/porting/families/nemotron_diar.md` capability table (8-speaker /
  10 ms / profile rows, OUT OF SCOPE entries with reasons).
- Seed `tests/golden/nemotron_diar/<variant>.manifest.json`: frontend block
  mirrors `tests/golden/sortformer/...manifest.json`, `source_model.hf_revision`
  set, `reference.kind=nemo` with `revision` TBD at Stage 2,
  `diarization.max_speakers: 8`, `frame_period_seconds: 0.01`, and the profile
  table above.
- **Gate A.**

### Stage 2 - oracle
- New env `scripts/envs/nemotron_diar` cloned from `scripts/envs/sortformer`;
  choose the NeMo pin by loading the checkpoint (try `6967f48fda2a`, fall back
  to `cf724ac`) and record it in the manifest's `reference.revision`. Do not
  touch existing envs.
- Extend `scripts/diar/run_reference_sortformer_nemo.py`: `_apply_preset`
  (`:45-56`) already writes `m.sortformer_modules.{chunk_len,fifo_len,
  spkcache_len,spkcache_update_period,chunk_right_context}` - the Nemotron yaml
  uses the same names, so reuse it and add the Nemotron profile names
  (`very_high`/`low`/`very_low`/`ultra_low`/`asr_la*`) as `PRESETS` entries, or
  probe NeMo's `configure_diar_streaming` / `forward_streaming` if the pinned
  NeMo exposes a profile API for this model. Keep `--limit` smoke support.
- Contract tensor dumps at a fixed window: `enc.mel.in` (frontend lock), pre-encode
  out, `embed_norm` out, per-layer hidden at layers 0/15/30 (or an
  attention-score dump to pin the scale convention), `final_norm` out,
  `encoder_proj` out, post-upsample `[T*8,192]`, head logits, final 10 ms probs,
  and one AOSC compress event (reuse the
  `TRANSCRIBE_SORTFORMER_COMPRESS_DUMP` pattern at `stream.cpp:380-400`).
- **Pins that must be answered by dump or NeMo source, not by reading audio.cpp:**
  RoPE position base (window-relative vs absolute), attention scale
  convention, silence-embedding source (`learnable_sil_emb` vs online mean -
  read NeMo `sortformer_modules.py` at the pinned rev), what consumes
  `sil_threshold`, mel frame count rule at the tail, `>=` vs `>` at threshold,
  and whether NeMo's *offline* path is the same chunked loop audio.cpp
  implements (`session.cpp:444-499`) or a full-context forward.
- Post-processing: save the yaml NeMo uses (expect it to be the existing
  `scripts/diar/postprocessing/diar_streaming_sortformer_4spk-v2_dihard3-dev.yaml`;
  if the checkpoint ships its own, add
  `scripts/diar/postprocessing/nemotron-3-diarization_dihard3-dev.yaml`).
- Provisional tolerances (finalize after the first parity run): max |Δprob|
  <= 2e-3, mean <= 5e-5, frame flips at 0.5 <= 1 per 20 s (audio.cpp's
  measured CPU-F32 band: 8.7e-4 / 2.1e-5 / 0 at `very_high`; 1.6e-3 / 2.6e-5 /
  1 at `low`), DER gate `reference + 0.01` as in the Sortformer port.
- Outputs: filled golden manifest, `tests/tolerances/nemotron_diar.json`,
  `reports/wer/<variant>-REF.*.score.json`.

### Stage 3 - convert
- New `scripts/convert-nemotron-diar.py` (pattern: `convert-sortformer.py`,
  which loads the model **through NeMo** at `:171` - prefer that path so yaml +
  state_dict come from the pinned NeMo; keep audio.cpp's tarfile path as the
  fallback if NeMo cannot load it). Output: reference dtype only,
  `models/nemotron_diar/nemotron_diar-<REFDTYPE>.gguf`.
- Tensor table (verbatim state_dict names):
  `encoder.pre_encode.proj`, `encoder.embed_norm`,
  `encoder.layers.{i}.{norm1,attn.w_qkv,attn.out_proj,norm2,ffn.net.0,ffn.net.3}`
  x31, `encoder.final_norm`,
  `sortformer_modules.{encoder_proj,subpixel_upsample,first_hidden_to_hidden,single_hidden_to_spks,learnable_sil_emb}`,
  `preprocessor.fb`.
- KVs: `stt.nemotron_diar.*` hparams (top-level + `encoder.*` incl.
  `ff_expansion`-derived `intermediate_size`, `qkv_bias`, `qk_norm`,
  `pre_block_norm`, `rope_base`, `rotary_fraction`, `layer_norm_eps`; `head.*`),
  `stt.frontend.*` (Sortformer block clone incl. `pre_emphasis`), and the
  **full** `stt.nemotron_diar.stream.*` block (14 non-cadence fields). Load-time
  asserts in `capabilities.cpp` mirror `assets.cpp:46-47` + `streaming.cpp:101-105`.
- Quant rule: keep `preprocessor.fb` F32 in every quant (the reference
  converter passes `--keep-type preprocessor.fb=f32`); confirm how
  `convert-sortformer.py` encodes the same rule and mirror it.
- **Gate B.**

### Stage 4 - C++
- New `src/arch/nemotron_diar/` - `model.cpp`, `stream.cpp`,
  `capabilities.cpp`, `weights.cpp`, `nemotron_diar.h` - registered in
  `CMakeLists.txt:55` + `:196` `_srcs_*` + `TRANSCRIBE_ENABLE_ARCH_NEMOTRON_DIAR`,
  and `src/transcribe-arch.cpp` (pattern at `:145`).
- Extract the shared AOSC host module (`src/arch/diar/aosc.*`) with an injected
  silence vector; point both families at it only if the diff stays readable
  (Sortformer keeps its online `get_silence_profile`). Otherwise copy and note
  the divergence in both file headers.
- Graphs: (A) stacking + pre-encode Linear; (B) embed_norm -> 31 pre-LN RoPE
  layers with `[B,1,T,T]` attention mask + `[B,T,1]` frame mask -> final_norm ->
  encoder_proj -> mask mul -> Conv1d subpixel upsample -> ReLU -> Linear -> ReLU
  -> Linear -> sigmoid. Drive them from a `run_diar_streaming_core` clone of
  `model.cpp:802-974` (host concat `[spkcache|fifo|chunk]`, per-chunk window
  with left/right context, AOSC update) with the profile capacity padding of
  `session.cpp:279-285`. Clone rather than parameterize: the Sortformer core is
  also driven by the parakeet multitalker bundle (`model.cpp:801-802`,
  `DiarStreamScratch`) and a second family's numerics must not flow through a
  shipped bundle path.
- Memory: size the arena from the 644-frame capacity (512 MB - 1 GB), not the
  inherited 32 MB; consider `ggml_gallocr` per audio.cpp commit `f5a55528`
  (allocate once; RoPE cos/sin marked graph input *and* output so they survive
  reallocation). Teardown lint: only `transcribe::safe_*`
  (`tests/lint_teardown.cmake`).
- Debug hooks, mirroring Sortformer: `TRANSCRIBE_NEMOTRON_DIAR_OFFLINE_DUMP`,
  `TRANSCRIBE_NEMOTRON_DIAR_STREAM_PROFILE`, `VALIDATE_NEMOTRON_DIAR_PROFILE`,
  wired at `scripts/validate.py:412-421` and `:468-471`; add `nemotron_diar` to
  the `--revision` family list at `validate.py:385`.
- Public surface: `include/transcribe/nemotron_diar.h`, RUN-slot ext registered
  as `NMDZ`. Fields: profile enum named after **this** model's profiles
  (`very_high`/`low`/`very_low`/`ultra_low`, `asr_la*` deferred to the
  multitalker wiring) - do not reuse Sortformer's
  `VERY_HIGH_LATENCY/HIGH/LOW` vocabulary (`model.cpp:1115-1121`), the sets are
  not isomorphic - plus `threshold`, `min_frames`, `pad_frames`.
- Decode: generalize `probs_to_speaker_segments` (threshold/min/pad/confidence,
  10 ms `ms_per_frame`) rather than a second copy with literals.
- Resolve the capability table; run `uv run scripts/validate.py all --family
  nemotron_diar --variant nemotron-3-diarization --backend cpu --report` + DER
  parity. **Stage 4 gate: ref-dtype AMI DER within `reference + 0.01` and prob
  parity inside the Stage-2 band.**

### Stage 5 - quants
- From the reference GGUF: F16, Q8_0, Q6_K, Q5_K_M, Q4_K_M (same matrix
  convention as Sortformer's shipped three, widened only if Stage-4 numerics
  allow); `preprocessor.fb` stays F32. audio.cpp's data point: on Nemotron,
  CUDA Q8_0 sat at 4.3% vs 4.2% CPU F32 (no cliff) - unlike Sortformer, where
  uniform F16/Q8_0 was rejected for long-stream drift. Re-verify on our
  harness; do not inherit their conclusion.
- CLI smoke per file; push to a private HF repo; preliminary 512-utterance DER.

### Stage 6 - bench
- Publication rigs as always (Apple M4 Max, AMD Ryzen 7 PRO 4750U) plus the
  CUDA box. Report RTFx **per latency profile** - profiles differ by >10x in
  compute/audio-seconds (`asr_la0` is one 80 ms frame per step and is slow),
  so this is a matrix, not a single number.

### Stage 7 - DER sweep (the "WER" stage for a diarizer)
- Precondition on a fresh checkout: `uv run scripts/diar/ingest_ami.py --config
  ihm --split test` then `uv run scripts/diar/fetch_ami_forced_alignment.py`
  (the `-fa` manifests are generated, not committed).
- Per profile and per shipped dtype:
  `uv run --project scripts/envs/nemotron_diar scripts/diar/run_cpp_<fam>.py
  --manifest samples/diar/ami-ihm-test-fa.manifest.jsonl --gguf models/... --profile
  <profile> --postprocessing-yaml scripts/diar/postprocessing/<oracle yaml>
  --pred-dir ... --out ...` (clone of `run_cpp_sortformer.py`, which post-processes
  the dumped `diar.probs` with **NeMo's own `ts_vad_post_processing`** so only the
  probabilities can differ), then `uv run scripts/diar/score_der.py --manifest
  samples/diar/ami-ihm-test-fa.manifest.jsonl --pred-dir ... --out ...`.
- Hard-gated for ref-dtype, human-reviewed for quants. Consider DIHARD-III and
  CALLHOME-Part2 as secondary reads (the upstream numbers above are what users
  compare against; a callhome postprocessing yaml already exists for
  Sortformer).

### Stage 8 - ship
- `docs/porting/families/nemotron_diar.md` filled, model card, HF card YAML,
  catalog entry `catalog/nemotron_diar*.json`
  (`capabilities.diarize.max_speakers: 8`), rendered README upload. Re-check
  `hf_revision` (upstream moves fast).

## Risks / open questions

1. **License** openmdw-1.1 unreviewed here; blocks Stage 8 at the latest.
2. **Scope policy** - see above; blocks Stage 4 until signed off.
3. **NeMo pin may not support this checkpoint** (released 2026-09-23; our pin
   is the Sortformer/parakeet one). Resolution path is in Stage 2: try
   `6967f48fda2a`, fall back to audio.cpp's validated `cf724ac`, record it.
   If neither loads the model, Stage 2 cannot start - surface this before Gate A.
4. **Parity traps** (each gets an explicit Stage-2 dump): RoPE position base,
   attention scale, silence-embedding source, `sil_threshold` consumer, tail
   frame count, `>=` vs `>`, offline = chunked vs full-context, `qkv_bias`/
   `qk_norm` flag values, `layer_norm_eps`/`upsample_kernel_size` provenance.
5. **Reference dtype** unknown until Stage 1 measures the state_dict (F32
   staging file vs BF16-native artifacts); Stage 2 must also record the dtype
   NeMo's forward runs in, or the tolerance band is meaningless.
6. **Long-stream drift**: 1e-5 input noise moves output ~1.1% DER-equivalent.
   Gates must be band-based, not exact-match; no zero-tolerance flip gate.
7. **Backend memory**: 31-layer graph over 644-frame windows needs a
   ~1 GB-class arena (vs our inherited 32 MB/chunk). Profile on the 8 GB CUDA
   box early; this is what audio.cpp's `f5a55528` fix was about.
8. **>4 speakers in the ABI**: `transcribe_speaker_segment` is generic, but
   downstream consumers (CLI formatting, moss/granite marker mapping,
   Handy_V2) were written against 4. Audit at Stage 4.
9. **Multitalker interop**: parakeet's layer-0 speaker kernel consumes a
   `T x 4` Sortformer tensor; an 8-column 10 ms timeline is a consumer-side
   change. OUT OF SCOPE with that reason.

## First three actions

1. `uv run scripts/intake.py inspect --repo nvidia/Nemotron-3-Diarization
   --family nemotron_diar --variant nemotron-3-diarization ...` (Gate A draft).
2. Write the Stage-1 measurement script and run it against
   `Nemotron-3-Diarization.nemo`: dumps `model_config.yaml`, state_dict keys +
   dtypes, and the four flag/float keys audio.cpp hardcodes - settles the
   tensor table, reference dtype, and hparams in one pass.
3. Settle the scope decision (wire as parakeet multitalker provider vs policy
   edit) - this decides whether Stage 4 can start - and sanity-check that the
   pinned NeMo can load the checkpoint (risk 3) before Gate A is signed.
