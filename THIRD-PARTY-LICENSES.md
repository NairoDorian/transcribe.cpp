# Third-party licenses

transcribe.cpp is MIT-licensed (see [`LICENSE`](LICENSE)). It vendors and links
the third-party components below, each under its own permissive license. The
authoritative license text for each ships in-tree at the path noted; copies are
reproduced here for convenience.

| Component | License | Vendored at | Pin |
|-----------|---------|-------------|-----|
| ggml      | MIT     | [`ggml/LICENSE`](ggml/LICENSE) | see [`ggml/UPSTREAM`](ggml/UPSTREAM) |
| Silero VAD | MIT | [`src/third_party/silero_vad/LICENSE`](src/third_party/silero_vad/LICENSE) | see [`src/third_party/silero_vad/UPSTREAM`](src/third_party/silero_vad/UPSTREAM) |
| miniz     | MIT     | [`src/third_party/miniz/LICENSE`](src/third_party/miniz/LICENSE) | see [`src/third_party/miniz/UPSTREAM`](src/third_party/miniz/UPSTREAM) |
| cJSON     | MIT     | [`src/third_party/cJSON/LICENSE`](src/third_party/cJSON/LICENSE) | vendored `cJSON.c` / `cJSON.h` |
| Earshot VAD (network weights, front-end tables) | MIT (dual MIT OR Apache-2.0 upstream; used under MIT) | [`licenses/earshot-MIT.txt`](licenses/earshot-MIT.txt) | pykeio/earshot 1.2.2 (`src/earshot-weights.inl`, `src/transcribe-earshot-*`) |
| RNNoise model and algorithm, via nnnoiseless | BSD-3-Clause | [`licenses/rnnoise-nnnoiseless-BSD-3-Clause.txt`](licenses/rnnoise-nnnoiseless-BSD-3-Clause.txt) | nnnoiseless 0.5.2 (`src/rnnoise-weights.inl`, `src/transcribe-rnnoise.*`) |
| audio.cpp (Earshot SIMD kernels, graph-optimizer subset) | Apache-2.0 | [`licenses/audio-cpp-Apache-2.0.txt`](licenses/audio-cpp-Apache-2.0.txt) | 0xShug0/audio.cpp #1095 / #1102 (`src/transcribe-earshot-engine.cpp`, `src/transcribe-earshot-kernels.inc`, `src/transcribe-graph-opt.h`) |

Prebuilt artifacts (the Swift xcframework, the native Python wheels, and the
npm platform packages) carry these same texts alongside the binaries — see each
binding's packaging for the bundled `LICENSE.ggml` / `LICENSE.miniz` / `LICENSE.silero_vad` files.

Conversion-time tooling also includes `scripts/lib/phonon.py`, adapted from
Fermion Research's Apache-2.0 container reader at
`FermionResearch/Phonon-2@7e153e4054c0a10db6d47d6fff5c062d4c77c154`.
Copyright 2026 Fermion Research. Modifications add strict record validation,
pinned artifact handling and F32 conversion. Its license text is
[`scripts/lib/phonon-LICENSE-Apache-2.0.txt`](scripts/lib/phonon-LICENSE-Apache-2.0.txt).
It is not linked into native binaries. Model-weight attribution and licenses
are separate; see [`docs/models/phonon-2.md`](docs/models/phonon-2.md).

---

## ggml

The tensor library underlying transcribe.cpp. Vendored under `ggml/`.

```
MIT License

Copyright (c) 2023-2026 The ggml authors

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

---

## miniz

A single-file deflate/inflate (zlib-subset) codec. transcribe.cpp uses it only
for Whisper's temperature-fallback compression-ratio heuristic; it replaces the
previous system-zlib dependency. Vendored under `src/third_party/miniz/`.

miniz is MIT-licensed. (The comment block at the top of the amalgamated
`miniz.c` / `miniz.h` still reads "public domain" — a stale artifact from the
project's pre-relicensing history; upstream's current and authoritative license
is the MIT text below, also shipped at `src/third_party/miniz/LICENSE`.)

```
Copyright 2013-2014 RAD Game Tools and Valve Software
Copyright 2010-2014 Rich Geldreich and Tenacious Software LLC

All Rights Reserved.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
```

---

## Silero VAD

The offline segmentation in `src/transcribe-vad.cpp` is
translated from `silero_vad/utils_vad.py`, pinned to silero-vad 6.2.3. The
authoritative text is `src/third_party/silero_vad/LICENSE`; native artifacts
include it as `LICENSE.silero_vad`.

```
MIT License

Copyright (c) 2020-present Silero Team

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

---

## Components added by this fork

These ship inside libtranscribe. Their license texts, copyright notices
included, are kept verbatim in the files linked below rather than reproduced
here. `cmake --install` copies each of them into
`share/licenses/transcribe/` as `LICENSE.<name>`, so they travel with every
installed binary.

- **cJSON** (MIT), [`src/third_party/cJSON/LICENSE`](src/third_party/cJSON/LICENSE),
  installed as `LICENSE.cjson`. A small JSON parser vendored under
  `src/third_party/cJSON/`; it reads the R2T2 package manifest.
- **Earshot VAD** (pykeio/earshot 1.2.2, dual MIT OR Apache-2.0, used here
  under MIT), [`licenses/earshot-MIT.txt`](licenses/earshot-MIT.txt), installed
  as `LICENSE.earshot`. The network behind `transcribe_vad_*`,
  `transcribe_vad_gate_*` and the stream fast path. Its weights are embedded as
  `src/earshot-weights.inl` and its front-end tables as
  `src/transcribe-earshot-tables.inl`.
- **RNNoise via nnnoiseless** (BSD-3-Clause),
  [`licenses/rnnoise-nnnoiseless-BSD-3-Clause.txt`](licenses/rnnoise-nnnoiseless-BSD-3-Clause.txt),
  installed as `LICENSE.rnnoise`. `transcribe_denoise_*` runs classic RNNoise
  (xiph/rnnoise). The algorithm in `src/transcribe-rnnoise.cpp` is ported from
  the nnnoiseless 0.5.2 Rust crate, itself a port of xiph/rnnoise, and the
  bundled model is embedded as `src/rnnoise-weights.inl`. BSD-3-Clause
  requires its notice in the documentation of binary redistributions; ship
  `LICENSE.rnnoise` with any build.
- **audio.cpp** (Apache-2.0),
  [`licenses/audio-cpp-Apache-2.0.txt`](licenses/audio-cpp-Apache-2.0.txt),
  installed as `LICENSE.audio-cpp`. Two pieces are ported or adapted from it:
  - the Earshot SIMD kernels, FFT, minGRU scan and CPU detection
    (`src/transcribe-earshot-engine.cpp`, `src/transcribe-earshot-kernels.inc`),
    from 0xShug0/audio.cpp #1095 / #1102;
  - a subset of its `graph_optimizer.cpp`, in `src/transcribe-graph-opt.h`.

  Each adapted file states its origin and the changes made.
