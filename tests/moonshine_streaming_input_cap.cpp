// moonshine_streaming_input_cap.cpp - real-model gated test of the
// moonshine_streaming input cap.
//
// The adapter adds pos_emb[frame] to every encoder frame from a table of
// dec_max_position_embeddings (4096) rows: 4096 encoder frames = 81.92 s of
// audio per decode. Past it the model is undefined (the reference raises
// IndexError), and the port used to read the table out of bounds - a
// GGML_ASSERT abort on CPU, garbage on Metal. This pins the contract
// (docs/input-limits.md), on the default backend and on CPU:
//
//   1. capabilities.max_audio_ms reports the 81.92 s wall.
//   2. One-shot: exactly 4096 frames runs; 4097 is TRANSCRIBE_ERR_INPUT_TOO_LONG,
//      rejected before encoding, with no transcript and no abort.
//   3. transcribe_run_batch: an over-cap row is a per-utterance
//      INPUT_TOO_LONG while the other rows complete and the whole call
//      returns OK. On a GPU backend this goes through the batched path.
//   4. Streaming past the cap: every feed and finalize return OK, the stream
//      is flagged transcribe_was_truncated(), audio past the cap is ignored,
//      and the final transcript equals one-shot over the first 81.92 s
//      (byte for byte on the default backend; the invariants above on CPU).
//
// Gating:
//   - TRANSCRIBE_BUILD_REAL_MODEL_TESTS (CMake, default OFF) builds it.
//   - At runtime, TRANSCRIBE_MOONSHINE_STREAMING_TINY_GGUF points at a tiny
//     GGUF. If unset/missing (or a sample is missing), exits 77 ("skipped").

#include "transcribe.h"
#include "transcribe/moonshine_streaming.h"
#include "wav.h"

#include <sys/stat.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <string>
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

#define CHECK_EQ_INT(actual, expected)                                                                           \
    do {                                                                                                         \
        const long long _a = static_cast<long long>(actual);                                                     \
        const long long _e = static_cast<long long>(expected);                                                   \
        if (_a != _e) {                                                                                          \
            std::fprintf(stderr, "FAIL %s:%d: %s = %lld, expected %lld\n", __FILE__, __LINE__, #actual, _a, _e); \
            ++g_failures;                                                                                        \
        }                                                                                                        \
    } while (0)

// 4096 encoder frames x 4 embedder frames x 80 samples.
constexpr int k_cap_samples = 4096 * 4 * 80;

bool file_exists(const std::string & path) {
    struct stat st{};
    return ::stat(path.c_str(), &st) == 0;
}

bool load_sample(const std::string & name, std::vector<float> & pcm) {
    const std::string path = std::string(TRANSCRIBE_TEST_SAMPLES_DIR) + "/" + name;
    if (!file_exists(path)) {
        std::fprintf(stderr, "skipping: %s missing\n", path.c_str());
        return false;
    }
    std::string err;
    if (!transcribe_cli::load_wav_mono_16k(path, pcm, err) || pcm.empty()) {
        std::fprintf(stderr, "failed to load %s: %s\n", path.c_str(), err.c_str());
        return false;
    }
    return true;
}

std::string full_text(transcribe_session * s) {
    const char * t = transcribe_full_text(s);
    return t == nullptr ? std::string() : std::string(t);
}

// Stream `pcm` in 2 s chunks. Returns the final transcript; every step must
// succeed even past the cap.
std::string stream_all(transcribe_session * s, const std::vector<float> & pcm) {
    transcribe_run_params rp;
    transcribe_run_params_init(&rp);
    transcribe_stream_params sp;
    transcribe_stream_params_init(&sp);
    transcribe_moonshine_streaming_stream_ext ms;
    transcribe_moonshine_streaming_stream_ext_init(&ms);
    ms.min_decode_interval_ms = 1000;
    sp.family                 = &ms.ext;
    CHECK(transcribe_stream_begin(s, &rp, &sp) == TRANSCRIBE_OK);

    constexpr size_t k_chunk = 32000;
    for (size_t pos = 0; pos < pcm.size(); pos += k_chunk) {
        const size_t             take = std::min(k_chunk, pcm.size() - pos);
        transcribe_stream_update upd;
        transcribe_stream_update_init(&upd);
        CHECK(transcribe_stream_feed(s, pcm.data() + pos, static_cast<int>(take), &upd) == TRANSCRIBE_OK);
    }
    transcribe_stream_update fin;
    transcribe_stream_update_init(&fin);
    CHECK(transcribe_stream_finalize(s, &fin) == TRANSCRIBE_OK);
    return full_text(s);
}

void check_backend(const char *               model_path,
                   transcribe_backend_request backend,
                   const char *               label,
                   bool                       compare_text,
                   const std::vector<float> & pcm_short,
                   const std::vector<float> & pcm_long) {
    std::fprintf(stdout, "-- %s\n", label);
    transcribe_model_load_params mp;
    transcribe_model_load_params_init(&mp);
    mp.backend                      = backend;
    struct transcribe_model * model = nullptr;
    if (transcribe_model_load_file(model_path, &mp, &model) != TRANSCRIBE_OK) {
        std::fprintf(stderr, "FAIL %s: model load failed\n", label);
        ++g_failures;
        return;
    }

    // 1. Advertised limit.
    transcribe_capabilities caps;
    transcribe_capabilities_init(&caps);
    CHECK(transcribe_model_get_capabilities(model, &caps) == TRANSCRIBE_OK);
    CHECK_EQ_INT(caps.max_audio_ms, 81920);

    transcribe_session_params sp;
    transcribe_session_params_init(&sp);
    struct transcribe_session * s = nullptr;
    if (transcribe_session_init(model, &sp, &s) != TRANSCRIBE_OK) {
        std::fprintf(stderr, "FAIL %s: session init failed\n", label);
        ++g_failures;
        transcribe_model_free(model);
        return;
    }

    // 2. One-shot at and just past the cap. A single extra 80-sample frame
    //    (after padding) pushes T_enc to 4097.
    const std::vector<float> at_cap(pcm_long.begin(), pcm_long.begin() + k_cap_samples);
    const std::vector<float> over_cap(pcm_long.begin(), pcm_long.begin() + k_cap_samples + 1);
    // At the cap the input is accepted; whether the tiny model's greedy decode
    // then finishes cleanly is a property of the audio, not of the cap.
    const transcribe_status  at_cap_st = transcribe_run(s, at_cap.data(), static_cast<int>(at_cap.size()), nullptr);
    CHECK(at_cap_st == TRANSCRIBE_OK || at_cap_st == TRANSCRIBE_ERR_OUTPUT_TRUNCATED ||
          at_cap_st == TRANSCRIBE_ERR_OUTPUT_REPETITION);
    const std::string at_cap_text = full_text(s);
    CHECK(!at_cap_text.empty());
    CHECK(transcribe_run(s, over_cap.data(), static_cast<int>(over_cap.size()), nullptr) ==
          TRANSCRIBE_ERR_INPUT_TOO_LONG);
    CHECK(full_text(s).empty());

    // 3. Batch: the over-cap row fails alone.
    const float * pcms[2] = { pcm_short.data(), pcm_long.data() };
    const int     lens[2] = { static_cast<int>(pcm_short.size()), static_cast<int>(pcm_long.size()) };
    CHECK(transcribe_run_batch(s, pcms, lens, 2, nullptr) == TRANSCRIBE_OK);
    CHECK_EQ_INT(transcribe_batch_n_results(s), 2);
    CHECK(transcribe_batch_status(s, 0) == TRANSCRIBE_OK);
    CHECK(transcribe_batch_status(s, 1) == TRANSCRIBE_ERR_INPUT_TOO_LONG);
    const char * row0 = transcribe_batch_full_text(s, 0);
    CHECK(row0 != nullptr && row0[0] != '\0');

    // 4. Streaming far past the cap.
    const std::string streamed = stream_all(s, pcm_long);
    CHECK(transcribe_was_truncated(s) == true);
    // On CUDA/Metal the stream's windows and the one-shot graph go through the
    // same kernels, so the two transcripts match byte for byte and that is the
    // assertion worth making: it proves the stream consumed exactly the first
    // 81.92 s and nothing else. On CPU, ggml picks shape-dependent kernel paths,
    // so a windowed run and a single-pass run can round differently and the
    // decoder may move a comma; there the invariants above are the assertion.
    if (compare_text && streamed != at_cap_text) {
        std::fprintf(stderr,
                     "FAIL %s: streamed transcript differs from one-shot over the first 81.92 s\n  stream: %s\n"
                     "  one-shot: %s\n",
                     label, streamed.c_str(), at_cap_text.c_str());
        ++g_failures;
    }

    // A stream that ends inside the cap is not truncated by it.
    const std::string short_stream = stream_all(s, pcm_short);
    CHECK(transcribe_was_truncated(s) == false);
    CHECK(!short_stream.empty());

    transcribe_session_free(s);
    transcribe_model_free(model);
}

}  // namespace

int main() {
    const char * model_path = std::getenv("TRANSCRIBE_MOONSHINE_STREAMING_TINY_GGUF");
    if (model_path == nullptr || *model_path == '\0' || !file_exists(model_path)) {
        std::fprintf(stderr,
                     "skipping: TRANSCRIBE_MOONSHINE_STREAMING_TINY_GGUF unset "
                     "or missing\n");
        return 77;
    }

    std::vector<float> pcm_short, pcm_long;
    if (!load_sample("jfk.wav", pcm_short)) {
        return 77;
    }
    if (!load_sample("dots-full.wav", pcm_long)) {  // ~306 s, far past the cap
        return 77;
    }
    if (pcm_long.size() <= static_cast<size_t>(k_cap_samples) + 1) {
        std::fprintf(stderr, "dots-full.wav is shorter than the cap; test needs > 81.92 s\n");
        return 1;
    }
    // 90 s is enough to cross the 81.92 s cap and cheap enough that the CPU
    // pass (a full greedy decode per feed) does not dominate the suite.
    pcm_long.resize(static_cast<int>(90 * 16000));

    check_backend(model_path, TRANSCRIBE_BACKEND_AUTO, "default backend", /*exact_text=*/true, pcm_short, pcm_long);
    check_backend(model_path, TRANSCRIBE_BACKEND_CPU, "cpu", /*exact_text=*/false, pcm_short, pcm_long);

    if (g_failures > 0) {
        std::fprintf(stderr, "moonshine_streaming_input_cap: %d failures\n", g_failures);
        return EXIT_FAILURE;
    }
    std::fprintf(stdout, "moonshine_streaming_input_cap: ok\n");
    return EXIT_SUCCESS;
}
