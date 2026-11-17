// Speech gate (transcribe_vad_gate_*) and batch scoring
// (transcribe_vad_score_frames), through the public C API.
#include "transcribe.h"
#include "wav.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#ifndef TRANSCRIBE_EARSHOT_FIXTURES
#    error "TRANSCRIBE_EARSHOT_FIXTURES must point at tests/fixtures/earshot"
#endif

namespace {

int g_failures = 0;

#define CHECK(cond)                                                              \
    do {                                                                         \
        if (!(cond)) {                                                           \
            std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); \
            ++g_failures;                                                        \
        }                                                                        \
    } while (0)

struct Run {
    std::vector<float> audio;
    uint64_t           speech_ms = 0;
    int                starts    = 0;
    int                ends      = 0;
    int64_t            frames    = 0;
};

// Feed `pcm` in chunks of `chunk` samples, then finish.
Run run_gate(const std::vector<float> & pcm, size_t chunk, const transcribe_vad_gate_params * params = nullptr) {
    Run                   r;
    transcribe_vad_gate * gate = nullptr;
    CHECK(transcribe_vad_gate_init(params, &gate) == TRANSCRIBE_OK);
    transcribe_vad_gate_update u;
    transcribe_vad_gate_update_init(&u);
    const auto take = [&r, &u] {
        r.audio.insert(r.audio.end(), u.audio, u.audio + u.n_audio);
        r.speech_ms = u.speech_ms;
        r.starts += u.speech_started ? 1 : 0;
        r.ends += u.speech_ended ? 1 : 0;
        r.frames += u.n_frames;
    };
    for (size_t i = 0; i < pcm.size(); i += chunk) {
        const size_t n = std::min(chunk, pcm.size() - i);
        CHECK(transcribe_vad_gate_feed(gate, pcm.data() + i, static_cast<int64_t>(n), &u) == TRANSCRIBE_OK);
        take();
    }
    CHECK(transcribe_vad_gate_finish(gate, &u) == TRANSCRIBE_OK);
    take();
    transcribe_vad_gate_free(gate);
    return r;
}

std::vector<float> load(const char * name) {
    std::vector<float> pcm;
    std::string        err;
    const std::string  path = std::string(TRANSCRIBE_EARSHOT_FIXTURES) + "/" + name;
    if (!transcribe_cli::load_wav_mono_16k(path, pcm, err)) {
        std::fprintf(stderr, "cannot load %s: %s\n", path.c_str(), err.c_str());
        ++g_failures;
    }
    return pcm;
}

void test_silence_releases_nothing() {
    const std::vector<float> silence(16000 * 2, 0.0f);
    const Run                r = run_gate(silence, 256);
    CHECK(r.audio.empty());
    CHECK(r.speech_ms == 0);
    CHECK(r.starts == 0);
}

void test_speech(const std::vector<float> & speech) {
    // 1 s of silence, the clip, 1 s of silence.
    std::vector<float> pcm(16000, 0.0f);
    pcm.insert(pcm.end(), speech.begin(), speech.end());
    pcm.insert(pcm.end(), 16000, 0.0f);
    const Run base = run_gate(pcm, 256);
    CHECK(base.starts >= 1);
    CHECK(base.speech_ms > 1000);
    // Never more audio than went in: nothing is released twice.
    CHECK(base.audio.size() <= pcm.size());
    CHECK(base.audio.size() > speech.size() / 2);
    CHECK(base.frames == static_cast<int64_t>((pcm.size() + 255) / 256));

    // Framing invariance: any chunk size gives the same output.
    for (size_t chunk : { size_t(1), size_t(7), size_t(100), size_t(1000), size_t(4096) }) {
        const Run r = run_gate(pcm, chunk);
        CHECK(r.audio == base.audio);
        CHECK(r.speech_ms == base.speech_ms);
        CHECK(r.starts == base.starts && r.ends == base.ends);
    }

    // A longer hangover never releases less; no pre-roll and no hangover
    // releases no more than the voiced frames themselves.
    transcribe_vad_gate_params p;
    transcribe_vad_gate_params_init(&p);
    p.hangover_ms = 1650;
    CHECK(run_gate(pcm, 256, &p).audio.size() >= base.audio.size());
    p.preroll_ms  = 0;
    p.hangover_ms = 0;
    p.onset_ms    = 0;
    CHECK(run_gate(pcm, 256, &p).audio.size() <= base.audio.size());
}

void test_finish_releases_only_real_samples(const std::vector<float> & speech) {
    // An odd length ends mid-frame; while in speech the last partial frame is
    // released with its real samples only.
    std::vector<float>         pcm(speech.begin(), speech.begin() + static_cast<long>(speech.size() / 256 * 256 - 100));
    transcribe_vad_gate_params p;
    transcribe_vad_gate_params_init(&p);
    p.hangover_ms = 5000;  // stay in speech to the end
    const Run r   = run_gate(pcm, 256, &p);
    CHECK(r.audio.size() <= pcm.size());
    CHECK(r.audio.size() % 256 == 156);
}

void test_reset_and_threshold(const std::vector<float> & speech) {
    transcribe_vad_gate * gate = nullptr;
    CHECK(transcribe_vad_gate_init(nullptr, &gate) == TRANSCRIBE_OK);
    transcribe_vad_gate_update u;
    transcribe_vad_gate_update_init(&u);
    CHECK(transcribe_vad_gate_feed(gate, speech.data(), static_cast<int64_t>(speech.size()), &u) == TRANSCRIBE_OK);
    const std::vector<float> first(u.audio, u.audio + u.n_audio);
    const uint64_t           first_ms = u.speech_ms;
    transcribe_vad_gate_reset(gate);
    CHECK(transcribe_vad_gate_feed(gate, speech.data(), static_cast<int64_t>(speech.size()), &u) == TRANSCRIBE_OK);
    CHECK(std::vector<float>(u.audio, u.audio + u.n_audio) == first);
    CHECK(u.speech_ms == first_ms);

    // Live threshold: nothing is speech at 0.95 that was not at 0.5.
    transcribe_vad_gate_reset(gate);
    CHECK(transcribe_vad_gate_set_threshold(gate, 0.95f) == TRANSCRIBE_OK);
    CHECK(transcribe_vad_gate_feed(gate, speech.data(), static_cast<int64_t>(speech.size()), &u) == TRANSCRIBE_OK);
    CHECK(u.speech_ms <= first_ms);
    CHECK(transcribe_vad_gate_set_threshold(gate, 1.5f) == TRANSCRIBE_ERR_INVALID_ARG);
    CHECK(transcribe_vad_gate_set_hangover_ms(gate, 1650) == TRANSCRIBE_OK);

    transcribe_vad_gate_tail tail;
    transcribe_vad_gate_tail_init(&tail);
    CHECK(transcribe_vad_gate_get_tail(gate, &tail) == TRANSCRIBE_OK);
    CHECK(tail.withheld_voiced_frames <= tail.withheld_frames);
    transcribe_vad_gate_free(gate);
}

// An explicit exit threshold survives threshold changes: lowering the
// threshold clamps the exit under it, raising it again restores the caller's
// exit. Regression: the clamp ratcheted, so after 0.3 -> 0.6 the gate kept
// exiting at 0.3 instead of the requested 0.4.
void test_explicit_exit_survives_threshold_changes(const std::vector<float> & speech) {
    std::vector<float> pcm(16000, 0.0f);
    pcm.insert(pcm.end(), speech.begin(), speech.end());
    pcm.insert(pcm.end(), 16000, 0.0f);

    const auto run = [&](float threshold, float exit, const std::vector<float> & retune) {
        transcribe_vad_gate_params p;
        transcribe_vad_gate_params_init(&p);
        p.threshold                = threshold;
        p.exit_threshold           = exit;
        p.hangover_ms              = 0;  // expose the exit edge directly
        transcribe_vad_gate * gate = nullptr;
        CHECK(transcribe_vad_gate_init(&p, &gate) == TRANSCRIBE_OK);
        for (const float t : retune) {
            CHECK(transcribe_vad_gate_set_threshold(gate, t) == TRANSCRIBE_OK);
        }
        transcribe_vad_gate_update u;
        transcribe_vad_gate_update_init(&u);
        CHECK(transcribe_vad_gate_feed(gate, pcm.data(), static_cast<int64_t>(pcm.size()), &u) == TRANSCRIBE_OK);
        Run r;
        r.audio.assign(u.audio, u.audio + u.n_audio);
        r.speech_ms = u.speech_ms;
        transcribe_vad_gate_free(gate);
        return r;
    };

    const Run direct  = run(0.6f, 0.4f, {});
    const Run retuned = run(0.5f, 0.4f, { 0.3f, 0.6f });
    CHECK(retuned.audio == direct.audio);
    CHECK(retuned.speech_ms == direct.speech_ms);
    // The clip does exercise the exit edge: exiting at 0.3 keeps more audio.
    const Run low_exit = run(0.6f, 0.3f, {});
    CHECK(low_exit.audio != direct.audio || low_exit.speech_ms != direct.speech_ms);
}

void test_validation() {
    transcribe_vad_gate *      gate = nullptr;
    transcribe_vad_gate_params p;
    transcribe_vad_gate_params_init(&p);
    p.threshold = 0.01f;
    CHECK(transcribe_vad_gate_init(&p, &gate) == TRANSCRIBE_ERR_INVALID_ARG);
    CHECK(gate == nullptr);
    p.threshold   = 0.5f;
    p.struct_size = 0;
    CHECK(transcribe_vad_gate_init(&p, &gate) == TRANSCRIBE_ERR_BAD_STRUCT_SIZE);
    CHECK(transcribe_vad_gate_init(nullptr, nullptr) == TRANSCRIBE_ERR_INVALID_ARG);
    CHECK(transcribe_vad_gate_init(nullptr, &gate) == TRANSCRIBE_OK);
    const float nan_frame[3] = { 0.0f, NAN, 0.0f };
    CHECK(transcribe_vad_gate_feed(gate, nan_frame, 3, nullptr) == TRANSCRIBE_ERR_INVALID_ARG);
    CHECK(transcribe_vad_gate_feed(gate, nullptr, 0, nullptr) == TRANSCRIBE_OK);
    CHECK(transcribe_vad_gate_feed(gate, nullptr, 5, nullptr) == TRANSCRIBE_ERR_INVALID_ARG);
    transcribe_vad_gate_free(gate);
    transcribe_vad_gate_free(nullptr);
    transcribe_vad_gate_reset(nullptr);
}

void test_score_frames(const std::vector<float> & speech) {
    const size_t       n = speech.size() / 256;
    std::vector<float> batch(n);
    CHECK(transcribe_vad_score_frames(speech.data(), static_cast<int64_t>(speech.size()), 4, batch.data()) ==
          TRANSCRIBE_OK);
    transcribe_vad *   vad = transcribe_vad_init(0.5f);
    std::vector<float> live(n);
    for (size_t f = 0; f < n; ++f) {
        live[f] = transcribe_vad_predict_frame(vad, speech.data() + f * 256);
    }
    transcribe_vad_free(vad);
    CHECK(batch == live);
    CHECK(transcribe_vad_score_frames(nullptr, 0, 1, nullptr) == TRANSCRIBE_OK);
    CHECK(transcribe_vad_score_frames(nullptr, 10, 1, batch.data()) == TRANSCRIBE_ERR_INVALID_ARG);
}

}  // namespace

int main() {
    const std::vector<float> speech = load("2086-149220-0033.wav");
    if (speech.empty()) {
        return EXIT_FAILURE;
    }
    test_silence_releases_nothing();
    test_speech(speech);
    test_finish_releases_only_real_samples(speech);
    test_reset_and_threshold(speech);
    test_explicit_exit_survives_threshold_changes(speech);
    test_validation();
    test_score_frames(speech);
    if (g_failures == 0) {
        std::printf("vad_gate_unit: OK\n");
    }
    return g_failures == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
