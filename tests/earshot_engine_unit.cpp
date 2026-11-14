// Earshot engine (src/transcribe-earshot-engine.*): parity with the upstream
// Rust crate's recorded scores, bit-identity across kernel sets, batch vs
// streaming, threads, reset.
#include "transcribe-earshot-engine.h"
#include "wav.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#ifndef TRANSCRIBE_EARSHOT_FIXTURES
#    error "TRANSCRIBE_EARSHOT_FIXTURES must point at tests/fixtures/earshot"
#endif

namespace {

namespace es = transcribe::earshot;

int g_failures = 0;

#define CHECK(cond, ...)                                                         \
    do {                                                                         \
        if (!(cond)) {                                                           \
            std::fprintf(stderr, "FAIL %s:%d: %s: ", __FILE__, __LINE__, #cond); \
            std::fprintf(stderr, __VA_ARGS__);                                   \
            std::fprintf(stderr, "\n");                                          \
            ++g_failures;                                                        \
        }                                                                        \
    } while (0)

struct Clip {
    std::string        name;
    std::vector<float> pcm;
    std::vector<float> reference;
};

bool load_clip(const std::string & name, Clip & clip) {
    const std::string base = std::string(TRANSCRIBE_EARSHOT_FIXTURES) + "/" + name;
    std::string       err;
    if (!transcribe_cli::load_wav_mono_16k(base + ".wav", clip.pcm, err)) {
        std::fprintf(stderr, "cannot load %s.wav: %s\n", name.c_str(), err.c_str());
        return false;
    }
    std::ifstream in(base + ".scores.txt");
    std::string   line;
    while (std::getline(in, line)) {
        if (!line.empty() && line[0] != '#') {
            clip.reference.push_back(std::stof(line));
        }
    }
    clip.name = name;
    return !clip.reference.empty();
}

std::vector<float> stream_scores(const std::vector<float> & pcm, es::Isa isa) {
    es::Detector       det(isa);
    std::vector<float> out;
    for (size_t i = 0; i + es::k_frame_samples <= pcm.size(); i += es::k_frame_samples) {
        out.push_back(det.predict(pcm.data() + i));
    }
    return out;
}

std::vector<es::Isa> supported_sets() {
    std::vector<es::Isa> sets = { es::Isa::Baseline };
    for (es::Isa isa : { es::Isa::Avx2, es::Isa::Avx512 }) {
        if (es::isa_supported(isa)) {
            sets.push_back(isa);
        }
    }
    return sets;
}

// The upstream crate computes in a different order (C library log/exp, its
// own FFT); audio.cpp measured <= 7.5e-5 per frame on these clips.
void test_reference_parity(const std::vector<Clip> & clips) {
    for (es::Isa isa : supported_sets()) {
        for (const Clip & clip : clips) {
            const std::vector<float> got = stream_scores(clip.pcm, isa);
            CHECK(got.size() == clip.reference.size(), "%s/%s: %zu frames vs %zu", es::isa_name(isa), clip.name.c_str(),
                  got.size(), clip.reference.size());
            float worst = 0.0f;
            for (size_t i = 0; i < std::min(got.size(), clip.reference.size()); ++i) {
                // The recorded reference has six decimals.
                worst = std::max(worst, std::fabs(got[i] - clip.reference[i]));
            }
            CHECK(worst <= 2e-4f, "%s/%s: max |score - upstream| = %g", es::isa_name(isa), clip.name.c_str(),
                  static_cast<double>(worst));
            std::printf("  %-8s %-20s max diff vs upstream %.2e\n", es::isa_name(isa), clip.name.c_str(),
                        static_cast<double>(worst));
        }
    }
}

void test_kernel_sets_identical(const std::vector<Clip> & clips) {
    for (const Clip & clip : clips) {
        const std::vector<float> base = stream_scores(clip.pcm, es::Isa::Baseline);
        for (es::Isa isa : supported_sets()) {
            const std::vector<float> got = stream_scores(clip.pcm, isa);
            CHECK(got.size() == base.size() && std::memcmp(got.data(), base.data(), got.size() * sizeof(float)) == 0,
                  "%s differs from baseline on %s", es::isa_name(isa), clip.name.c_str());
        }
    }
}

// Several 4096-frame blocks, so the block carry-over and the threaded scan
// split are exercised.
void test_batch_matches_stream(const std::vector<Clip> & clips) {
    std::vector<float> long_pcm;
    while (long_pcm.size() < 3 * 4096 * es::k_frame_samples + 1000) {
        for (const Clip & clip : clips) {
            long_pcm.insert(long_pcm.end(), clip.pcm.begin(), clip.pcm.end());
        }
    }
    const int64_t n_frames = static_cast<int64_t>(long_pcm.size()) / es::k_frame_samples;
    for (es::Isa isa : supported_sets()) {
        const std::vector<float> stream = stream_scores(long_pcm, isa);
        for (int threads : { 1, 2, 3, 4, 8 }) {
            std::vector<float> batch(static_cast<size_t>(n_frames), -1.0f);
            es::score_frames(long_pcm.data(), static_cast<int64_t>(long_pcm.size()), threads, batch.data(), isa);
            CHECK(std::memcmp(batch.data(), stream.data(), batch.size() * sizeof(float)) == 0,
                  "%s batch with %d threads differs from streaming", es::isa_name(isa), threads);
        }
    }
}

void test_reset_and_silence() {
    std::vector<float> silence(es::k_frame_samples * 40, 0.0f);
    es::Detector       det;
    float              last = 1.0f;
    for (size_t i = 0; i < silence.size(); i += es::k_frame_samples) {
        last = det.predict(silence.data() + i);
    }
    CHECK(last < 0.5f, "silence scored %g", static_cast<double>(last));

    Clip clip;
    if (!load_clip("ls-6930-75918-0000", clip)) {
        ++g_failures;
        return;
    }
    const std::vector<float> fresh = stream_scores(clip.pcm, es::Isa::Auto);
    for (size_t i = 0; i + es::k_frame_samples <= clip.pcm.size(); i += es::k_frame_samples) {
        det.predict(clip.pcm.data() + i);
    }
    det.reset();
    std::vector<float> again;
    for (size_t i = 0; i + es::k_frame_samples <= clip.pcm.size(); i += es::k_frame_samples) {
        again.push_back(det.predict(clip.pcm.data() + i));
    }
    CHECK(again == fresh, "reset does not return the detector to its initial state");
}

}  // namespace

int main() {
    std::vector<Clip> clips(2);
    if (!load_clip("ls-6930-75918-0000", clips[0]) || !load_clip("2086-149220-0033", clips[1])) {
        return EXIT_FAILURE;
    }
    std::printf("earshot kernel sets on this CPU:");
    for (es::Isa isa : supported_sets()) {
        std::printf(" %s", es::isa_name(isa));
    }
    std::printf(" (auto = %s)\n", es::isa_name(es::resolve_isa(es::Isa::Auto)));
    test_reference_parity(clips);
    test_kernel_sets_identical(clips);
    test_batch_matches_stream(clips);
    test_reset_and_silence();
    if (g_failures == 0) {
        std::printf("earshot_engine_unit: OK\n");
    }
    return g_failures == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
