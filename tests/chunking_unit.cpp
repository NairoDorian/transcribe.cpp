// tests/chunking_unit.cpp - Unit test for long audio chunk planning and timestamp rebasing.

#include "transcribe-chunking.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
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

}  // namespace

int main() {
    using namespace transcribe::chunking;

    // -------------------------------------------------------------------------
    // 1. Uniform Audio Chunk Planner Test
    // -------------------------------------------------------------------------
    {
        AudioChunkSpec spec;
        spec.chunk_samples = 480000;  // 30s at 16 kHz
        spec.hop_samples   = 448000;  // 28s hop (2s overlap)

        // 60s of audio: 960,000 samples
        const int64_t total_samples = 960000;
        auto          spans         = plan_audio_chunks(total_samples, spec);

        CHECK(spans.size() == 3);
        CHECK(spans[0].index == 0);
        CHECK(spans[0].start_sample == 0);
        CHECK(spans[0].valid_samples == 480000);
        CHECK(spans[0].keep_start_sample == 0);

        CHECK(spans[1].index == 1);
        CHECK(spans[1].start_sample == 448000);
        CHECK(spans[1].valid_samples == 480000);

        // Verify continuity: keep_end of span[0] matches keep_start of span[1]
        CHECK(spans[0].keep_end_sample == spans[1].keep_start_sample);
        CHECK(spans[1].keep_end_sample == spans[2].keep_start_sample);
        CHECK(spans.back().keep_end_sample == total_samples);
    }

    // -------------------------------------------------------------------------
    // 2. Quiet Energy Audio Chunk Planner Test
    // -------------------------------------------------------------------------
    {
        // Generate 40s of audio (640,000 samples at 16 kHz):
        // 0s - 25s: loud speech (sine wave, amplitude 0.8)
        // 25s - 27s: silence (amplitude 0.0) -> samples 400,000 to 432,000
        // 27s - 40s: loud speech (amplitude 0.8)
        const int64_t      n_samples = 640000;
        std::vector<float> pcm(n_samples, 0.0f);
        for (int64_t i = 0; i < 400000; ++i) {
            pcm[i] = 0.8f * std::sin(2.0f * 3.14159f * 440.0f * static_cast<float>(i) / 16000.0f);
        }
        // [400000, 432000] is pure silence
        for (int64_t i = 432000; i < n_samples; ++i) {
            pcm[i] = 0.8f * std::sin(2.0f * 3.14159f * 440.0f * static_cast<float>(i) / 16000.0f);
        }

        // Target chunk: 30s (480,000 samples), search window 6s (96,000 samples)
        // Search window covers [480,000 - 96,000, 480,000] = [384,000, 480,000]
        // The silence at [400,000, 432,000] falls right within this search window!
        auto spans = plan_quiet_energy_audio_chunks(pcm.data(), n_samples, 480000, 96000, -35.0f);

        CHECK(spans.size() == 2);
        CHECK(spans[0].start_sample == 0);
        // Cut point must land cleanly in the silent region [400,000, 432,000]
        CHECK(spans[0].keep_end_sample >= 400000);
        CHECK(spans[0].keep_end_sample <= 432000);
        CHECK(spans[1].start_sample == spans[0].keep_end_sample);
        CHECK(spans[1].keep_end_sample == n_samples);
    }

    // -------------------------------------------------------------------------
    // 3. Timestamp Rebasing and ResultSet Merging Test
    // -------------------------------------------------------------------------
    {
        // Chunk 0: covers [0s, 30s] (0 samples start)
        transcribe_session::ResultSet rs0;
        rs0.has_result = true;
        rs0.full_text  = "Hello world";
        rs0.raw_text   = "Hello world";
        rs0.tokens     = {
            { 1, "Hello", 0.95f, 500,  1000, 0, 0 },
            { 2, "world", 0.98f, 1100, 1800, 0, 1 },
        };
        rs0.words = {
            { "Hello", 500,  1000, 0, 0, 1 },
            { "world", 1100, 1800, 0, 1, 1 },
        };
        rs0.segments = {
            { "Hello world", 500, 1800, 0, 2, 0, 2, 0 },
        };

        // Chunk 1: covers [30s, 60s] (480,000 samples start at 16kHz -> 30,000 ms offset)
        transcribe_session::ResultSet rs1;
        rs1.has_result = true;
        rs1.full_text  = "this is a test";
        rs1.raw_text   = "this is a test";
        rs1.tokens     = {
            { 3, "this", 0.92f, 200,  600,  0, 0 },
            { 4, "is",   0.94f, 700,  900,  0, 1 },
            { 5, "a",    0.91f, 1000, 1100, 0, 2 },
            { 6, "test", 0.96f, 1200, 1700, 0, 3 },
        };
        rs1.words = {
            { "this", 200,  600,  0, 0, 1 },
            { "is",   700,  900,  0, 1, 1 },
            { "a",    1000, 1100, 0, 2, 1 },
            { "test", 1200, 1700, 0, 3, 1 },
        };
        rs1.segments = {
            { "this is a test", 200, 1700, 0, 4, 0, 4, 0 },
        };

        std::vector<transcribe_session::ResultSet> chunk_results       = { rs0, rs1 };
        std::vector<int64_t>                       chunk_start_samples = { 0, 480000 };

        auto merged = merge_chunk_results(chunk_results, chunk_start_samples, 16000);

        CHECK(merged.has_result);
        CHECK(merged.tokens.size() == 6);
        CHECK(merged.words.size() == 6);
        CHECK(merged.segments.size() == 2);
        CHECK(merged.full_text == "Hello world this is a test");

        // First chunk timestamps preserved:
        CHECK(merged.tokens[0].t0_ms == 500);
        CHECK(merged.tokens[0].t1_ms == 1000);
        CHECK(merged.tokens[0].seg_index == 0);
        CHECK(merged.tokens[0].word_index == 0);

        // Second chunk tokens rebased by +30,000 ms:
        CHECK(merged.tokens[2].t0_ms == 30200);
        CHECK(merged.tokens[2].t1_ms == 30600);
        CHECK(merged.tokens[2].seg_index == 1);   // was 0, offset by 1 segment
        CHECK(merged.tokens[2].word_index == 2);  // was 0, offset by 2 words

        // Second chunk words rebased:
        CHECK(merged.words[2].t0_ms == 30200);
        CHECK(merged.words[2].t1_ms == 30600);
        CHECK(merged.words[2].seg_index == 1);
        CHECK(merged.words[2].first_token == 2);  // was 0, offset by 2 tokens

        // Second chunk segment rebased:
        CHECK(merged.segments[1].t0_ms == 30200);
        CHECK(merged.segments[1].t1_ms == 31700);
        CHECK(merged.segments[1].first_word == 2);
        CHECK(merged.segments[1].first_token == 2);
    }

    if (g_failures > 0) {
        std::fprintf(stderr, "FAIL: %d check(s) failed in chunking_unit\n", g_failures);
        return EXIT_FAILURE;
    }

    std::printf("PASS: chunking_unit\n");
    return EXIT_SUCCESS;
}
