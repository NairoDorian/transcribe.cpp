// Unit test for the parakeet RNN-T language-allowlist mask: a blocked token
// can never win the argmax, every other score is untouched, and an empty
// mask is exactly the plain argmax (the byte-identical guarantee).
#include "arch/parakeet/decoder.h"

#include <cstdint>
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

using transcribe::parakeet::argmax_range_masked;

void test_masked_argmax() {
    // id 2 is a disallowed locale tag with the top score; id 4 is next.
    const std::vector<float> logits = { 0.1f, 0.3f, 9.0f, -1.0f, 5.0f, 0.2f };
    const int                n      = static_cast<int>(logits.size());

    CHECK(argmax_range_masked(logits.data(), n, nullptr) == 2);
    const std::vector<uint8_t> none;
    CHECK(argmax_range_masked(logits.data(), n, &none) == 2);

    std::vector<uint8_t> block_tag = { 0, 0, 1 };  // shorter than n: tail unblocked
    CHECK(argmax_range_masked(logits.data(), n, &block_tag) == 4);

    // Blocking a token that would not win changes nothing.
    std::vector<uint8_t> block_other = { 0, 1, 0, 1, 0, 0 };
    CHECK(argmax_range_masked(logits.data(), n, &block_other) == 2);

    // Ties keep the first occurrence among the unblocked ids.
    const std::vector<float> tie = { 3.0f, 7.0f, 7.0f, 7.0f };
    std::vector<uint8_t>     b1  = { 0, 1, 0, 0 };
    CHECK(argmax_range_masked(tie.data(), 4, &b1) == 2);

    // Everything blocked has a defined answer (the plain argmax).
    std::vector<uint8_t> all = { 1, 1, 1, 1, 1, 1 };
    CHECK(argmax_range_masked(logits.data(), n, &all) == 2);
}

}  // namespace

int main() {
    test_masked_argmax();
    if (g_failures == 0) {
        std::printf("parakeet_language_mask_unit: OK\n");
    }
    return g_failures == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
