#ifndef TRANSCRIBE_R2T2_H
#define TRANSCRIBE_R2T2_H
#include "transcribe.h"

/* R2T2 native decoding cadence, independent of PCM feed packet duration.
 * Every integer millisecond in [80, 2000] is accepted; default is 320.
 * Values are copied at stream_begin and cannot change an active stream.
 */
#define TRANSCRIBE_EXT_KIND_R2T2_STREAM 0x32543252u
struct transcribe_r2t2_stream_ext {
    struct transcribe_ext ext;
    uint32_t chunk_size_ms;
};
#endif
