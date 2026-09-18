#ifndef EARSHOT_H
#define EARSHOT_H

#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct ESVoiceActivityDetector ESVoiceActivityDetector;

ESVoiceActivityDetector * es_vad_new(void);
void                      es_vad_free(ESVoiceActivityDetector * vad);
void                      es_vad_reset(ESVoiceActivityDetector * vad);
float                     es_vad_predict_f32(ESVoiceActivityDetector * vad, const float * frame_256);
float                     es_vad_predict_i16(ESVoiceActivityDetector * vad, const int16_t * frame_256);

#ifdef __cplusplus
}
#endif

#endif /* EARSHOT_H */
