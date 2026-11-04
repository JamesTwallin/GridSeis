// The spectrum model: how clean the grid is now compared with the last seven days, from the
// grid frequency's spectrum alone. No clock and no internet: everything comes from the board's
// own readings, and the weights are fixed when the firmware is built.
//
// Every 5 minutes (model_update):
//   1. 13 spectrum bands from 0.01 to 0.5 Hz, without the 0.258 Hz hum (ring_compute_bands);
//   2. each band ranked against its own last seven days, one value kept per 30 minutes;
//   3. each band's rank averaged over the last 1, 6 and 24 hours;
//   4. a weighted sum of those 39 averages: the predicted cleanliness, 0 = dirtiest half-hour of
//      the week, 1 = cleanest;
//   5. the figures' position from it, -1 (power station fully out) to +1 (turbine fully out).
//
// The weights are in model_weights.h, written by hardware/tools/train_fft_model.py, which mirrors
// this file step for step. Pure C, no ESP-IDF dependencies, so it can be tested on a PC.
#pragma once
#include <stdbool.h>
#include <stdint.h>

#include "model_weights.h"

#define MODEL_RANKS_N 288   // 24 hours of 5-minute ranks: the longest average

typedef struct {
    uint32_t version;                            // MODEL_STATE_VERSION, so an old save is ignored
    uint32_t count;                              // 5-minute slots seen, valid or not
    int32_t kept_head, kept_n;                   // the weekly band history, one value per 30 minutes
    float kept[MODEL_NBANDS][MODEL_THIN_N];
    int32_t rank_head, rank_n;                   // the last 24 hours of band ranks (NaN when unranked)
    float ranks[MODEL_NBANDS][MODEL_RANKS_N];
} model_state_t;

#define MODEL_STATE_VERSION 1

void model_init(model_state_t *m);

// One 5-minute slot. bands_mhz is ignored when valid is false (the 30 minutes had too many bad
// seconds); the slot still counts, as in training. Returns the predicted cleanliness (0 to 1,
// roughly) and sets *position to -1..+1, or returns NAN and sets *position to NAN while the model
// is still filling its history (about 36 hours from empty).
float model_update(model_state_t *m, const float *bands_mhz, bool valid, float *position);

// Kept values (one per 30 minutes) a band needs before it is placed against its history: 3 hours.
// The network was trained on histories this short; the weighted sum's 24-hour averages take longer anyway.
#define MODEL_RANK_MIN_KEPT 6

// The on-board network's input (nn.h): 24 rows (row h = the hour that ended h hours ago) of 13 band
// placings. Hours with no placings yet are filled with 0.5, as in training. False until the latest
// hour has placings, about 3 hours from empty.
bool model_nn_input(const model_state_t *m, float *out);

// Days of band history held, for the serial report.
float model_history_days(const model_state_t *m);
