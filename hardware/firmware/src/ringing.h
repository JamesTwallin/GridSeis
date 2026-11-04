// The ringing gauge: how loudly the grid rings at 4 to 10 second periods, ranked against
// the last seven days.
//
// Every second the measured frequency goes into a 30-minute buffer. Every 5 minutes the
// buffer is turned into a spectrum with the same recipe as the GridSeis analysis
// (mean removed, Hann window, amplitude = |DFT| * 2 / sum(window)), and the ringing is
// the mean amplitude between 0.1 and 0.25 Hz. That band stops just below the 0.258 Hz
// hum, so the hum is excluded.
//
// The absolute size of the ringing depends on the antenna and the local noise, so the
// gauge does not use it directly. It uses the ringing's rank among the last seven days of
// values: loud ringing means a light, windy, low-carbon grid; quiet ringing a heavy,
// gas-heavy, high-carbon one.
//
// Pure C, no ESP-IDF dependencies, so it can be tested on a PC.
#pragma once
#include <stdbool.h>
#include <stdint.h>

#define RING_WIN 1800                  // seconds in one spectrum (30 minutes)
#define RING_HOP 300                   // seconds between spectra (5 minutes)
#define RING_HIST (7 * 24 * 3600 / RING_HOP)   // 2016 values = seven days
#define RING_F_LO 0.10                 // Hz
#define RING_F_HI 0.25                 // Hz
#define RING_MIN_VALID 0.95            // fraction of valid seconds needed for a spectrum
#define RING_MIN_HIST (24 * 3600 / RING_HOP)   // one day of history before the gauge moves

typedef struct {
    float sec[RING_WIN];   // per-second frequency, Hz, circular
    uint8_t ok[RING_WIN];  // 1 if that second was valid
    int head;              // index where the next second goes
    int filled;            // seconds stored, up to RING_WIN
    float hist[RING_HIST]; // ringing values, mHz, circular
    int hist_head;
    int hist_n;
} ring_state_t;

void ring_init(ring_state_t *r);
void ring_push_second(ring_state_t *r, double f_hz, bool valid);

// Ringing of the last 30 minutes in mHz. Returns false if the buffer is not full or too
// many seconds were invalid.
bool ring_compute(const ring_state_t *r, float *out_mhz);

// Mean spectrum amplitude, mHz, in each of nb bands of DFT bins k_lo[b]..k_hi[b] (bin k is k / 1800 Hz),
// leaving out bins skip_lo..skip_hi. Same recipe as ring_compute. Used by the spectrum model (model.h).
bool ring_compute_bands(const ring_state_t *r, int nb, const int *k_lo, const int *k_hi, int skip_lo, int skip_hi,
                        float *out_mhz);

void ring_hist_push(ring_state_t *r, float v);

// Rank of v among the stored history, 0 (quietest of the week) to 1 (loudest).
float ring_percentile(const ring_state_t *r, float v);
