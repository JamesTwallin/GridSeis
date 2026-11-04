// Mains frequency from a wire antenna, one reading per second, by IQ demodulation.
//
// The ADC samples the antenna at FREQ_FS samples a second. Each second of samples is
// correlated with a 50 Hz cosine and sine. Because one second holds exactly 50 cycles of
// the reference, the 150 Hz and 250 Hz harmonics that dominate antenna pickup and any DC
// offset cancel out. The phase of the result moves by 2*pi*(f - 50) per second, so the
// change in phase from one second to the next gives the average frequency over that second.
//
// Pure C, no ESP-IDF dependencies, so it can be tested on a PC (hardware/tools/host_test.c).
#pragma once
#include <stdbool.h>
#include <stdint.h>

#define FREQ_FS 2000          // ADC samples per second; must be a multiple of 50
#define FREQ_NOMINAL 50.0     // Hz; this firmware is for 50 Hz grids (GB, Ireland, Continental Europe)
#define FREQ_MIN_AMPLITUDE 8  // ADC counts of 50 Hz signal below which a second is marked invalid

typedef struct {
    double prev_phase;    // radians, phase of the previous second
    bool have_prev;       // false after start-up or a dropped block
    double amplitude;     // ADC counts, peak amplitude of the 50 Hz component this second
    double freq_hz;       // average frequency over this second
    bool valid;           // true when freq_hz can be trusted
} freq_state_t;

void freq_init(freq_state_t *s);

// Forget the previous phase, e.g. after the ADC driver reports dropped samples.
void freq_reset_phase(freq_state_t *s);

// Process exactly FREQ_FS consecutive samples (one second). Returns s->valid.
bool freq_process_second(freq_state_t *s, const uint16_t *x, int n);
