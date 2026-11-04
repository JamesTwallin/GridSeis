#include "freq.h"

#include <math.h>

#define SAMPLES_PER_CYCLE (FREQ_FS / 50)

static float s_cos[SAMPLES_PER_CYCLE];
static float s_sin[SAMPLES_PER_CYCLE];
static bool s_tables_ready = false;

static void make_tables(void)
{
    for (int k = 0; k < SAMPLES_PER_CYCLE; k++) {
        double a = 2.0 * M_PI * (double)k / (double)SAMPLES_PER_CYCLE;
        s_cos[k] = (float)cos(a);
        s_sin[k] = (float)sin(a);
    }
    s_tables_ready = true;
}

void freq_init(freq_state_t *s)
{
    if (!s_tables_ready) make_tables();
    s->prev_phase = 0.0;
    s->have_prev = false;
    s->amplitude = 0.0;
    s->freq_hz = FREQ_NOMINAL;
    s->valid = false;
}

void freq_reset_phase(freq_state_t *s)
{
    s->have_prev = false;
}

bool freq_process_second(freq_state_t *s, const uint16_t *x, int n)
{
    if (!s_tables_ready) make_tables();
    s->valid = false;
    if (n != FREQ_FS) {          // a partial second breaks the phase chain
        s->have_prev = false;
        return false;
    }

    // Correlate with the reference. The reference restarts every second; that is exact
    // because FREQ_FS samples hold a whole number of 50 Hz cycles.
    double i_sum = 0.0, q_sum = 0.0;
    int k = 0;
    for (int m = 0; m < n; m++) {
        double v = (double)x[m];
        i_sum += v * s_cos[k];
        q_sum += v * s_sin[k];
        if (++k == SAMPLES_PER_CYCLE) k = 0;
    }

    // For x = A cos(2 pi f t + p0): I ~ (A n / 2) cos(theta), Q ~ -(A n / 2) sin(theta),
    // where theta advances by 2 pi (f - 50) per second. So theta = atan2(-Q, I).
    s->amplitude = 2.0 * sqrt(i_sum * i_sum + q_sum * q_sum) / (double)n;
    double phase = atan2(-q_sum, i_sum);

    if (s->have_prev) {
        double d = phase - s->prev_phase;
        while (d > M_PI) d -= 2.0 * M_PI;
        while (d <= -M_PI) d += 2.0 * M_PI;
        s->freq_hz = FREQ_NOMINAL + d / (2.0 * M_PI);   // one-second block
        s->valid = (s->amplitude >= FREQ_MIN_AMPLITUDE) && fabs(s->freq_hz - FREQ_NOMINAL) < 0.5;
    }
    s->prev_phase = phase;
    s->have_prev = (s->amplitude >= FREQ_MIN_AMPLITUDE);
    return s->valid;
}
