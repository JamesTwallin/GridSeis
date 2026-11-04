#include "ringing.h"

#include <math.h>
#include <string.h>

#define K_LO ((int)ceil(RING_F_LO * RING_WIN - 1e-9))
#define K_HI ((int)floor(RING_F_HI * RING_WIN + 1e-9))

static float s_hann[RING_WIN];
static float s_cos[RING_WIN];
static float s_sin[RING_WIN];
static double s_hann_sum = 0.0;
static bool s_tables_ready = false;
static float s_work[RING_WIN];

static void make_tables(void)
{
    s_hann_sum = 0.0;
    for (int n = 0; n < RING_WIN; n++) {
        // numpy.hanning(N): 0.5 - 0.5 cos(2 pi n / (N - 1))
        s_hann[n] = (float)(0.5 - 0.5 * cos(2.0 * M_PI * n / (RING_WIN - 1)));
        s_hann_sum += s_hann[n];
        s_cos[n] = (float)cos(2.0 * M_PI * n / RING_WIN);
        s_sin[n] = (float)sin(2.0 * M_PI * n / RING_WIN);
    }
    s_tables_ready = true;
}

void ring_init(ring_state_t *r)
{
    if (!s_tables_ready) make_tables();
    memset(r, 0, sizeof(*r));
}

void ring_push_second(ring_state_t *r, double f_hz, bool valid)
{
    r->sec[r->head] = (float)f_hz;
    r->ok[r->head] = valid ? 1 : 0;
    r->head = (r->head + 1) % RING_WIN;
    if (r->filled < RING_WIN) r->filled++;
}

// Unrolls the last 30 minutes into s_work: gaps interpolated, mean removed, in mHz, Hann window applied.
static bool prepare(const ring_state_t *r)
{
    if (!s_tables_ready) make_tables();
    if (r->filled < RING_WIN) return false;

    // Unroll the circular buffer into time order, counting valid seconds.
    int n_ok = 0;
    for (int n = 0; n < RING_WIN; n++) {
        int i = (r->head + n) % RING_WIN;
        s_work[n] = r->ok[i] ? r->sec[i] : NAN;
        n_ok += r->ok[i];
    }
    if (n_ok < RING_MIN_VALID * RING_WIN) return false;

    // Fill gaps by linear interpolation, holding the ends.
    int last = -1;
    for (int n = 0; n < RING_WIN; n++) {
        if (isnan(s_work[n])) continue;
        if (last < 0) {
            for (int m = 0; m < n; m++) s_work[m] = s_work[n];
        } else if (n - last > 1) {
            for (int m = last + 1; m < n; m++)
                s_work[m] = s_work[last] + (s_work[n] - s_work[last]) * (float)(m - last) / (float)(n - last);
        }
        last = n;
    }
    for (int m = last + 1; m < RING_WIN; m++) s_work[m] = s_work[last];

    // Remove the mean, convert to mHz, apply the window.
    double mean = 0.0;
    for (int n = 0; n < RING_WIN; n++) mean += s_work[n];
    mean /= RING_WIN;
    for (int n = 0; n < RING_WIN; n++) s_work[n] = (float)((s_work[n] - mean) * 1000.0) * s_hann[n];

    return true;
}

bool ring_compute(const ring_state_t *r, float *out_mhz)
{
    if (!prepare(r)) return false;

    // Direct DFT of the bins in the band: about half a million multiply-adds.
    double total = 0.0;
    for (int k = K_LO; k <= K_HI; k++) {
        double re = 0.0, im = 0.0;
        int idx = 0;
        for (int n = 0; n < RING_WIN; n++) {
            re += s_work[n] * s_cos[idx];
            im -= s_work[n] * s_sin[idx];
            idx += k;
            if (idx >= RING_WIN) idx -= RING_WIN;
        }
        total += sqrt(re * re + im * im) * 2.0 / s_hann_sum;
    }
    *out_mhz = (float)(total / (K_HI - K_LO + 1));
    return true;
}

void ring_hist_push(ring_state_t *r, float v)
{
    r->hist[r->hist_head] = v;
    r->hist_head = (r->hist_head + 1) % RING_HIST;
    if (r->hist_n < RING_HIST) r->hist_n++;
}

float ring_percentile(const ring_state_t *r, float v)
{
    if (r->hist_n == 0) return 0.5f;
    int below = 0, equal = 0;
    for (int i = 0; i < r->hist_n; i++) {
        if (r->hist[i] < v) below++;
        else if (r->hist[i] == v) equal++;
    }
    return ((float)below + 0.5f * (float)equal) / (float)r->hist_n;
}

bool ring_compute_bands(const ring_state_t *r, int nb, const int *k_lo, const int *k_hi, int skip_lo, int skip_hi,
                        float *out_mhz)
{
    if (!prepare(r)) return false;
    // Single-precision DFT: the ESP32-S3 has a float unit but no double one. About 1.5 million
    // multiply-adds for the 0.01-0.5 Hz bands.
    for (int b = 0; b < nb; b++) {
        float total = 0.0f;
        int count = 0;
        for (int k = k_lo[b]; k <= k_hi[b]; k++) {
            if (k >= skip_lo && k <= skip_hi) continue;
            float re = 0.0f, im = 0.0f;
            int idx = 0;
            for (int n = 0; n < RING_WIN; n++) {
                re += s_work[n] * s_cos[idx];
                im -= s_work[n] * s_sin[idx];
                idx += k;
                if (idx >= RING_WIN) idx -= RING_WIN;
            }
            total += sqrtf(re * re + im * im) * (float)(2.0 / s_hann_sum);
            count++;
        }
        out_mhz[b] = count ? total / (float)count : NAN;
    }
    return true;
}
