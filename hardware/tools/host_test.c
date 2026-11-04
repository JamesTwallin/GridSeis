// Host-side test of the firmware's signal processing (freq.c and ringing.c, compiled unchanged).
//
// Reads one grid frequency per line (Hz, one per second, e.g. a day of NESO data), synthesises
// what the antenna would give the ADC at 2000 samples a second, and runs the firmware on it:
//
//   x = 2048 + A cos(phi) + H cos(3 phi + 0.7) + Gaussian noise, clipped to 0..4095, rounded
//   phi advances by 2 pi f / 2000 per sample, with f the reading for that second
//
// Usage: host_test FREQ_FILE A H NOISE OUT_FILE
// Writes "SEC,second,truth_hz,recovered_hz,valid" and "RING,second,ringing_truth_mhz,ringing_recovered_mhz".
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "freq.h"
#include "ringing.h"

static unsigned long long s_rng = 88172645463325252ULL;
static double uniform(void)
{
    s_rng ^= s_rng << 13; s_rng ^= s_rng >> 7; s_rng ^= s_rng << 17;
    return ((s_rng >> 11) + 0.5) / 9007199254740992.0;
}
static double gauss(void) { return sqrt(-2.0 * log(uniform())) * cos(2.0 * M_PI * uniform()); }

static ring_state_t ring_truth, ring_rec;

int main(int argc, char **argv)
{
    if (argc != 6) { fprintf(stderr, "usage: host_test FREQ_FILE A H NOISE OUT_FILE\n"); return 2; }
    FILE *in = fopen(argv[1], "r");
    FILE *out = fopen(argv[5], "w");
    if (!in || !out) { fprintf(stderr, "cannot open files\n"); return 2; }
    double A = atof(argv[2]), H = atof(argv[3]), noise = atof(argv[4]);

    freq_state_t fs;
    freq_init(&fs);
    ring_init(&ring_truth);
    ring_init(&ring_rec);
    static uint16_t block[FREQ_FS];
    double phi = 1.234, f;
    long sec = 0;
    while (fscanf(in, "%lf", &f) == 1) {
        for (int m = 0; m < FREQ_FS; m++) {
            double v = 2048.0 + A * cos(phi) + H * cos(3.0 * phi + 0.7) + noise * gauss();
            phi += 2.0 * M_PI * f / FREQ_FS;
            if (phi > 2.0 * M_PI) phi -= 2.0 * M_PI;
            if (v < 0) v = 0;
            if (v > 4095) v = 4095;
            block[m] = (uint16_t)lround(v);
        }
        bool ok = freq_process_second(&fs, block, FREQ_FS);
        ring_push_second(&ring_truth, f, true);
        ring_push_second(&ring_rec, fs.freq_hz, ok);
        sec++;
        fprintf(out, "SEC,%ld,%.6f,%.6f,%d\n", sec, f, fs.freq_hz, ok);
        if (sec % RING_HOP == 0) {
            float rt, rr;
            bool a = ring_compute(&ring_truth, &rt), b = ring_compute(&ring_rec, &rr);
            if (a && b) fprintf(out, "RING,%ld,%.6f,%.6f\n", sec, rt, rr);
        }
    }
    fclose(in);
    fclose(out);
    return 0;
}
