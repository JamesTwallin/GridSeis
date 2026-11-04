// Host-side test of the spectrum model (ringing.c and model.c, compiled unchanged).
//
// Reads one grid frequency per line (Hz, one per second; "nan" for a bad second), runs it through
// the firmware's 30-minute spectrum every 5 minutes and the model, and writes one line per slot:
//   MODEL,second,valid,band0..band12 (mHz),prediction,position
// tools/check_model_c.py runs the same seconds through the Python training pipeline and compares.
//
// Usage: host_model_test FREQ_FILE OUT_FILE
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "model.h"
#include "ringing.h"

static ring_state_t s_ring;
static model_state_t s_model;

int main(int argc, char **argv)
{
    if (argc != 3) { fprintf(stderr, "usage: host_model_test FREQ_FILE OUT_FILE\n"); return 2; }
    FILE *in = fopen(argv[1], "r"), *out = fopen(argv[2], "w");
    if (!in || !out) { fprintf(stderr, "cannot open files\n"); return 1; }
    ring_init(&s_ring);
    model_init(&s_model);
    char line[64];
    long sec = 0;
    float bands[MODEL_NBANDS];
    while (fgets(line, sizeof line, in)) {
        double f = strtod(line, NULL);
        bool ok = !isnan(f);
        ring_push_second(&s_ring, ok ? f : 50.0, ok);
        sec++;
        if (sec % RING_HOP != 0 || s_ring.filled < RING_WIN) continue;
        bool valid = ring_compute_bands(&s_ring, MODEL_NBANDS, MODEL_K_LO, MODEL_K_HI, MODEL_HUM_K_LO, MODEL_HUM_K_HI, bands);
        float pos;
        float pred = model_update(&s_model, bands, valid, &pos);
        fprintf(out, "MODEL,%ld,%d", sec, valid);
        for (int b = 0; b < MODEL_NBANDS; b++) fprintf(out, ",%.6g", valid ? bands[b] : NAN);
        fprintf(out, ",%.6f,%.6f\n", pred, pos);
    }
    fclose(in); fclose(out);
    return 0;
}
