#include "model.h"

#include <math.h>
#include <string.h>

void model_init(model_state_t *m)
{
    memset(m, 0, sizeof(*m));
    m->version = MODEL_STATE_VERSION;
}

float model_history_days(const model_state_t *m)
{
    return (float)m->kept_n * MODEL_THIN * 300.0f / 86400.0f;
}

// Rank of x among the kept values of band b, 0 (lowest of the week) to 1 (highest).
static float band_rank(const model_state_t *m, int b, float x)
{
    int below = 0, equal = 0;
    for (int i = 0; i < m->kept_n; i++) {
        float k = m->kept[b][i];
        if (k < x) below++;
        else if (k == x) equal++;
    }
    return ((float)below + 0.5f * (float)equal) / (float)m->kept_n;
}

// Mean of the last len ranks of band b, skipping NaNs; NaN unless at least half are present.
static float mean_rank(const model_state_t *m, int b, int len)
{
    int have = len < m->rank_n ? len : m->rank_n;
    float sum = 0.0f;
    int n = 0;
    for (int i = 1; i <= have; i++) {
        float v = m->ranks[b][(m->rank_head - i + MODEL_RANKS_N) % MODEL_RANKS_N];
        if (!isnan(v)) { sum += v; n++; }
    }
    return n >= len / 2 ? sum / (float)n : NAN;
}

float model_update(model_state_t *m, const float *bands_mhz, bool valid, float *position)
{
    // 2. rank each band against its week, then (one slot in six) keep it.
    for (int b = 0; b < MODEL_NBANDS; b++) {
        float r = NAN;
        if (valid && !isnan(bands_mhz[b]) && m->kept_n >= MODEL_RANK_MIN_KEPT) r = band_rank(m, b, bands_mhz[b]);
        m->ranks[b][m->rank_head] = r;
    }
    m->rank_head = (m->rank_head + 1) % MODEL_RANKS_N;
    if (m->rank_n < MODEL_RANKS_N) m->rank_n++;
    if (valid && m->count % MODEL_THIN == MODEL_THIN - 1) {
        for (int b = 0; b < MODEL_NBANDS; b++) m->kept[b][m->kept_head] = bands_mhz[b];
        m->kept_head = (m->kept_head + 1) % MODEL_THIN_N;
        if (m->kept_n < MODEL_THIN_N) m->kept_n++;
    }
    m->count++;

    // 3 and 4. averages over 1, 6 and 24 hours, weighted.
    float pred = MODEL_W0;
    for (int b = 0; b < MODEL_NBANDS; b++) {
        for (int a = 0; a < MODEL_NAVG; a++) {
            float v = mean_rank(m, b, MODEL_AVG_LEN[a]);
            if (isnan(v)) { *position = NAN; return NAN; }
            pred += MODEL_W[b][a] * v;
        }
    }

    // 5. the figures' position, on the fixed scale set in training.
    float p = pred < MODEL_P_MID ? (pred - MODEL_P_MID) / (MODEL_P_MID - MODEL_P_LO)
                                 : (pred - MODEL_P_MID) / (MODEL_P_HI - MODEL_P_MID);
    *position = p < -1.0f ? -1.0f : (p > 1.0f ? 1.0f : p);
    return pred;
}

bool model_nn_input(const model_state_t *m, float *out)
{
    // Row h, column b: band b's mean placing over the hour that ended h hours ago (12 five-minute
    // ranks, NaNs skipped, at least 6 needed), as tools/tinyml_inputs.py builds it.
    for (int h = 0; h < 24; h++) {
        for (int b = 0; b < MODEL_NBANDS; b++) {
            float sum = 0.0f;
            int n = 0;
            for (int k = 0; k < 12; k++) {
                int back = 12 * h + k + 1;
                if (back > m->rank_n) break;
                float v = m->ranks[b][(m->rank_head - back + MODEL_RANKS_N) % MODEL_RANKS_N];
                if (!isnan(v)) { sum += v; n++; }
            }
            if (n < 6) {
                if (h == 0) return false;   // the latest hour must be real
                out[h * MODEL_NBANDS + b] = 0.5f;   // older hours not yet seen: neutral, as in training
            } else {
                out[h * MODEL_NBANDS + b] = sum / (float)n;
            }
        }
    }
    return true;
}
