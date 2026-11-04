"""Train the weather house's spectrum model and write its weights into the firmware.

The board is offline, so the model sees only what the board measures: the grid frequency's spectrum.
No clock, no carbon intensity, no retraining after it is flashed.

Every 5 minutes, from the last 30 minutes of one-second frequency (same recipe as the ringing):
  1. 13 spectrum bands, log-spaced from 0.01 to 0.5 Hz, with the 0.250-0.266 Hz hum left out;
  2. each band ranked against its own last seven days, keeping one value per 30 minutes (336 values),
     so a cleaner grid or a different antenna position cancels out;
  3. each band's rank averaged over the last 1, 6 and 24 hours: 39 numbers;
  4. a weighted sum of those 39 numbers plus a constant: the predicted cleanliness, where 0 means the
     dirtiest half-hour of the week and 1 the cleanest;
  5. the figures' position from that prediction, on a fixed scale set in training: fully out for the
     10% most extreme predictions at each end, both inside at the middle. (Ranking the prediction
     against its own week again was tried and lost a fifth of the agreement.)

Target: where each half-hour's NESO carbon intensity sits in its previous seven days. Home pickup is
stood in for by 2.5 mHz of white noise on each one-second reading.

First checks the method by training on July 2021 to December 2023 and testing, frozen, on 2024 to
September 2026 against the current one-band rule. Then trains on everything and writes
firmware/src/model_weights.h and tools/train_fft_model_results.json.

Usage: python hardware/tools/train_fft_model.py"""
import json, os, sys
import numpy as np, pandas as pd
from scipy.stats import spearmanr

HERE = os.path.dirname(os.path.abspath(__file__)); HW = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import fft_model_test as T  # noqa: E402  (band_series, band_bins, ci_clean_rank, weekly_rank, EDGES, HUM)

NB = len(T.EDGES) - 1
THIN = 6            # keep one band value in six (one per 30 minutes) for the weekly ranking
THIN_N = 336        # seven days of kept values
THIN_MIN = 48       # one day of kept values before a band is ranked
AVG = (12, 72, 288)  # 1, 6 and 24 hours of 5-minute ranks
HEADER = os.path.join(HW, 'firmware', 'src', 'model_weights.h')


def thin_rank(v, thin=THIN, n=THIN_N, nmin=THIN_MIN):
    """Rank each 5-minute value against the last n kept values (every thin-th value, taken after it is
    ranked), exactly as model.c does. NaN values are not kept and not ranked."""
    out = np.full(len(v), np.nan, np.float32); kept = np.full(n, np.nan, np.float32); head = 0; nk = 0
    for i, x in enumerate(v):
        if np.isnan(x): continue
        if nk >= nmin:
            k = kept[:nk] if nk < n else kept
            out[i] = ((k < x).sum() + 0.5 * (k == x).sum()) / len(k)
        if i % thin == thin - 1:
            kept[head] = x; head = (head + 1) % n; nk = min(nk + 1, n)
    return out


def rolling_mean_ranks(r):
    """Mean of the last 12, 72 and 288 ranks (NaNs skipped), needing at least half of them, as model.c."""
    s = pd.Series(r)
    return [s.rolling(w, min_periods=w // 2).mean().values.astype(np.float32) for w in AVG]


def features():
    fp = f'{T.CACHE}/model_features_thin.parquet'
    if os.path.exists(fp): return pd.read_parquet(fp)
    B = T.band_series(); F = {}
    for i in range(NB):
        r = thin_rank(B[f'b{i}'].values.astype(np.float32))
        for w, m in zip(AVG, rolling_mean_ranks(r)): F[f'b{i}_{w // 12}h'] = m
        print('band', i, flush=True)
    r = T.weekly_rank(B['ring'].values, 2016, 288)   # the current firmware rule, for comparison
    F['rule'] = r
    F = pd.DataFrame(F, index=B.index); F.to_parquet(fp); return F


def joined(F):
    ci, tg = T.ci_clean_rank()
    hh = F.index.floor('30min') - pd.Timedelta(minutes=0)
    # a 5-minute value labelled t covers the 30 minutes before t; it belongs to the half-hour it ends in
    hh = (F.index - pd.Timedelta(seconds=1)).floor('30min')
    j = F.copy(); j['ci'] = ci.reindex(hh).values; j['target'] = tg.reindex(hh).values
    return j


def fit(tr, cols):
    X = np.c_[tr[cols].values, np.ones(len(tr))]
    return np.linalg.lstsq(X, tr.target.values, rcond=None)[0]


def predict(df, cols, w):
    return np.c_[df[cols].values, np.ones(len(df))] @ w


def score(pred5, j5):
    """Half-hourly comparison of a 5-minute series with the real weekly cleanliness rank."""
    d = pd.DataFrame({'p': pred5, 'target': j5.target.values, 'ci': j5.ci.values}, index=j5.index)
    h = d.resample('30min', label='left').mean().dropna()
    fifth = pd.qcut(h.p, 5, labels=False, duplicates='drop')
    return {'agreement': round(float(spearmanr(h.p, h.target).statistic), 3),
            'station_out_and_dirtier_than_week': round(float((h.target[fifth == 0] < 0.5).mean()), 3),
            'turbine_out_and_cleaner_than_week': round(float((h.target[fifth == 4] > 0.5).mean()), 3),
            'mean_gco2_by_fifth_station_to_turbine': h.ci.groupby(fifth).mean().round(0).tolist()}


def position(pred, lo, mid, hi):
    """Step 5: -1 (power station fully out) to +1 (turbine fully out), as model.c does."""
    return np.clip(np.where(pred < mid, (pred - mid) / (mid - lo), (pred - mid) / (hi - mid)), -1, 1)


def main():
    F = features(); j = joined(F)
    cols = [c for c in F.columns if c.startswith('b')]
    ok = j[cols + ['target']].notna().all(axis=1)
    res = {'bands_hz': [round(float(x), 4) for x in T.EDGES], 'hum_left_out_hz': list(T.HUM), 'check': {}}

    # ---- check: train to 2023, frozen test on 2024 to 2026
    tr = j[ok & (j.index < '2024-01-01')]; w = fit(tr, cols)
    lo, mid, hi = np.percentile(predict(tr, cols, w), [10, 50, 90])
    te = j[j.index >= '2024-01-01']; tok = te[cols].notna().all(axis=1)
    pred = np.where(tok, predict(te.fillna(0), cols, w), np.nan)
    disp = position(pred, lo, mid, hi)
    for name, (a, b) in {'2024': ('2024', '2025'), '2025': ('2025', '2026'), '2026 to Sep': ('2026', '2026-10'),
                         'all 2024 to Sep 2026': ('2024', '2026-10')}.items():
        m = (te.index >= a) & (te.index < b)
        sub = te[m]; keep = ~np.isnan(pred[m]) & ~np.isnan(disp[m]) & sub.rule.notna().values & sub.target.notna().values
        s5 = sub[keep]
        res['check'][name] = {'current rule': score(s5.rule.values, s5),
                              'model': score(pred[m][keep], s5),
                              'model, as the figures show it': score(disp[m][keep], s5)}
        print(name, {k: v['agreement'] for k, v in res['check'][name].items()}, flush=True)

    # ---- final: train on everything, write the header
    w = fit(j[ok], cols)
    lo, mid, hi = np.percentile(predict(j[ok], cols, w), [10, 50, 90])
    res['position_scale'] = {'fully_station': float(lo), 'centre': float(mid), 'fully_turbine': float(hi)}
    res['weights'] = {c: round(float(x), 5) for c, x in zip(cols + ['constant'], w)}
    bins = T.band_bins()
    with open(HEADER, 'w') as f:
        f.write('// Generated by hardware/tools/train_fft_model.py - do not edit by hand.\n')
        f.write(f'// Trained on NESO one-second frequency and carbon intensity, July 2021 to September 2026,\n')
        f.write('// with 2.5 mHz of noise added to stand in for a home measurement.\n#pragma once\n\n')
        f.write(f'#define MODEL_NBANDS {NB}\n#define MODEL_THIN {THIN}\n#define MODEL_THIN_N {THIN_N}\n#define MODEL_THIN_MIN {THIN_MIN}\n')
        f.write(f'#define MODEL_NAVG {len(AVG)}\n')
        f.write('static const int MODEL_AVG_LEN[MODEL_NAVG] = {' + ', '.join(map(str, AVG)) + '};\n\n')
        f.write('// DFT bins (k / 1800 Hz) in each band; a band is bins k_lo..k_hi minus the hum bins.\n')
        f.write('static const int MODEL_K_LO[MODEL_NBANDS] = {' + ', '.join(str(int(b[0])) for b in bins) + '};\n')
        f.write('static const int MODEL_K_HI[MODEL_NBANDS] = {' + ', '.join(str(int(b[-1])) for b in bins) + '};\n')
        hum = [k for k in range(int(bins[0][0]), int(bins[-1][-1]) + 1) if T.HUM[0] <= k / 1800 <= T.HUM[1]]
        f.write(f'#define MODEL_HUM_K_LO {hum[0]}\n#define MODEL_HUM_K_HI {hum[-1]}\n\n')
        f.write('// Weight of each band\'s 1 h, 6 h and 24 h average rank, then the constant.\n')
        f.write('static const float MODEL_W[MODEL_NBANDS][MODEL_NAVG] = {\n')
        for i in range(NB): f.write('    {' + ', '.join(f'{w[i * 3 + k]:.6f}f' for k in range(3)) + '},\n')
        f.write('};\n' + f'static const float MODEL_W0 = {w[-1]:.6f}f;\n\n')
        f.write('// Figure position: predictions at or below P_LO put the power station fully out, at P_MID both\n')
        f.write('// stay inside, at or above P_HI the turbine is fully out (10th, 50th, 90th percentiles in training).\n')
        f.write(f'#define MODEL_P_LO {lo:.6f}f\n#define MODEL_P_MID {mid:.6f}f\n#define MODEL_P_HI {hi:.6f}f\n')
    json.dump(res, open(os.path.join(HERE, 'train_fft_model_results.json'), 'w'), indent=1)
    print('wrote', HEADER)


if __name__ == '__main__':
    main()
