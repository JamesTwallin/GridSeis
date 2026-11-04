"""Could a small model on the board beat the weather house's one-band rule, using only the frequency
spectrum and no clock?

The board is offline: it cannot fetch carbon intensity, cannot be retrained, and has no time of day.
So every input here is something the board could compute from its own readings:

- 13 spectrum bands from 0.01 to 0.5 Hz (log-spaced), from the same 30-minute window every 5 minutes as
  the firmware, with the 0.250-0.266 Hz hum left out;
- each band ranked against its own previous seven days (the firmware's rule, applied to every band), so a
  cleaner grid or a different antenna position cancels out;
- the average of each band's rank over the last 1, 6 and 24 hours.

The target is where the half-hour's carbon intensity sits in its own previous seven days, the same thing
the house shows. Home pickup is stood in for by 2.5 mHz of white noise on every one-second reading.

Trained on July 2021 to December 2023, then frozen and tested on 2024 to September 2026. Compared with the
current rule (one band, 0.1 to 0.25 Hz, ranked against its week) on exactly the same half-hours.

Usage: python hardware/tools/fft_model_test.py
Writes hardware/tools/fft_model_results.json and caches the band series on D:."""
import json, os, sys
import numpy as np, pandas as pd
from numpy.lib.stride_tricks import sliding_window_view
from scipy.stats import spearmanr
from xgboost import XGBRegressor

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from desk_check import load_month, WIN, HOP, KLO, KHI  # noqa: E402

CACHE = 'D:/gridseis_hum/hardware_test'
MONTHS = [(y, m) for y in range(2021, 2027) for m in range(1, 13) if (2021, 7) <= (y, m) <= (2026, 9)]
NOISE_MHZ = 2.5
HUM = (0.250, 0.266)
TRAIN_END, TEST_FROM = '2024-01-01', '2024-01-01'
EDGES = np.geomspace(0.01, 0.5, 14)


def band_bins():
    k = np.arange(WIN // 2 + 1); f = k / WIN; out = []
    for lo, hi in zip(EDGES[:-1], EDGES[1:]):
        sel = (f >= lo) & (f < hi) & ~((f >= HUM[0]) & (f <= HUM[1]))
        out.append(k[sel])
    return out


def band_series():
    fp = f'{CACHE}/bands_{NOISE_MHZ:g}mHz.parquet'
    if os.path.exists(fp): return pd.read_parquet(fp)
    rng = np.random.default_rng(7); w = np.hanning(WIN); bins = band_bins(); out = []
    for y, m in MONTHS:
        f = load_month(y, m); start = pd.Timestamp(y, m, 1); end = start + pd.offsets.MonthBegin(1)
        full = f.reindex(pd.date_range(start, end, freq='1s', inclusive='left'))
        valid = full.notna().values; x = full.interpolate(limit_direction='both').values
        x = x + rng.normal(0, NOISE_MHZ / 1000, len(x))
        ends = np.arange(WIN, len(x) + 1, HOP); B = np.full((len(ends), len(bins) + 1), np.nan, np.float32)
        for c in range(0, len(ends), 3000):
            e = ends[c:c + 3000]; idx = e[:, None] - WIN + np.arange(WIN)[None, :]
            X = x[idx]; X = (X - X.mean(axis=1, keepdims=True)) * 1000
            A = np.abs(np.fft.rfft(X * w, axis=1)) * 2 / w.sum()
            cols = [A[:, b].mean(axis=1) for b in bins] + [A[:, KLO:KHI + 1].mean(axis=1)]   # last = the firmware's band
            ok = valid[idx].mean(axis=1) >= 0.95
            B[c:c + 3000] = np.where(ok[:, None], np.stack(cols, axis=1), np.nan)
        out.append(pd.DataFrame(B, index=full.index[ends - 1] + pd.Timedelta(seconds=1),
                                columns=[f'b{i}' for i in range(len(bins))] + ['ring']))
        print('bands', y, m, flush=True)
    df = pd.concat(out); df.to_parquet(fp); return df


def weekly_rank(v, n_prev, n_min):
    """Rank of each value among the previous n_prev values (firmware rule), NaN until n_min exist."""
    a = np.concatenate([np.full(n_prev, np.nan), v.astype(np.float32)]); out = np.full(len(v), np.nan)
    win = sliding_window_view(a, n_prev + 1)
    for c in range(0, len(v), 20000):
        W_ = win[c:c + 20000]; last = W_[:, -1:]; prev = W_[:, :-1]; n = np.sum(~np.isnan(prev), axis=1)
        rk = (np.sum(prev < last, axis=1) + 0.5 * np.sum(prev == last, axis=1)) / np.maximum(n, 1)
        out[c:c + 20000] = np.where((n >= n_min) & ~np.isnan(last[:, 0]), rk, np.nan)
    return out


def features():
    fp = f'{CACHE}/band_ranks_{NOISE_MHZ:g}mHz.parquet'
    if os.path.exists(fp): return pd.read_parquet(fp)
    B = band_series()
    R = pd.DataFrame({c: weekly_rank(B[c].values, 2016, 288) for c in B.columns}, index=B.index)
    print('ranks done', flush=True)
    F = {}
    for c in B.columns:
        F[f'{c}_now'] = R[c]
        for h in (1, 6, 24): F[f'{c}_{h}h'] = R[c].rolling(h * 12, min_periods=h * 6).mean()
    F = pd.DataFrame(F); F.to_parquet(fp); return F


def ci_clean_rank():
    fuel = pd.read_csv('D:/gridseis_hum/df_fuel_ckan.csv', usecols=['DATETIME', 'CARBON_INTENSITY'])
    ci = pd.Series(fuel.CARBON_INTENSITY.values, index=pd.to_datetime(fuel.DATETIME)).sort_index()
    ci = ci[~ci.index.duplicated()]; ci = ci['2021-06-01':].asfreq('30min')
    # cleanliness rank: share of the previous 7 days that were dirtier (1 = cleanest of the week)
    rank_dirty = weekly_rank(-ci.values, 336, 300)
    return ci, pd.Series(rank_dirty, ci.index)


def report(name, pred, j):
    rho = spearmanr(pred, j.target).statistic
    fifth = pd.qcut(pred, 5, labels=False, duplicates='drop')
    ci_by = j.ci.groupby(fifth).mean().round(0).tolist()
    tgt_by = j.target.groupby(fifth).mean().round(2).tolist()
    lo, hi = fifth == fifth.min(), fifth == fifth.max()
    return {'agreement_with_real_weekly_rank': round(float(rho), 3),
            'power_station_fifth_dirtier_than_week_median': round(float((j.target[lo] < 0.5).mean()), 3),
            'turbine_fifth_cleaner_than_week_median': round(float((j.target[hi] > 0.5).mean()), 3),
            'mean_ci_by_fifth_dirty_to_clean': ci_by, 'mean_real_rank_by_fifth': tgt_by}


def main():
    F = features()
    Fh = F.resample('30min', label='left').mean()
    ci, target = ci_clean_rank()
    j = Fh.join(pd.DataFrame({'ci': ci, 'target': target}), how='inner').dropna()
    feats = [c for c in F.columns if not c.startswith('ring')]
    tr, te = j[j.index < TRAIN_END], j[j.index >= TEST_FROM]
    print('train', len(tr), 'test', len(te), flush=True)
    res = {'features': len(feats), 'train_rows': len(tr), 'test_rows': len(te), 'periods': {}}
    models = {
        'small trees (60 trees, depth 3)': XGBRegressor(n_estimators=60, max_depth=3, learning_rate=0.1, subsample=0.8),
        'larger trees (300 trees, depth 5)': XGBRegressor(n_estimators=300, max_depth=5, learning_rate=0.05, subsample=0.8, colsample_bytree=0.7),
    }
    preds = {'current rule (one band, now)': te['ring_now'].values,
             'current rule, 1 h average': te['ring_1h'].values}
    for name, mdl in models.items():
        mdl.fit(tr[feats], tr.target); preds[name] = mdl.predict(te[feats])
        if 'small' in name:
            imp = pd.Series(mdl.feature_importances_, feats).sort_values(ascending=False)
            res['small_model_top_features'] = imp.head(10).round(3).to_dict()
            mdl.save_model(f'{CACHE}/fft_small_model.json')
    # linear model on the same features: the tiniest option
    X = np.c_[tr[feats].values, np.ones(len(tr))]; coef = np.linalg.lstsq(X, tr.target.values, rcond=None)[0]
    preds['straight-line mix of bands'] = np.c_[te[feats].values, np.ones(len(te))] @ coef
    periods = {'2024': ('2024-01-01', '2025-01-01'), '2025': ('2025-01-01', '2026-01-01'),
               '2026 to Sep': ('2026-01-01', '2026-10-01'), 'all 2024 to 2026': ('2024-01-01', '2026-10-01')}
    for pname, (a, b) in periods.items():
        m = (te.index >= a) & (te.index < b); res['periods'][pname] = {}
        for name, p in preds.items():
            res['periods'][pname][name] = report(name, pd.Series(p[m], te.index[m]), te[m])
        print(pname, {k: v['agreement_with_real_weekly_rank'] for k, v in res['periods'][pname].items()}, flush=True)
    pd.DataFrame(preds, index=te.index).join(te[['ci', 'target']]).to_parquet(f'{CACHE}/fft_model_preds.parquet')
    json.dump(res, open(os.path.join(HERE, 'fft_model_results.json'), 'w'), indent=1)


if __name__ == '__main__':
    main()
