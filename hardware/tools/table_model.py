"""A lookup-table model for the weather house: carbon intensity from the weekly ringing rank and the
time of day, trained on NESO data, and a test of whether it can keep itself up to date.

1. Runs the firmware's gauge rule (30-minute ringing every 5 minutes, ranked against the previous seven
   days) over every month of NESO one-second frequency from July 2021 to September 2026, with white
   noise added to stand in for a home measurement. Cached on D:.
2. Trains tables on the first year (July 2021 to June 2022): by rank only, by time of day only, and by
   rank and time of day together, at a few sizes.
3. Tests them on the four years after, two ways: left fixed, and updated every half hour by nudging the
   table cell towards the real figure once it is known (what a WiFi-connected gauge could do).

Usage: python hardware/tools/table_model.py
Writes hardware/tools/table_model_results.json."""
import json, os, sys
import numpy as np, pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

HERE = os.path.dirname(os.path.abspath(__file__)); HW = os.path.dirname(HERE)
sys.path.insert(0, HERE)
from desk_check import load_month, WIN, HOP, KLO, KHI  # noqa: E402
CACHE = 'D:/gridseis_hum/hardware_test'
MONTHS = [(y, m) for y in range(2021, 2027) for m in range(1, 13) if (2021, 7) <= (y, m) <= (2026, 9)]
NOISE_MHZ = 2.5                      # typical wire pickup, from the host test
TRAIN_END = '2022-07-01'
ALPHA = 0.05                         # how far each update moves a table cell towards the real figure


# ---------------------------------------------------------------- 1. the gauge rule over five years
def ringing_series(noise_mhz):
    fp = f'{CACHE}/ringing_{noise_mhz:g}mHz.parquet'
    if os.path.exists(fp): return pd.read_parquet(fp).ring
    rng = np.random.default_rng(7); w = np.hanning(WIN); out = []
    for y, m in MONTHS:
        f = load_month(y, m); start = pd.Timestamp(y, m, 1); end = start + pd.offsets.MonthBegin(1)
        full = f.reindex(pd.date_range(start, end, freq='1s', inclusive='left'))
        valid = full.notna().values; x = full.interpolate(limit_direction='both').values
        if noise_mhz: x = x + rng.normal(0, noise_mhz / 1000, len(x))
        ends = np.arange(WIN, len(x) + 1, HOP); ring = np.full(len(ends), np.nan)
        for c in range(0, len(ends), 4000):
            e = ends[c:c + 4000]; idx = e[:, None] - WIN + np.arange(WIN)[None, :]
            X = x[idx]; X = (X - X.mean(axis=1, keepdims=True)) * 1000
            r = (np.abs(np.fft.rfft(X * w, axis=1))[:, KLO:KHI + 1] * 2 / w.sum()).mean(axis=1)
            ok = valid[idx].mean(axis=1) >= 0.95; ring[c:c + 4000] = np.where(ok, r, np.nan)
        out.append(pd.Series(ring, index=full.index[ends - 1] + pd.Timedelta(seconds=1))); print('ringing', y, m, flush=True)
    s = pd.concat(out).rename('ring'); s.to_frame().to_parquet(fp); return s


def weekly_rank(r):
    """Firmware rule: rank of each value among the previous 2016 (seven days), needing 288 (one day)."""
    a = np.concatenate([np.full(2016, np.nan), r.values.astype(np.float32)]); out = np.full(len(r), np.nan)
    win = sliding_window_view(a, 2017)
    for c in range(0, len(r), 20000):
        W_ = win[c:c + 20000]; last = W_[:, -1:]; prev = W_[:, :-1]; n = np.sum(~np.isnan(prev), axis=1)
        rk = (np.sum(prev < last, axis=1) + 0.5 * np.sum(prev == last, axis=1)) / np.maximum(n, 1)
        out[c:c + 20000] = np.where((n >= 288) & ~np.isnan(last[:, 0]), rk, np.nan)
    return pd.Series(out, index=r.index)


def half_hourly(noise_mhz):
    rank = weekly_rank(ringing_series(noise_mhz))
    g = rank.resample('30min', label='left').mean()
    fuel = pd.read_csv('D:/gridseis_hum/df_fuel_ckan.csv', usecols=['DATETIME', 'CARBON_INTENSITY'])
    ci = pd.Series(fuel.CARBON_INTENSITY.values, index=pd.to_datetime(fuel.DATETIME)).sort_index()
    ci = ci[~ci.index.duplicated()]
    j = pd.concat([g.rename('rank'), ci.rename('ci')], axis=1).dropna()
    j['hour'] = j.index.tz_localize('UTC').tz_convert('Europe/London').hour
    return j


# ---------------------------------------------------------------- 2 and 3. tables, fixed and self-updating
def cells(j, n_rank, n_time):
    rb = np.minimum((j['rank'].values * n_rank).astype(int), n_rank - 1) if n_rank else np.zeros(len(j), int)
    tb = (j.hour.values * n_time // 24) if n_time else np.zeros(len(j), int)
    return rb * max(n_time, 1) + tb


def train_table(j, n_rank, n_time):
    c = cells(j, n_rank, n_time); size = max(n_rank, 1) * max(n_time, 1)
    tab = pd.Series(j.ci.values).groupby(c).median().reindex(range(size))
    return tab.fillna(j.ci.median()).values


def run(j_test, table, n_rank, n_time, alpha):
    """Predict each half hour from the table, then (if alpha > 0) move that cell towards the real value."""
    tab = table.astype(float).copy(); c = cells(j_test, n_rank, n_time); pred = np.empty(len(c)); y = j_test.ci.values
    for i, (k, v) in enumerate(zip(c, y)):
        pred[i] = tab[k]
        if alpha: tab[k] += alpha * (v - tab[k])
    return pred, tab


def scores(y, p):
    return dict(typical_error_g=float(np.mean(np.abs(y - p))), share_explained=float(1 - np.sum((y - p) ** 2) / np.sum((y - y.mean()) ** 2)))


MODELS = {'time of day only (8 blocks)': (0, 8), 'rank only (5 bands)': (5, 0), 'rank only (10 bands)': (10, 0),
          'rank 5 x time 8': (5, 8), 'rank 5 x time 24': (5, 24), 'rank 10 x time 8': (10, 8), 'rank 10 x time 24': (10, 24)}


if __name__ == '__main__':
    res = {}
    for noise in (NOISE_MHZ, 0.0):
        j = half_hourly(noise); tr, te = j[j.index < TRAIN_END], j[j.index >= TRAIN_END]
        print(f'noise {noise} mHz: {len(tr)} training half hours, {len(te)} test half hours, test spread {te.ci.std():.0f} g')
        out = {}
        for name, (nr, nt) in MODELS.items():
            tab = train_table(tr, nr, nt)
            fixed, _ = run(te, tab, nr, nt, 0); upd, _ = run(te, tab, nr, nt, ALPHA)
            per_year = {}
            for yr in sorted(set(te.index.year)):
                m = te.index.year == yr; y = te.ci.values[m]
                per_year[int(yr)] = dict(fixed=scores(y, fixed[m]), updating=scores(y, upd[m]), mean_ci=float(y.mean()))
            out[name] = dict(fixed=scores(te.ci.values, fixed), updating=scores(te.ci.values, upd), per_year=per_year)
            f_, u_ = out[name]['fixed'], out[name]['updating']
            print(f"  {name:28s} fixed {f_['typical_error_g']:5.1f} g {f_['share_explained']:6.2f}   updating {u_['typical_error_g']:5.1f} g {u_['share_explained']:6.2f}", flush=True)
        res[f'{noise:g} mHz'] = out
        if noise == NOISE_MHZ: jn = j
    # ways of keeping the 5-band rank table up to date that leave the sensor doing the work
    j = jn; tr, te = j[j.index < TRAIN_END], j[j.index >= TRAIN_END]; y = te.ci.values; extra = {}
    tab = train_table(tr, 5, 0)
    for a in (0.02, 0.05, 0.1):
        extra[f'updated every half hour, step {a:g}'] = scores(y, run(te, tab, 5, 0, a)[0])
    # updated only during a two-week calibration window at the start of each quarter, fixed in between
    cal = np.array([(t.month % 3 == 1) and t.day <= 14 for t in te.index]); t_ = tab.astype(float).copy(); pred = np.empty(len(te))
    for i, (k, v) in enumerate(zip(cells(te, 5, 0), y)):
        pred[i] = t_[k]
        if cal[i]: t_[k] += 0.05 * (v - t_[k])
    extra['updated two weeks each quarter'] = scores(y, pred)
    # band offsets from the training year, plus last month's average carbon intensity (one number a month)
    off = pd.Series(tr.ci.values - tr.ci.groupby(tr.index.to_period('M')).transform('mean').values).groupby(cells(tr, 5, 0)).median().values
    monthly = j.ci.groupby(j.index.to_period('M')).mean(); last = monthly.shift(1).reindex(te.index.to_period('M')).values
    extra["last month's average plus the rank band's offset"] = scores(y, last + off[cells(te, 5, 0)])
    extra["last month's average alone"] = scores(y, last)
    # no sensor at all: show the previous half hour's published figure
    prev = j.ci.shift(1).reindex(te.index).values; ok = ~np.isnan(prev)
    extra['no sensor: previous half hour from the internet'] = scores(y[ok], prev[ok])
    for k, v in extra.items(): print(f"  {k:52s} {v['typical_error_g']:5.1f} g {v['share_explained']:6.2f}")
    res['ways of updating, 2.5 mHz'] = extra; res['rank_offsets_g'] = [float(v) for v in off]
    res['table_trained_2021_22'] = [float(v) for v in tab]
    json.dump(res, open(os.path.join(HERE, 'table_model_results.json'), 'w'), indent=1)
