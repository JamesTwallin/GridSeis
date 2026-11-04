"""Desk checks for the weather-house firmware, on real NESO data.

1. Host test: compiles the firmware's freq.c and ringing.c for the PC (via `python -m ziglang cc`),
   feeds them a synthesised antenna signal built from one day of NESO one-second frequency, and
   compares the recovered frequency and ringing with the truth and with a numpy reference that
   uses the same recipe as the GridSeis analysis.
2. Gauge check: runs the firmware's gauge rule (30-minute ringing every 5 minutes, ranked against
   the previous seven days) over a full year of NESO one-second frequency, with extra white noise
   added to stand in for a noisier home measurement, and compares the gauge with NESO's carbon
   intensity.

Usage: python hardware/tools/desk_check.py
Needs D:/gridseis_hum/freq_parquet/fnew-YYYY-M.parquet and D:/gridseis_hum/df_fuel_ckan.csv.
Writes hardware/tools/desk_check_results.json and hardware/images/gauge_check.png."""
import json, os, subprocess, sys
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__)); HW = os.path.dirname(HERE); FW = os.path.join(HW, 'firmware', 'src')
WORK = 'D:/gridseis_hum/hardware_test'; os.makedirs(WORK, exist_ok=True)
sys.path.insert(0, os.path.join(HW, '..', 'scripts', 'primer'))
FREQ_DIRS = ['D:/gridseis_hum/freq_parquet', os.path.join(HW, '..', 'raw_data')]
WIN, HOP, KLO, KHI = 1800, 300, 180, 450


def load_month(y, m):
    for d in FREQ_DIRS:
        fp = f'{d}/fnew-{y}-{m}.parquet'
        if os.path.exists(fp):
            df = pd.read_parquet(fp); t = pd.to_datetime(df.dtm, format='mixed', errors='coerce')
            s = pd.Series(pd.to_numeric(df.f, errors='coerce').values, index=t).dropna(); s = s[(s > 49) & (s < 51)]
            return s[~s.index.duplicated()].sort_index()
    raise FileNotFoundError(f'{y}-{m}')


def ringing_numpy(f_hz):
    """Same recipe as ringing.c and the GridSeis spectra: mean removed, Hann window, |rfft|*2/sum(w), mean of bins 0.1-0.25 Hz."""
    x = (np.asarray(f_hz, float) - np.mean(f_hz)) * 1000; w = np.hanning(WIN)
    a = np.abs(np.fft.rfft(x * w)) * 2 / w.sum(); return a[KLO:KHI + 1].mean()


# ---------------------------------------------------------------- 1. host test of the firmware C code
def host_test():
    exe = os.path.join(WORK, 'host_test.exe')
    cmd = [sys.executable, '-m', 'ziglang', 'cc', '-O2', '-std=gnu17', '-I', FW, os.path.join(HERE, 'host_test.c'), os.path.join(FW, 'freq.c'), os.path.join(FW, 'ringing.c'), '-o', exe, '-lm']
    subprocess.run(cmd, check=True); print('compiled firmware DSP for the host')
    day = load_month(2025, 1)['2025-01-15'].resample('1s').mean().interpolate(limit_direction='both')
    fin = os.path.join(WORK, 'day.txt'); np.savetxt(fin, day.values, fmt='%.5f')
    out = {}
    # A: 50 Hz amplitude in ADC counts; H: 150 Hz harmonic; noise: ADC noise in counts (1 count ~ 0.8 mV at 12 dB attenuation)
    for label, A, H, noise in [('strong pickup', 300, 450, 20), ('typical pickup', 60, 90, 20), ('weak pickup', 15, 25, 20)]:
        fo = os.path.join(WORK, f'out_{A}.txt'); subprocess.run([exe, fin, str(A), str(H), str(noise), fo], check=True)
        rows = [l.strip().split(',') for l in open(fo)]
        sec = pd.DataFrame([r[1:] for r in rows if r[0] == 'SEC'], columns=['s', 'truth', 'rec', 'ok']).astype(float)
        rng = pd.DataFrame([r[1:] for r in rows if r[0] == 'RING'], columns=['s', 'truth', 'rec']).astype(float)
        good = sec[sec.ok == 1]; err = (good.rec - good.truth) * 1000
        ref = [ringing_numpy(day.values[int(s) - WIN:int(s)]) for s in rng.s]
        out[label] = dict(adc_amplitude_counts=A, harmonic_counts=H, noise_counts=noise, valid_share=float(sec.ok.mean()),
                          freq_error_mhz_sd=float(err.std()), freq_error_mhz_mean=float(err.mean()),
                          ringing_truth_c_vs_numpy_max_rel_diff=float(np.max(np.abs(rng.truth.values - ref) / np.array(ref))),
                          ringing_recovered_over_truth_median=float(np.median(rng.rec / rng.truth)))
        print(label, json.dumps(out[label]))
    return out


# ---------------------------------------------------------------- 2. the gauge over a year, with extra noise
def gauge_check(noise_levels=(0.0, 1.0, 2.0, 5.0)):
    f = pd.concat([load_month(y, m) for y, m in [(2024, 10), (2024, 11), (2024, 12)] + [(2025, m) for m in range(1, 10)]])
    full = f.reindex(pd.date_range('2024-10-01', '2025-10-01', freq='1s', inclusive='left'))
    valid = full.notna().values; vals = full.interpolate(limit_direction='both').values
    fuel = pd.read_csv('D:/gridseis_hum/df_fuel_ckan.csv', usecols=['DATETIME', 'CARBON_INTENSITY']); ci = pd.Series(fuel.CARBON_INTENSITY.values, index=pd.to_datetime(fuel.DATETIME)).sort_index()
    ends = np.arange(WIN, len(vals) + 1, HOP); t_end = full.index[ends - 1] + pd.Timedelta(seconds=1)
    okw = np.array([valid[e - WIN:e].mean() >= 0.95 for e in ends]); w = np.hanning(WIN); rng = np.random.default_rng(1)
    res = {}; curves = {}
    for sd in noise_levels:
        x = vals + (rng.normal(0, sd / 1000, len(vals)) if sd else 0)
        ring = np.empty(len(ends))
        for c in range(0, len(ends), 4000):
            idx = ends[c:c + 4000][:, None] - WIN + np.arange(WIN)[None, :]; X = x[idx]; X = (X - X.mean(axis=1, keepdims=True)) * 1000
            ring[c:c + 4000] = (np.abs(np.fft.rfft(X * w, axis=1))[:, KLO:KHI + 1] * 2 / w.sum()).mean(axis=1)
        r = pd.Series(np.where(okw, ring, np.nan), index=t_end)
        # firmware rule: rank of the new value among the previous seven days (2016 values), needs one day of history
        rank = r.rolling(2017, min_periods=289).apply(lambda a: (np.sum(a[:-1] < a[-1]) + 0.5 * np.sum(a[:-1] == a[-1])) / np.sum(~np.isnan(a[:-1])) if not np.isnan(a[-1]) else np.nan, raw=True)
        gauge = rank.resample('30min', label='left').mean()   # value for each settlement period
        j = pd.concat([gauge.rename('gauge'), ci.rename('ci')], axis=1).dropna()
        q = pd.qcut(j.gauge, 5, labels=False); typ = j.groupby(q).ci.median()
        res[f'{sd:g} mHz'] = dict(n=int(len(j)), rank_corr=float(j.gauge.corr(j.ci, method='spearman')),
                                 ci_when_turbine_out=float(typ.iloc[-1]), ci_when_station_out=float(typ.iloc[0]), ci_by_fifth=[float(v) for v in typ.values])
        curves[sd] = typ.values; print('noise', sd, 'mHz', json.dumps(res[f'{sd:g} mHz']))
    return res, curves


if __name__ == '__main__':
    results = {'host_test': host_test()}
    g, curves = gauge_check(); results['gauge_check'] = g
    json.dump(results, open(os.path.join(HERE, 'desk_check_results.json'), 'w'), indent=1)
    from style import tidy, title, INK  # noqa: E402
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(9, 4.4), gridspec_kw=dict(top=0.72, bottom=0.16, left=0.08, right=0.97))
    cols = ['#0f766e', '#2563eb', '#c2410c', '#6b7280']
    for (sd, typ), c in zip(curves.items(), cols): ax.plot(range(1, 6), typ, color=c, lw=2, marker='o', label=f'extra measurement noise {sd:g} mHz')
    ax.set_xticks(range(1, 6)); ax.set_xticklabels(['power station\nfully out', 'power station\nhalf out', 'both\nin', 'turbine\nhalf out', 'turbine\nfully out'])
    ax.set_ylabel('typical NESO carbon intensity, gCO$_2$/kWh'); ax.set_ylim(0); tidy(ax); ax.legend(frameon=False, fontsize=8.5)
    title(fig, 'Weather-house position against carbon intensity, Oct 2024 to Sep 2025', y=0.955, size=16)
    fig.text(0.02, 0.93, 'The firmware rule run on a year of NESO one-second frequency: 30-minute ringing every 5 minutes, ranked against the previous' + chr(10) +
             'seven days, split into five positions of the figures. White noise is added to stand in for a noisier home measurement.', fontsize=10, color=INK, va='top', linespacing=1.35)
    fig.text(0.02, 0.015, 'Data: NESO system frequency (1 s); NESO generation mix carbon intensity', fontsize=8, color='grey')
    fig.savefig(os.path.join(HW, 'images', 'gauge_check.png'), dpi=200, bbox_inches='tight', pad_inches=0.25); print('saved gauge_check.png')
