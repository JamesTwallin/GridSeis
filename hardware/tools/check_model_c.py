"""Check the firmware's spectrum model (model.c, ringing.c) against the Python training pipeline.

Takes 10 days of NESO one-second frequency (late September 2026) with 2.5 mHz of noise and a
40-minute gap of bad seconds, runs it through host_model_test (the firmware's C code compiled for the
PC with `python -m ziglang cc`) and through train_fft_model.py's steps, and compares the band
amplitudes, the predictions and the figure positions.

Usage: python hardware/tools/check_model_c.py"""
import json, os, subprocess, sys, tempfile
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__)); SRC = os.path.join(os.path.dirname(HERE), 'firmware', 'src')
sys.path.insert(0, HERE)
import fft_model_test as T  # noqa: E402
import train_fft_model as M  # noqa: E402

work = tempfile.mkdtemp()
f = T.load_month(2026, 9)
f = f.reindex(pd.date_range('2026-09-20', '2026-09-30', freq='1s', inclusive='left')).interpolate(limit_direction='both')
x = f.values + np.random.default_rng(3).normal(0, 0.0025, len(f))
x[5 * 86400:5 * 86400 + 2400] = np.nan          # a 40-minute gap: three slots must come out invalid
np.savetxt(os.path.join(work, 'freq.txt'), x, fmt='%.6f')
exe = os.path.join(work, 'host_model_test.exe')
subprocess.run([sys.executable, '-m', 'ziglang', 'cc', '-O2', '-I', SRC, os.path.join(HERE, 'host_model_test.c'),
                os.path.join(SRC, 'model.c'), os.path.join(SRC, 'ringing.c'), '-o', exe], check=True)
subprocess.run([exe, os.path.join(work, 'freq.txt'), os.path.join(work, 'out.txt')], check=True)
c = pd.read_csv(os.path.join(work, 'out.txt'), header=None)
c.columns = ['tag', 'sec', 'valid'] + [f'b{i}' for i in range(M.NB)] + ['pred', 'pos']

# Python: same windows, numpy FFT in double precision
W = T.WIN; w = np.hanning(W); bins = T.band_bins(); rows = []
for e in c.sec.values:
    seg = x[e - W:e]
    if np.isnan(seg).mean() > 0.05: rows.append([np.nan] * M.NB); continue
    seg = pd.Series(seg).interpolate(limit_direction='both').values
    A = np.abs(np.fft.rfft((seg - seg.mean()) * 1000 * w)) * 2 / w.sum()
    rows.append([A[b].mean() for b in bins])
P = np.array(rows, np.float32)
res = json.load(open(os.path.join(HERE, 'train_fft_model_results.json')))
wts = res['weights']; sc = res['position_scale']
pred = np.full(len(P), wts['constant'])
for i in range(M.NB):
    for wlen, avg in zip(M.AVG, M.rolling_mean_ranks(M.thin_rank(P[:, i]))):
        pred = pred + wts[f'b{i}_{wlen // 12}h'] * avg
pos = M.position(pred, sc['fully_station'], sc['centre'], sc['fully_turbine'])

bc = c[[f'b{i}' for i in range(M.NB)]].values
rel = np.nanmax(np.abs(bc - P) / P)
both = ~np.isnan(pred) & ~np.isnan(c.pred.values)
print(f'slots {len(c)}, invalid in C {int((c.valid == 0).sum())}, invalid in Python {int(np.isnan(P[:, 0]).sum())}')
print(f'band amplitudes: largest relative difference {rel:.2e}')
print(f'predictions: {both.sum()} in both, {int(np.isnan(pred).sum())} not ready in Python, {int(c.pred.isna().sum())} in C')
print(f'prediction: largest difference {np.max(np.abs(pred[both] - c.pred.values[both])):.2e}, '
      f'position: largest difference {np.max(np.abs(pos[both] - c.pos.values[both])):.2e}')
