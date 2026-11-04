"""Build the on-board network's inputs from the band data on D: (run in the main environment, which has
xgboost for fft_model_test; tinyml_train.py then runs in the TensorFlow environment).

Only the board's own spectrum goes in: 13 bands, each placed against its own last seven days (one value
kept per 30 minutes, as model.c). One row per half hour:
  A  52 numbers: each band's latest placing and its 1, 6 and 24 hour averages (the weighted sum uses the
     39 averages)
  B  312 numbers: each band's hourly average placing for each of the last 24 hours (the network's input,
     as model_nn_input in model.c builds it)
plus the target (where the half hour's NESO carbon intensity sits in its previous seven days) and the
carbon intensity itself.

Writes D:/gridseis_hum/hardware_test/tinyml_inputs.npz.
Usage: python hardware/tools/tinyml_inputs.py          steady-state inputs
       python hardware/tools/tinyml_inputs.py warm     inputs for a board that has just started"""
import os, sys
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import fft_model_test as T  # noqa: E402
import train_fft_model as M  # noqa: E402

OUT = f'{T.CACHE}/tinyml_inputs.npz'


def band_ranks():
    fp = f'{T.CACHE}/band_thin_ranks.parquet'
    if os.path.exists(fp): return pd.read_parquet(fp)
    B = T.band_series()
    R = pd.DataFrame({f'b{i}': M.thin_rank(B[f'b{i}'].values.astype(np.float32)) for i in range(M.NB)}, index=B.index)
    R.to_parquet(fp); return R


def main():
    R = band_ranks()
    hourly = {c: R[c].rolling(12, min_periods=6).mean() for c in R.columns}   # mean of the last hour, every 5 min
    A, Bseq = [], []
    for c in R.columns:
        A += [R[c], hourly[c], R[c].rolling(72, min_periods=36).mean(), R[c].rolling(288, min_periods=144).mean()]
        for h in range(24): Bseq.append(hourly[c].shift(12 * h))   # the hour that ended h hours ago
    A = pd.concat(A, axis=1).values.astype(np.float32); Bs = pd.concat(Bseq, axis=1).values.astype(np.float32)
    j = M.joined(pd.DataFrame(index=R.index))
    keep = (~np.isnan(A).any(1)) & (~np.isnan(Bs).any(1)) & j.target.notna().values & (R.index.minute % 30 == 0)
    np.savez(OUT, A=A[keep], B=Bs[keep], y=j.target.values[keep].astype(np.float32), ci=j.ci.values[keep],
             t=R.index[keep].values)
    print('wrote', OUT, int(keep.sum()), 'half hours')


if __name__ == '__main__' and len(sys.argv) == 1:
    main()


# ---------------------------------------------------------------- inputs for a board that has just started
WARM_MIN_KEPT = 6     # kept band values (one per 30 min) before a band is placed: 3 hours


def warmup_inputs(seed=11):
    """The network's input as a board sees it in its first days: the five years are cut into stretches
    that each start as if the board had just been switched on (empty history), 1 to 8 days long, with
    some 20 to 40 day stretches for the steady state. Bands are placed against the stretch's own history
    once 6 values are kept; hours with no placings yet are given 0.5, as model_nn_input does.
    Writes tinyml_warm_inputs.npz with the hours since switch-on for each row (for scoring only; the
    network never sees it)."""
    fp = f'{T.CACHE}/tinyml_warm_inputs.npz'
    B = T.band_series(); n = len(B); rng = np.random.default_rng(seed)
    starts = [0]
    while starts[-1] < n:
        days = rng.uniform(20, 40) if rng.random() < 0.3 else rng.uniform(1, 8)
        starts.append(starts[-1] + int(days * 288))
    starts = np.array(starts[:-1] + [n]); seg = np.searchsorted(starts, np.arange(n), side='right') - 1
    elapsed = (np.arange(n) - starts[seg]).astype(np.int32)   # 5-minute slots since switch-on
    R = np.full((n, M.NB), np.nan, np.float32)
    for a, b in zip(starts[:-1], starts[1:]):
        for i in range(M.NB):
            R[a:b, i] = M.thin_rank(B[f'b{i}'].values[a:b].astype(np.float32), nmin=WARM_MIN_KEPT)
    print('placings done', flush=True)
    # hourly means within each stretch: mean of the last 12 slots, at least 6 present, else 0.5
    Rdf = pd.DataFrame(R)
    Xh = np.full((n, 24, M.NB), 0.5, np.float32)
    for h in range(24):
        shifted = Rdf.groupby(seg).shift(12 * h)
        m = shifted.groupby(seg).rolling(12, min_periods=6).mean().reset_index(level=0, drop=True).sort_index().values
        Xh[:, h, :] = np.where(np.isnan(m), 0.5, m)
    hour0_ok = ~np.isnan(Rdf.groupby(seg).rolling(12, min_periods=6).mean().reset_index(level=0, drop=True).sort_index().values).any(1)
    j = M.joined(pd.DataFrame(index=B.index))
    keep = hour0_ok & j.target.notna().values & (B.index.minute % 30 == 0)
    np.savez(fp, X=Xh[keep], y=j.target.values[keep].astype(np.float32), ci=j.ci.values[keep],
             t=B.index[keep].values, elapsed_h=elapsed[keep] / 12.0)
    print('wrote', fp, int(keep.sum()), 'half hours')


if __name__ == '__main__' and len(sys.argv) > 1 and sys.argv[1] == 'warm':
    warmup_inputs()
