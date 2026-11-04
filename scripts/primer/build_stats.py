"""Half-hourly statistics of GB grid frequency from NESO 1-second data (parquet caches on D: and in GridSeis/raw_data).
Output: D:/gridseis_hum/freq_hh_stats.parquet  (one row per half hour)"""
import glob, os, numpy as np, pandas as pd
SRC = glob.glob('D:/gridseis_hum/freq_parquet/fnew-*.parquet') + glob.glob('C:/Users/User/Documents/GitHub/GridSeis/raw_data/fnew-*.parquet')
OUT = 'D:/gridseis_hum/freq_hh_stats.parquet'
def month_key(p): y, m = os.path.basename(p)[5:-8].split('-'); return int(y) * 100 + int(m)
files = {}
for p in SRC: files.setdefault(month_key(p), p)   # D: copy wins if both exist
rows = []
for k in sorted(files):
    df = pd.read_parquet(files[k]); df['dtm'] = pd.to_datetime(df.dtm, format='mixed', errors='coerce'); df['f'] = pd.to_numeric(df.f, errors='coerce')
    df = df.dropna().drop_duplicates('dtm').set_index('dtm').sort_index(); f = df.f[(df.f > 49) & (df.f < 51)]
    f = f.asfreq('1s'); d1 = f.diff(); d10 = (f - f.shift(10)) / 10
    g = f.groupby(pd.Grouper(freq='30min'))
    s = pd.DataFrame({'n': g.count(), 'mean': g.mean(), 'std': g.std(), 'min': g.min(), 'max': g.max(),
                      'rocof_rms_mHz_s': d1.groupby(pd.Grouper(freq='30min')).std() * 1000,
                      'rocof10_max_mHz_s': d10.abs().groupby(pd.Grouper(freq='30min')).max() * 1000,
                      'outside_100mHz_s': (f.sub(50).abs() > 0.1).groupby(pd.Grouper(freq='30min')).sum(),
                      'excursion_mHz': (g.max() - g.min()) * 1000})
    s = s[s.n >= 1500]; rows.append(s); print(k, len(s), flush=True)
S = pd.concat(rows).sort_index(); S = S[~S.index.duplicated()]; S.to_parquet(OUT); print('saved', S.shape, S.index.min(), S.index.max())
