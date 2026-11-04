"""One half hour of grid frequency split into the waves inside it, and its spectrum: a plain-language Fourier primer figure.
Usage: fourier_primer_plot.py [wide]
Output: plots/primer/waves.png (run from the repository root)"""
import os, sys, numpy as np, pandas as pd
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'primer'))
from style import tidy, title, INK, GREY, HEAVY  # noqa: E402
import matplotlib.pyplot as plt, matplotlib.dates as mdates; plt.rcParams['axes.unicode_minus'] = False
WIDE = len(sys.argv) > 1 and sys.argv[1] == 'wide'; TEAL = '#0f766e'; ORANGE = '#c2410c'
OUT = 'plots/primer'; os.makedirs(OUT, exist_ok=True); FS = (9, 4.4)
FREQ_DIRS = ['D:/gridseis_hum/freq_parquet', 'raw_data']
T0 = pd.Timestamp('2024-12-09 02:00:00')                      # a night half hour with the hum on and the grid light
for d in FREQ_DIRS:
    fp = f'{d}/fnew-2024-12.parquet'
    if os.path.exists(fp):
        df = pd.read_parquet(fp); df['dtm'] = pd.to_datetime(df.dtm, format='mixed', errors='coerce'); df['f'] = pd.to_numeric(df.f, errors='coerce')
        f = df.dropna().drop_duplicates('dtm').set_index('dtm').sort_index().f; break
seg = f.reindex(pd.date_range(T0, periods=1800, freq='s')).interpolate(limit_direction='both')
raw = (seg.values - 50) * 1000; dev = raw - raw.mean(); N = 1800
freqs = np.fft.rfftfreq(N, d=1.0); win = np.hanning(N); spec = np.abs(np.fft.rfft(dev * win) * (2.0 / win.sum()))
F = np.fft.rfft(dev)
def band(lo, hi):
    G = F.copy(); G[(freqs < lo) | (freqs > hi)] = 0; return np.fft.irfft(G, n=N)
bands = [('slow swings, periods over 30 s', band(0.0005, 1 / 30), HEAVY, (0.0005, 1 / 30)),
         ('the ringing, 4 to 10 s periods', band(0.1, 0.25), TEAL, (0.1, 0.25)),
         ('the hum, 3.88 s period', band(0.255, 0.261), ORANGE, (0.255, 0.261))]
t = seg.index; w = slice(600, 900)                            # five minutes from the middle of the half hour
fig = plt.figure(figsize=FS)
if WIDE:
    gs = fig.add_gridspec(4, 2, width_ratios=[1.25, 1], hspace=0.4, wspace=0.28, top=0.70, bottom=0.19, left=0.075, right=0.985); left = [fig.add_subplot(gs[i, 0]) for i in range(4)]; axS = fig.add_subplot(gs[:, 1])
else:
    g1 = fig.add_gridspec(4, 1, hspace=0.45, top=0.80, bottom=0.50, left=0.1, right=0.97); g2 = fig.add_gridspec(1, 1, top=0.40, bottom=0.10, left=0.1, right=0.97)
    left = [fig.add_subplot(g1[i, 0]) for i in range(4)]; axS = fig.add_subplot(g2[0, 0])
rows = [('the frequency as published, minus 50 Hz', raw, INK)] + [(n, y, c) for n, y, c, _ in bands]
from matplotlib.ticker import MaxNLocator
for ax, (name, y, c), last in zip(left, rows, [False, False, False, True]):
    ax.plot(t[w], y[w], color=c, lw=0.9); ax.set_ylabel('mHz', fontsize=8); tidy(ax); ax.tick_params(labelsize=7.5)
    ax.text(0.01, 1.02, name, transform=ax.transAxes, va='bottom', fontsize=8.5, color=c)
    if name.startswith('the frequency'): lo, hi = y[w].min(), y[w].max(); pad = (hi - lo) * 0.3; ax.set_ylim(lo - pad, hi + pad)
    else: m = np.abs(y[w]).max() * 1.3; ax.set_ylim(-m, m)
    ax.yaxis.set_major_locator(MaxNLocator(3)); ax.xaxis.set_major_locator(mdates.MinuteLocator(interval=1))
    if last: ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M')); ax.set_xlabel(f'{T0:%d %b %Y}, UTC', fontsize=8.5)
    else: ax.tick_params(labelbottom=False)
    ax.set_xlim(t[w][0], t[w][-1])
left[0].set_title('Five minutes of one half hour, and three of the waves inside it', loc='left', fontsize=9.5 if WIDE else 10, pad=16)
ax = axS
ok = freqs >= 0.005; ax.plot(freqs[ok], spec[ok], color=INK, lw=0.7, zorder=3)
for name, _, c, (lo, hi) in bands: ax.axvspan(lo, hi, color=c, alpha=0.18 if c != ORANGE else 0.35, lw=0, zorder=1)
ax.set_xscale('log'); ax.set_yscale('log'); tidy(ax); ax.tick_params(labelsize=7.5)
tk = [0.01, 0.02, 0.05, 0.1, 0.2, 0.5]; ax.set_xticks(tk); ax.set_xticklabels([f'{x:g} Hz' + chr(10) + f'({1/x:g} s)' for x in tk], fontsize=7.5); ax.set_xticks([], minor=True)
ax.set_xlabel('how fast the wave repeats (left: slow, right: fast)', fontsize=8.5); ax.set_ylabel('size of the wave, mHz', fontsize=8.5)
ax.set_xlim(0.005, 0.5); ax.set_ylim(0.01, 30); ax.set_yticks([0.01, 0.1, 1, 10]); ax.set_yticklabels(['0.01', '0.1', '1', '10']); ax.set_yticks([], minor=True)
ax.text(0.012, 8, 'slow swings', color=HEAVY, fontsize=8.5); ax.text(0.105, 3, 'the ringing', color=TEAL, fontsize=8.5)
ax.annotate('the hum', xy=(0.258, spec[np.argmin(np.abs(freqs - 0.258))]), xytext=(0.3, 2), fontsize=8.5, color=ORANGE, arrowprops=dict(arrowstyle='->', color=ORANGE, lw=0.8))
ax.set_title('The same half hour as a spectrum: the size of the wave at every speed', loc='left', fontsize=9.5 if WIDE else 10)
title(fig, 'One half hour of frequency, and the waves inside it', y=0.955 if WIDE else 0.952, size=18 if WIDE else 20)
sub = ('Any wiggly line is the sum of smooth waves of different speeds. The Fourier transform works out how big the wave at\n'
       'each speed is. Left: the raw signal and three of the waves it contains, pulled out by keeping only one range of speeds.\n'
       'Right: the size of every wave in that half hour, which is the spectrum used throughout this document.')
fig.text(0.02, 0.93 if WIDE else 0.938, sub, fontsize=10.5 if WIDE else 11.5, color=INK, va='top', linespacing=1.35 if WIDE else 1.4)
fig.text(0.02, 0.015 if WIDE else 0.012, f'Data: NESO system frequency, 1-second readings, {T0:%d %B %Y} {T0:%H:%M} to {T0 + pd.Timedelta(minutes=30):%H:%M} UTC; 30-minute spectrum, same method as every spectrum in this document', fontsize=8 if WIDE else 8.5, color='grey')
fig.savefig(f'{OUT}/waves.png', dpi=200, bbox_inches='tight', pad_inches=0.25); print('saved', OUT, 'hum peak', round(float(spec[np.argmin(np.abs(freqs - 0.258))]), 3), 'gaps', int(f.reindex(pd.date_range(T0, periods=1800, freq='s')).isna().sum()))
