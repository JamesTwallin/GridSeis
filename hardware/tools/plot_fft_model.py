"""Figures for the spectrum model write-up (hardware/README.md, "The spectrum model").

Uses the frozen check from train_fft_model.py: weights fitted on July 2021 to December 2023, then
run unchanged over January 2024 to September 2026. Needs the feature cache that script builds on D:.

Writes hardware/images/model_weeks.png, model_by_year.png and model_bands.png.
Usage: python hardware/tools/plot_fft_model.py"""
import os, sys
import numpy as np, pandas as pd
from scipy.stats import spearmanr

HERE = os.path.dirname(os.path.abspath(__file__)); HW = os.path.dirname(HERE)
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HW, '..', 'scripts', 'primer'))
import train_fft_model as M  # noqa: E402
import fft_model_test as T  # noqa: E402
from style import tidy, title, INK, GREY  # noqa: E402
import matplotlib.pyplot as plt, matplotlib.dates as mdates  # noqa: E402

TEAL, ORANGE = '#0f766e', '#c2410c'
OUT = os.path.join(HW, 'images')


def frozen_check():
    F = M.features(); j = M.joined(F)
    cols = [c for c in F.columns if c.startswith('b')]
    ok = j[cols + ['target']].notna().all(axis=1)
    w = M.fit(j[ok & (j.index < '2024-01-01')], cols)
    te = j[(j.index >= '2024-01-01') & ok & j.rule.notna()].copy()
    te['model'] = M.predict(te, cols, w)
    lo, mid, hi = np.percentile(M.predict(j[ok & (j.index < '2024-01-01')], cols, w), [10, 50, 90])
    te['shown'] = (M.position(te.model.values, lo, mid, hi) + 1) / 2   # 0 power station fully out, 1 turbine fully out
    h = te[['model', 'shown', 'rule', 'target', 'ci']].resample('30min', label='left').mean().dropna()
    return h, w, cols


def weeks_figure(h):
    # a fortnight whose agreement is closest to the typical fortnight, so the example is not picked to flatter
    starts = pd.date_range('2024-01-01', '2026-09-01', freq='14D'); scores = []
    for s in starts:
        x = h[s:s + pd.Timedelta(days=14)]
        if len(x) > 600: scores.append((spearmanr(x.model, x.target).statistic, s))
    typ = np.median([a for a, _ in scores]); s = min(scores, key=lambda t: abs(t[0] - typ))[1]
    x = h[s:s + pd.Timedelta(days=14)]
    fig, axes = plt.subplots(2, 1, figsize=(12, 6.2), sharex=True)
    for ax, col, colour, label in [(axes[0], 'rule', GREY, 'current rule: one band, as the figures show it'),
                                   (axes[1], 'shown', TEAL, 'spectrum model: 13 bands, as the figures show it')]:
        ax.plot(x.index, x.target, color=ORANGE, lw=1.8, label='real: carbon intensity against its previous 7 days')
        ax.plot(x.index, x[col], color=colour, lw=1.8, label=label)
        ax.set_ylim(0, 1); ax.set_yticks([0, 0.5, 1]); ax.set_yticklabels(['dirtiest /\npower station out', 'middle /\nboth inside', 'cleanest /\nturbine out'])
        tidy(ax); ax.legend(frameon=False, loc='upper left', ncols=2, fontsize=9, bbox_to_anchor=(0, 1.13))
    axes[1].xaxis.set_major_formatter(mdates.DateFormatter('%a %d %b'))
    title(fig, f'Real cleanliness and the two gauges, {s:%d %b} to {s + pd.Timedelta(days=13):%d %b %Y}', y=0.99, size=15)
    fig.text(0.01, -0.01, 'Half-hourly. The model was trained on July 2021 to December 2023 and not changed after. '
             'The fortnight is the one with the most typical match in 2024 to 2026.\n'
             'Data: NESO system frequency (1 s) with 2.5 mHz of noise added; NESO carbon intensity.', fontsize=8, color='grey')
    fig.tight_layout(rect=(0, 0.02, 1, 0.95))
    fig.savefig(os.path.join(OUT, 'model_weeks.png'), dpi=200, bbox_inches='tight', pad_inches=0.25)


def by_year_figure(h):
    years = [('2024', '2024'), ('2025', '2025'), ('2026', '2026 (to Sep)')]
    fig, (a, b) = plt.subplots(1, 2, figsize=(12, 4.8), gridspec_kw={'width_ratios': [1, 1.25]})
    xs = np.arange(len(years))
    for k, (col, colour, label) in enumerate([('rule', GREY, 'current rule'), ('model', TEAL, 'spectrum model')]):
        v = [spearmanr(h.loc[y, col], h.loc[y, 'target']).statistic for y, _ in years]
        a.bar(xs + (k - 0.5) * 0.36, v, 0.34, color=colour, label=label)
        for xi, vi in zip(xs, v): a.text(xi + (k - 0.5) * 0.36, vi + 0.01, f'{vi:.2f}', ha='center', fontsize=9, color=INK)
    a.set_xticks(xs); a.set_xticklabels([l for _, l in years]); a.set_ylim(0, 1)
    a.set_ylabel('how closely the gauge follows the real week\n(1 would be a perfect match)')
    a.legend(frameon=False, loc='upper right'); tidy(a)
    a.set_title('Match with real cleanliness, by year', loc='left', fontsize=12, color=INK)
    labels = ['power station\nfully out', 'power station\nhalf out', 'both inside', 'turbine\nhalf out', 'turbine\nfully out']
    for k, (col, colour, label) in enumerate([('rule', GREY, 'current rule'), ('model', TEAL, 'spectrum model')]):
        fifth = pd.qcut(h[col], 5, labels=False)
        g = h.ci.groupby(fifth).mean()
        b.bar(np.arange(5) + (k - 0.5) * 0.36, g.values, 0.34, color=colour, label=label)
        for xi, vi in zip(range(5), g.values): b.text(xi + (k - 0.5) * 0.36, vi + 2, f'{vi:.0f}', ha='center', fontsize=8.5, color=INK)
    b.set_xticks(range(5)); b.set_xticklabels(labels, fontsize=9)
    b.set_ylabel('average NESO carbon intensity, gCO$_2$/kWh'); tidy(b); b.legend(frameon=False, loc='upper right')
    b.set_title('Carbon intensity at each figure position, 2024 to Sep 2026', loc='left', fontsize=12, color=INK)
    fig.text(0.01, -0.03, 'Each figure position is a fifth of the half-hours. Model trained on July 2021 to December 2023 '
             'and not changed after. Data: NESO system frequency (1 s) with 2.5 mHz of noise added; NESO carbon intensity.',
             fontsize=8, color='grey')
    fig.tight_layout(); fig.savefig(os.path.join(OUT, 'model_by_year.png'), dpi=200, bbox_inches='tight', pad_inches=0.25)


def bands_figure(w):
    e = T.EDGES; total = np.array([w[i * 3:i * 3 + 3].sum() for i in range(M.NB)])
    labels = [f'{1 / hi:.0f}-{1 / lo:.0f} s' if 1 / hi >= 10 else f'{1 / hi:.1f}-{1 / lo:.1f} s' for lo, hi in zip(e[:-1], e[1:])]
    fig, ax = plt.subplots(figsize=(12, 4.4))
    ax.bar(range(M.NB), total, color=[TEAL if t > 0 else ORANGE for t in total], width=0.7)
    ax.axhline(0, color=INK, lw=0.8); ax.set_xticks(range(M.NB)); ax.set_xticklabels(labels, rotation=35, ha='right', fontsize=9)
    ax.set_xlabel('wobble period (one band of the frequency spectrum; 3.8-4.0 s, where the hum sits, is left out)')
    ax.set_ylabel('weight in the model\n(1 h + 6 h + 24 h averages)'); tidy(ax)
    ax.text(0.99, 0.95, 'above zero: louder than usual for the week counts towards cleaner', transform=ax.transAxes, ha='right', color=TEAL, fontsize=9)
    ax.text(0.99, 0.06, 'below zero: louder than usual counts towards dirtier', transform=ax.transAxes, ha='right', color=ORANGE, fontsize=9)
    title(fig, 'Weight the model gives each band of the spectrum', y=0.99, size=15)
    fig.text(0.01, -0.02, 'Final weights, trained on July 2021 to September 2026. '
             'Data: NESO system frequency (1 s) with 2.5 mHz of noise added; NESO carbon intensity.', fontsize=8, color='grey')
    fig.tight_layout(rect=(0, 0, 1, 0.94)); fig.savefig(os.path.join(OUT, 'model_bands.png'), dpi=200, bbox_inches='tight', pad_inches=0.25)


def main():
    h, _, cols = frozen_check()
    weeks_figure(h); by_year_figure(h)
    import json
    wd = json.load(open(os.path.join(HERE, 'train_fft_model_results.json')))['weights']
    bands_figure(np.array([wd[c] for c in cols]))
    print('saved model_weeks.png, model_by_year.png, model_bands.png')


if __name__ == '__main__':
    main()
