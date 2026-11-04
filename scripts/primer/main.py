"""Grid frequency primer: how GB frequency behaviour changes with the generation mix.
Figures are written next to this file. Data: NESO 1-second frequency (cached as parquet), NESO historic generation mix,
NESO system inertia outturn.  Run build_stats.py first for the half-hourly frequency statistics."""
import os, glob, numpy as np, pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from style import tidy, title, source, save, FUEL_COLOURS, INK, GREY, HEAVY, LIGHT

HERE = os.path.dirname(os.path.abspath(__file__)); os.chdir(HERE)
FUEL = 'D:/gridseis_hum/df_fuel_ckan.csv'
INERTIA = 'C:/Users/User/AppData/Local/Temp/claude/C--Users-User-Documents-GitHub-GridSeis/041b8924-ca2c-4ad9-9990-a681826366c3/scratchpad/hum/inertia_raw.parquet'
STATS = 'D:/gridseis_hum/freq_hh_stats.parquet'
SPECTRA = 'D:/gridseis_hum/spectra'
FREQ_DIRS = ['D:/gridseis_hum/freq_parquet', 'C:/Users/User/Documents/GitHub/GridSeis/raw_data']
SRC_FREQ = 'Data: NESO system frequency, 1-second readings'


def load_fuel():
    d = pd.read_csv(FUEL); d.index = pd.to_datetime(d.DATETIME); d = d.drop(columns='DATETIME')
    d['SPINNING'] = d[['GAS', 'COAL', 'NUCLEAR', 'BIOMASS', 'HYDRO', 'OTHER']].sum(axis=1)
    d['INVERTER'] = d[['WIND', 'SOLAR', 'IMPORTS', 'STORAGE']].sum(axis=1)
    d['SPINNING_perc'] = 100 * d.SPINNING / d.GENERATION; d['WS_perc'] = d.WIND_perc + d.SOLAR_perc
    return d


def load_inertia():
    i = pd.read_parquet(INERTIA)
    ts = pd.to_datetime(i['Settlement Date']).dt.tz_localize('Europe/London').dt.tz_convert('UTC').dt.tz_localize(None)
    i['ts'] = ts + pd.to_timedelta((i['Settlement Period'] - 1) * 30, unit='min')
    return i.drop_duplicates('ts').set_index('ts')['Outturn Inertia'].sort_index()


def have_freq_month(y, m):
    return any(os.path.exists(f'{d}/fnew-{y}-{m}.parquet') for d in FREQ_DIRS)


def load_freq_month(y, m):
    for d in FREQ_DIRS:
        fp = f'{d}/fnew-{y}-{m}.parquet'
        if os.path.exists(fp):
            df = pd.read_parquet(fp); df['dtm'] = pd.to_datetime(df.dtm, format='mixed', errors='coerce'); df['f'] = pd.to_numeric(df.f, errors='coerce')
            df = df.dropna().drop_duplicates('dtm').set_index('dtm').sort_index(); return df.f[(df.f > 49) & (df.f < 51)]
    return None


# ---------------------------------------------------------------- figure 1: a heavy day and a light day
def fig_two_days(fuel, inertia):
    daily = inertia.resample('D').mean(); ws = fuel.WS_perc.resample('D').mean().reindex(daily.index)
    ok = [d for d in daily.index if have_freq_month(d.year, d.month)]
    daily = daily.loc[ok]; ws = ws.loc[ok]
    recent = daily[daily.index.year >= 2023]; heavy_day = recent.idxmax(); light_day = recent[(recent >= 90) & (ws.loc[recent.index] >= 45)].idxmin()
    fig, axes = plt.subplots(2, 1, figsize=(11, 7.5), sharey=True)
    for ax, day, col, label in [(axes[0], heavy_day, HEAVY, 'heavy'), (axes[1], light_day, LIGHT, 'light')]:
        f = load_freq_month(day.year, day.month)[str(day.date())]; dev = (f - 50) * 1000
        ax.axhspan(-200, 200, color='#f3f4f6', zorder=0); ax.axhline(0, color=GREY, lw=0.8)
        ax.plot(dev.index, dev.values, color=col, lw=0.5)
        for h in pd.date_range(day, day + pd.Timedelta(days=1), freq='30min'): ax.axvline(h, color=GREY, lw=0.3, alpha=0.5)
        ax.set_ylim(-320, 320); ax.set_xlim(day, day + pd.Timedelta(days=1)); ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        ax.set_ylabel('mHz from 50 Hz'); tidy(ax)
        ax.set_title(f'{day:%A %d %B %Y}: {label} grid.  Inertia {daily[day]:.0f} GVA·s, wind and solar {ws[day]:.0f}% of generation, '
                     f'frequency spread {dev.std():.0f} mHz', loc='left', fontsize=11)
        ax.text(0.995, 0.96, 'grey band: the normal operating range, 50 Hz ± 200 mHz', transform=ax.transAxes, ha='right', va='top', fontsize=9, color='grey')
    axes[1].set_xlabel('time (UTC)'); source(axes[1], SRC_FREQ + '; NESO system inertia; NESO generation mix')
    title(fig, 'Grid frequency on a heavy day and a light day', y=0.93)
    fig.text(0.02, 0.895, 'With fewer heavy spinning generators connected, the frequency moves more sharply from second to second. Vertical lines are half hours.', fontsize=11)
    save(fig, 'fig1_two_days.png', top=0.84); return heavy_day, light_day


# ---------------------------------------------------------------- figure 2: the mix has changed
def fig_mix(fuel):
    cols = {'Gas': 'GAS', 'Coal': 'COAL', 'Nuclear': 'NUCLEAR', 'Biomass': 'BIOMASS', 'Hydro': 'HYDRO', 'Other': 'OTHER', 'Wind': 'WIND', 'Solar': 'SOLAR', 'Imports': 'IMPORTS', 'Storage': 'STORAGE'}
    y = fuel.resample('YS').mean(); y = y[y.index.year >= 2009]; share = pd.DataFrame({k: 100 * y[v] / y.GENERATION for k, v in cols.items()})
    fig, axes = plt.subplots(2, 1, figsize=(11, 8), gridspec_kw={'height_ratios': [2.2, 1]}, sharex=True)
    ax = axes[0]; ax.stackplot(share.index, [share[k].values for k in cols], labels=list(cols), colors=[FUEL_COLOURS[k] for k in cols], alpha=0.9)
    ax.set_ylim(0, 100); ax.set_ylabel('share of generation, %'); tidy(ax, ygrid=False)
    ax.legend(loc='upper left', bbox_to_anchor=(1.0, 1.0), frameon=False, title='bottom: spinning machines\ntop: inverter-connected', title_fontsize=9, alignment='left')
    ax.set_title('Annual average generation mix', loc='left', fontsize=11)
    ax = axes[1]; sp = 100 * y.SPINNING / y.GENERATION; ax.plot(sp.index, sp.values, color=HEAVY, lw=2.5)
    ax.fill_between(sp.index, sp.values, 100, color=LIGHT, alpha=0.15); ax.set_ylim(0, 100); ax.set_ylabel('%'); tidy(ax)
    ax.text(sp.index[-1], sp.iloc[-1] - 8, 'generation from spinning machines' + chr(10) + '(gas, coal, nuclear, biomass, hydro)', fontsize=9, color=HEAVY, ha='right', va='top')
    ax.text(sp.index[-1], 96, 'wind, solar, imports and batteries,' + chr(10) + 'connected through electronics', fontsize=9, color=LIGHT, ha='right', va='top')
    ax.xaxis.set_major_locator(mdates.YearLocator(2)); ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y')); ax.set_xlabel('')
    source(ax, 'Data: NESO historic generation mix (half-hourly, averaged by year). 2026 is January to October.')
    title(fig, 'Share of generation from spinning machines, 2009 to 2026', y=0.94)
    fig.text(0.02, 0.905, f'Spinning share: {sp.iloc[0]:.0f}% in 2009, {sp.iloc[-1]:.0f}% in 2026.', fontsize=11)
    save(fig, 'fig2_mix.png', top=0.86)


# ---------------------------------------------------------------- figure 3: inertia
def fig_inertia(fuel, inertia):
    j = pd.DataFrame({'H': inertia, 'ws': fuel.WS_perc.reindex(inertia.index), 'sp': fuel.SPINNING_perc.reindex(inertia.index)}).dropna()
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.2))
    ax = axes[0]; ax.scatter(j.ws, j.H, s=1, alpha=0.3, color=INK, edgecolors='none')
    b = j.groupby(pd.cut(j.ws, np.arange(0, 85, 5)), observed=True).H.median(); ax.plot([iv.mid for iv in b.index], b.values, color=LIGHT, lw=2.5, label='median')
    ax.set_xlabel('wind and solar share of generation, %'); ax.set_ylabel('system inertia, GVA·s'); ax.set_ylim(0, 400); tidy(ax)
    ax.set_title('Every half hour, April 2022 to July 2026', loc='left', fontsize=11); ax.legend(frameon=False, loc='upper right')
    ax = axes[1]; m = inertia.resample('MS').agg(['min', 'median', 'max'])
    ax.fill_between(m.index, m['min'], m['max'], color=GREY, alpha=0.3, label='monthly range'); ax.plot(m.index, m['median'], color=HEAVY, lw=2, label='monthly median'); ax.plot(m.index, m['min'], color=LIGHT, lw=1.5, label='monthly minimum')
    ax.set_ylim(0, 400); ax.set_ylabel('system inertia, GVA·s'); tidy(ax); ax.legend(frameon=False, loc='upper right'); ax.xaxis.set_major_locator(mdates.YearLocator()); ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax.set_title('Inertia over time', loc='left', fontsize=11)
    source(axes[0], 'Data: NESO system inertia outturn; NESO generation mix')
    title(fig, 'System inertia against wind and solar share, and over time', y=0.97)
    fig.text(0.02, 0.905, 'Inertia is the kinetic energy stored in connected generators. It is what slows frequency down in the first second after a shock.', fontsize=11)
    save(fig, 'fig3_inertia.png', top=0.84)


# ---------------------------------------------------------------- figure 4: how the frequency behaves vs inertia
def fig_behaviour(stats, inertia, fuel):
    j = stats.join(inertia.rename('H'), how='inner').join(fuel.WS_perc, how='inner'); j = j[(j.n >= 1700)].copy(); j['std_mHz'] = j['std'] * 1000
    bins = np.arange(60, 381, 40); j['Hbin'] = pd.cut(j.H, bins); g = j.groupby('Hbin', observed=True)
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 5))
    for ax, col, lab, ttl in [(axes[0], 'std_mHz', 'spread of frequency over the half hour, mHz', 'Over minutes: how far it wanders'),
                              (axes[1], 'rocof_rms_mHz_s', 'typical second-to-second change, mHz per second', 'Over seconds: how much it jitters')]:
        q = g[col].quantile([0.25, 0.5, 0.75]).unstack(); x = [iv.mid for iv in q.index]
        ax.fill_between(x, q[0.25], q[0.75], color=LIGHT, alpha=0.2, label='middle half of half hours'); ax.plot(x, q[0.5], color=LIGHT, lw=2.5, marker='o', label='median')
        ax.set_xlabel('system inertia, GVA·s'); ax.set_ylabel(lab); ax.set_ylim(0); tidy(ax); ax.set_title(ttl, loc='left', fontsize=11)
        ax.invert_xaxis()
    axes[0].legend(frameon=False, loc='upper right')
    source(axes[0], SRC_FREQ + ' (half-hourly statistics, 2022 to 2026); NESO system inertia. Inertia increases to the left, so the lighter grid is on the right.')
    title(fig, 'Frequency spread and jitter at each level of inertia', y=0.97)
    fig.text(0.02, 0.9, f'{len(j):,} half hours, 2022 to 2026. Slow drift is held in check by response services, which have grown; the second-to-second jitter is what inertia used to damp.', fontsize=11)
    save(fig, 'fig4_behaviour.png', top=0.82)
    return j


# ---------------------------------------------------------------- figure 5: by year, and the half-hour sawtooth
def fig_year_and_sawtooth(stats, fuel):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))
    ax = axes[0]; s = stats[stats.n >= 1700]; yrs = sorted(s.index.year.unique()); data = [s[s.index.year == y]['std'].values * 1000 for y in yrs]
    bp = ax.boxplot(data, tick_labels=[str(y) for y in yrs], showfliers=False, patch_artist=True, widths=0.6, medianprops=dict(color=INK, lw=2))
    for p in bp['boxes']: p.set(facecolor=LIGHT, alpha=0.35, edgecolor=LIGHT)
    ax.set_ylabel('standard deviation of frequency in each half hour, mHz'); tidy(ax); ax.set_title('Spread of frequency, by year (July 2021 to September 2026)', loc='left', fontsize=11)
    ws = fuel.WS_perc.resample('YS').mean(); ws = ws[ws.index.year.isin(yrs)]
    for k, y in enumerate(yrs): ax.text(k + 1, ax.get_ylim()[1] * 0.02, f'wind+solar\n{ws[ws.index.year == y].iloc[0]:.0f}%', ha='center', fontsize=8, color='grey')
    # sawtooth: mean deviation by second of the half hour, for heavy and light half hours of 2025
    ax = axes[1]; f = pd.concat([load_freq_month(2025, m) for m in (1, 2, 6, 7)]); dev = (f - 50) * 1000
    hh = dev.index.floor('30min'); sec = ((dev.index - hh).total_seconds()).astype(int)
    ws_hh = fuel.WS_perc.reindex(hh); light = (ws_hh.values > 45); heavy = (ws_hh.values < 20)
    for mask, col, lab in [(heavy, HEAVY, 'half hours with wind+solar under 20%'), (light, LIGHT, 'half hours with wind+solar over 45%')]:
        prof = pd.Series(dev.values[mask]).groupby(sec[mask]).mean(); ax.plot(prof.index / 60, prof.values, color=col, lw=2, label=lab)
    ax.axhline(0, color=GREY, lw=0.8); ax.set_xlabel('minutes into the half hour'); ax.set_ylabel('average frequency deviation, mHz'); tidy(ax); ax.legend(frameon=False, loc='lower right')
    ax.set_title('The half-hour sawtooth: average shape of frequency within a half hour, 2025', loc='left', fontsize=11); ax.set_xlim(0, 30)
    source(axes[0], SRC_FREQ + '; NESO generation mix. Boxes show the middle half of half hours, whiskers the bulk of the rest.')
    title(fig, 'Frequency spread by year, and the average shape of a half hour', y=0.97)
    fig.text(0.02, 0.9, 'Generators change output on the half hour, because that is how the market is settled. The frequency dips or overshoots at each boundary while they catch up.', fontsize=11)
    save(fig, 'fig5_year_sawtooth.png', top=0.84)


# ---------------------------------------------------------------- figure 6: the spectrum
def fig_spectrum(inertia):
    mags, tss = [], []
    for fp in sorted(glob.glob(f'{SPECTRA}/*.npz')):
        z = np.load(fp); mags.append(z['mag']); tss.append(z['ts']); freqs = z['freqs']
    M = np.vstack(mags); tsv = np.concatenate(tss); ts = pd.to_datetime(tsv, unit='ns' if tsv.max() > 1e13 else 's'); H = inertia.reindex(ts).values; ok = ~np.isnan(H)
    M, H, ts = M[ok], H[ok], ts[ok]; lo, hi = np.nanpercentile(H, [10, 90])
    fig, ax = plt.subplots(figsize=(11, 6))
    for mask, col, lab in [(H >= hi, HEAVY, f'heaviest 10% of half hours (inertia above {hi:.0f} GVA·s)'), (H <= lo, LIGHT, f'lightest 10% of half hours (inertia below {lo:.0f} GVA·s)')]:
        ax.plot(freqs, np.median(M[mask], axis=0) * 1000, color=col, lw=2, label=lab)
    ax.set_xscale('log'); ax.set_yscale('log'); ax.set_ylabel('typical size of the wobble, mHz'); tidy(ax)
    tk = [0.01, 0.02, 0.05, 0.1, 0.2, 0.5]; ax.set_xticks(tk); ax.set_xticklabels([f'{t:g} Hz' + chr(10) + f'({1/t:g} s period)' for t in tk]); ax.set_xticks([], minor=True)
    ax.set_xlabel('how fast the wobble repeats (left: slow swings over minutes; right: fast swings over seconds)')
    for per in [60, 30, 15, 12, 10, 8, 6, 5]: ax.axvline(1 / per, color=GREY, lw=0.4, alpha=0.6)
    ax.text(1 / 10, ax.get_ylim()[1] * 0.55, 'thin lines: round-number periods' + chr(10) + '(60, 30, 15, 12, 10, 8, 6, 5 s), where' + chr(10) + 'clocked devices leave spikes', fontsize=9, color='grey', ha='center')
    ax.legend(frameon=False, loc='lower left'); ytk = [0.05, 0.1, 0.2, 0.5, 1, 2]; ax.set_yticks(ytk); ax.set_yticklabels([f'{t:g}' for t in ytk]); ax.set_yticks([], minor=True)
    ax.annotate('a single 5 MW machine somewhere on the\nnetwork, oscillating every 3.9 s since 2022', xy=(0.258, 1000 * np.median(M[H <= lo], axis=0)[np.argmin(np.abs(freqs - 0.258))]), xytext=(0.55, 0.62), textcoords='axes fraction',
                fontsize=9, color=INK, arrowprops=dict(arrowstyle='->', color=INK, lw=0.8))
    ax.axvspan(0.3, 0.6, color='#fde68a', alpha=0.3); ax.text(0.42, ax.get_ylim()[1] * 0.6, 'where the grid' + chr(39) + 's own' + chr(10) + 'natural modes live', ha='center', fontsize=9, color='#92400e')
    source(ax, SRC_FREQ + ' (30-minute spectra, 2021 to 2026); NESO system inertia')
    title(fig, 'Size of the frequency wobbles, heavy v light grid', y=0.97)
    fig.text(0.02, 0.9, 'Typical spectrum of the frequency signal. Slow swings (left) are smaller on the light grid; everything faster than about 20 seconds (right) is larger.', fontsize=11)
    save(fig, 'fig6_spectrum.png', top=0.84)


# ---------------------------------------------------------------- figure 7: the ringing as a carbon gauge
def fig_ringing(fuel):
    mags, tss = [], []
    for fp in sorted(glob.glob(f'{SPECTRA}/*.npz')):
        z = np.load(fp); mags.append(z['mag']); tss.append(z['ts']); freqs = z['freqs']
    M = np.vstack(mags); tsv = np.concatenate(tss); ts = pd.to_datetime(tsv, unit='ns' if tsv.max() > 1e13 else 's')
    band = (freqs >= 0.1) & (freqs <= 0.25)   # fast band below the 0.258 Hz hum and the 0.3 to 0.4 Hz winter bursts, which are forced oscillations, not ringing
    r = pd.Series(M[:, band].mean(axis=1) * 1000, index=ts, name='ring'); r = r[~r.index.duplicated()].sort_index()
    j = pd.DataFrame({'ring': r}).join(fuel[['CARBON_INTENSITY', 'WS_perc']], how='inner').dropna()
    rho = j.ring.corr(j.CARBON_INTENSITY, method='spearman')
    fig = plt.figure(figsize=(13, 6)); gs = fig.add_gridspec(2, 2, width_ratios=[1, 1.3], hspace=0.15, wspace=0.25)
    ax = fig.add_subplot(gs[:, 0]); dec = pd.qcut(j.ring, 20, duplicates='drop'); g = j.groupby(dec, observed=True)
    x = g.ring.median(); q = g.CARBON_INTENSITY.quantile([0.25, 0.5, 0.75]).unstack()
    ax.fill_between(x, q[0.25], q[0.75], color=LIGHT, alpha=0.2, label='middle half of half hours'); ax.plot(x, q[0.5], color=LIGHT, lw=2.5, marker='o', label='typical carbon intensity')
    ax.set_xlabel('ringing: average size of the fast wobbles, 4 to 10 s periods, mHz'); ax.set_ylabel('carbon intensity, gCO' + chr(8322) + '/kWh'); ax.set_ylim(0); tidy(ax); ax.legend(frameon=False, loc='upper right')
    ax.set_title(f'{len(j):,} half hours, 2021 to 2026, in twenty equal groups by ringing', loc='left', fontsize=11)
    ax.text(0.02, 0.04, 'the louder the ringing, the cleaner the grid', transform=ax.transAxes, fontsize=10, color=INK)
    w = j.resample('W').median()
    ax1 = fig.add_subplot(gs[0, 1]); ax1.plot(w.index, w.ring, color=LIGHT, lw=1.5); ax1.set_ylabel('ringing, mHz'); tidy(ax1); ax1.set_title('Week by week, 2021 to 2026', loc='left', fontsize=11); ax1.tick_params(labelbottom=False)
    ax2 = fig.add_subplot(gs[1, 1], sharex=ax1); ax2.plot(w.index, w.CARBON_INTENSITY, color=HEAVY, lw=1.5); ax2.set_ylabel('carbon intensity, gCO' + chr(8322) + '/kWh'); tidy(ax2); ax2.set_ylim(0)
    ax2.xaxis.set_major_locator(mdates.YearLocator()); ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    for a, col in [(ax1, LIGHT), (ax2, HEAVY)]: a.spines['left'].set_color(col); a.yaxis.label.set_color(col); a.tick_params(axis='y', colors=col)
    source(ax, SRC_FREQ + ' (30-minute spectra, average amplitude 0.1 to 0.25 Hz); NESO generation mix carbon intensity')
    title(fig, 'Ringing against carbon intensity, 2021 to 2026', y=0.97)
    fig.text(0.02, 0.9, 'Left: typical carbon intensity of half hours grouped by how loudly the grid rings. Right: the two series week by week, ringing above, carbon intensity below.', fontsize=11)
    save(fig, 'fig7_ringing_vs_carbon.png', top=0.84)
    for y in sorted(j.index.year.unique()):
        k = j[j.index.year == y]; print(y, 'rank correlation', round(k.ring.corr(k.CARBON_INTENSITY, method='spearman'), 2))


if __name__ == '__main__':
    fuel = load_fuel(); inertia = load_inertia()
    fig_mix(fuel); fig_inertia(fuel, inertia)
    print('example days:', fig_two_days(fuel, inertia))
    if os.path.exists(STATS):
        stats = pd.read_parquet(STATS); fig_behaviour(stats, inertia, fuel); fig_year_and_sawtooth(stats, fuel)
    fig_spectrum(inertia); fig_ringing(fuel)
