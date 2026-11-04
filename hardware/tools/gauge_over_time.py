"""Does the weather house's clean/dirty gauge hold up over time? Runs the firmware's gauge rule over July
2021 to September 2026 (with 2.5 mHz of noise, as a home measurement) and, year by year, compares the
figure position with NESO's carbon intensity.

The gauge reads the week, so the fair test is against the week: when the station is fully out, was the
grid dirtier than the median of the previous seven days? When the turbine is fully out, was it cleaner?

Usage: python hardware/tools/gauge_over_time.py   (uses the cache written by table_model.py)
Writes hardware/tools/gauge_over_time.csv (by year) and gauge_by_month.csv (by calendar month, leaving out
the two weeks-long episodes of extra jitter in the NESO record)."""
import os
import numpy as np, pandas as pd
from table_model import half_hourly

HERE = os.path.dirname(os.path.abspath(__file__))

if __name__ == '__main__':
    j = half_hourly(2.5)
    j['pos'] = np.minimum((j['rank'] * 5).astype(int), 4)          # five figure positions, station full = 0
    j['dirtier_than_week'] = j.ci > j.ci.rolling('7D').median().shift(1)
    rows = []
    for yr, g in j.groupby(j.index.year):
        med = g.groupby('pos').ci.median().reindex(range(5)).round()
        rows.append(dict(year=yr, half_hours=len(g), average_ci=round(g.ci.mean()),
                         station_full=med[0], station_half=med[1], both_in=med[2], turbine_half=med[3], turbine_full=med[4],
                         station_full_and_dirtier_than_week_pct=round(100 * g[g.pos == 0].dirtier_than_week.mean()),
                         turbine_full_and_cleaner_than_week_pct=round(100 * (~g[g.pos == 4].dirtier_than_week).mean())))
    out = pd.DataFrame(rows); out.to_csv(os.path.join(HERE, 'gauge_over_time.csv'), index=False); print(out.to_string(index=False))
    jit = ((j.index >= '2024-12-13') & (j.index < '2025-02-21')) | ((j.index >= '2025-12-24') & (j.index < '2026-01-10'))
    k = j[~jit]; rows = []
    for mo, g in k.groupby(k.index.month):
        ch, tu = g[g.pos == 0], g[g.pos == 4]
        rows.append(dict(month=mo, station_full=round(ch.ci.median()), turbine_full=round(tu.ci.median()),
                         station_full_and_dirtier_than_week_pct=round(100 * ch.dirtier_than_week.mean()),
                         turbine_full_and_cleaner_than_week_pct=round(100 * (~tu.dirtier_than_week).mean())))
    out = pd.DataFrame(rows); out.to_csv(os.path.join(HERE, 'gauge_by_month.csv'), index=False); print(out.to_string(index=False))
