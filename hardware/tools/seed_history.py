"""Prime the weather house with a week of history, so the figures move from the first reading
instead of staying inside for the first 24 hours.

Takes the last seven days of a month of NESO one-second frequency, computes the same 30-minute
ringing every 5 minutes as the firmware, writes it as the firmware's saved history (the "hist"
blob in the "gridseis" NVS namespace) and flashes it over USB. The firmware itself is not touched.
The board's own readings replace the seed over the following week.

The firmware measures the ringing at NESO's level (hardware/README.md, "What the board hears"),
so the week is written as it is. Pass --scale to multiply it if your readings sit at a different level.

Usage: python seed_history.py [--port COM3] [--month 2026-9] [--scale 1.0] [--dry-run]

Needs: pyserial, pandas, pyarrow, cryptography (for ESP-IDF's nvs_partition_gen.py), and a
PlatformIO install with the ESP-IDF framework and esptool.
"""
import argparse, glob, os, struct, subprocess, sys, tempfile
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__)); HW = os.path.dirname(HERE)
FREQ_DIRS = ['D:/gridseis_hum/freq_parquet', os.path.join(HW, '..', 'raw_data')]
PIO = os.path.expanduser('~/.platformio')
NVS_GEN = os.path.join(PIO, 'packages', 'framework-espidf', 'components', 'nvs_flash', 'nvs_partition_generator', 'nvs_partition_gen.py')
ESPTOOL = os.path.join(PIO, 'packages', 'tool-esptoolpy', 'esptool.py')
NVS_OFFSET = '0x9000'   # ESP-IDF's default single-app partition table: nvs at 0x9000, 0x6000 long
NVS_SIZE = '0x6000'
WIN, HOP, HIST = 1800, 300, 2016   # must match ringing.h
F_LO, F_HI = 0.10, 0.25


def find_month(month):
    for d in FREQ_DIRS:
        files = sorted(glob.glob(os.path.join(d, 'fnew-*.parquet')))
        if not files: continue
        if month is None: return files[-1]
        for f in files:
            if f.endswith(f'fnew-{month}.parquet'): return f
    sys.exit('no NESO frequency parquet found; run scripts/fetch_frequency.py first')


def week_of_ringing(path):
    df = pd.read_parquet(path)
    s = df.set_index(pd.to_datetime(df['dtm'], format='mixed'))['f'].sort_index()
    s = s[~s.index.duplicated()].asfreq('1s').interpolate(limit=60)
    end = s.index[-1].floor('5min'); start = end - pd.Timedelta(days=7)
    klo, khi = int(np.ceil(F_LO * WIN - 1e-9)), int(np.floor(F_HI * WIN + 1e-9)); w = np.hanning(WIN)
    vals = []
    for t in pd.date_range(start, end, freq='5min')[1:]:
        x = s[t - pd.Timedelta(seconds=WIN - 1):t].values
        if len(x) != WIN or np.isnan(x).any(): continue
        x = (x - x.mean()) * 1000
        vals.append((np.abs(np.fft.rfft(x * w))[klo:khi + 1] * 2 / w.sum()).mean())
    v = np.array(vals[-HIST:], dtype=np.float32)
    if len(v) < HIST: sys.exit(f'only {len(v)} ringing values in the last week of {path}; need {HIST}')
    return v, start, end


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--port', default='COM3'); ap.add_argument('--month', default=None)
    ap.add_argument('--scale', type=float, default=1.0); ap.add_argument('--dry-run', action='store_true')
    a = ap.parse_args()
    path = find_month(a.month)
    v, start, end = week_of_ringing(path)
    v = (v * a.scale).astype(np.float32)
    print(f'{path}: {len(v)} ringing values {start} to {end} UTC, median {np.median(v):.3f} mHz (scale {a.scale:g})')
    work = tempfile.mkdtemp()
    with open(os.path.join(work, 'hist.bin'), 'wb') as f:
        f.write(struct.pack('<ii', 0, len(v)) + v.astype('<f4').tobytes())   # saved_hist_t in main.c: head, n, hist[]
    with open(os.path.join(work, 'nvs.csv'), 'w') as f:
        f.write('key,type,encoding,value\ngridseis,namespace,,\nhist,file,binary,hist.bin\n')
    nvs = os.path.join(work, 'nvs.bin')
    subprocess.run([sys.executable, NVS_GEN, 'generate', 'nvs.csv', nvs, NVS_SIZE], cwd=work, check=True)
    if a.dry_run: print('dry run: image at', nvs); return
    subprocess.run([sys.executable, ESPTOOL, '--chip', 'esp32s3', '-p', a.port, '--after', 'hard_reset',
                    'write_flash', NVS_OFFSET, nvs], check=True)
    print('done: the board restarts and should report "loaded 2016 ringing values (7.0 days) from flash"')


if __name__ == '__main__':
    main()
