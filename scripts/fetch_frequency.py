"""Download NESO 1-second system frequency CSVs (fnew-YYYY-M.csv) and convert to parquet.
Usage: fetch_frequency.py CSVDIR PARQUETDIR YYYY-M [YYYY-M ...]"""
import sys, os, time, requests, pandas as pd
from concurrent.futures import ThreadPoolExecutor
CSV, PQ, months = sys.argv[1], sys.argv[2], sys.argv[3:]
pkg = requests.get('https://api.neso.energy/api/3/action/package_show?id=system-frequency-data', headers={'User-Agent':'Mozilla/5.0'}, timeout=60).json()
urls = {r['url'].rsplit('/',1)[-1]: r['url'] for r in pkg['result']['resources']}
def pull(m):
    out = f'{PQ}/fnew-{m}.parquet'
    if os.path.exists(out): return f'{m} skip'
    name = f'fnew-{m}.csv'
    if name not in urls:
        name = f'fnew-{m}.zip' if f'fnew-{m}.zip' in urls else (f'f-{m}.zip' if f'f-{m}.zip' in urls else (f'f-{m}.csv' if f'f-{m}.csv' in urls else None))
    if name is None: return f'{m} NOT LISTED'
    csv = f'{CSV}/{name}'
    for a in range(4):
        try:
            with requests.get(urls[name], headers={'User-Agent':'Mozilla/5.0'}, stream=True, timeout=600) as r:
                r.raise_for_status()
                with open(csv, 'wb') as f:
                    for chunk in r.iter_content(1<<20): f.write(chunk)
            if csv.endswith('.zip'):
                import zipfile
                with zipfile.ZipFile(csv) as z:
                    inner = [n for n in z.namelist() if n.lower().endswith('.csv')][0]
                    with z.open(inner) as fh: df = pd.read_csv(fh)
            else:
                df = pd.read_csv(csv)
            df.columns = [c.strip() for c in df.columns]
            if 'dtm' not in df.columns: df = df.rename(columns={df.columns[0]:'dtm', df.columns[1]:'f'})
            df[['dtm','f']].to_parquet(out, index=False); os.remove(csv)
            return f'{m} ok {len(df)} rows'
        except Exception as e:
            err = e; time.sleep(5*(a+1))
    return f'{m} FAIL {err}'
with ThreadPoolExecutor(3) as ex:
    for res in ex.map(pull, months): print(res, flush=True)
print('ALL DONE')
