# 사용: python scripts/build_aggtrades_size3_tape_1m_20260925.py ETHUSDT 2023-01-01  → tmp/whale/tape/<SYM>/ (research_eth_fused_signal_votes_gate_20260925.py 가 읽는다)
"""binance.vision aggTrades → 1분 × 크기 3구간(대시보드 경계: 리테일 <$10k · 중형 $10k~$100k · 고래 ≥$100k).
두 단위: row = aggTrade 한 줄(대시보드 TakerOrderAggregator 와 같은 (가격·방향·ms)) · ord = (ms·방향) 으로 가격 넘어 묶은 주문.
출력: tmp/whale/tape/<SYM>/<day>.parquet — 분 시작 라벨."""
import io, os, sys, zipfile, urllib.request, urllib.error, time
import numpy as np, pandas as pd
from concurrent.futures import ThreadPoolExecutor
SYM = sys.argv[1] if len(sys.argv) > 1 else 'ETHUSDT'
URL = "https://data.binance.vision/data/futures/um/daily/aggTrades/{s}/{s}-aggTrades-{d}.zip"
OUT = f'tmp/whale/tape/{SYM}'; os.makedirs(OUT, exist_ok=True)
EDGES = [10_000, 100_000]

def fetch(day):
    for k in range(4):
        try: return urllib.request.urlopen(URL.format(s=SYM, d=day), timeout=180).read()
        except urllib.error.HTTPError as e:
            if e.code == 404: return None
        except Exception: pass
        time.sleep(3 * (k + 1))
    return None

def buckets(b, side, nt, tag):
    cls = np.searchsorted(EDGES, nt, side='right')   # 0 리테일 1 중형 2 고래
    cols = {}
    for k, nm in enumerate(['ret', 'mid', 'whl']):
        m = cls == k
        d = pd.DataFrame({'b': b[m], 'sn': side[m] * nt[m], 'nt': nt[m]}).groupby('b').agg(sn=('sn', 'sum'), nt=('nt', 'sum'), n=('nt', 'size'))
        d.columns = [f'{tag}_{nm}_{c}' for c in d.columns]; cols[nm] = d
    return pd.concat(cols.values(), axis=1)

def day_features(blob):
    z = zipfile.ZipFile(io.BytesIO(blob))
    df = pd.read_csv(z.open(z.namelist()[0]), usecols=['price', 'quantity', 'transact_time', 'is_buyer_maker'])
    if df.empty: return None
    t = df.transact_time.to_numpy(np.int64); px = df.price.to_numpy(float); q = df.quantity.to_numpy(float)
    side = np.where(df.is_buyer_maker.to_numpy(bool), -1.0, 1.0); nt = px * q; b = t // 60000
    out = pd.DataFrame({'b': b, 'px': px}).groupby('b').px.agg(o='first', h='max', l='min', c='last')
    out = out.join(buckets(b, side, nt, 'row'))
    o = pd.DataFrame({'t': t, 's': side, 'nt': nt}).groupby(['t', 's'], sort=False).nt.sum().reset_index()
    out = out.join(buckets(o.t.values // 60000, o.s.values, o.nt.values, 'ord'))
    out = out.fillna(0.0); out.index = pd.to_datetime(out.index * 60000, unit='ms', utc=True)
    return out

if __name__ == '__main__':
    days = [d.strftime('%Y-%m-%d') for d in pd.date_range(sys.argv[2] if len(sys.argv) > 2 else '2023-01-01', pd.Timestamp.utcnow().tz_localize(None).normalize() - pd.Timedelta('1D'), freq='D')]
    todo = [d for d in days if not os.path.exists(f'{OUT}/{d}.parquet')]
    print(f'[{SYM}] {len(days)} days, todo {len(todo)}', flush=True); t0 = time.time(); miss = []
    # 🔴 ex.map 은 전부 즉시 제출 → 받은 zip 이 메모리에 쌓인다(1,000일×수십MB). 12일씩 끊는다.
    chunks = [todo[k:k + 12] for k in range(0, len(todo), 12)]
    with ThreadPoolExecutor(6) as ex:
        for i, (d, blob) in enumerate((d, b) for ch in chunks for d, b in zip(ch, list(ex.map(fetch, ch)))):
            if blob is None: miss.append(d); continue
            f = day_features(blob)
            if f is not None: f.to_parquet(f'{OUT}/{d}.parquet')
            if i % 50 == 0: print(f'[{SYM}] {i}/{len(todo)} {(time.time()-t0)/(i+1):.2f}s/day', flush=True)
    print(f'[{SYM}] done, missing {miss}', flush=True)
