#!/usr/bin/env python3
"""스테이블 개별 발행 사건 → ETH 장중. 사전 등록 docs/prereg/stablecoin_mint_event_intraday_20261025.md.

    python3 scripts/research_stablecoin_mint_intraday_20261025.py --fetch   # 체인 사건 + 1분봉 채움
    python3 scripts/research_stablecoin_mint_intraday_20261025.py --test    # 한 번만
실주문 없음. 공개 API(TronGrid·Blockscout·data.binance.vision)만.
"""
import io, json, sys, time, zipfile
from pathlib import Path
import numpy as np, pandas as pd, requests

ROOT = Path("/home/kbj20/crypto-scalping")
OUT = ROOT / "tmp" / "stablecoin_mint_20261025"
K1 = ROOT / "data" / "binance_vision" / "klines1m"
START, END = "2024-01-01", "2026-10-24"
T0, T1 = int(pd.Timestamp(START).value // 10**6), int(pd.Timestamp(END).value // 10**6)
M1 = 60_000
MIN_USD, GAP = 50e6, 6 * 3_600_000
HS = (15, 60, 240, 1440)
SPLITS = dict(train=("2024-01-01", "2026-01-01"), val=("2026-01-01", "2026-06-01"), test=("2026-06-01", "2026-10-24"))
TRON = "TR7NHqjeKQxGTCi8q8ZY4pL8otSzgjLj6t"
ETH_LOGS = [("USDT", "mint", "0xdac17f958d2ee523a2206206994597c13d831ec7", "0xcb8241adb0c3fdb35b70c24ce35c5eb0c17af7431c99f827d44a445ca624176a"),
            ("USDT", "burn", "0xdac17f958d2ee523a2206206994597c13d831ec7", "0x702d5967f45f6513a38ffc42d6ba9bf230bd40e8f53b16363c7eb4fd2deb9a44"),
            ("USDC", "mint", "0xa0b86991c6218b36c1d19d4a2e9eb0ce3606eb48", "0xab8530f87dc9b59234c4623bf917212bb2536d647574c8e7e5da92c2ede0c9f8"),
            ("USDC", "burn", "0xa0b86991c6218b36c1d19d4a2e9eb0ce3606eb48", "0xcc16f5dbb4873280815c1ee09dbd06736cffcc184412cf7a71a0fdb75d397ca5")]


def get(url, **kw):
    for i in range(10):
        try:
            r = requests.get(url, timeout=60, **kw)
            if r.status_code == 200:
                return r
        except requests.RequestException:
            pass
        time.sleep(5 * 2 ** min(i, 4))                                    # 429 는 길게 쉰다(최대 80초)
    raise RuntimeError(url)


def tron(name):
    rows, url = [], f"https://api.trongrid.io/v1/contracts/{TRON}/events"
    p = dict(event_name=name, limit=200, min_block_timestamp=T0, max_block_timestamp=T1, order_by="block_timestamp,asc")
    while True:
        d = get(url, params=p).json()
        rows += [dict(fam="USDT", chain="tron", kind="mint" if name == "Issue" else "burn", ts_ms=int(e["block_timestamp"]),
                      amount_usd=int(e["result"]["amount"]) / 1e6, tx=e["transaction_id"]) for e in d["data"]]
        fp = d.get("meta", {}).get("fingerprint")
        if not fp:
            return rows
        p["fingerprint"] = fp; time.sleep(.3)


RPCS = ["https://rpc.mevblocker.io", "https://gateway.tenderly.co/public/mainnet"]   # 무료·키 없음·getLogs 1만 블록 허용(10-25 실측)


def eth(fam, kind, addr, topic, b0=18_900_000, step=10_000):
    rows, n = [], 0
    top = b0 + (T1 // 1000 - 1_703_900_000) // 12 + 20_000                 # 블록 18.9M ≈ 2023-12-30, 12초 블록 — 넉넉히(빈 창은 비용 없음)
    while b0 <= top:
        b1 = min(b0 + step - 1, top)
        q = dict(jsonrpc="2.0", id=1, method="eth_getLogs", params=[dict(fromBlock=hex(b0), toBlock=hex(b1), address=addr, topics=[topic])])
        for i in range(10):
            try:
                d = requests.post(RPCS[(n + i) % 2], json=q, timeout=60).json()
                if "result" in d:
                    break
            except (requests.RequestException, ValueError):
                pass
            time.sleep(3 * (i + 1))
        else:
            raise RuntimeError((fam, kind, b0))
        rows += [dict(fam=fam, chain="eth", kind=kind, ts_ms=int(e["blockTimestamp"], 16) * 1000,
                      amount_usd=int(e["data"][:66], 16) / 1e6, tx=e["transactionHash"]) for e in d["result"]]
        b0, n = b1 + 1, n + 1; time.sleep(.2)
    return rows


def vision(sym, kind, tag):
    u = f"https://data.binance.vision/data/futures/um/{kind}/klines/{sym}/1m/{sym}-1m-{tag}.zip"
    with zipfile.ZipFile(io.BytesIO(get(u).content)) as z:
        df = pd.read_csv(z.open(z.namelist()[0]), header=None, usecols=[0, 4], names=["t", "c"], low_memory=False)
    df = df[pd.to_numeric(df.t, errors="coerce").notna()]                 # 머리줄 있는 파일
    return df.astype({"t": "int64", "c": "float64"})


def prices(sym):
    parts = []
    if sym == "ETHUSDT":
        parts += [pd.read_parquet(f, columns=["t", "c"]) for f in sorted(K1.glob("ETHUSDT-1m-*.parquet"))]
        last = pd.Timestamp(int(pd.concat(parts).t.max()), unit="ms").normalize() + pd.Timedelta(days=1)
    else:
        parts += [vision(sym, "monthly", m.strftime("%Y-%m")) for m in pd.date_range(START, "2026-09-01", freq="MS")]
        last = pd.Timestamp("2026-10-01")
    for d in pd.date_range(last, pd.Timestamp(END) - pd.Timedelta(days=1), freq="D"):
        parts.append(vision(sym, "daily", d.strftime("%Y-%m-%d")))
    k = pd.concat(parts).drop_duplicates("t").sort_values("t")
    return k[(k.t >= T0 - 86_400_000) & (k.t < T1 + 2 * 86_400_000)].reset_index(drop=True)


def fetch():
    OUT.mkdir(parents=True, exist_ok=True)
    def cached(name, fn):                                                   # 출처별 캐시 — 중간에 끊겨도 받은 것은 남김
        p = OUT / f"raw_{name}.parquet"
        if not p.exists():
            pd.DataFrame(fn()).to_parquet(p)
        return pd.read_parquet(p)
    parts = [cached(f"tron_{n}", lambda n=n: tron(n)) for n in ("Issue", "Redeem")]
    for a in ETH_LOGS:
        parts.append(cached(f"eth_{a[0]}_{a[1]}", lambda a=a: eth(*a))); print(a[:2], len(parts[-1]), flush=True)
    ev = pd.concat(parts, ignore_index=True); ev = ev[(ev.ts_ms >= T0) & (ev.ts_ms < T1)].sort_values("ts_ms").reset_index(drop=True)
    ev.to_parquet(OUT / "events.parquet")
    print(ev.groupby(["fam", "chain", "kind"]).agg(n=("tx", "size"), big=("amount_usd", lambda a: int((a >= MIN_USD).sum())), usd_bn=("amount_usd", lambda a: round(a.sum() / 1e9, 1))))
    for s in ("ETHUSDT", "BTCUSDT"):
        k = prices(s); k.to_parquet(OUT / f"px_{s[:3]}.parquet")
        print(s, len(k), "틈", int((np.diff(k.t) != M1).sum()), pd.to_datetime(k.t.iloc[[0, -1]], unit="ms").tolist(), flush=True)


def dedupe(ts):
    keep, last = [], -10**18
    for x in ts:
        if x - last > GAP:
            keep.append(x); last = x
    return np.array(keep, np.int64)


def selftest():
    assert dedupe([0, 1, GAP, GAP + 1, 3 * GAP]).tolist() == [0, GAP + 1, 3 * GAP]
    t = np.arange(0, 10) * M1; c = np.exp(np.arange(10.0))
    s1 = np.searchsorted(t, 2 * M1 + 5_000, side="right") - 1          # 사건 2분 05초 → 분 2, 기준가 close[2]
    assert s1 == 2 and np.isclose(np.log(c[s1 + 3] / c[s1]), 3.0)


def test():
    selftest()
    rp = OUT / "report.json"; assert not rp.exists(), "판정은 한 번만 연다"
    ev = pd.read_parquet(OUT / "events.parquet"); ev = ev[ev.amount_usd >= MIN_USD]
    px = {c: pd.read_parquet(OUT / f"px_{c}.parquet") for c in ("ETH", "BTC")}
    rng = np.random.default_rng(2026102520)
    rep = dict(min_usd=MIN_USD, gap_h=6, splits=SPLITS)

    def analyse(coin, ts, H):
        t, c = px[coin].t.to_numpy(), np.log(px[coin].c.to_numpy())
        def ret(tt):
            s1 = np.searchsorted(t, tt, side="right") - 1; ok = (s1 >= 0) & (s1 + H < len(t))
            s1c = np.clip(s1, 0, len(t) - 1 - H)
            ok &= (t[s1c] == tt - tt % M1) & (t[s1c + H] == t[s1c] + H * M1)          # 1분 틈이 끼면 버림
            return np.where(ok, c[s1c + H] - c[s1c], np.nan)
        r = ret(ts); out = {}
        for nm, (a, b) in dict(all=(START, END), **SPLITS).items():
            m = (ts >= pd.Timestamp(a).value // 10**6) & (ts < pd.Timestamp(b).value // 10**6) & np.isfinite(r)
            if m.sum() < 3:
                out[nm] = dict(n=int(m.sum())); continue
            # 귀무: 각 사건과 같은 분할 기간(전체면 사건이 속한 분할) · 같은 UTC 시 의 무작위 분
            null = np.zeros(2000)
            for sn, (sa, sb) in SPLITS.items():
                mm = m & (ts >= pd.Timestamp(sa).value // 10**6) & (ts < pd.Timestamp(sb).value // 10**6)
                if not mm.any():
                    continue
                cand = t[(t >= pd.Timestamp(sa).value // 10**6) & (t < pd.Timestamp(sb).value // 10**6 - H * M1)]
                byh = {h: cand[(cand // 3_600_000) % 24 == h] for h in range(24)}
                for h in np.unique((ts[mm] // 3_600_000) % 24):
                    k = int(((ts[mm] // 3_600_000) % 24 == h).sum())
                    draw = byh[h][rng.integers(0, len(byh[h]), (2000, k))]
                    null += np.nansum(ret(draw.ravel() + 30_000).reshape(2000, k), 1)
            null /= m.sum(); mu = float(r[m].mean())
            out[nm] = dict(n=int(m.sum()), days=int(len(np.unique(ts[m] // 86_400_000))), mean_bp=mu * 1e4,
                           hit=float((r[m] > 0).mean()), null_p50_bp=float(np.median(null) * 1e4), null_p95_bp=float(np.percentile(null, 95) * 1e4),
                           pct=float((null < mu).mean() * 100), long_net_bp=mu * 1e4 - 10)
        return out

    for fam in ("USDT", "USDC"):
        for kind in ("mint", "burn"):
            e = ev[(ev.fam == fam) & (ev.kind == kind)]; ts = dedupe(e.ts_ms.to_numpy())
            rep[f"{fam}_{kind}"] = {coin: {f"H{H}": analyse(coin, ts, H) for H in HS} for coin in ("ETH", "BTC")}
            rep[f"{fam}_{kind}"]["events_raw"], rep[f"{fam}_{kind}"]["events"] = int(len(e)), int(len(ts))
    P = rep["USDT_mint"]["ETH"]["H60"]
    rep["G1"] = bool(P["all"].get("pct", 0) > 95)
    rep["G2"] = bool(P["val"].get("pct", 0) > 50 and P["test"].get("pct", 0) > 50)
    rep["G3"] = bool(rep["USDT_mint"]["BTC"]["H60"]["all"].get("pct", 0) > 90)
    rep["verdict"] = "후보(전진 기록)" if rep["G1"] and rep["G2"] and rep["G3"] else "불통과 — 개별 발행 장중 축 닫음"
    rp.write_text(json.dumps(rep, indent=1, ensure_ascii=False)); print(json.dumps(rep, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    a = sys.argv[1:]
    if a[:1] == ["--fetch"]: fetch()
    elif a[:1] == ["--test"]: test()
    elif a[:1] == ["--selftest"]: selftest(); print("ok")
    else: print(__doc__)
