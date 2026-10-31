"""XRP 단독 «7일 예측»(코인베이스 프리미엄 C1a) 판정 + «지금 추세» 서술 -- 사전등록 docs/prereg/xrp_trend_cb7d_20261031.md.
실행: python scripts/research_xrp_trend_cb7d_20261031.py [--selftest]
"""
import io, json, sys, time, urllib.request, zipfile
from pathlib import Path

import numpy as np
import pandas as pd

WT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WT))
from dashboard.dir_signals import cb_premium_z  # noqa: E402  화면과 같은 식
from dashboard.trend_rule import LOOKBACKS, TARGET_VOL, MAX_LEVERAGE, VOL_WINDOW  # noqa: E402

RAW = Path("/home/kbj20/crypto-scalping/.claude/worktrees/medium-long-term-forecast-model-669354/tmp")
CACHE = WT / "data/xrp_trend_cb7d_cache"
OUT = WT / "tmp/xrp_trend_cb7d_20261031"
LAST = pd.Timestamp("2026-10-30")
H, K, BLOCK, NB = 7, 1.0, 14, 2000


def _get(url):
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return r.read()


def bn_close(sym):
    s = pd.read_csv(RAW / f"trend_mid_long_20261015/raw/bn_{sym}.csv", index_col=0, parse_dates=True).close
    CACHE.mkdir(parents=True, exist_ok=True)
    for d in pd.date_range(s.index[-1] + pd.Timedelta(days=1), LAST):
        f = CACHE / f"bn_{sym}_{d.date()}.json"
        if not f.exists():
            try:
                z = zipfile.ZipFile(io.BytesIO(_get(f"https://data.binance.vision/data/spot/daily/klines/{sym}USDT/1d/{sym}USDT-1d-{d.date()}.zip")))
                row = pd.read_csv(z.open(z.namelist()[0]), header=None).iloc[0]
                f.write_text(json.dumps(float(row[4])))
            except Exception as e:  # noqa: BLE001  빠진 날은 NaN(라벨·z 가 알아서 건너뜀)
                print("missing", sym, d.date(), e)
                continue
        s.loc[d] = json.loads(f.read_text())
    return s.sort_index()


def cb_btc():
    s = pd.read_csv(RAW / "short_fresh_20261015/raw/cb_BTC.csv", index_col=0, parse_dates=True).close
    s = s[s.index < pd.Timestamp("2026-10-15")]   # 10-15 행은 덜 찬 봉(거래량 73)
    f = CACHE / "cb_BTC_tail.json"
    if not f.exists():
        q = f"granularity=86400&start=2026-10-10T00:00:00Z&end={(LAST + pd.Timedelta(days=1)).date()}T00:00:00Z"
        f.write_text(_get(f"https://api.exchange.coinbase.com/products/BTC-USD/candles?{q}").decode())
    for r in json.loads(f.read_text()):
        d = pd.Timestamp(r[0], unit="s")
        if pd.Timestamp("2026-10-15") <= d <= LAST:
            s.loc[d] = float(r[4])
    return s.sort_index()


def label(c):
    return np.sign(c.shift(-H) / c - 1)


def block_hits(hit, rng):
    a = hit.values.astype(float)
    n = len(a)
    nbk = int(np.ceil(n / BLOCK))
    bs = [a[((rng.integers(0, n, nbk)[:, None] + np.arange(BLOCK)) % n).ravel()].mean() for _ in range(NB)]
    return float(np.percentile(bs, 2.5)), float(np.percentile(bs, 10))


def cb_eval(z, c, lo, hi, rng):
    st = pd.Series(np.where(z >= K, 1, np.where(z <= -K, -1, 0)), index=z.index)
    y = label(c).reindex(st.index)
    m = (st != 0) & y.notna() & (y != 0) & (st.index >= lo) & (st.index <= hi)
    hit = (st[m] == y[m])
    lo95, lo90 = block_hits(hit, rng)
    yy, ss = y[m].values, st[m].values
    rnd = [np.mean(rng.permutation(ss) == yy) for _ in range(1000)]
    yr = {str(g): f"{int(v.sum())}/{len(v)}" for g, v in hit.groupby(hit.index.year)}
    base = (y[(y != 0) & y.notna() & (y.index >= lo) & (y.index <= hi)] > 0).mean()
    return {"hit": float(hit.mean()), "n": f"{int(hit.sum())}/{len(hit)}", "ci95_lo": lo95, "lo90_one_sided": lo90,
            "rand95": float(np.percentile(rnd, 95)), "coverage": float(m.sum() / max(1, ((y.index >= lo) & (y.index <= hi) & y.notna()).sum())),
            "always_up": float(base), "by_year": yr}


def trend_desc(c):
    sig = sum(np.sign(c - c.shift(L)) for L in LOOKBACKS) / len(LOOKBACKS)
    y = label(c)
    r1 = np.log(c).diff()
    vol = r1.rolling(VOL_WINDOW).std() * np.sqrt(365)
    size = (sig * TARGET_VOL / vol).clip(-MAX_LEVERAGE, MAX_LEVERAGE)
    out = {}
    for nm, lo in (("all", "2018-01-01"), ("2024~", "2024-01-01"), ("2026~", "2026-01-01")):
        m = (sig.index >= lo) & y.notna() & sig.notna()
        up = {f"{v:+.1f}": f"{int((y[m & (sig == v)] > 0).sum())}/{int((m & (sig == v)).sum())}" for v in sorted(sig[m].unique())}
        hit = (np.sign(sig[m]) == y[m]).mean()
        pnl = (size.shift(1) * r1)[sig.index >= lo].dropna()
        bh = r1[sig.index >= lo].dropna()
        sh = lambda x: float(x.mean() / x.std() * np.sqrt(365))
        out[nm] = {"up7_by_signal": up, "hit7": float(hit), "sharpe_rule": sh(pnl), "sharpe_hold": sh(bh)}
    return out


def selftest():
    idx = pd.date_range("2020-01-01", periods=60)
    c = pd.Series(np.arange(1, 61, dtype=float), idx)
    assert label(c).iloc[0] == 1 and np.isnan(label(c).iloc[-1])
    st = pd.Series([1, -1, 0, 1], index=idx[:4])
    y = pd.Series([1, 1, 1, -1], index=idx[:4])
    m = st != 0
    assert (st[m] == y[m]).mean() == 1 / 3
    print("selftest ok")


def main():
    selftest()
    if "--selftest" in sys.argv:
        return
    rng = np.random.default_rng(20261031)
    cl = {s: bn_close(s) for s in ("BTC", "ETH", "XRP")}
    z = cb_premium_z(cb_btc(), cl["BTC"])
    rep = {"z_last": [str(z.dropna().index[-1].date()), float(z.dropna().iloc[-1])], "cb7d": {}, "trend": {}}
    for s in ("XRP", "ETH", "BTC"):
        rep["cb7d"][s] = {"holdout_2024_1023": cb_eval(z, cl[s], "2024-01-01", "2026-10-23", rng),
                          "new_1016_1023": cb_eval(z, cl[s], "2026-10-16", "2026-10-23", rng)}
        rep["trend"][s] = trend_desc(cl[s])
    x = rep["cb7d"]["XRP"]["holdout_2024_1023"]
    rep["verdict_xrp_cb7d"] = {"strict": x["hit"] >= 0.60 and x["ci95_lo"] > 0.50, "candidate": x["lo90_one_sided"] > 0.50,
                               "beats_random95": x["hit"] > x["rand95"]}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "report.json").write_text(json.dumps(rep, ensure_ascii=False, indent=1))
    print(json.dumps(rep, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
