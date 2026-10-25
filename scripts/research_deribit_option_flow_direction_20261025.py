#!/usr/bin/env python3
"""Deribit ETH 옵션 델타 가중 순매수 → ETH 4h 방향. 사전 등록 docs/prereg/deribit_option_flow_direction_20261025.md.

    python3 scripts/research_deribit_option_flow_direction_20261025.py --fetch   # history.deribit.com 일별 저장(이어받기)
    python3 scripts/research_deribit_option_flow_direction_20261025.py --test    # 한 번만
실주문 없음. 2025-12 는 z 정규화 준비 구간으로만 쓴다(평가 표본은 2026 만).
"""
import json, sys, time
from math import erf, sqrt
from pathlib import Path
import numpy as np, pandas as pd, requests

ROOT = Path("/home/kbj20/crypto-scalping")
OUT = ROOT / "tmp" / "deribit_optflow_20261025"
PX = ROOT / "tmp" / "stablecoin_mint_20261025" / "px_ETH.parquet"
H4, M1 = 4 * 3_600_000, 60_000
VAL, TEST = ("2026-01-01", "2026-06-01"), ("2026-06-01", "2026-10-24")
URL = "https://history.deribit.com/api/v2/public/get_last_trades_by_currency_and_time"


def fetch_day(day):
    p = OUT / "trades" / f"{day:%Y-%m-%d}.parquet"
    if p.exists():
        return
    a, b = int(day.value // 10**6), int((day + pd.Timedelta(days=1)).value // 10**6)
    rows, seen = [], set()
    while a < b:
        for i in range(6):
            try:
                r = requests.get(URL, params=dict(currency="ETH", kind="option", start_timestamp=a, end_timestamp=b - 1, count=1000, sorting="asc"), timeout=60)
                if r.status_code == 200:
                    break
            except requests.RequestException:
                pass
            time.sleep(2 + 3 * i)
        else:
            raise RuntimeError(day)
        res = r.json()["result"]; tr = res["trades"]
        for x in tr:
            if x["trade_id"] not in seen:
                seen.add(x["trade_id"])
                rows.append(dict(timestamp=x["timestamp"], instrument_name=x["instrument_name"], direction=x["direction"], amount=x["amount"],
                                 iv=x.get("iv", np.nan), index_price=x["index_price"], block=bool(x.get("block_trade_id")), liq=bool(x.get("liquidation"))))
        if not res.get("has_more") or not tr:
            break
        a = tr[-1]["timestamp"] if tr[-1]["timestamp"] > a else a + 1    # 같은 ms 중복은 seen 으로
        time.sleep(.15)
    pd.DataFrame(rows).to_parquet(p)


def fetch():
    (OUT / "trades").mkdir(parents=True, exist_ok=True)
    for d in pd.date_range("2025-12-01", "2026-10-23", freq="D"):
        fetch_day(d)
        if d.day == 1:
            print(d.date(), flush=True)


def ncdf(x):
    return 0.5 * (1 + np.vectorize(erf)(x / sqrt(2)))


def deltas(df):
    p = df.instrument_name.str.split("-", expand=True)
    exp = pd.to_datetime(p[1], format="%d%b%y").astype("int64") // 10**6 + 8 * 3_600_000
    K, cp = p[2].astype(float), p[3]
    T = (exp - df.timestamp) / (365.25 * 86_400_000); sig = df.iv / 100
    d1 = (np.log(df.index_price / K) + 0.5 * sig**2 * T) / (sig * np.sqrt(T))
    d = ncdf(d1.to_numpy()) - (cp == "P").to_numpy()
    return np.where((T > 0) & (sig > 0), d, np.nan)


def selftest():
    df = pd.DataFrame(dict(instrument_name=["ETH-4JAN26-3000-C", "ETH-4JAN26-3000-P"], timestamp=[1767225600000] * 2, iv=[50.0] * 2, index_price=[3000.0] * 2))
    d = deltas(df)
    assert 0.5 < d[0] < 0.53 and np.isclose(d[0] - d[1], 1.0)        # ATM 콜 ≈ 0.5+, 풋-콜 패리티 차 1


def partial_ic(x, y, z):
    rx, ry, rz = (pd.Series(v).rank().to_numpy() for v in (x, y, z))
    res = lambda a: a - np.polyval(np.polyfit(rz, a, 1), rz)
    return float(np.corrcoef(res(rx), res(ry))[0, 1])


def test():
    selftest()
    rp = OUT / "report.json"; assert not rp.exists(), "판정은 한 번만 연다"
    tr = pd.concat([pd.read_parquet(f) for f in sorted((OUT / "trades").glob("*.parquet"))], ignore_index=True)
    tr = tr[tr.instrument_name.str.match(r"^ETH-\d+[A-Z]{3}\d{2}-\d+-[CP]$")]
    tr["d"] = deltas(tr); tr = tr[np.isfinite(tr.d)]
    tr["dnb"] = np.where(tr.direction == "buy", 1, -1) * tr.d * tr.amount
    tr["bin"] = (tr.timestamp // H4 + 1) * H4                     # 창 [t−4h, t) → 결정 시각 t
    px = pd.read_parquet(PX); pc = pd.Series(np.log(px.c.to_numpy()), index=px.t.to_numpy())
    grid = np.arange(pd.Timestamp("2025-12-01").value // 10**6, pd.Timestamp("2026-10-24").value // 10**6 + 1, H4)
    close_at = lambda t: pc.reindex(t - M1).to_numpy()              # t 직전 분 종가

    def series(m):
        s = tr[m].groupby("bin").dnb.sum().reindex(grid, fill_value=0.0)
        return (s / s.rolling(180, min_periods=180).std().shift(1)).to_numpy()
    base = ~tr.block & ~tr.liq
    Z = dict(main=series(base), calls=series(base & tr.instrument_name.str.endswith("C")),
             puts=series(base & tr.instrument_name.str.endswith("P")), with_blocks=series(~tr.liq))
    past = close_at(grid) - close_at(grid - H4)
    ins = lambda a, b: (grid >= pd.Timestamp(a).value // 10**6) & (grid < pd.Timestamp(b).value // 10**6)
    ic = lambda x, y, m: float(pd.Series(x[m]).corr(pd.Series(y[m]), method="spearman"))
    rep = dict(trades=int(len(tr)), blocks=int(tr.block.sum()), first=str(pd.to_datetime(tr.timestamp.min(), unit="ms")), last=str(pd.to_datetime(tr.timestamp.max(), unit="ms")))
    for hn, H in dict(H1=3_600_000, H4=H4, H24=24 * 3_600_000).items():
        y = close_at(grid + H) - close_at(grid)
        for zn, z in Z.items():
            ok = np.isfinite(z) & np.isfinite(y) & np.isfinite(past)
            m26 = ok & ins("2026-01-01", "2026-10-24")
            r = dict(n=int(m26.sum()), ic_all=ic(z, y, m26), ic_val=ic(z, y, ok & ins(*VAL)), ic_test=ic(z, y, ok & ins(*TEST)),
                     partial_ic=partial_ic(z[m26], y[m26], past[m26]))
            if hn == "H4" and zn == "main":
                rng = np.random.default_rng(2026102521); idx = np.flatnonzero(m26); n = len(idx)
                ks = rng.integers(42, n - 42, 1000)
                null = np.array([pd.Series(z[idx][np.roll(np.arange(n), k)]).corr(pd.Series(y[idx]), method="spearman") for k in ks])
                r.update(null_p95=float(np.percentile(null, 95)), pct=float((null < r["ic_all"]).mean() * 100))
                pos = np.where(np.abs(z) > 1, np.sign(z), 0.0)[idx]; net = pos * y[idx] - 10e-4 * (pos != 0)
                rate = float((pos != 0).mean()); rnd = []
                for _ in range(1000):
                    q = np.where(rng.random(n) < rate, rng.choice([-1.0, 1.0], n), 0.0)
                    rnd.append(float((q * y[idx] - 10e-4 * (q != 0)).sum()))
                r.update(trades=int((pos != 0).sum()), net_bp_per_trade=float(net[pos != 0].mean() * 1e4) if (pos != 0).any() else 0.0,
                         total_bp=float(net.sum() * 1e4), random_pct=float((np.array(rnd) < net.sum()).mean() * 100))
            rep[f"{hn}_{zn}"] = r
    P = rep["H4_main"]
    rep["G1"] = bool(P["pct"] > 95); rep["G2"] = bool(P["ic_val"] > 0 and P["ic_test"] > 0); rep["G3"] = bool(P["partial_ic"] > 0)
    rep["verdict"] = "후보(전진 판정)" if rep["G1"] and rep["G2"] and rep["G3"] else "불통과 — 옵션 순매수 방향 닫음"
    rp.write_text(json.dumps(rep, indent=1, ensure_ascii=False)); print(json.dumps(rep, indent=1, ensure_ascii=False))


FWD0, FWD_DAY = pd.Timestamp("2026-10-24"), pd.Timestamp("2027-03-24")   # 사후 매도 꼬리 전진 창 [FWD0, FWD_DAY)


def forward():
    """사후(10-25) 매도 꼬리 z<−1 → 다음 4h 하락. 판정일 전 거절. 문서 docs/prereg/deribit_option_sell_tail_forward_20261025.md."""
    import research_stablecoin_mint_intraday_20261025 as S
    assert pd.Timestamp.utcnow().tz_localize(None) >= FWD_DAY, f"판정일 {FWD_DAY.date()} 전 — 전진 자료를 보지 않는다"
    rp = OUT / f"forward_sell_tail_{FWD_DAY.date()}.json"; assert not rp.exists(), "판정은 한 번만"
    for d in pd.date_range(FWD0, FWD_DAY - pd.Timedelta(days=1), freq="D"):
        fetch_day(d)
    tr = pd.concat([pd.read_parquet(f) for f in sorted((OUT / "trades").glob("*.parquet"))], ignore_index=True)
    tr = tr[tr.instrument_name.str.match(r"^ETH-\d+[A-Z]{3}\d{2}-\d+-[CP]$") & ~tr.block & ~tr.liq]
    tr["d"] = deltas(tr); tr = tr[np.isfinite(tr.d) & (tr.timestamp < FWD_DAY.value // 10**6)]
    tr["dnb"] = np.where(tr.direction == "buy", 1, -1) * tr.d * tr.amount; tr["bin"] = (tr.timestamp // H4 + 1) * H4
    px = pd.concat([pd.read_parquet(PX)] + [S.vision("ETHUSDT", "daily", f"{d:%Y-%m-%d}") for d in pd.date_range(FWD0, FWD_DAY - pd.Timedelta(days=1), freq="D")])
    pc = pd.Series(np.log(px.drop_duplicates("t").set_index("t").c)).sort_index()
    grid = np.arange(pd.Timestamp("2025-12-01").value // 10**6, FWD_DAY.value // 10**6 + 1, H4)
    s = tr.groupby("bin").dnb.sum().reindex(grid, fill_value=0.0); z = (s / s.rolling(180, min_periods=180).std().shift(1)).to_numpy()
    y = pc.reindex(grid + H4 - M1).to_numpy() - pc.reindex(grid - M1).to_numpy()
    ok = np.isfinite(z) & np.isfinite(y) & (grid >= FWD0.value // 10**6); ev = ok & (z < -1)
    rng = np.random.default_rng(2026102522); pool = y[ok]; k = int(ev.sum())
    null = np.array([pool[rng.choice(len(pool), k, replace=False)].mean() for _ in range(2000)])
    mu = float(y[ev].mean()); pct = float((null > mu).mean() * 100)                 # 숏 기준: 무작위보다 더 내릴수록 높음
    rep = dict(window=[str(FWD0.date()), str(FWD_DAY.date())], events=k, mean_bp=mu * 1e4, all_mean_bp=float(pool.mean() * 1e4),
               short_net_bp=-mu * 1e4 - 10, null_pct=pct, verdict="후보(옵션 매도 꼬리)" if pct > 95 and k >= 30 else "불통과")
    rp.write_text(json.dumps(rep, indent=1, ensure_ascii=False)); print(rep)


if __name__ == "__main__":
    a = sys.argv[1:]
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    if a[:1] == ["--fetch"]: fetch()
    elif a[:1] == ["--test"]: test()
    elif a[:1] == ["--forward"]: forward()
    elif a[:1] == ["--selftest"]: selftest(); print("ok")
    else: print(__doc__)
