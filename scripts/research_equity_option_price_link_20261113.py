"""주식 옵션 문헌 3지표(CW·SKEW·VRP) → 다음 1·5일 수익 횡단면 IC — 사전등록 docs/prereg/equity_option_price_link_20261113.md (9bc9e840).

    python scripts/research_equity_option_price_link_20261113.py fetch      # DoltHub 체인·변동성 이력 + Yahoo 일봉 → tmp/equity_optlink_20261113/
    python scripts/research_equity_option_price_link_20261113.py run        # 주 판정 6칸 + 보조 S1·S2 (S3 는 cboe 폴더가 있으면)
    python scripts/research_equity_option_price_link_20261113.py selftest
"""
from __future__ import annotations

import gzip
import json
import sys
import time
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
D = ROOT / "tmp" / "equity_optlink_20261113"
SYMS = ["MU", "SNDK", "NVDA", "AMD", "AVGO", "INTC", "QCOM", "AMAT", "LRCX", "MRVL", "TXN", "KLAC",
        "AAPL", "MSFT", "AMZN", "META", "GOOGL", "TSLA", "NFLX", "PLTR"]
START, END = date(2026, 1, 2), date(2026, 11, 12)
SPLIT_H2 = pd.Timestamp("2026-07-01")
SIGN = {"CW": 1, "SKEW": -1, "VRP": -1}      # 문헌 부호
HZ = (1, 5)
SEED, NBOOT, NPLAC, BLOCK = 20261113, 2000, 500, 10
DOLT = "https://www.dolthub.com/api/v1alpha1/post-no-preference/options/master"
UA = {"User-Agent": "Mozilla/5.0"}


# ── 내려받기 ─────────────────────────────────────────────────────────
def dolt(q: str, tries: int = 4) -> list[dict]:
    for k in range(tries):
        try:
            with urllib.request.urlopen(urllib.request.Request(DOLT + "?" + urllib.parse.urlencode({"q": q}), headers=UA), timeout=120) as r:
                d = json.loads(r.read())
            if d.get("query_execution_status") == "Success":
                return d.get("rows", [])
            err = d.get("query_execution_message")
        except Exception as e:  # noqa: BLE001 -- 재시도
            err = repr(e)
        time.sleep(2 + 3 * k)
    raise RuntimeError(f"dolt 실패: {err} · {q[:120]}")


def fetch() -> None:
    (D / "chain").mkdir(parents=True, exist_ok=True)
    vh = []
    for s in SYMS:
        rows = dolt(f"SELECT date, act_symbol, iv_current, hv_current FROM volatility_history WHERE act_symbol='{s}' AND date BETWEEN '{START - timedelta(days=10)}' AND '{END}'")
        vh += rows
        print(s, "vol_history", len(rows), flush=True)
    pd.DataFrame(vh).to_csv(D / "vol_history.csv", index=False)
    jobs = [(r["act_symbol"], r["date"]) for r in vh if r["date"] >= str(START - timedelta(days=10))]

    def one(job):
        s, d = job
        p = D / "chain" / s / f"{d}.json.gz"
        if p.exists():
            return 0
        rows = dolt(f"SELECT expiration, strike, call_put, bid, ask, vol, delta FROM option_chain WHERE date='{d}' AND act_symbol='{s}' "
                    f"AND expiration BETWEEN DATE_ADD('{d}', INTERVAL 7 DAY) AND DATE_ADD('{d}', INTERVAL 60 DAY)")
        p.parent.mkdir(exist_ok=True)
        p.write_bytes(gzip.compress(json.dumps(rows).encode()))
        return 1
    n = 0
    with ThreadPoolExecutor(4) as ex:     # ponytail: 공개 API 예의상 4줄 동시 -- 느리면 그대로 기다린다
        for i, k in enumerate(ex.map(one, jobs)):
            n += k
            if i % 200 == 0:
                print("chain", i, "/", len(jobs), flush=True)
    print("chain 새로 받음", n)
    px = []
    for s in SYMS + ["SOXL", "SPY"]:
        u = f"https://query1.finance.yahoo.com/v8/finance/chart/{s}?range=2y&interval=1d"
        with urllib.request.urlopen(urllib.request.Request(u, headers=UA), timeout=60) as r:
            res = json.loads(r.read())["chart"]["result"][0]
        q = res["indicators"]["quote"][0]
        px.append(pd.DataFrame({"sym": s, "date": pd.to_datetime(res["timestamp"], unit="s").tz_localize("UTC").tz_convert("America/New_York").date,
                                "close": q["close"], "adj": res["indicators"]["adjclose"][0]["adjclose"]}))
        time.sleep(0.5)
    pd.concat(px).dropna().to_csv(D / "prices.csv", index=False)
    for s in ("SOXL", "SNDK"):   # S3 용 5분봉(Yahoo 는 60일까지)
        u = f"https://query1.finance.yahoo.com/v8/finance/chart/{s}?range=30d&interval=5m"
        with urllib.request.urlopen(urllib.request.Request(u, headers=UA), timeout=60) as r:
            res = json.loads(r.read())["chart"]["result"][0]
        pd.DataFrame({"ts": res["timestamp"], "close": res["indicators"]["quote"][0]["close"]}).dropna().to_csv(D / f"{s}_5m.csv", index=False)
    print("prices ok")


# ── 지표 ─────────────────────────────────────────────────────────────
def chain_features(rows: list[dict], d: date, S: float) -> dict:
    """한 종목·한 날 체인 → CW·SKEW·ATM IV·패리티 현물. 문서 정의 그대로."""
    out = {"CW": np.nan, "SKEW": np.nan, "ATM": np.nan, "PAR": np.nan, "cw_n": 0, "cw_diff": np.nan}
    if not rows or not S > 0:
        return out
    c = pd.DataFrame(rows)
    for k in ("strike", "bid", "ask", "vol"):
        c[k] = pd.to_numeric(c[k], errors="coerce")
    c["days"] = (pd.to_datetime(c["expiration"]) - pd.Timestamp(d)).dt.days
    c["m"] = c["strike"] / S
    w = c[(c["days"] >= 10) & (c["days"] <= 60)]
    piv = w.pivot_table(index=["expiration", "strike"], columns="call_put", values="vol", aggfunc="first")
    if {"Call", "Put"} <= set(piv.columns):
        pr = piv.reset_index()
        pr = pr[(pr["strike"] / S >= 0.9) & (pr["strike"] / S <= 1.1) & (pr["Call"] > 0) & (pr["Put"] > 0)]
        if len(pr):
            out["cw_n"], out["cw_diff"] = len(pr), float((pr["Call"] != pr["Put"]).mean())
            if len(pr) >= 3 and out["cw_diff"] >= 0.5:
                out["CW"] = float((pr["Call"] - pr["Put"]).mean())
    best = None
    for e, g in w.groupby("expiration"):
        p = g[(g["call_put"] == "Put") & (g["m"] >= 0.80) & (g["m"] <= 0.95) & (g["vol"] > 0)]
        k = g[(g["call_put"] == "Call") & (g["m"] >= 0.95) & (g["m"] <= 1.05) & (g["vol"] > 0)]
        if len(p) and len(k):
            dd = abs(int(g["days"].iloc[0]) - 30)
            if best is None or dd < best[0]:
                pv = p.iloc[(p["m"] - 0.95).abs().argsort().iloc[0]]["vol"]
                kv = k.iloc[(k["m"] - 1.0).abs().argsort().iloc[0]]["vol"]
                a = g[(g["vol"] > 0)].copy()
                a["dk"] = (a["m"] - 1).abs()
                atm = a[a["dk"] == a["dk"].min()]["vol"].mean()
                best = (dd, float(pv - kv), float(atm))
    if best:
        out["SKEW"], out["ATM"] = best[1], best[2]
    near = c[(c["days"] >= 7)]
    if len(near):
        e0 = near["days"].min()
        g = near[(near["days"] == e0) & (near["bid"] > 0) & (near["ask"] > 0)]
        mid = g.assign(mid=(g["bid"] + g["ask"]) / 2).pivot_table(index="strike", columns="call_put", values="mid", aggfunc="first").dropna()
        if len(mid) and {"Call", "Put"} <= set(mid.columns):
            mid = mid.iloc[np.argsort(np.abs(mid.index.values - S))[:5]]
            out["PAR"] = float(np.median(mid["Call"] - mid["Put"] + mid.index.values))
    return out


def build_panel() -> tuple[pd.DataFrame, pd.DataFrame]:
    px = pd.read_csv(D / "prices.csv", parse_dates=["date"])
    vh = pd.read_csv(D / "vol_history.csv", parse_dates=["date"])
    rows = []
    for _, r in vh.iterrows():
        s, d = r["act_symbol"], r["date"].date()
        p = D / "chain" / s / f"{d}.json.gz"
        cl = px[(px["sym"] == s) & (px["date"] == pd.Timestamp(d))]
        S = float(cl["close"].iloc[0]) if len(cl) else np.nan
        f = chain_features(json.loads(gzip.decompress(p.read_bytes())) if p.exists() else [], d, S)
        rows.append({"sym": s, "date": pd.Timestamp(d), "S": S, "VRP": float(r["iv_current"]) - float(r["hv_current"])
                     if pd.notna(r["iv_current"]) and pd.notna(r["hv_current"]) else np.nan, **f})
    return pd.DataFrame(rows), px


def g0(panel: pd.DataFrame, px: pd.DataFrame) -> dict:
    """패리티 현물이 어느 날 종가와 맞나 -- d−1·d·d+1 의 |로그 차| 중앙(2026 분할 종목 제외)."""
    out = {}
    for lag in (-1, 0, 1):
        diffs = []
        for s, g in panel.dropna(subset=["PAR"]).groupby("sym"):
            c = px[px["sym"] == s].set_index("date")["close"].sort_index()
            r = c.pct_change().abs()
            if (r[(r.index >= pd.Timestamp(START))] > 0.4).any():
                continue          # 액면분할 의심
            idx = list(c.index)
            for _, row in g.iterrows():
                if row["date"] not in c.index:
                    continue
                j = idx.index(row["date"]) + lag
                if 0 <= j < len(idx):
                    diffs.append(abs(np.log(row["PAR"] / c.iloc[j])))
        out[lag] = float(np.median(diffs)) if diffs else np.nan
    return out


def add_targets(panel: pd.DataFrame, px: pd.DataFrame, lag: int) -> pd.DataFrame:
    out = []
    for s, g in panel.groupby("sym"):
        a = px[px["sym"] == s].set_index("date")["adj"].sort_index()
        la = np.log(a)
        g = g.copy()
        pos = a.index.get_indexer(g["date"])
        for h in HZ:
            v = np.full(len(g), np.nan)
            ok = (pos >= 0) & (pos + lag + h < len(a))
            v[ok] = la.values[pos[ok] + lag + h] - la.values[pos[ok] + lag]
            g[f"R{h}"] = v
        pr = np.full(len(g), np.nan)
        ok = (pos >= 5)
        pr[ok] = la.values[pos[ok]] - la.values[pos[ok] - 5]
        g["P5"] = pr
        out.append(g)
    return pd.concat(out)


# ── 통계 ─────────────────────────────────────────────────────────────
def daily_ic(df: pd.DataFrame, x: str, y: str, min_n: int = 10, resid_on: str | None = None) -> pd.Series:
    vals = {}
    for d, g in df.dropna(subset=[x, y] + ([resid_on] if resid_on else [])).groupby("date"):
        if len(g) < min_n:
            continue
        xv = g[x].values
        if resid_on:
            A = np.c_[np.ones(len(g)), g[resid_on].values]
            xv = xv - A @ np.linalg.lstsq(A, xv, rcond=None)[0]
        vals[d] = spearmanr(xv, g[y].values)[0]
    return pd.Series(vals).sort_index()


def block_boot(s: np.ndarray, rng, nb: int = NBOOT, blk: int = BLOCK) -> np.ndarray:
    n = len(s)
    k = int(np.ceil(n / blk))
    st = rng.integers(0, max(1, n - blk + 1), size=(nb, k))
    idx = (st[:, :, None] + np.arange(blk)).reshape(nb, -1)[:, :n]
    return s[idx].mean(1)


def holm(ps: dict) -> dict:
    ks = sorted(ps, key=ps.get)
    m, out, run = len(ks), {}, 0.0
    for i, k in enumerate(ks):
        run = max(run, min(1.0, (m - i) * ps[k]))
        out[k] = run
    return out


def run() -> None:
    panel, px = build_panel()
    panel = panel[(panel["date"] >= pd.Timestamp(START)) & (panel["date"] <= pd.Timestamp(END))]
    gz = g0(panel, px)
    lag = 0 if (np.nanargmin([gz[-1], gz[0], gz[1]]) == 1 and gz[0] < 0.01) else 1
    print("G0 |log(패리티/종가)| 중앙 d−1·d·d+1:", {k: round(v, 5) for k, v in gz.items()}, "→ 수익 시작 늦춤", lag)
    df = add_targets(panel, px, lag)
    print("종목", df["sym"].nunique(), sorted(df["sym"].unique()), "· 날", df["date"].nunique())
    for f in SIGN:
        print(f"  {f} 유효 종목·일 {df[f].notna().sum()}/{len(df)} ({df[f].notna().mean():.0%})")
    print("  CW 콜≠풋 IV 쌍 비율 중앙", round(float(df["cw_diff"].median()), 3))
    rng = np.random.default_rng(SEED)
    res, ps = {}, {}
    for f, sg in SIGN.items():
        df[f + "s"] = df[f] * sg
        cw_ok = f != "CW" or df[f].notna().mean() >= 0.5
        for h in HZ:
            key = f"{f}·R{h}"
            if not cw_ok:
                res[key] = {"verdict": "판정 불가(CW 자료 문턱)"}
                continue
            ic = daily_ic(df, f + "s", f"R{h}")
            bm = block_boot(ic.values, rng)
            p = float((bm <= 0).mean())
            h1, h2 = ic[ic.index < SPLIT_H2].mean(), ic[ic.index >= SPLIT_H2].mean()
            icr = daily_ic(df, f + "s", f"R{h}", resid_on="P5")
            bmr = block_boot(icr.values, rng)
            plac = []
            for _ in range(NPLAC):
                sh = df[["date", f + "s"]].copy()
                sh[f + "s"] = sh.groupby("date")[f + "s"].transform(lambda v: rng.permutation(v.values))
                plac.append(daily_ic(df.assign(**{f + "s": sh[f + "s"].values}), f + "s", f"R{h}").mean())
            pct = float((np.array(plac) < ic.mean()).mean() * 100)
            # 상·하위 5종목 다음 수익 차
            sp, win = [], 0
            for d, g in df.dropna(subset=[f + "s", f"R{h}"]).groupby("date"):
                if len(g) < 10:
                    continue
                g = g.sort_values(f + "s")
                v = g[f"R{h}"].tail(5).mean() - g[f"R{h}"].head(5).mean()
                sp.append(v)
                win += v > 0
            res[key] = {"IC": ic.mean(), "lo90": float(np.quantile(bm, 0.10)), "lo80": float(np.quantile(bm, 0.20)),
                        "ci95": (float(np.quantile(bm, 0.025)), float(np.quantile(bm, 0.975))), "p": p, "n_days": len(ic),
                        "H1": h1, "H2": h2, "IC_resid": icr.mean(), "lo90_resid": float(np.quantile(bmr, 0.10)), "plac_pct": pct,
                        "spread_bp": float(np.mean(sp) * 1e4), "win": f"{win}/{len(sp)}", "n_per_day": float(df.dropna(subset=[f + 's', f'R{h}']).groupby('date').size().mean())}
            ps[key] = p
    hp = holm(ps)
    print("\n== 주 판정(부호 = 문헌대로 양수) ==")
    for k, r in res.items():
        if "IC" not in r:
            print(k, r["verdict"])
            continue
        ok = [hp[k] < 0.10, r["H1"] > 0 and r["H2"] > 0, r["IC_resid"] > 0 and r["lo90_resid"] > 0, r["plac_pct"] >= 90]
        cand = r["lo80"] > 0 and r["H1"] > 0 and r["H2"] > 0
        r["verdict"] = "통과" if all(ok) else ("통과 후보" if cand else "불통과")
        print(f"{k}: IC̄ {r['IC']:+.4f} [단측90 하한 {r['lo90']:+.4f} · 양측95 {r['ci95'][0]:+.4f}~{r['ci95'][1]:+.4f}] p {r['p']:.3f} Holm {hp[k]:.3f}"
              f" · 날 {r['n_days']}·날당 {r['n_per_day']:.1f}종목 · H1 {r['H1']:+.4f} H2 {r['H2']:+.4f} · 통제 {r['IC_resid']:+.4f}(하한 {r['lo90_resid']:+.4f})"
              f" · 위약 {r['plac_pct']:.0f}분위 · 상하위5 차 {r['spread_bp']:+.1f}bp 승 {r['win']} · 조건 {ok} → {r['verdict']}")
    # S1
    print("\n== S1 MU·SNDK 시계열(판정 안 씀) ==")
    for s in ("MU", "SNDK"):
        g = df[df["sym"] == s].sort_values("date")
        for f in SIGN:
            for h in HZ:
                x = g[[f + "s", f"R{h}"]].dropna()
                if len(x) < 30:
                    print(s, f, h, "표본", len(x))
                    continue
                r = spearmanr(x.iloc[:, 0], x.iloc[:, 1])[0]
                rk = x.rank().values
                rk = (rk - rk.mean(0)) / rk.std(0)
                bm = block_boot(rk[:, 0] * rk[:, 1], rng)
                print(f"  {s} {f}→R{h}: ρ {r:+.3f} (90% {np.quantile(bm, .05):+.3f}~{np.quantile(bm, .95):+.3f}) n {len(x)}")
    # S2 동시: ΔATM vs 같은 날 수익
    print("\n== S2 동시 상관(예측 아님) ==")
    dd = []
    for s, g in df.sort_values("date").groupby("sym"):
        a = px[px["sym"] == s].set_index("date")["adj"].sort_index()
        g = g.copy()
        g["dATM"] = g["ATM"].diff()
        prev = a.shift(1)
        g["R0"] = np.log(a.reindex(g["date"]).values / prev.reindex(g["date"]).values)
        g.loc[g["date"].diff().dt.days > 5, "dATM"] = np.nan
        dd.append(g)
    dd = pd.concat(dd)
    ic0 = daily_ic(dd, "dATM", "R0")
    mu = dd[dd["sym"] == "MU"][["dATM", "R0"]].dropna()
    print(f"  횡단면 ΔATM IV ↔ 같은 날 수익 IC̄ {ic0.mean():+.3f} (날 {len(ic0)}) · MU 시계열 ρ {spearmanr(mu['dATM'], mu['R0'])[0]:+.3f} (n {len(mu)})")
    json.dump({"g0": gz, "lag": lag, "res": {k: {kk: (vv if not isinstance(vv, (np.floating,)) else float(vv)) for kk, vv in v.items()} for k, v in res.items()},
               "holm": hp}, open(D / "result.json", "w"), ensure_ascii=False, default=float, indent=1)
    s3()


def s3() -> None:
    """S3 자체 CBOE 10분 체인 -- 서술만. tmp/.../cboe/<SYM>/<날짜>/<시각>.json.gz(서버 사본)."""
    base = D / "cboe"
    if not base.exists():
        print("\nS3: cboe 사본 없음 -- 건너뜀")
        return
    print("\n== S3 자체 CBOE 10분(서술만 · 판정 안 함) ==")
    for sym in ("SOXL", "SNDK"):
        rec = []
        for p in sorted((base / sym).glob("*/*.json.gz")):
            try:
                d = json.loads(gzip.decompress(p.read_bytes()))["data"]
            except Exception:  # noqa: BLE001
                continue
            pv = sum(float(o.get("volume") or 0) for o in d["options"] if o["option"][-9] == "P")
            cv = sum(float(o.get("volume") or 0) for o in d["options"] if o["option"][-9] == "C")
            rec.append({"t": pd.Timestamp(d["last_trade_time"]), "S": float(d["current_price"]), "iv": float(d.get("iv30") or np.nan),
                        "ps": pv / (pv + cv) if pv + cv > 0 else np.nan})
        r = pd.DataFrame(rec).drop_duplicates("t").sort_values("t").set_index("t")
        r = r[(r.index.time >= pd.Timestamp("09:40").time()) & (r.index.time <= pd.Timestamp("16:00").time())]
        same = r.index.to_series().diff().dt.total_seconds().between(300, 900)
        dr = np.log(r["S"]).diff()[same]
        div = r["iv"].diff()[same]
        x = pd.concat([div, dr], axis=1).dropna()
        fwd = np.log(r["S"]).shift(-6) - np.log(r["S"])
        sameday = pd.Series(r.index.date, index=r.index)
        fwd[sameday.shift(-6) != sameday] = np.nan
        dps = r["ps"].diff()[same]
        y = pd.concat([dps, fwd], axis=1).dropna()
        print(f"  {sym}: 날 {len(set(r.index.date))} · 10분 ΔIV30 ↔ 같은 10분 수익 ρ {spearmanr(x.iloc[:, 0], x.iloc[:, 1])[0]:+.3f} (n {len(x)})"
              f" · 풋 거래 비중 변화 → 다음 60분 수익 ρ {spearmanr(y.iloc[:, 0], y.iloc[:, 1])[0]:+.3f} (n {len(y)})")


def selftest() -> None:
    d = date(2026, 3, 2)
    rows = []
    for k in (90, 95, 100, 105, 110):
        for cp, v in (("Call", 0.40 + 0.001 * k), ("Put", 0.38 + 0.001 * k + (0.05 if k == 95 else 0))):
            rows.append({"expiration": "2026-04-01", "strike": str(k), "call_put": cp, "bid": "1", "ask": "1.2", "vol": str(v), "delta": "0"})
    f = chain_features(rows, d, 100.0)
    assert f["cw_n"] == 5 and abs(f["CW"] - (0.02 * 5 - 0.05) / 5) < 1e-9, f       # 콜 − 풋 평균 = (0.02×5 − 0.05)/5
    assert abs(f["SKEW"] - ((0.38 + 0.095 + 0.05) - (0.40 + 0.100))) < 1e-9, f       # 풋 K 95 − 콜 K 100
    same = [dict(r, vol="0.5") for r in rows]
    assert np.isnan(chain_features(same, d, 100.0)["CW"])                            # 콜 = 풋 IV 인 날은 무효
    assert abs(chain_features(rows, d, 100.0)["PAR"] - 100.0) < 1e-9                  # 중간가 같으면 K 의 중앙 = 100
    s = pd.DataFrame({"date": pd.to_datetime(["2026-01-05"] * 10), "x": np.arange(10.0), "y": np.arange(10.0) * 2, "z": np.arange(10.0)})
    assert abs(daily_ic(s, "x", "y").iloc[0] - 1) < 1e-12
    assert holm({"a": 0.01, "b": 0.04, "c": 0.5}) == {"a": 0.03, "b": 0.08, "c": 0.5}
    bm = block_boot(np.ones(50), np.random.default_rng(0), nb=10)
    assert np.allclose(bm, 1)
    print("selftest OK")


if __name__ == "__main__":
    {"fetch": fetch, "run": run, "selftest": selftest}[sys.argv[1]]()
