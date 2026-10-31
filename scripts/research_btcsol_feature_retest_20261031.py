"""ETH(·XRP) 전용 기능 3개를 BTC·SOL 로 재검정 -- 사전등록 docs/prereg/btcsol_feature_retest_20261031.md (bff662b4).
정의는 새로 만들지 않고 기존 연구 함수를 import 한다(데이터만 바꿈).
  python -I scripts/research_btcsol_feature_retest_20261031.py duel|openrev|gamma|selftest
"""
import glob, json, os, sys, zipfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
BV = Path("/home/kbj20/crypto-scalping/data/binance_vision")
OUT = Path("/home/kbj20/crypto-scalping/tmp/btcsol_retest_20261031")
SALV = "/home/kbj20/crypto-scalping/tmp/worktree_salvage_20260930/scalping-entry-analysis-c0c47c/files/tmp/whale/tape"
M5 = 300_000


def m1(sym, with_2021=True, cols=("t", "c")):
    """vision 1분봉(2021 월 zip + 2022~ 월 parquet + 2026-10 일 parquet) → t 정렬·중복 제거."""
    parts = []
    if with_2021:
        for f in sorted((BV / "klines1m_futures_pre2022").glob(f"{sym}-1m-2021-*.zip")):
            z = zipfile.ZipFile(f)
            d = pd.read_csv(z.open(z.namelist()[0]), header=None, usecols=[0, 4, 5, 9], names=["t", "c", "v", "tb"])
            parts.append(d.apply(pd.to_numeric, errors="coerce").dropna()[list(cols)])
    for sub in ("klines1m", "klines1m_daily"):
        parts += [pd.read_parquet(f, columns=list(cols)) for f in sorted((BV / sub).glob(f"{sym}-1m-20*.parquet"))]
    return pd.concat(parts).drop_duplicates("t").astype({"t": "int64"}).sort_values("t")


def to5(k):
    """완결 5분 종가(1분 5개 다 있는 칸만). index = 봉 시작 ms."""
    g = k.groupby(k["t"] // M5 * M5)["c"]
    return g.last()[g.size() == 5]


# ── §1 고래 맞대결 BTC ──────────────────────────────────────────────────────────────────────
def duel():
    import research_whale_duel_solxrp_20261004 as W
    E = W.load(glob.glob(f"{W.ETH_TAPE}/*.parquet"))
    eth = W.report(E, "row", full=False)
    assert abs(eth["all"]["mean"] - 6.17) < 0.01, f"양성대조 실패 ETH {eth['all']['mean']}"
    del E
    T = W.load(glob.glob(f"{SALV}/BTCUSDT/*.parquet"))
    a = W.report(T, "row"); o = W.report(T, "ord", full=False)
    D = W.duel(T, "row"); ts, v, dis = D["ts"], D["v"], D["dis"]
    seg = {nm: W.bci(v[dis & m], D["day"][dis & m]) for nm, m in (
        ("2024", ts.year == 2024), ("2025", ts.year == 2025),
        ("2026-01~05", (ts >= pd.Timestamp("2026-01-01", tz="UTC")) & (ts < pd.Timestamp("2026-06-01", tz="UTC"))),
        ("2026-06~", ts >= pd.Timestamp("2026-06-01", tz="UTC")))}
    y, ci = a["years"], a["main"]["ci"]
    C = 6.2
    verdict = {"strict": bool(ci[0] - C > 0 and y.get(2025, -99) - C > 0 and y.get(2026, -99) - C > 0),
               "candidate": bool(ci[0] > 0 and y.get(2025, -1) > 0 and y.get(2026, -1) > 0),
               "net_maker6.2": {"mean": round(a["main"]["mean"] - C, 2), "ci": [round(ci[0] - C, 2), round(ci[1] - C, 2)],
                                "2025": round(y.get(2025, np.nan) - C, 2), "2026": round(y.get(2026, np.nan) - C, 2)}}
    # 무결성: 테이프 5분 종가·총 순매수 vs 로컬 1분봉
    k = m1("BTCUSDT", False, ("t", "c", "v", "tb")); k = k[(k.t >= T.index[0].value // 10**6) & (k.t <= T.index[-1].value // 10**6)]
    kb = k.t // M5 * M5
    kk = pd.DataFrame({"c": k.groupby(kb).c.last(), "net": ((2 * k.tb - k.v) * k.c).groupby(kb).sum()})
    tb5 = (T.index.asi8 // 10**6) // M5 * M5
    tt = pd.DataFrame({"c": T.c.groupby(tb5).last(), "net": sum(T[f"row_{g}_sn"] for g in W.GRP).groupby(tb5).sum()})
    j = tt.join(kk, rsuffix="_k", how="inner").dropna()
    sh = {g: T[f"row_{g}_nt"] for g in W.GRP}; tot = sum(sh.values())
    integ = {"bars5": int(len(j)), "close_match": round(float((np.abs(j.c / j.c_k - 1) < 1e-9).mean()), 5),
             "net_corr": round(float(np.corrcoef(j.net, j.net_k)[0, 1]), 5), "no_trade_minutes": int(T.c.isna().sum()),
             "whale_share_by_year": {int(yr): round(float(sh["whl"][tot.index.year == yr].sum() / tot[tot.index.year == yr].sum()), 3)
                                     for yr in sorted(set(tot.index.year))}}
    res = {"eth_control": eth, "btc": a, "btc_ord": o, "btc_split": seg, "verdict": verdict, "integrity": integ,
           "bands_usd": {"retail_lt": 10_000, "whale_ge": 100_000}}
    _save("duel", res)


# ── §2 09시 개장 되돌림 BTC·SOL ──────────────────────────────────────────────────────────────
def openrev():
    import research_xrp_parity_20261020 as XP
    files = {}
    for sym in ("BTCUSDT", "SOLUSDT", "ETHUSDT"):
        p = OUT / f"{sym}_5m_open.parquet"
        if not p.exists():
            s = to5(m1(sym, False))
            OUT.mkdir(parents=True, exist_ok=True)
            pd.DataFrame({"timestamp": pd.to_datetime(s.index, unit="ms"), "close": s.to_numpy()}).to_parquet(p)
        files[sym] = p
    XP.OUT = OUT                                                                  # 선례 report 를 덮어쓰지 않게
    src = {"XRPUSDT": Path("/home/kbj20/crypto-scalping/tmp/xrp_parity_20261020/vol4h/XRPUSDT_5m.parquet"),
           "ETHUSDT": Path.home() / "backups/crypto-scalping-server/data/live/vol_panel/ETHUSDT_5m.parquet",   # 선례 순서(난수 재현)
           "BTCUSDT": files["BTCUSDT"], "SOLUSDT": files["SOLUSDT"], "ETHUSDT_vision1m": files["ETHUSDT"]}
    XP.openrev(src)
    rep = json.loads((OUT / "open_rev/report.json").read_text())
    for sym in rep:
        if not isinstance(rep[sym], dict) or "summary" not in rep[sym]:
            continue
        s = rep[sym]["summary"]; pb = rep[sym]["random_window_placebo"]["2025~"]
        rep[sym]["candidate_0bp"] = {"net0_2025~_ci_low>0": s["2025~"]["net0"]["ci95"][0] > 0, "net0_2025>0": s["2025"]["net0"]["mean_bp"] > 0,
                                     "net0_2026>0": s["2026"]["net0"]["mean_bp"] > 0, "gross_above_placebo_q95": pb["pctile"] > 95}
        rep[sym]["candidate_0bp"]["PASS"] = all(rep[sym]["candidate_0bp"].values())
    # 보조: 공통 분할 2026-01~05 / 2026-06~ (규칙 고정, 고르기 없음)
    for sym in ("BTCUSDT", "SOLUSDT"):
        D = XP.daily_table(XP.grid5(pd.read_parquet(src[sym])), 0)
        D = D[D.pos != 0]; dt = pd.to_datetime(D.day, unit="ms")
        rep[sym]["split_2026"] = {nm: {"n": int(m.sum()), "gross_bp": round(float((D.pos * D.fwd)[m].mean()), 2)}
                                  for nm, m in (("2026-01~05", (dt >= "2026-01-01") & (dt < "2026-06-01")), ("2026-06~", dt >= "2026-06-01"))}
    _save("openrev", rep)


# ── §3 실현 감마 BTC·SOL ────────────────────────────────────────────────────────────────────
def load_sym_5m(sym):
    """XRP 판 load_xrp_5m 과 같은 규칙: index = 봉 끝 ms, ETH 연구 격자로 reindex."""
    import research_eth_realized_gamma_regime_20261002 as G
    s = to5(m1(sym))
    s.index = s.index.astype("int64") + M5
    return s.reindex(np.arange(G.D0 + M5, G.D1 + M5, M5, dtype="int64"))


def gamma():
    import research_eth_realized_gamma_regime_20261002 as G
    import research_xrp_realized_gamma_regime_20261031 as X
    # 원 연구 ETH 1분 parquet 은 다른 워크트리 tmp 에만 있다(사본 md5 동일)
    G.ROOT = next(Path(p).parents[2] for p in sorted(glob.glob("/home/kbj20/crypto-scalping/.claude/worktrees/*/tmp/opt_validate_20261002/ethusdt_1m.parquet")))
    rep = {"eth": X.run(np.log(G.load_5m()), "ETH")}
    xrp = X.run(np.log(load_sym_5m("XRPUSDT")), "XRP")
    rep["xrp_control"] = {k: xrp[k] for k in ("h3_pct_coef", "tercile_cut", "tercile_ratio", "n_disc")}
    for sym in ("BTCUSDT", "SOLUSDT"):
        r = X.run(np.log(load_sym_5m(sym)), sym[:3]); h = r["h3_pct_coef"]
        r["verdict"] = {"strict": h["est"] > 0 and h["lo90_one_sided"] > 0, "candidate": h["est"] > 0}
        rep[sym[:3].lower()] = r
    _save("gamma", rep)


def _save(name, rep):
    OUT.mkdir(parents=True, exist_ok=True)
    s = json.dumps(rep, ensure_ascii=False, indent=1, default=str)
    (OUT / f"report_{name}.json").write_text(s); print(s[:6000], flush=True)


def selftest():
    t0 = 1_700_000_000_000 // M5 * M5
    k = pd.DataFrame({"t": t0 + 60_000 * np.r_[0:5, 5:9, 10:15], "c": np.arange(14.0)})   # 둘째 칸 1분 빠짐
    s = to5(k)
    assert list(s.index) == [t0, t0 + 2 * M5] and list(s) == [4.0, 13.0], s
    print("selftest ok")


if __name__ == "__main__":
    os.environ.setdefault("OMP_NUM_THREADS", "3")
    {"duel": duel, "openrev": openrev, "gamma": gamma, "selftest": selftest}[sys.argv[1]]()
