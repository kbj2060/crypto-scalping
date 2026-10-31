"""XRP 실현 감마 국면 -- 사전등록 docs/prereg/xrp_realized_gamma_regime_20261031.md (88e61c64).
ETH 연구(research_eth_realized_gamma_regime_20261002)의 panel·boot_ols·pct_of 를 그대로 쓰고 데이터만 XRP.
  python scripts/research_xrp_realized_gamma_regime_20261031.py [--eth-only]
"""
import json, sys, zipfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import research_eth_realized_gamma_regime_20261002 as G  # noqa: E402

BV = Path("/home/kbj20/crypto-scalping/data/binance_vision")
OUT = ROOT / "tmp/xrp_realized_gamma_20261031"
M5 = G.M5


def load_xrp_5m() -> pd.Series:
    parts = []
    for f in sorted((BV / "klines1m_futures_pre2022").glob("XRPUSDT-1m-2021-*.zip")):
        z = zipfile.ZipFile(f)
        d = pd.read_csv(z.open(z.namelist()[0]), header=None, usecols=[0, 4], names=["t", "c"])
        parts.append(d.apply(pd.to_numeric, errors="coerce").dropna())
    for f in sorted((BV / "klines1m").glob("XRPUSDT-1m-20*.parquet")):
        d = pd.read_parquet(f, columns=["t", "c"])
        parts.append(d.rename(columns={"c": "c"}))
    m1 = pd.concat(parts).drop_duplicates("t").astype({"t": "int64"})
    g = m1.groupby(m1["t"] // M5 * M5)["c"]
    s = g.last()[g.size() == 5]
    s.index = s.index.astype("int64") + M5
    return s.reindex(np.arange(G.D0 + M5, G.D1 + M5, M5, dtype="int64"))


def run(logp: pd.Series, name: str) -> dict:
    d = G.panel(logp, 12)
    per = G.periods(d)
    disc, hold = d[per["discovery"]], d[per["holdout"]]
    ref = disc["rg1_12h"].to_numpy()
    q = {p: float(np.nanpercentile(ref, p)) for p in (1, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 99)}
    t1, t2 = float(np.nanquantile(ref, 1 / 3)), float(np.nanquantile(ref, 2 / 3))
    # 주 판정: H3 pct 계수(보류)
    y = np.log(hold["rv_next4h"].to_numpy())
    L = np.log(hold[["rv1h", "rv24h", "rv7d"]].to_numpy())
    p = G.pct_of(hold["rg1_12h"].to_numpy(), ref)
    full, boot, nd = G.boot_ols(np.c_[np.ones(len(hold)), L, p], y, hold["day"].to_numpy())
    rng = np.random.default_rng(20261031)
    ps = p.copy(); ok = np.isfinite(ps); ps[ok] = rng.permutation(ps[ok])
    fs, bs, _ = G.boot_ols(np.c_[np.ones(len(hold)), L, ps], y, hold["day"].to_numpy())
    # 화면 배수: HAR 발견 구간 OLS → 보류 적용, RG 삼분위별 두 정의
    yd = np.log(disc["rv_next4h"].to_numpy()); Ld = np.log(disc[["rv1h", "rv24h", "rv7d"]].to_numpy())
    md = np.isfinite(yd) & np.isfinite(Ld).all(1)
    beta = np.linalg.lstsq(np.c_[np.ones(md.sum()), Ld[md]], yd[md], rcond=None)[0]
    pred = np.c_[np.ones(len(hold)), L] @ beta
    rg = hold["rg1_12h"].to_numpy()
    mh = np.isfinite(y) & np.isfinite(pred) & np.isfinite(rg)
    terc = {}
    for nm, sel in (("rev", rg <= t1), ("mid", (rg > t1) & (rg < t2)), ("trend", rg >= t2)):
        s = mh & sel
        terc[nm] = {"n": int(s.sum()), "a_geo": float(np.exp(np.mean(y[s] - pred[s]))),
                    "b_sum": float(np.exp(y[s]).sum() / np.exp(pred[s]).sum())}
    return {"name": name, "ref_pct": q, "tercile_cut": [t1, t2], "n_disc": int(np.isfinite(ref).sum()),
            "h3_pct_coef": {"est": float(full[4]), "lo90_one_sided": float(np.percentile(boot[:, 4], 10)),
                            "ci95": [float(np.percentile(boot[:, 4], 2.5)), float(np.percentile(boot[:, 4], 97.5))], "n_days": nd},
            "placebo_shuffle": {"est": float(fs[4]), "ci95": [float(np.percentile(bs[:, 4], 2.5)), float(np.percentile(bs[:, 4], 97.5))]},
            "tercile_ratio": terc}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rep = {"eth": run(np.log(G.load_5m()), "ETH")}
    print(json.dumps(rep["eth"], ensure_ascii=False, indent=1))
    if "--eth-only" in sys.argv:
        (OUT / "report_eth.json").write_text(json.dumps(rep, ensure_ascii=False, indent=1))
        return
    rep["xrp"] = run(np.log(load_xrp_5m()), "XRP")
    x = rep["xrp"]["h3_pct_coef"]
    rep["verdict_xrp"] = {"strict": x["est"] > 0 and x["lo90_one_sided"] > 0, "candidate": x["est"] > 0}
    print(json.dumps({k: rep[k] for k in ("xrp", "verdict_xrp")}, ensure_ascii=False, indent=1))
    (OUT / "report.json").write_text(json.dumps(rep, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
