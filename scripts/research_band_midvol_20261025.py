#!/usr/bin/env python3
"""청산 띠 핑퐁 — 중간 변동성에서만. 사전 등록 docs/prereg/band_pingpong_midvol_20261025.md. 한 번만 연다.

    python3 scripts/research_band_midvol_20261025.py
"""
import glob, json, sys
from pathlib import Path
import numpy as np, pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import research_rl_v12_band_gate_20261025 as V   # noqa: E402

OUT = V.ROOT / "tmp" / "band_midvol_20261025"
LO, HI = 17.913, 27.226
P, F, R = ("2022-01-02", "2023-12-31"), ("2026-08-22", "2026-09-13"), ("2026-06-01", "2026-08-21")


def main():
    rp = OUT / "report.json"; assert not rp.exists(), "한 번만 연다"
    OUT.mkdir(parents=True, exist_ok=True)
    K = pd.concat([pd.read_parquet(f, columns=["t", "h", "l", "c"]) for f in sorted(glob.glob(str(V.K1 / "ETHUSDT-1m-*.parquet")))]).drop_duplicates("t").sort_values("t").reset_index(drop=True)
    t = K.t.to_numpy(np.int64); print("1분", len(K), "틈", int((np.diff(t) != V.M1).sum()), flush=True)
    E = pd.read_parquet(V.EDGES); ei = pd.Index(E.t.to_numpy()).get_indexer(t - t % V.M5 - V.M5)
    pick = lambda col: np.where(ei >= 0, E[col].to_numpy()[np.maximum(ei, 0)], np.nan)
    ev = V.simulate(t, K.h.to_numpy(), K.l.to_numpy(), K.c.to_numpy(), pick("dn_edge"), pick("up_edge"))
    c5 = K[t % V.M5 == V.M5 - V.M1].assign(b=lambda d: d.t - (V.M5 - V.M1)).set_index("b").c
    ev["sig24"] = (np.log(c5).diff().rolling(288).std().shift(1) * 1e4).reindex(ev.bar_t).to_numpy()
    ev.to_parquet(OUT / "events_all.parquet")
    rep = {"rule": dict(lo=LO, hi=HI)}
    for name, p in (("P", P), ("F", F), ("R_reused", R)):
        e = ev[V.span(ev, p)].reset_index(drop=True); net = e.net.to_numpy(); nx = V.nxt_index(e.bar_t.to_numpy(), e.exit_t.to_numpy())
        day = pd.to_datetime(e.exit_t, unit="ms").dt.floor("D"); days = pd.date_range(p[0], p[1], freq="D")
        s = e.sig24.to_numpy()
        masks = dict(teacher=np.ones(len(e), bool), M=(s >= LO) & (s < HI), low=s < LO, high=s >= HI)

        def st(m):
            k = V.seq(m, nx); d = pd.Series(net[k], index=day.iloc[k]).groupby(level=0).sum().reindex(days, fill_value=0.0); eq = d.cumsum()
            yr = pd.Series(net[k], index=day.iloc[k].dt.year.to_numpy()).groupby(level=0).agg(["size", "mean"]).round(2)
            return dict(trades=int(len(k)), total_bp=float(net[k].sum()), mean_bp=float(net[k].mean()) if len(k) else 0.0,
                        win=float((net[k] > 0).mean()) if len(k) else 0.0, sl_share=float((e.reason.to_numpy()[k] == "sl").mean()) if len(k) else 0.0,
                        mdd_bp=float((eq - eq.cummax()).min()), by_year={int(y): r.tolist() for y, r in yr.iterrows()}), d
        out = {}
        for nm, m in masks.items():
            out[nm], d = st(m)
            if nm == "M":
                dM = d
        rate = float(masks["M"].mean()); rng = np.random.default_rng(1)
        rnd = np.array([V.total(rng.random(len(e)) < rate, net, nx) for _ in range(1000)])
        lo, hi = V.block_ci(dM.values)
        out.update(events=int(len(e)), M_rate=rate, random_same_rate=dict(p50=float(np.median(rnd)), p95=float(np.percentile(rnd, 95)),
                   pct=float((rnd < out["M"]["total_bp"]).mean())), M_daily_mean_ci95=[lo, hi])
        rep[name] = out
    m = rep["P"]
    rep["H1"] = bool(m["M"]["mean_bp"] > 0 and m["M_daily_mean_ci95"][0] > 0)
    rep["H2"] = bool(m["M"]["total_bp"] > m["random_same_rate"]["p95"])
    rep["H3"] = bool(rep["F"]["M"]["total_bp"] > 0)
    rep["verdict"] = "후보(전진 기록 필요)" if rep["H1"] and rep["H2"] and rep["H3"] else "불통과 — 중간 변동성 띠 핑퐁 닫음"
    rp.write_text(json.dumps(rep, indent=1, ensure_ascii=False, default=float)); print(json.dumps(rep, indent=1, ensure_ascii=False, default=float))


if __name__ == "__main__":
    main()
