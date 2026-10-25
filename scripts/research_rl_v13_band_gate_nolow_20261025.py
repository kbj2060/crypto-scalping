#!/usr/bin/env python3
"""RL v13 — 저변동 뺀 띠 교사 + PPO 쉬기 · 164 피쳐 · 상대 기준. 사전 등록 docs/prereg/rl_v13_band_gate_nolow_20261025.md.

    python3 scripts/research_rl_v13_band_gate_nolow_20261025.py --build
    python3 scripts/research_rl_v13_band_gate_nolow_20261025.py --train SEED
    python3 scripts/research_rl_v13_band_gate_nolow_20261025.py --test
v12 의 사건·환경·PPO 를 그대로 쓴다(train 은 V.load/V.OUT 을 바꿔 끼워 호출). 실주문 없음.
"""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import research_rl_v12_band_gate_20261025 as V   # noqa: E402

OUT = V.ROOT / "tmp" / "rl_v13_20261025"
ALL = V.ROOT / "tmp" / "band_midvol_20261025" / "events_all.parquet"
SEEDS = [67331423, 1958407701, 1263307844, 1782551387, 662055058]
LO = 17.913
P, F, R = ("2022-01-02", "2023-12-31"), ("2026-08-22", "2026-09-13"), ("2026-06-01", "2026-08-21")


def build():
    OUT.mkdir(parents=True, exist_ok=True)
    ev = pd.read_parquet(ALL).sort_values("bar_t").reset_index(drop=True)
    bt, net, xt = ev.bar_t.to_numpy(), ev.net.to_numpy(), ev.exit_t.to_numpy()
    o = np.argsort(xt); past = np.searchsorted(xt[o], bt, side="right"); cs = np.r_[0, np.cumsum(net[o])]
    ev["rec5_raw"] = np.where(past >= 5, (cs[past] - cs[np.maximum(past - 5, 0)]) / 5, 0)        # 전체 교사(저변동 포함) 기준
    ev["cnt24"] = (np.arange(len(bt)) - np.searchsorted(bt, bt - 86_400_000)) / 10
    z = np.load(V.FEAT); tc = pd.to_datetime(z["tc"], unit="ms")
    cov = lambda a, b: np.isfinite(z["F"][(tc >= a) & (tc < b)]).mean(0)
    keep = (cov("2022-01-02", "2024-01-01") >= .95) & (cov("2024-01-01", "2026-01-01") >= .95)
    ev = ev[ev.sig24 >= LO].reset_index(drop=True)
    fi = pd.Index(z["tc"]).get_indexer(ev.bar_t.to_numpy())
    Fm = np.where((fi >= 0)[:, None], z["F"][np.maximum(fi, 0)][:, keep], np.nan).astype(np.float32)
    print("사건(저변동 뺌)", len(ev), "열", int(keep.sum()), "피쳐 맞춤", float((fi >= 0).mean()), flush=True)
    for nm, p in (("P", P), ("F", F), ("R", R), ("train", V.TRAIN)):
        m = V.span(ev, p); print(nm, int(m.sum()), "결측률", round(float(np.isnan(Fm[m]).mean()), 3), flush=True)
    ev.to_parquet(OUT / "events.parquet"); np.save(OUT / "F202.npy", Fm); (OUT / "names.json").write_text(json.dumps(z["names"][keep].tolist()))


def load(hint_rho=None, seed=0):
    ev = pd.read_parquet(OUT / "events.parquet"); Fm = np.load(OUT / "F202.npy").astype(np.float64)
    tr = V.span(ev, V.TRAIN); sd = float(ev.net[tr].std()); smed = float(np.nanmedian(ev.sig24[tr]))
    G = np.c_[ev.side, ev.width_bp / ev.sig24, ev.tp_bp / ev.sig24, np.log(ev.sig24 / smed), ev.rec5_raw / sd, ev.cnt24]
    X = np.c_[Fm, G]
    med = np.nanmedian(X[tr], 0); iqr = np.nanpercentile(X[tr], 75, 0) - np.nanpercentile(X[tr], 25, 0)
    X = np.nan_to_num(np.clip((X - med) / np.where(iqr > 0, iqr, 1), -5, 5)).astype(np.float32)
    return ev, X, ev.net.to_numpy() / sd, V.nxt_index(ev.bar_t.to_numpy(), ev.exit_t.to_numpy()), sd, smed


def test():
    from stable_baselines3 import PPO
    rp = OUT / "test_report.json"; assert not rp.exists(), "판정은 한 번만 연다"
    ev, X, r, nx, sd, _ = load(); net = ev.net.to_numpy(); va = V.span(ev, V.VAL)
    picks = {s: max([c for c in json.loads((OUT / "real" / f"s{s}" / "run.json").read_text())["ckpt"] if c["step"] > 0], key=lambda z: z["val_total"])["step"] for s in SEEDS}
    models = {s: PPO.load(OUT / "real" / f"s{s}" / f"{st}.zip", device="cpu") for s, st in picks.items()}
    rep = dict(seeds=SEEDS, picks=picks)
    for name, p in (("P", P), ("F", F), ("R_reused", R), ("VAL", V.VAL)):
        m = V.span(ev, p); E = ev[m].reset_index(drop=True); n, nxs = net[m], V.sub(m, nx)
        votes = {s: mo.predict(X[m], deterministic=True)[0].astype(bool) for s, mo in models.items()}
        rl = np.sum(list(votes.values()), 0) >= 3; teach = np.ones(len(n), bool)
        day = pd.to_datetime(E.exit_t, unit="ms").dt.floor("D"); days = pd.date_range(p[0], p[1], freq="D")
        daily = lambda mk: (lambda k: pd.Series(n[k], index=day.iloc[k]).groupby(level=0).sum().reindex(days, fill_value=0.0))(V.seq(mk, nxs))

        def st(mk):
            k = V.seq(mk, nxs); eq = daily(mk).cumsum()
            return dict(trades=int(len(k)), total_bp=float(n[k].sum()), mean_bp=float(n[k].mean()) if len(k) else 0.0,
                        win=float((n[k] > 0).mean()) if len(k) else 0.0, mdd_bp=float((eq - eq.cummax()).min()))
        vis = np.zeros(len(rl), bool); i = 0
        while i < len(rl):
            vis[i] = True; i = nxs[i] if rl[i] else i + 1
        rate = float(rl[vis].mean()); rng = np.random.default_rng(1)
        rnd = np.array([V.total(rng.random(len(n)) < rate, n, nxs) for _ in range(1000)])
        d = daily(rl) - daily(teach); lo, hi = V.block_ci(d.values)
        t = E.bar_t.to_numpy(); gap = lambda mk: n[mk].mean() - n[~mk].mean() if mk.any() and (~mk).any() else 0.0
        shift = [gap(rl[np.argsort(np.argsort((t - t[0] + k * 86_400_000) % (t[-1] - t[0] + 1)))]) for k in range(1, 81)]
        rep[name] = dict(events=int(len(n)), RL=st(rl), teacher=st(teach), take_rate=rate,
                         random_same_rate=dict(p50=float(np.median(rnd)), p95=float(np.percentile(rnd, 95)), pct=float((rnd < st(rl)["total_bp"]).mean())),
                         RL_minus_teacher_bp_day=float(d.mean()), ci95=[lo, hi],
                         all_events=dict(taken_mean=float(n[rl].mean()) if rl.any() else None, skipped_mean=float(n[~rl].mean()) if (~rl).any() else None,
                                         gap=float(gap(rl)), circular_shift_pct=float((np.array(shift) < gap(rl)).mean())),
                         per_seed={s: V.total(v, n, nxs) - V.total(teach, n, nxs) for s, v in votes.items()})
    Pq = rep["P"]
    rep["G1"] = bool(Pq["RL"]["total_bp"] > Pq["random_same_rate"]["p95"])
    rep["G2"] = bool(Pq["RL_minus_teacher_bp_day"] > 0)
    rep["G3"] = bool(sum(rep["VAL"]["per_seed"][s] > 0 and Pq["per_seed"][s] > 0 for s in SEEDS) >= 4)
    rep["verdict"] = "후보(상대 기준)" if rep["G1"] and rep["G2"] and rep["G3"] else "불통과"
    rp.write_text(json.dumps(rep, indent=1, ensure_ascii=False, default=float)); print(json.dumps(rep, indent=1, ensure_ascii=False, default=float))


if __name__ == "__main__":
    a = sys.argv[1:]
    if a[:1] == ["--build"]: build()
    elif a[:1] == ["--train"]:
        assert int(a[1]) in SEEDS; V.OUT, V.load = OUT, load; V.train(int(a[1]))
    elif a[:1] == ["--test"]: test()
    else: print(__doc__)
