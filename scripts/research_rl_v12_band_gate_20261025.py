#!/usr/bin/env python3
"""RL v12 — 청산 띠 교사(방향) + PPO «쉬기» 게이트 · 202 피쳐. 사전 등록 docs/prereg/rl_v12_band_teacher_gate_20261025.md.

    python3 scripts/research_rl_v12_band_gate_20261025.py --selftest
    python3 scripts/research_rl_v12_band_gate_20261025.py --build
    python3 scripts/research_rl_v12_band_gate_20261025.py --plant RHO SEED
    python3 scripts/research_rl_v12_band_gate_20261025.py --train SEED
    python3 scripts/research_rl_v12_band_gate_20261025.py --test
실주문 없음.
"""
import glob, json, sys
from pathlib import Path
import numpy as np, pandas as pd
import gymnasium as gym

ROOT = Path(__file__).resolve().parents[1]
WT = ROOT.parent if ROOT.parent.name == "worktrees" else ROOT / ".claude" / "worktrees"
EDGES = WT / "footprint-liquidation-distance-indicator-4a9790/tmp/liq_band_sweep_20261012/edges_long.parquet"
FEAT = WT / "agent-skills-maintenance-refactor-8afcd4/tmp/feature_catalog_20261022/feat202_ETH_5m_ext.npz"
K1 = Path("/home/kbj20/crypto-scalping/data/binance_vision/klines1m")
OUT = ROOT / "tmp" / "rl_v12_20261025"
SEEDS = [1620288271, 1832909907, 1912233838, 689954156, 745035243]
TRAIN, VAL, TEST = ("2024-01-01", "2025-12-31"), ("2026-01-01", "2026-05-31"), ("2026-06-01", "2026-08-21")
M1, M5, HOLD, SL = 60_000, 300_000, 1440, 0.05
MAKER, TAKER = 2.0, 5.0
TOTAL, CKPT, EP = 300_000, 30_000, 200


def simulate(t, h, l, c, dn, up):
    """1분 배열 + 분마다 행 k−1 띠 → 사건 목록(봉마다 첫 닿음, 평탄 여부와 무관하게 전부)."""
    bar = t - t % M5
    hd, hu = l <= dn, h >= up                                   # nan 비교는 False
    idx = np.flatnonzero(hd | hu)
    first = idx[np.r_[True, bar[idx][1:] != bar[idx][:-1]]] if len(idx) else idx
    rows = []
    for i in first:
        if i == 0 or i + 1 + HOLD > len(t) or (hd[i] and hu[i]):
            continue
        s = 1 if hd[i] else -1
        e = min(dn[i], c[i - 1]) if s > 0 else max(up[i], c[i - 1])
        tp = up[i] if s > 0 else dn[i]
        if not np.isfinite(tp) or s * (tp - e) <= 0:
            continue
        sl = e * (1 - s * SL); seg = slice(i + 1, i + 1 + HOLD)
        a = (l[seg] <= sl) if s > 0 else (h[seg] >= sl)
        b = (h[seg] >= tp) if s > 0 else (l[seg] <= tp)
        ja = a.argmax() if a.any() else HOLD; jb = b.argmax() if b.any() else HOLD
        if ja < HOLD and ja <= jb:
            x, why, j = sl, "sl", ja
        elif jb < HOLD:
            x, why, j = tp, "tp", jb
        else:
            x, why, j = c[i + HOLD], "time", HOLD - 1
        g = s * (x - e) / e * 1e4
        rows.append(dict(bar_t=int(bar[i]), side=s, entry=e, tp=tp, exit_t=int(t[i + 1 + j] + M1), gross=g, reason=why,
                         net=g - MAKER - (MAKER if why == "tp" else TAKER), width_bp=(up[i] - dn[i]) / c[i - 1] * 1e4, tp_bp=s * (tp - e) / e * 1e4))
    return pd.DataFrame(rows)


def selftest():
    t = np.arange(3000) * M1; c = np.full(3000, 100.0); h, l = c + .01, c - .01
    dn, up = np.full(3000, 99.0), np.full(3000, 101.0); dn[:5] = up[:5] = np.nan
    l[10] = 98.9                                                # 롱 닿음(봉 2)
    h[20] = 101.2                                               # 익절(체결 다음 분부터)
    E = simulate(t, h, l, c, dn, up)
    r = E.iloc[0]; assert r.side == 1 and r.entry == 99.0 and r.reason == "tp" and r.exit_t == 21 * M1, r
    assert abs(r.net - ((101 - 99) / 99 * 1e4 - 4)) < 1e-9
    l2 = l.copy(); l2[11] = 94.0                                # 손절 먼저
    r2 = simulate(t, h, l2, c, dn, up).iloc[0]; assert r2.reason == "sl" and abs(r2.gross + 500) < 1e-9, r2
    hh = h.copy(); hh[10] = 101.5                               # 같은 분 양쪽 → 버림
    assert simulate(t, hh, l, c, dn, up).bar_t.iloc[0] != 10 * M1 - 10 * M1 % M5
    nx = nxt_index(np.array([0, 5, 10, 15]) * M5, np.array([12, 7, 16, 99]) * M5)
    assert nx.tolist() == [3, 2, 4, 4], nx
    assert seq(np.array([1, 1, 0, 1]), nx).tolist() == [0, 3]
    print("selftest OK")


def nxt_index(bar_t, exit_t):
    return np.searchsorted(bar_t, exit_t, side="left")


def seq(mask, nx):
    out, i = [], 0
    while i < len(mask):
        if mask[i]:
            out.append(i); i = nx[i]
        else:
            i += 1
    return np.array(out, int)


def build():
    OUT.mkdir(parents=True, exist_ok=True)
    K = pd.concat([pd.read_parquet(f, columns=["t", "h", "l", "c"]) for f in sorted(glob.glob(str(K1 / "ETHUSDT-1m-*.parquet")))
                   if "2023-12" <= Path(f).stem[-7:] <= "2026-09"]).drop_duplicates("t").sort_values("t").reset_index(drop=True)
    t = K.t.to_numpy(np.int64); print("1분", len(K), "틈", int((np.diff(t) != M1).sum()), flush=True)
    E = pd.read_parquet(EDGES); ei = pd.Index(E.t.to_numpy()).get_indexer(t - t % M5 - M5)
    pick = lambda col: np.where(ei >= 0, E[col].to_numpy()[np.maximum(ei, 0)], np.nan)
    ev = simulate(t, K.h.to_numpy(), K.l.to_numpy(), K.c.to_numpy(), pick("dn_edge"), pick("up_edge"))
    c5 = K[t % M5 == M5 - M1].assign(b=lambda d: d.t - (M5 - M1)).set_index("b").c
    s24 = np.log(c5).diff().rolling(288).std().shift(1)
    ev["sig24"] = s24.reindex(ev.bar_t).to_numpy() * 1e4
    ev = ev[(ev.bar_t >= pd.Timestamp(TRAIN[0]).value // 10**6) & (ev.bar_t < pd.Timestamp(TEST[1]).value // 10**6 + 86_400_000)].reset_index(drop=True)
    z = np.load(FEAT); fi = pd.Index(z["tc"]).get_indexer(ev.bar_t.to_numpy())
    print("사건", len(ev), "피쳐 맞춤", float((fi >= 0).mean()), flush=True)
    F = np.where((fi >= 0)[:, None], z["F"][np.maximum(fi, 0)], np.nan).astype(np.float32)
    ev.to_parquet(OUT / "events.parquet"); np.save(OUT / "F202.npy", F); (OUT / "names.json").write_text(json.dumps(z["names"].tolist()))
    print(ev.groupby(pd.to_datetime(ev.bar_t, unit="ms").dt.year).net.agg(["size", "mean"]).round(2), flush=True)


def span(ev, p):
    a, b = pd.Timestamp(p[0]).value // 10**6, pd.Timestamp(p[1]).value // 10**6 + 86_400_000
    return ((ev.bar_t >= a) & (ev.bar_t < b)).to_numpy()


def load(hint_rho=None, seed=0):
    ev = pd.read_parquet(OUT / "events.parquet"); F = np.load(OUT / "F202.npy").astype(np.float64)
    tr = span(ev, TRAIN); sd = float(ev.net[tr].std())
    bt, net = ev.bar_t.to_numpy(), ev.net.to_numpy()
    past = np.searchsorted(ev.exit_t.to_numpy()[np.argsort(ev.exit_t.to_numpy())], bt, side="right")   # t_k 전에 끝난 사건 수
    cs = np.r_[0, np.cumsum(net[np.argsort(ev.exit_t.to_numpy())])]
    rec5 = np.where(past >= 5, (cs[past] - cs[np.maximum(past - 5, 0)]) / 5, 0) / sd
    cnt = (np.arange(len(bt)) - np.searchsorted(bt, bt - 86_400_000)) / 10
    smed = float(np.nanmedian(ev.sig24[tr]))
    G = np.c_[ev.side, ev.width_bp / ev.sig24, ev.tp_bp / ev.sig24, np.log(ev.sig24 / smed), rec5, cnt]
    X = np.c_[F, G]
    if hint_rho is not None:
        rng = np.random.default_rng(seed); X = np.c_[X, hint_rho * (net - net[tr].mean()) / sd + np.sqrt(1 - hint_rho**2) * rng.standard_normal(len(net))]
    med = np.nanmedian(X[tr], 0); iqr = np.nanpercentile(X[tr], 75, 0) - np.nanpercentile(X[tr], 25, 0)
    X = np.nan_to_num(np.clip((X - med) / np.where(iqr > 0, iqr, 1), -5, 5)).astype(np.float32)
    return ev, X, net / sd, nxt_index(bt, ev.exit_t.to_numpy()), sd, smed


class Env(gym.Env):
    def __init__(self, X, r, nx, seed):
        self.X, self.r, self.nx, self.n = X, r, nx, len(r); self.rng = np.random.default_rng(seed)
        self.observation_space = gym.spaces.Box(-5, 5, (X.shape[1],), np.float32); self.action_space = gym.spaces.Discrete(2)

    def reset(self, seed=None, options=None):
        self.i, self.k = int(self.rng.integers(0, self.n - EP)), 0; return self.X[self.i], {}

    def step(self, a):
        rew = float(self.r[self.i]) if a == 1 else 0.0
        self.i = int(self.nx[self.i]) if a == 1 else self.i + 1; self.k += 1
        done = self.k >= EP or self.i >= self.n
        return (np.zeros(self.X.shape[1], np.float32) if done else self.X[self.i]), rew, done, False, {}


def sub(m, nx):
    """전체 배열 기준 nx 를 부분 구간 m(연속) 안으로 옮김."""
    j = np.flatnonzero(m); return np.minimum(nx[j], j[-1] + 1) - j[0]


def total(mask, net, nx):
    return float(net[seq(mask, nx)].sum())


def train(seed, rho=None):
    import torch as th
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv
    th.set_num_threads(1); th.manual_seed(seed)
    ev, X, r, nx, sd, _ = load(rho, seed)
    tr, va = span(ev, TRAIN), span(ev, VAL); net = ev.net.to_numpy()
    Xtr, rtr, ntr = X[tr], r[tr], sub(tr, nx); Xva, nva, nxva = X[va], net[va], sub(va, nx)
    venv = DummyVecEnv([lambda i=i: Env(Xtr, rtr, ntr, seed * 10 + i) for i in range(4)])
    m = PPO("MlpPolicy", venv, learning_rate=1e-4, gamma=0.9, gae_lambda=0.95, clip_range=0.2, ent_coef=0.01, n_steps=512, batch_size=256, n_epochs=10,
            policy_kwargs=dict(net_arch=[128, 64], activation_fn=th.nn.Tanh), seed=seed, verbose=0, device="cpu")
    od = OUT / (f"plant{rho}" if rho is not None else "real") / f"s{seed}"; od.mkdir(parents=True, exist_ok=True)
    teach = total(np.ones(len(nva), bool), nva, nxva); res = []
    for step in range(0, TOTAL + 1, CKPT):
        if step:
            m.learn(CKPT, reset_num_timesteps=False)
        m.save(od / f"{step}.zip"); mk = m.predict(Xva, deterministic=True)[0].astype(bool)
        res.append(dict(step=step, val_total=total(mk, nva, nxva), take=float(mk.mean())))
        print(f"{od.parent.name} s{seed} 걸음 {step:>6} 검증 총 {res[-1]['val_total']:+.0f}bp (교사 {teach:+.0f}) 들어감 {res[-1]['take']:.2f}", flush=True)
        (od / "run.json").write_text(json.dumps(dict(teacher_val=teach, ckpt=res), indent=1))


def block_ci(d, n=4000, blk=5, seed=0):
    d = np.asarray(d); rng = np.random.default_rng(seed); nb = int(np.ceil(len(d) / blk)); st = np.arange(len(d) - blk + 1)
    m = [d[(rng.choice(st, nb)[:, None] + np.arange(blk)).ravel()[:len(d)]].mean() for _ in range(n)]
    return float(np.quantile(m, .025)), float(np.quantile(m, .975))


def test():
    from stable_baselines3 import PPO
    from sklearn.ensemble import HistGradientBoostingRegressor
    rp = OUT / "test_report.json"; assert not rp.exists(), "시험은 한 번만 연다"
    ev, X, r, nx, sd, smed = load(); net = ev.net.to_numpy()
    tr, va, te = (span(ev, p) for p in (TRAIN, VAL, TEST))
    picks = {s: max([c for c in json.loads((OUT / "real" / f"s{s}" / "run.json").read_text())["ckpt"] if c["step"] > 0], key=lambda z: z["val_total"])["step"] for s in SEEDS}
    models = {s: PPO.load(OUT / "real" / f"s{s}" / f"{st}.zip", device="cpu") for s, st in picks.items()}
    P = lambda Z: {s: m.predict(Z, deterministic=True)[0].astype(bool) for s, m in models.items()}
    Pva, Pte = P(X[va]), P(X[te]); nva, nte, nxva, nxte = net[va], net[te], sub(va, nx), sub(te, nx)
    rl = np.sum(list(Pte.values()), 0) >= 3
    hgb = HistGradientBoostingRegressor(max_iter=300, learning_rate=.05, early_stopping=False, random_state=0).fit(X[tr], r[tr])
    pv, pt = hgb.predict(X[va]), hgb.predict(X[te])
    q = max(range(0, 100, 10), key=lambda q: total(pv > np.percentile(pv, q) - (q == 0), nva, nxva))
    E = ev[te].reset_index(drop=True); day = pd.to_datetime(E.exit_t, unit="ms").dt.floor("D")
    days = pd.date_range(TEST[0], TEST[1], freq="D")

    def daily(mask):
        k = seq(mask, nxte); return pd.Series(nte[k], index=day.iloc[k]).groupby(level=0).sum().reindex(days, fill_value=0.0)

    def stats(mask):
        k = seq(mask, nxte); d = daily(mask); eq = d.cumsum()
        return dict(trades=int(len(k)), total_bp=float(nte[k].sum()), mean_bp=float(nte[k].mean()) if len(k) else 0.0,
                    win=float((nte[k] > 0).mean()) if len(k) else 0.0, mdd_bp=float((eq - eq.cummax()).min()))
    teach = np.ones(len(nte), bool)
    vis = np.zeros(len(rl), bool); i = 0                        # RL 이 실제로 마주친 사건
    while i < len(rl):
        vis[i] = True; i = nxte[i] if rl[i] else i + 1
    took_rate = float(rl[vis].mean())
    rng = np.random.default_rng(1); rnd = np.array([total(rng.random(len(nte)) < took_rate, nte, nxte) for _ in range(1000)])
    dR, dT = daily(rl), daily(teach); lo, hi = block_ci((dR - dT).values)
    per_seed = {s: dict(step=picks[s], val=total(Pva[s], nva, nxva) - total(np.ones(len(nva), bool), nva, nxva),
                        test=total(Pte[s], nte, nxte) - total(teach, nte, nxte)) for s in SEEDS}
    g1 = (dR - dT).mean() > 0 and lo > 0; g2 = stats(rl)["total_bp"] > np.percentile(rnd, 95); g3 = sum(v["val"] > 0 and v["test"] > 0 for v in per_seed.values()) >= 4
    Et = E.assign(take=rl)
    rep = dict(seeds=SEEDS, picks=picks, test_days=len(days), events_test=int(len(nte)), sd_train=sd,
               RL=stats(rl), teacher=stats(teach), lowvol=stats(E.sig24.to_numpy() <= smed), hgb=dict(q=q, **stats(pt > np.percentile(pv, q) - (q == 0))),
               random_same_rate=dict(rate=took_rate, p50=float(np.median(rnd)), p95=float(np.percentile(rnd, 95))),
               RL_minus_teacher_bp_day=float((dR - dT).mean()), ci95=[lo, hi], G1=bool(g1), G2=bool(g2), G3=bool(g3), per_seed=per_seed,
               take_by=dict(side=Et.groupby("side")["take"].mean().round(3).to_dict(),
                            sig_q=Et.groupby(pd.qcut(Et.sig24, 5, labels=False))["take"].mean().round(3).tolist(),
                            width_q=Et.groupby(pd.qcut(Et.width_bp, 5, labels=False))["take"].mean().round(3).tolist()))
    rep["G4_RL_total_positive"] = rep["RL"]["total_bp"] > 0
    rep["verdict"] = "후보(전진 판정 필요)" if g1 and g2 and g3 else "불통과"
    rp.write_text(json.dumps(rep, indent=1, ensure_ascii=False, default=float)); print(json.dumps(rep, indent=1, ensure_ascii=False, default=float))


if __name__ == "__main__":
    a = sys.argv[1:]
    if a[:1] == ["--selftest"]: selftest()
    elif a[:1] == ["--build"]: build()
    elif a[:1] == ["--plant"]: assert int(a[2]) in SEEDS; train(int(a[2]), float(a[1]))
    elif a[:1] == ["--train"]: assert int(a[1]) in SEEDS; train(int(a[1]))
    elif a[:1] == ["--test"]: test()
    else: print(__doc__)
