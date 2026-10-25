#!/usr/bin/env python3
"""RL v10 — 롱 보유 기본 + PPO 노출 줄이기 m∈{0,.5,1} · 미분 샤프 보상. 사전 등록 docs/prereg/rl_v10_hold_derisk_20261025.md.

    python3 scripts/research_rl_v10_hold_derisk_20261025.py --train SEED
    python3 scripts/research_rl_v10_hold_derisk_20261025.py --test
입력은 v9 표(tmp/rl_v9_20261025/inputs.parquet). 실주문 없음.
"""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
import gymnasium as gym

sys.path.insert(0, str(Path(__file__).resolve().parent))
import research_rl_v9_residual_sizing_20261025 as V9   # noqa: E402

OUT = V9.ROOT / "tmp" / "rl_v10_20261025"
SEEDS = [1894794404, 1787215481, 1118651229, 770382219, 1250441416]
MS, COST, ETA = np.array([0.0, 0.5, 1.0]), 4.0, 1 / 720
COLS = ["z", "sigrel", "mupct", "r24", "mdd30"]


def load():
    D = V9.load()
    lp = D.r1h_bp.cumsum().shift(1).fillna(0)                                     # 결정 시각까지의 로그 가격(bp)
    D["mdd30"] = (lp - lp.rolling(720, min_periods=1).max()) / 100                # 30일 고점 대비 %
    return D


def run(d, m):
    r = d.r1h_bp.to_numpy(); pos = 0.0; pnl = np.empty(len(d))
    for t in range(len(d)):
        pnl[t] = m[t] * r[t] - COST * abs(m[t] - pos); pos = m[t]
    return pd.Series(pnl, index=d.index)


class Env(gym.Env):
    def __init__(self, d, sd, seed, ep=720):
        self.r, self.F, self.sd, self.ep, self.n = d.r1h_bp.to_numpy(), d[COLS].to_numpy(np.float32), sd, ep, len(d)
        self.rng = np.random.default_rng(seed)
        self.observation_space = gym.spaces.Box(-10, 10, (8,), np.float32); self.action_space = gym.spaces.Discrete(3)

    def obs(self):
        return np.clip(np.r_[self.F[self.t, :4], self.pos, (self.eq - self.peak) / (self.sd * 10), self.F[self.t, 4], self.hold / 24], -10, 10).astype(np.float32)

    def reset(self, seed=None, options=None):
        self.t = int(self.rng.integers(0, self.n - self.ep)) if self.ep else 0; self.end = self.t + self.ep if self.ep else self.n
        self.pos, self.eq, self.peak, self.hold, self.A, self.B = 1.0, 0.0, 0.0, 0, 0.0, 1.0; return self.obs(), {}

    def apply(self, m):
        pnl = m * self.r[self.t] - COST * abs(m - self.pos); self.hold = self.hold + 1 if m == self.pos else 0
        self.pos, self.eq = m, self.eq + pnl; self.peak = max(self.peak, self.eq); self.t += 1; return pnl

    def step(self, act):
        R = self.apply(MS[act]) / self.sd; dA, dB = R - self.A, R * R - self.B
        var = self.B - self.A**2; dsr = (self.B * dA - 0.5 * self.A * dB) / var**1.5 if var > 1e-6 else 0.0   # Moody·Saffell 미분 샤프
        self.A += ETA * dA; self.B += ETA * dB; done = self.t >= self.end
        return (np.zeros(8, np.float32) if done else self.obs()), float(np.clip(dsr, -10, 10)), done, False, {}


def policy_m(models, d, sd):
    env = Env(d, sd, 0, ep=0); o, _ = env.reset(); ms = np.empty(len(d))
    for t in range(len(d)):
        ms[t] = MS[[int(m.predict(o, deterministic=True)[0]) for m in models]].mean(); env.apply(ms[t])
        if env.t < len(d):
            o = env.obs()
    return ms


def train(seed):
    import torch as th
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv
    assert seed in SEEDS; th.set_num_threads(1); th.manual_seed(seed)
    D = load(); tr, va = V9.span(D, V9.TRAIN), V9.span(D, V9.VAL); sd = float(tr.r1h_bp.std())
    venv = DummyVecEnv([lambda i=i: Env(tr, sd, seed * 10 + i) for i in range(4)])
    m = PPO("MlpPolicy", venv, learning_rate=1e-4, gamma=0.9, gae_lambda=0.95, clip_range=0.2, ent_coef=0.01, n_steps=512, batch_size=256, n_epochs=10,
            policy_kwargs=dict(net_arch=[64, 64], activation_fn=th.nn.Tanh), seed=seed, verbose=0, device="cpu")
    od = OUT / f"s{seed}"; od.mkdir(parents=True, exist_ok=True); res = []; hv = V9.sharpe(run(va, np.ones(len(va))))
    for step in range(0, V9.TOTAL + 1, V9.CKPT):
        if step:
            m.learn(V9.CKPT, reset_num_timesteps=False)
        m.save(od / f"{step}.zip"); ms = policy_m([m], va, sd); s = run(va, ms)
        res.append(dict(step=step, val_sharpe=V9.sharpe(s), m_mean=float(ms.mean())))
        print(f"s{seed} 걸음 {step:>6} 검증 샤프 {res[-1]['val_sharpe']:+.2f} (보유 {hv:+.2f}) 평균 노출 {res[-1]['m_mean']:.2f}", flush=True)
        (od / "run.json").write_text(json.dumps(res, indent=1))


def sharpe_diff_ci(a, b, n=4000, blk=5, seed=0):
    a, b = np.asarray(a), np.asarray(b); rng = np.random.default_rng(seed); nb = int(np.ceil(len(a) / blk)); st = np.arange(len(a) - blk + 1); out = []
    for _ in range(n):
        i = (rng.choice(st, nb)[:, None] + np.arange(blk)).ravel()[:len(a)]; x, y = a[i], b[i]
        out.append(x.mean() / x.std() * np.sqrt(365) - y.mean() / y.std() * np.sqrt(365))
    return float(np.quantile(out, .025)), float(np.quantile(out, .975))


def test():
    from stable_baselines3 import PPO
    rp = OUT / "test_report.json"; assert not rp.exists(), "시험은 한 번만 연다"
    D = load(); tr, va, te = (V9.span(D, p) for p in (V9.TRAIN, V9.VAL, V9.TEST)); sd = float(tr.r1h_bp.std())
    picks = {s: max(json.loads((OUT / f"s{s}" / "run.json").read_text()), key=lambda z: z["val_sharpe"])["step"] for s in SEEDS}
    models = {s: PPO.load(OUT / f"s{s}" / f"{st}.zip", device="cpu") for s, st in picks.items()}
    tgt = float(tr.sig.median()); vt = lambda d: MS[np.abs(np.minimum(1, tgt / d.sig.to_numpy())[:, None] - MS).argmin(1)]
    ms = policy_m(list(models.values()), te, sd); R, H, VT = run(te, ms), run(te, np.ones(len(te))), run(te, vt(te))
    dR, dH = V9.daily(R), V9.daily(H); lo, hi = sharpe_diff_ci(dR.values, dH.values)
    hv = V9.sharpe(run(va, np.ones(len(va)))); seed = {}
    for s, m in models.items():
        seed[s] = dict(step=picks[s], val=V9.sharpe(run(va, policy_m([m], va, sd))) - hv, test=V9.sharpe(run(te, policy_m([m], te, sd))) - V9.sharpe(H))
    g2 = sum(v["val"] > 0 and v["test"] > 0 for v in seed.values()); mdd = lambda s: float((V9.daily(s).cumsum() - V9.daily(s).cumsum().cummax()).min())
    F = te.assign(m=ms); learned = {c: F.groupby(pd.qcut(F[c], 5, labels=False, duplicates="drop")).m.mean().round(3).tolist() for c in COLS}
    diff = V9.sharpe(R) - V9.sharpe(H)
    rep = dict(seeds=SEEDS, picks=picks, test=dict(days=int(len(dR)), sharpe=dict(R=V9.sharpe(R), hold=V9.sharpe(H), VT=V9.sharpe(VT)), sharpe_diff=diff, ci95=[lo, hi],
               G1=bool(diff > 0 and lo > 0), G2_count=g2, G2=g2 >= 4, bp_day=dict(R=float(dR.mean()), hold=float(dH.mean()), VT=float(V9.daily(VT).mean())),
               mdd_bp=dict(R=mdd(R), hold=mdd(H), VT=mdd(VT)), mean_exposure=dict(R=float(ms.mean()), VT=float(vt(te).mean()))),
               val_hold_sharpe=hv, per_seed=seed, learned_m_by_quintile=learned)
    rep["verdict"] = "후보(전진 판정 필요)" if rep["test"]["G1"] and rep["test"]["G2"] else "불통과 — v10 닫음"
    rp.write_text(json.dumps(rep, indent=1, ensure_ascii=False, default=float)); print(json.dumps(rep, indent=1, ensure_ascii=False, default=float))


if __name__ == "__main__":
    a = sys.argv[1:]
    test() if a[:1] == ["--test"] else train(int(a[1])) if a[:1] == ["--train"] else print(__doc__)
