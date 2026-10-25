#!/usr/bin/env python3
"""RL v11 — v10 + 평균−분산 효용 보상 · 굴림 W2·W3. 사전 등록 docs/prereg/rl_v11_meanvar_derisk_20261025.md.

    python3 scripts/research_rl_v11_meanvar_derisk_20261025.py --train W2 SEED
    python3 scripts/research_rl_v11_meanvar_derisk_20261025.py --test
실주문 없음.
"""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import research_rl_v9_residual_sizing_20261025 as V9   # noqa: E402
import research_rl_v10_hold_derisk_20261025 as V10     # noqa: E402

OUT = V9.ROOT / "tmp" / "rl_v11_20261025"
SEEDS = [1489644453, 417192911, 737500341, 685376499, 355313916]
WIN = {"W2": (("2024-01-01", "2024-12-31"), ("2025-01-01", "2025-05-31"), ("2025-06-01", "2025-10-23")),
       "W3": (("2024-01-01", "2025-12-31"), ("2026-01-01", "2026-05-31"), ("2026-06-01", "2026-08-21"))}


def lam(tr, sd):
    y = tr.r1h_bp.to_numpy() / sd; m1, m2 = y.mean(), (y * y).mean()
    return (m1 / m2, False) if m1 > 0 else (0.01 / m2, True)          # 보유가 상수 노출 최적 · 음수 평균이면 대체(보고)


class Env(V10.Env):
    def __init__(self, d, sd, lm, seed, ep=720):
        super().__init__(d, sd, seed, ep); self.lm = lm

    def step(self, act):
        x = self.apply(V10.MS[act]) / self.sd; done = self.t >= self.end
        return (np.zeros(8, np.float32) if done else self.obs()), float(x - 0.5 * self.lm * x * x), done, False, {}


def train(w, seed):
    import torch as th
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv
    assert seed in SEEDS and w in WIN; th.set_num_threads(1); th.manual_seed(seed)
    D = V10.load(); tr, va, _ = (V9.span(D, p) for p in WIN[w]); sd = float(tr.r1h_bp.std()); lm, fb = lam(tr, sd)
    venv = DummyVecEnv([lambda i=i: Env(tr, sd, lm, seed * 10 + i) for i in range(4)])
    m = PPO("MlpPolicy", venv, learning_rate=1e-4, gamma=0.9, gae_lambda=0.95, clip_range=0.2, ent_coef=0.01, n_steps=512, batch_size=256, n_epochs=10,
            policy_kwargs=dict(net_arch=[64, 64], activation_fn=th.nn.Tanh), seed=seed, verbose=0, device="cpu")
    od = OUT / w / f"s{seed}"; od.mkdir(parents=True, exist_ok=True); res = []; hv = V9.sharpe(V10.run(va, np.ones(len(va))))
    print(w, "λ", round(lm, 5), "대체" if fb else "", flush=True)
    for step in range(0, V9.TOTAL + 1, V9.CKPT):
        if step:
            m.learn(V9.CKPT, reset_num_timesteps=False)
        m.save(od / f"{step}.zip"); ms = V10.policy_m([m], va, sd); s = V10.run(va, ms)
        res.append(dict(step=step, val_sharpe=V9.sharpe(s), m_mean=float(ms.mean()), m_sd=float(ms.std())))
        print(f"{w} s{seed} 걸음 {step:>6} 검증 샤프 {res[-1]['val_sharpe']:+.2f} (보유 {hv:+.2f}) 노출 평균 {res[-1]['m_mean']:.2f} 표준편차 {res[-1]['m_sd']:.2f}", flush=True)
        (od / "run.json").write_text(json.dumps(dict(lam=lm, lam_fallback=fb, ckpt=res), indent=1))


def test():
    from stable_baselines3 import PPO
    rp = OUT / "test_report.json"; assert not rp.exists(), "시험은 한 번만 연다"
    D = V10.load(); dR, dH, dV, rep = [], [], [], {"seeds": SEEDS, "windows": {}}
    for w, (p_tr, p_va, p_te) in WIN.items():
        tr, va, te = (V9.span(D, p) for p in (p_tr, p_va, p_te)); sd = float(tr.r1h_bp.std())
        picks = {s: max([c for c in json.loads((OUT / w / f"s{s}" / "run.json").read_text())["ckpt"] if c["step"] > 0], key=lambda z: z["val_sharpe"])["step"] for s in SEEDS}
        models = {s: PPO.load(OUT / w / f"s{s}" / f"{st}.zip", device="cpu") for s, st in picks.items()}
        tgt = float(tr.sig.median()); vt = V10.MS[np.abs(np.minimum(1, tgt / te.sig.to_numpy())[:, None] - V10.MS).argmin(1)]
        ms = V10.policy_m(list(models.values()), te, sd); R, H, VT = V10.run(te, ms), V10.run(te, np.ones(len(te))), V10.run(te, vt)
        dR.append(V9.daily(R)); dH.append(V9.daily(H)); dV.append(V9.daily(VT))
        hv, ht = V9.sharpe(V10.run(va, np.ones(len(va)))), V9.sharpe(H); seed = {}
        for s, m in models.items():
            seed[s] = dict(step=picks[s], val=V9.sharpe(V10.run(va, V10.policy_m([m], va, sd))) - hv, test=V9.sharpe(V10.run(te, V10.policy_m([m], te, sd))) - ht)
        F = te.assign(m=ms); learned = {c: F.groupby(pd.qcut(F[c], 5, labels=False, duplicates="drop")).m.mean().round(3).tolist() for c in V10.COLS}
        rep["windows"][w] = dict(picks=picks, sharpe=dict(R=V9.sharpe(R), hold=ht, VT=V9.sharpe(VT)), mean_exposure=float(ms.mean()), m_sd=float(ms.std()),
                                 seeds_pass=sum(v["val"] > 0 and v["test"] > 0 for v in seed.values()), per_seed=seed, learned_m_by_quintile=learned)
    dR, dH, dV = pd.concat(dR), pd.concat(dH), pd.concat(dV); sh = lambda x: float(x.mean() / x.std() * np.sqrt(365))
    lo, hi = V10.sharpe_diff_ci(dR.values, dH.values); mdd = lambda x: float((x.cumsum() - x.cumsum().cummax()).min())
    g1 = sh(dR) - sh(dH) > 0 and lo > 0; g2 = sh(dR) - sh(dV) > 0; g3 = all(v["seeds_pass"] >= 4 for v in rep["windows"].values())
    rep["pooled"] = dict(days=int(len(dR)), sharpe=dict(R=sh(dR), hold=sh(dH), VT=sh(dV)), R_minus_hold=sh(dR) - sh(dH), ci95=[lo, hi],
                         bp_day=dict(R=float(dR.mean()), hold=float(dH.mean()), VT=float(dV.mean())), mdd_bp=dict(R=mdd(dR), hold=mdd(dH), VT=mdd(dV)), G1=g1, G2=g2, G3=g3)
    rep["verdict"] = "후보(전진 판정 필요)" if g1 and g2 and g3 else "불통과 — RL 노출 줄이기 축 닫음"
    rp.write_text(json.dumps(rep, indent=1, ensure_ascii=False, default=float)); print(json.dumps(rep, indent=1, ensure_ascii=False, default=float))


if __name__ == "__main__":
    a = sys.argv[1:]
    test() if a[:1] == ["--test"] else train(a[1], int(a[2])) if a[:1] == ["--train"] else print(__doc__)
