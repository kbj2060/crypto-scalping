#!/usr/bin/env python3
"""RL v9 — 방향 HGB ÷ QX4 분산 닫힌 꼴 크기 정책(B) 위 잔차 PPO(R). 사전 등록 docs/prereg/rl_v9_residual_sizing_20261025.md.

    python3 scripts/research_rl_v9_residual_sizing_20261025.py --build      # μ̂·σ̂·r 1시간 표
    python3 scripts/research_rl_v9_residual_sizing_20261025.py --base       # B 격자 → 학습 구간 샤프 최대 고정
    python3 scripts/research_rl_v9_residual_sizing_20261025.py --train SEED # 사전 등록 시드만
    python3 scripts/research_rl_v9_residual_sizing_20261025.py --test       # 검증 체크포인트 선택 → 시험 1회
실주문 없음.
"""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
import gymnasium as gym

sys.path.insert(0, str(Path(__file__).resolve().parent))
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "tmp" / "rl_v9_20261025"
DIR = Path("/home/kbj20/crypto-scalping/.claude/worktrees/liquidation-price-feature-14eceb/tmp/best_dir_20261010")
TRAIN, VAL, TEST = ("2024-01-01", "2025-12-31"), ("2026-01-01", "2026-05-31"), ("2026-06-01", "2026-08-21")
SEEDS = [int(s) for s in np.random.default_rng(20261025).integers(1, 2**31 - 1, 5)]
COST, L, MS = 4.0, 2.0, np.array([0.0, 0.5, 1.0, 1.5, 2.0])
TOTAL, CKPT = 500_000, 50_000


def build():
    import live_vol4h_qx4_20261025 as Q
    P = pd.read_parquet(DIR / "panel.parquet", columns=["ts"]); ts = P.ts.to_numpy()
    ph = np.load(DIR / "pred_holdout_2023_24_4h_seed3.npy"); pf = np.load(DIR / "pred_FINAL_eth_4h.npy")[:, 0]
    mu = np.where(ts < np.datetime64("2025-01-01"), np.r_[ph, np.full(len(ts) - len(ph), np.nan)], pf)
    fr, oks = Q.load_frames()
    art = Q.fit(fr, Q.masks(oks, fr, pd.Timestamp("2022-01-08"), pd.Timestamp("2023-12-31 23:55"), pd.Timestamp("2024-01-01"), pd.Timestamp("2024-05-31 23:55")))
    F, _ = fr["ETH"]; ets, ok = oks["ETH"]; p = np.full(len(F), np.nan); p[ok] = Q.predict_log(art, "ETH", F[ok])
    sig = pd.Series(np.exp(p / 2), index=ets)
    c = Q.R.load("ETH")[0].c
    r1h = pd.Series(np.log(c.shift(-12).to_numpy() / c.to_numpy()) * 1e4, index=c.index)       # 결정 봉 마감 → 12봉 뒤 마감
    D = pd.DataFrame({"mu_bp": pd.Series(mu, index=pd.DatetimeIndex(ts))}).join(sig.rename("sig"), how="inner").join(r1h.rename("r1h_bp"), how="inner")
    D = D[(D.index.minute == 55) & (D.index >= "2024-01-01") & (D.index <= TEST[1] + " 23:55")].dropna()
    assert D.index.is_monotonic_increasing
    OUT.mkdir(parents=True, exist_ok=True); D.to_parquet(OUT / "inputs.parquet")
    for nm, (a, b) in {"train": TRAIN, "val": VAL, "test": TEST}.items():
        d = D.loc[a:b + " 23:55"]; print(nm, len(d), "행 · μ̂ ~ 다음 1h 순위상관", round(d.mu_bp.corr(d.r1h_bp, method="spearman"), 4),
                                         "· σ̂ ~ 다음 1h |r|", round(d.sig.corr(d.r1h_bp.abs(), method="spearman"), 3), "· σ̂ 중앙", round(d.sig.median() * 100, 3), "%")


def load():
    D = pd.read_parquet(OUT / "inputs.parquet")
    D["aim_raw"] = (D.mu_bp / 1e4) / D.sig**2                                      # 분수 켈리 μ/σ²(4h 단위), k 곱하기 전
    D["z"] = D.mu_bp / (D.sig * 1e4)
    D["sigrel"] = np.log(D.sig / D.sig.rolling(720, min_periods=24).median())
    D["mupct"] = D.mu_bp.expanding(24).rank(pct=True)
    D["r24"] = D.r1h_bp.shift(1).rolling(24, min_periods=1).sum() / (D.sig * 1e4 * np.sqrt(6))   # 결정 전까지 지난 24h
    return D.fillna(0)


def span(D, p):
    return D.loc[p[0]:p[1] + " 23:55"]


def run(d, k, a, b, m=None):
    """결정적 실행(m = 시각별 배수, 없으면 1 = B). 반환 = 시각별 순손익 bp, pos."""
    aim = np.clip(k * d.aim_raw.to_numpy(), -L, L); r = d.r1h_bp.to_numpy(); mm = np.ones(len(d)) if m is None else m
    pos = 0.0; pnl = np.empty(len(d)); ps = np.empty(len(d))
    for t in range(len(d)):
        tgt = np.clip(mm[t] * aim[t], -L, L); new = pos if abs(tgt - pos) < b else pos + a * (tgt - pos)
        pnl[t] = new * r[t] - COST * abs(new - pos); pos = new; ps[t] = pos
    return pd.Series(pnl, index=d.index), ps


def daily(s):
    return s.groupby(s.index.floor("D")).sum()


def sharpe(s):
    dd = daily(s); return float(dd.mean() / dd.std() * np.sqrt(365)) if dd.std() > 0 else 0.0


def base():
    D = load(); tr = span(D, TRAIN); rows = []
    for k in (.1, .25, .5, 1):
        for a in (1, .5, .25):
            for b in (0, .25):
                s, ps = run(tr, k, a, b)
                rows.append(dict(k=k, a=a, b=b, sharpe=sharpe(s), bp_day=float(daily(s).mean()), turnover_day=float(np.abs(np.diff(ps)).sum() / (len(ps) / 24)), mean_abs_pos=float(np.abs(ps).mean())))
    G = pd.DataFrame(rows).sort_values("sharpe", ascending=False); print(G.round(3).to_string(index=False))
    best = G.iloc[0].to_dict(); (OUT / "base.json").write_text(json.dumps({"best": best, "grid": rows}, indent=1)); print("B =", best)


def base_params():
    b = json.loads((OUT / "base.json").read_text())["best"]; return b["k"], b["a"], b["b"]


class Env(gym.Env):
    def __init__(self, d, k, a, b, sd, seed, ep=720):
        self.a, self.b, self.sd, self.ep, self.n = a, b, sd, ep, len(d); self.rng = np.random.default_rng(seed)
        self.aim = np.clip(k * d.aim_raw.to_numpy(), -L, L); self.r = d.r1h_bp.to_numpy()
        self.F = d[["z", "sigrel", "mupct", "r24"]].to_numpy(np.float32)
        self.observation_space = gym.spaces.Box(-10, 10, (8,), np.float32); self.action_space = gym.spaces.Discrete(len(MS))

    def obs(self):
        t = self.t; return np.clip(np.r_[self.F[t], self.pos, self.aim[t], (self.eq - self.peak) / (self.sd * 10), self.hold / 24], -10, 10).astype(np.float32)

    def reset(self, seed=None, options=None):
        self.t = int(self.rng.integers(0, self.n - self.ep)) if self.ep else 0; self.end = self.t + self.ep if self.ep else self.n
        self.pos, self.eq, self.peak, self.hold = 0.0, 0.0, 0.0, 0; return self.obs(), {}

    def apply(self, m):
        t = self.t; tgt = np.clip(m * self.aim[t], -L, L); new = self.pos if abs(tgt - self.pos) < self.b else self.pos + self.a * (tgt - self.pos)
        pnl = new * self.r[t] - COST * abs(new - self.pos); self.hold = self.hold + 1 if new != 0 and np.sign(new) == np.sign(self.pos) else 0
        self.pos, self.eq = new, self.eq + pnl; self.peak = max(self.peak, self.eq); self.t += 1; return pnl

    def step(self, act):
        pnl = self.apply(MS[act]); done = self.t >= self.end
        return (np.zeros(8, np.float32) if done else self.obs()), pnl / self.sd, done, False, {}


def policy_m(models, d, k, a, b, sd):
    """앙상블 결정적 실행: 시각마다 각 모델 argmax 의 m 평균 → 그 m 으로 상태 진행."""
    env = Env(d, k, a, b, sd, 0, ep=0); o, _ = env.reset(); ms = np.empty(len(d))
    for t in range(len(d)):
        ms[t] = MS[[int(m.predict(o, deterministic=True)[0]) for m in models]].mean(); env.apply(ms[t])
        if env.t < len(d):
            o = env.obs()
    return ms


def train(seed):
    import torch as th
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv
    assert seed in SEEDS, f"사전 등록 시드만: {SEEDS}"; th.set_num_threads(1); th.manual_seed(seed)
    D = load(); tr, va = span(D, TRAIN), span(D, VAL); k, a, b = base_params(); sd = float(tr.r1h_bp.std())
    venv = DummyVecEnv([lambda i=i: Env(tr, k, a, b, sd, seed * 10 + i) for i in range(4)])
    m = PPO("MlpPolicy", venv, learning_rate=1e-4, gamma=0.9, gae_lambda=0.95, clip_range=0.2, ent_coef=0.01, n_steps=512, batch_size=256,   # 4 환경 × 512 = 2048
            n_epochs=10, policy_kwargs=dict(net_arch=[64, 64], activation_fn=th.nn.Tanh), seed=seed, verbose=0, device="cpu")
    od = OUT / f"s{seed}"; od.mkdir(parents=True, exist_ok=True); res = []
    for step in range(0, TOTAL + 1, CKPT):
        if step:
            m.learn(CKPT, reset_num_timesteps=False)
        m.save(od / f"{step}.zip"); ms = policy_m([m], va, k, a, b, sd); s, _ = run(va, k, a, b, ms)
        res.append(dict(step=step, val_sharpe=sharpe(s), val_bp_day=float(daily(s).mean()), m_mean=float(ms.mean()), m_hist=np.bincount(np.searchsorted(MS, ms), minlength=5).tolist()))
        print(f"s{seed} 걸음 {step:>6} 검증 샤프 {res[-1]['val_sharpe']:+.2f} 일 {res[-1]['val_bp_day']:+.1f}bp m평균 {res[-1]['m_mean']:.2f} m분포 {res[-1]['m_hist']}", flush=True)
        (od / "run.json").write_text(json.dumps(res, indent=1))


def block_ci(x, n=4000, blk=5, seed=0):
    x = np.asarray(x); rng = np.random.default_rng(seed); nb = int(np.ceil(len(x) / blk)); st = np.arange(len(x) - blk + 1)
    bs = [x[(rng.choice(st, nb)[:, None] + np.arange(blk)).ravel()[:len(x)]].mean() for _ in range(n)]
    return float(np.quantile(bs, .025)), float(np.quantile(bs, .975))


def test():
    from stable_baselines3 import PPO
    rp = OUT / "test_report.json"; assert not rp.exists(), "시험은 한 번만 연다"
    D = load(); tr, va, te = span(D, TRAIN), span(D, VAL), span(D, TEST); k, a, b = base_params(); sd = float(tr.r1h_bp.std())
    picks = {s: max(json.loads((OUT / f"s{s}" / "run.json").read_text()), key=lambda z: z["val_sharpe"])["step"] for s in SEEDS}
    models = {s: PPO.load(OUT / f"s{s}" / f"{st}.zip", device="cpu") for s, st in picks.items()}
    bv, _ = run(va, k, a, b); bt, pb = run(te, k, a, b)
    ms_e = policy_m(list(models.values()), te, k, a, b, sd); rt, pr = run(te, k, a, b, ms_e)
    diff = daily(rt) - daily(bt); lo, hi = block_ci(diff.values)
    seed = {}
    for s, m in models.items():
        sv, _ = run(va, k, a, b, policy_m([m], va, k, a, b, sd)); st_, _ = run(te, k, a, b, policy_m([m], te, k, a, b, sd))
        seed[s] = dict(step=picks[s], val_diff=float((daily(sv) - daily(bv)).mean()), test_diff=float((daily(st_) - daily(bt)).mean()))
    g2 = sum(v["test_diff"] > 0 and v["val_diff"] > 0 for v in seed.values())
    hold = te.r1h_bp.copy(); mdd = lambda s: float((daily(s).cumsum() - daily(s).cumsum().cummax()).min())
    F = te.assign(m=ms_e); learned = {c: F.groupby(pd.qcut(F[c], 5, labels=False, duplicates="drop")).m.mean().round(3).tolist() for c in ("z", "sigrel", "r24")}
    rep = dict(seeds=SEEDS, base=dict(k=k, a=a, b=b), picks=picks,
               test=dict(days=int(len(diff)), R_minus_B_bp_day=float(diff.mean()), ci95=[lo, hi], G1=bool(diff.mean() > 0 and lo > 0), G2_count=g2, G2=g2 >= 4,
                         sharpe=dict(R=sharpe(rt), B=sharpe(bt), hold=sharpe(hold)), bp_day=dict(R=float(daily(rt).mean()), B=float(daily(bt).mean()), hold=float(daily(hold).mean())),
                         mdd_bp=dict(R=mdd(rt), B=mdd(bt), hold=mdd(hold)), mean_abs_pos=dict(R=float(np.abs(pr).mean()), B=float(np.abs(pb).mean())), m_mean=float(ms_e.mean())),
               val=dict(sharpe_B=sharpe(bv)), per_seed=seed, learned_m_by_quintile=learned)
    rep["verdict"] = "후보(전진 판정 필요)" if rep["test"]["G1"] and rep["test"]["G2"] else "불통과 — v9 닫음"
    rp.write_text(json.dumps(rep, indent=1, ensure_ascii=False, default=float)); print(json.dumps(rep, indent=1, ensure_ascii=False, default=float))


if __name__ == "__main__":
    a = sys.argv[1:]
    if a[:1] == ["--test"]:
        test(); sys.exit()
    if a[:1] == ["--build"]:
        build()
    elif a[:1] == ["--base"]:
        base()
    elif a[:1] == ["--train"]:
        train(int(a[1]))
    else:
        print(__doc__, "\n시드:", SEEDS)
