#!/usr/bin/env python3
"""RL v14 — 방향 모델 표본 밖 예측 + 변동성 크기를 PPO 가 배움. 사전 등록 docs/prereg/rl_v14_vol_sizing_20261026.md.

    python3 scripts/research_rl_v14_vol_sizing_20261026.py --build
    python3 scripts/research_rl_v14_vol_sizing_20261026.py --plant          # 장치 검증(1시드, 심은 신호)
    python3 scripts/research_rl_v14_vol_sizing_20261026.py --train SEED
    python3 scripts/research_rl_v14_vol_sizing_20261026.py --test           # 한 번만
실주문 없음.
"""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
import gymnasium as gym

ROOT = Path("/home/kbj20/crypto-scalping")
WT = Path(__file__).resolve().parents[1]
OUT = ROOT / "tmp" / "rl_v14_20261026"
DIR = WT / "data" / "live" / "eth_dir4h_oos_seed.parquet"
PANEL = ROOT / "data" / "binance_vision" / "panel" / "ETHUSDT.parquet"
TRAIN, VAL, TEST = ("2024-01-01", "2026-01-01"), ("2026-01-01", "2026-06-01"), ("2026-06-01", "2026-09-14 23:00")
SEEDS = [int(s) for s in np.random.default_rng(2026102614).integers(1, 2**31 - 1, 5)]
ACTS = np.array([-1.0, -0.5, 0.0, 0.5, 1.0]); COST, CAP, EP, TOTAL, CK = 4e-4, 2.0, 336, 300_000, 30_000


def build():
    OUT.mkdir(parents=True, exist_ok=True)
    p = pd.read_parquet(PANEL, columns=["timestamp", "close"]).set_index("timestamp").close.astype(float)
    p = p[~p.index.duplicated()].sort_index(); p = p.reindex(pd.date_range(p.index[0], p.index[-1], freq="5min")).ffill()
    r2 = np.log(p).diff() ** 2
    t = pd.date_range("2023-12-01", "2026-09-14 22:00", freq="1h")
    end = t - pd.Timedelta("5min")                                       # 정시 t 에 끝난 5분봉의 시작 시각
    sig = lambda n: np.sqrt(r2.rolling(n).sum() / (n / 12)).reindex(end).to_numpy()   # noqa: E731 — 1시간 환산 σ
    c = p.reindex(end).to_numpy(); c_next = p.reindex(end + pd.Timedelta("1h")).to_numpy()
    s24 = sig(288)
    d = pd.read_parquet(DIR).set_index("ts").pred.astype(float)
    F = pd.DataFrame(index=t)
    F["pred"] = d.reindex(end).to_numpy()
    for k, n in (("ls1h", 12), ("ls4h", 48), ("ls24h", 288), ("ls7d", 2016)):
        F[k] = sig(n)
    for k, h in (("m1d", 24), ("m7d", 168), ("m30d", 720), ("m4h", 4)):
        F[k] = np.log(c / p.reindex(end - pd.Timedelta(hours=h)).to_numpy()) / (s24 * np.sqrt(h))
    F["sig"], F["ret"] = s24, np.log(c_next / c)
    F.index.name = "t"; F.reset_index().to_parquet(OUT / "frame.parquet")
    print("행", len(F), "결측", F.isna().mean().round(4).to_dict(), "학습", int(span(F, TRAIN).sum()), flush=True)


def span(F, p):
    ix = F.index if isinstance(F.index, pd.DatetimeIndex) else pd.DatetimeIndex(F.t)
    return np.asarray((ix >= pd.Timestamp(p[0])) & (ix < pd.Timestamp(p[1])))


def load(rho=None, seed=0):
    F = pd.read_parquet(OUT / "frame.parquet").set_index("t").dropna()
    tr = span(F, TRAIN); sstar = float(np.median(F.sig[tr]))
    X = pd.DataFrame(index=F.index)
    X["pred_z"] = F.pred / F.pred[tr].std()
    for k in ("ls1h", "ls4h", "ls24h", "ls7d"):
        X[k] = np.log(F[k] / sstar)
    for k in ("m1d", "m7d", "m30d", "m4h"):
        X[k] = F[k]
    if rho is not None:                                                   # 장치 검증: 다음 1시간 수익을 심는다
        z = F.ret / F.sig; z = (z - z[tr].mean()) / z[tr].std()
        X["hint"] = rho * z + np.sqrt(1 - rho ** 2) * np.random.default_rng(seed).standard_normal(len(z))
    X = X.clip(-5, 5).astype(np.float32)
    size = np.minimum(sstar / F.sig.to_numpy(), CAP)
    return F, X, size, sstar


def pnl(notional, ret):
    prev = np.r_[0.0, notional[:-1]]
    return notional * ret - COST * np.abs(notional - prev)


def stats(x):
    eq = np.cumsum(x); s = x.std()
    return dict(sharpe=float(x.mean() / s * np.sqrt(8760)) if s > 0 else 0.0, total_bp=float(x.sum() * 1e4),
                mdd_bp=float((eq - np.maximum.accumulate(eq)).min() * 1e4), hours=int(len(x)))


SIGNED = ["pred_z", "m1d", "m7d", "m30d", "m4h", "hint"]   # v15 거울: 가격 방향을 뒤집으면 부호가 바뀌는 열


class Env(gym.Env):
    def __init__(self, X, size, ret, sstar, mask, seed, flipv=None):
        self.X, self.size, self.ret, self.s = X, size, ret, sstar
        self.idx = np.flatnonzero(mask); self.rng = np.random.default_rng(seed)
        self.flipv, self.f = flipv, 1.0                                   # flipv = 열마다 −1(부호 열)·1, None 이면 거울 없음(v14)
        self.observation_space = gym.spaces.Box(-5, 5, (X.shape[1] + 1,), np.float32); self.action_space = gym.spaces.Discrete(len(ACTS))

    def _obs(self):
        x = self.X[self.i] * self.flipv if self.f < 0 else self.X[self.i]
        return np.r_[x, self.n / CAP].astype(np.float32)

    def reset(self, seed=None, options=None):
        st = self.rng.integers(0, len(self.idx) - EP); self.i, self.end, self.n = self.idx[st], self.idx[st] + EP, 0.0
        self.f = -1.0 if self.flipv is not None and self.rng.random() < 0.5 else 1.0
        return self._obs(), {}

    def step(self, a):
        n = float(ACTS[a] * self.size[self.i]); r = (n * self.f * self.ret[self.i] - COST * abs(n - self.n)) / self.s
        self.n, self.i = n, self.i + 1
        return self._obs(), float(r), self.i >= self.end, False, {}


def rollout(model, X, size, mask):
    """마스크 구간을 처음부터 순서대로 — 자기 직전 명목을 상태에 넣는다."""
    out, n = [], 0.0
    for i in np.flatnonzero(mask):
        a, _ = model.predict(np.r_[X[i], n / CAP].astype(np.float32)[None], deterministic=True)
        n = float(ACTS[int(a[0])] * size[i]); out.append(n)
    return np.array(out)


def teachers(F, size, tr):
    hi, lo = np.quantile(F.pred[tr], .8), np.quantile(F.pred[tr], .2)
    return {"T1_long": size.copy(), "T2_dirmodel": np.where(F.pred > hi, 1.0, np.where(F.pred < lo, -1.0, 0.0)) * size,
            "T3_tsmom": np.sign(F.m30d.to_numpy()) * size}


def train(seed, rho=None, tag=None, mirror=False):
    from stable_baselines3 import PPO
    F, X, size, sstar = load(rho, seed); Xn, ret = X.to_numpy(), F.ret.to_numpy()
    tr, va = span(F, TRAIN), span(F, VAL)
    d = OUT / (tag or f"s{seed}"); d.mkdir(parents=True, exist_ok=True)
    flipv = np.array([-1.0 if c in SIGNED else 1.0 for c in X.columns], np.float32) if mirror else None
    m = PPO("MlpPolicy", Env(Xn, size, ret, sstar, tr, seed, flipv), seed=seed, verbose=0, n_steps=2048, batch_size=256, gamma=0.99,
            learning_rate=3e-4, ent_coef=0.01, policy_kwargs=dict(net_arch=[64, 64]), device="cpu")
    T = teachers(F, size, tr); best_t = max(stats(pnl(v[va], ret[va]))["sharpe"] for v in T.values())
    ck = []
    for step in range(CK, TOTAL + 1, CK):
        m.learn(CK, reset_num_timesteps=False); m.save(d / f"{step}.zip")
        s = stats(pnl(rollout(m, Xn, size, va), ret[va])); ck.append(dict(step=step, **s))
        print(f"[{tag or seed}] {step} 검증 샤프 {s['sharpe']:.2f} 총 {s['total_bp']:.0f}bp · 최고 교사 {best_t:.2f}", flush=True)
    (d / "run.json").write_text(json.dumps(dict(seed=seed, rho=rho, best_teacher_val_sharpe=best_t, ckpt=ck), indent=1))


def test(pre="s", report="test_report.json"):
    from stable_baselines3 import PPO
    rp = OUT / report; assert not rp.exists(), "판정은 한 번만"
    F, X, size, _ = load(); Xn, ret = X.to_numpy(), F.ret.to_numpy(); tr = span(F, TRAIN)
    T = teachers(F, size, tr); rep = dict(seeds=SEEDS)
    picks = {s: max(json.loads((OUT / f"{pre}{s}" / "run.json").read_text())["ckpt"], key=lambda c: c["sharpe"])["step"] for s in SEEDS}
    models = {s: PPO.load(OUT / f"{pre}{s}" / f"{st}.zip", device="cpu") for s, st in picks.items()}; rep["picks"] = picks
    per_seed = {}
    for name, p in (("VAL", VAL), ("TEST", TEST)):
        m = span(F, p); r = ret[m]
        seeds = {s: rollout(mo, Xn, size, m) for s, mo in models.items()}; ens = np.mean(list(seeds.values()), 0)
        tt = {k: stats(pnl(v[m], r)) for k, v in T.items()}; best = max(v["sharpe"] for v in tt.values())
        st = stats(pnl(ens, r)); rng = np.random.default_rng(1); n = len(ens)
        null = np.array([pnl(np.roll(ens, int(k)), r).sum() for k in rng.integers(168, n - 168, 1000)])
        acts = pd.Series(np.round(ens / np.where(size[m] > 0, size[m], 1), 2)).value_counts(normalize=True).head(6).round(3).to_dict()
        rep[name] = dict(RL=st, teachers=tt, best_teacher_sharpe=best, mean_abs_notional=float(np.abs(ens).mean()),
                         turnover_per_day=float(np.abs(np.diff(ens)).sum() / (n / 24)), shift_pct=float((null < pnl(ens, r).sum()).mean() * 100),
                         action_mix=acts, per_seed={s: stats(pnl(v, r))["sharpe"] for s, v in seeds.items()})
        per_seed[name] = (rep[name]["per_seed"], best)
        print(name, json.dumps(rep[name], ensure_ascii=False, default=float), flush=True)
    t = rep["TEST"]
    rep["G1"] = bool(t["RL"]["sharpe"] > t["best_teacher_sharpe"])
    rep["G2"] = bool(t["shift_pct"] > 95)
    rep["G3"] = bool(sum(per_seed["VAL"][0][s] > per_seed["VAL"][1] and per_seed["TEST"][0][s] > per_seed["TEST"][1] for s in SEEDS) >= 4)
    rep["verdict"] = "후보(상대 기준)" if rep["G1"] and rep["G2"] and rep["G3"] else "불통과"
    rp.write_text(json.dumps(rep, indent=1, ensure_ascii=False, default=float)); print({k: rep[k] for k in ("G1", "G2", "G3", "verdict")})


def selftest():
    n = np.array([0.0, 1.0, 1.0, -1.0]); r = np.array([.01, .01, -.01, .02])
    assert np.allclose(pnl(n, r), [0, .01 - COST, -.01, -.02 - 2 * COST])
    assert np.isclose(stats(np.array([.01, -.01, .01, -.01]))["mdd_bp"], -100)


if __name__ == "__main__":
    selftest(); a = sys.argv[1:]
    if a[:1] == ["--build"]: build()
    elif a[:1] == ["--plant"]: train(SEEDS[0], rho=0.3, tag="plant")
    elif a[:1] == ["--train"]: assert int(a[1]) in SEEDS; train(int(a[1]))
    elif a[:1] == ["--test"]: test()
    elif a[:1] == ["--plant-m"]: train(SEEDS[0], rho=0.3, tag="plant_m", mirror=True)        # v15 장치 검증
    elif a[:1] == ["--train-m"]: assert int(a[1]) in SEEDS; train(int(a[1]), tag=f"m{a[1]}", mirror=True)
    elif a[:1] == ["--test-v15"]: test("m", "test_report_v15.json")
    else: print(__doc__, SEEDS)
