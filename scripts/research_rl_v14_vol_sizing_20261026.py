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
SEEDS19 = [int(x) for x in np.random.default_rng(2026102619).integers(1, 2**31 - 1, 10)]   # v19 새 10시드
ACTS = np.array([-1.0, -0.5, 0.0, 0.5, 1.0]); COST, CAP, EP, TOTAL, CK = 4e-4, 2.0, 336, 300_000, 30_000


FUND = WT / "tmp" / "system_trend_ruleb_20261025" / "rest_fund_ETH.parquet"
BTCP = ROOT / "data" / "binance_vision" / "panel" / "BTCUSDT.parquet"


def build(start="2023-12-01", name="frame.parquet", extra=False):
    OUT.mkdir(parents=True, exist_ok=True)
    p = pd.read_parquet(PANEL, columns=["timestamp", "close"]).set_index("timestamp").close.astype(float)
    p = p[~p.index.duplicated()].sort_index(); p = p.reindex(pd.date_range(p.index[0], p.index[-1], freq="5min")).ffill()
    r2 = np.log(p).diff() ** 2
    t = pd.date_range(start, "2026-09-14 22:00", freq="1h")
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
    if extra:                                                             # v18: 파생·흐름·BTC (전부 t 에 끝난 봉까지)
        q = pd.read_parquet(PANEL, columns=["timestamp", "sum_open_interest", "sum_toptrader_long_short_ratio", "taker_buy_base", "volume"])
        q = q.drop_duplicates("timestamp").set_index("timestamp").sort_index().reindex(p.index).ffill()
        oi = q.sum_open_interest
        F["oi4h"] = np.log(oi / oi.shift(48)).reindex(end).to_numpy(); F["oi24h"] = np.log(oi / oi.shift(288)).reindex(end).to_numpy()
        F["oiz"] = ((oi - oi.rolling(2016).mean()) / oi.rolling(2016).std()).reindex(end).to_numpy()
        for k, n in (("tb1h", 12), ("tb4h", 48)):
            F[k] = (2 * q.taker_buy_base.rolling(n).sum() / q.volume.rolling(n).sum() - 1).reindex(end).to_numpy()
        ls = np.log(q.sum_toptrader_long_short_ratio)
        F["lsz"] = ((ls - ls.rolling(2016).mean()) / ls.rolling(2016).std()).reindex(end).to_numpy()
        fu = pd.read_parquet(FUND); fs = pd.Series(fu.rate.to_numpy(), index=pd.to_datetime(fu.t, unit="ms")).sort_index()
        F["fund"] = fs.reindex(fs.index.union(t)).ffill().reindex(t).to_numpy()     # t 시점까지 확정된 마지막 펀딩
        b = pd.read_parquet(BTCP, columns=["timestamp", "close"]).drop_duplicates("timestamp").set_index("timestamp").close.astype(float)
        b = b.reindex(p.index).ffill(); bs = np.sqrt((np.log(b).diff() ** 2).rolling(288).sum() / 24).reindex(end).to_numpy()
        bc = b.reindex(end).to_numpy()
        for k, h in (("b4h", 4), ("b24h", 24)):
            F[k] = np.log(bc / b.reindex(end - pd.Timedelta(hours=h)).to_numpy()) / (bs * np.sqrt(h))
        F["bvr"] = np.log(bs / s24)
    F.index.name = "t"; F.reset_index().to_parquet(OUT / name)
    print("행", len(F), "결측", F.isna().mean().round(4).to_dict(), "학습", int(span(F, TRAIN).sum()), flush=True)


def span(F, p):
    ix = F.index if isinstance(F.index, pd.DatetimeIndex) else pd.DatetimeIndex(F.t)
    return np.asarray((ix >= pd.Timestamp(p[0])) & (ix < pd.Timestamp(p[1])))


def load(rho=None, seed=0, frame="frame.parquet", train=TRAIN):
    F = pd.read_parquet(OUT / frame).set_index("t").dropna()
    tr = span(F, train); sstar = float(np.median(F.sig[tr]))
    X = pd.DataFrame(index=F.index)
    X["pred_z"] = F.pred / F.pred[tr].std()
    for k in ("ls1h", "ls4h", "ls24h", "ls7d"):
        X[k] = np.log(F[k] / sstar)
    for k in ("m1d", "m7d", "m30d", "m4h"):
        X[k] = F[k]
    for k in ("oi4h", "oi24h", "oiz", "tb1h", "tb4h", "lsz", "b4h", "b24h", "bvr"):   # v18 프레임에만 있는 열
        if k in F:
            X[k] = F[k]
    if "fund" in F:
        X["fund"] = F.fund / F.fund[tr].std()
    if rho is not None:                                                   # 장치 검증: 다음 1시간 수익을 심는다
        z = F.ret / F.sig; z = (z - z[tr].mean()) / z[tr].std()
        X["hint"] = rho * z + np.sqrt(1 - rho ** 2) * np.random.default_rng(seed).standard_normal(len(z))
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(-5, 5).astype(np.float32)   # v18: OI 0 봉의 log 무한 → 중립 0
    size = np.minimum(sstar / F.sig.to_numpy(), CAP)
    return F, X, size, sstar


def pnl(notional, ret):
    prev = np.r_[0.0, notional[:-1]]
    return notional * ret - COST * np.abs(notional - prev)


def stats(x):
    eq = np.cumsum(x); s = x.std()
    return dict(sharpe=float(x.mean() / s * np.sqrt(8760)) if s > 0 else 0.0, total_bp=float(x.sum() * 1e4),
                mdd_bp=float((eq - np.maximum.accumulate(eq)).min() * 1e4), hours=int(len(x)))


SIGNED = ["pred_z", "m1d", "m7d", "m30d", "m4h", "hint", "tb1h", "tb4h", "lsz", "fund", "b4h", "b24h"]   # v15 거울: 가격 방향을 뒤집으면 부호가 바뀌는 열


class Env(gym.Env):
    def __init__(self, X, size, ret, sstar, mask, seed, flipv=None, base=None):
        self.X, self.size, self.ret, self.s = X, size, ret, sstar
        self.base = np.zeros(len(ret)) if base is None else base          # v16: 교사(TSMOM 부호) 위 조정, None = v14·v15
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
        n = float(np.clip((self.f * self.base[self.i] + ACTS[a]) * self.size[self.i], -CAP, CAP))
        r = (n * self.f * self.ret[self.i] - COST * abs(n - self.n)) / self.s
        self.n, self.i = n, self.i + 1
        return self._obs(), float(r), self.i >= self.end, False, {}


def rollout(model, X, size, mask, base=None):
    """마스크 구간을 처음부터 순서대로 — 자기 직전 명목을 상태에 넣는다."""
    out, n = [], 0.0
    b = np.zeros(len(size)) if base is None else base
    for i in np.flatnonzero(mask):
        a, _ = model.predict(np.r_[X[i], n / CAP].astype(np.float32)[None], deterministic=True)
        n = float(np.clip((b[i] + ACTS[int(a[0])]) * size[i], -CAP, CAP)); out.append(n)
    return np.array(out)


def teachers(F, size, tr):
    hi, lo = np.quantile(F.pred[tr], .8), np.quantile(F.pred[tr], .2)
    return {"T1_long": size.copy(), "T2_dirmodel": np.where(F.pred > hi, 1.0, np.where(F.pred < lo, -1.0, 0.0)) * size,
            "T3_tsmom": np.sign(F.m30d.to_numpy()) * size}


def train(seed, rho=None, tag=None, mirror=False, residual=False, win=None, frame=None):
    from stable_baselines3 import PPO
    trp, vap = (WIN17[win][0], WIN17[win][1]) if win else (TRAIN, VAL)
    F, X, size, sstar = load(rho, seed, frame or ("frame_v17.parquet" if win else "frame.parquet"), trp); Xn, ret = X.to_numpy(), F.ret.to_numpy()
    tr, va = span(F, trp), span(F, vap)
    d = OUT / (tag or f"s{seed}"); d.mkdir(parents=True, exist_ok=True)
    flipv = np.array([-1.0 if c in SIGNED else 1.0 for c in X.columns], np.float32) if mirror else None
    base = np.sign(F.m30d.to_numpy()) if residual else None
    m = PPO("MlpPolicy", Env(Xn, size, ret, sstar, tr, seed, flipv, base), seed=seed, verbose=0, n_steps=2048, batch_size=256, gamma=0.99,
            learning_rate=3e-4, ent_coef=0.01, policy_kwargs=dict(net_arch=[64, 64]), device="cpu")
    T = teachers(F, size, tr); best_t = max(stats(pnl(v[va], ret[va]))["sharpe"] for v in T.values())
    ck = []
    for step in range(CK, TOTAL + 1, CK):
        m.learn(CK, reset_num_timesteps=False); m.save(d / f"{step}.zip")
        s = stats(pnl(rollout(m, Xn, size, va, base), ret[va])); ck.append(dict(step=step, **s))
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


WIN17 = {"R1": (("2023-01-01", "2024-01-01"), ("2024-01-01", "2024-06-01"), ("2024-06-01", "2024-10-24")),
         "R2": (("2023-01-01", "2025-01-01"), ("2025-01-01", "2025-06-01"), ("2025-06-01", "2025-10-24")),
         "R3": (TRAIN, VAL, TEST)}                                         # v17 굴림 3창 (학습, 검증=보고만, 시험)


def test_v17(prefix="v17", frame="frame_v17.parquet", report="test_report_v17.json", seeds_=None):
    """v17: 창마다 마지막(30만) 체크포인트 · H1 3창 이어 붙인 샤프 RL > T3 그리고 > max(T1,T2) · H2 창별 RL > T3 ≥ 2/3 · H3 조정 원형이동 > 90분위."""
    from stable_baselines3 import PPO
    rp = OUT / report; assert not rp.exists(), "판정은 한 번만"
    SD = seeds_ or SEEDS
    rep, P = dict(seeds=SD), {k: [] for k in ("RL", "T1_long", "T2_dirmodel", "T3_tsmom")}; parts = []
    for w, (trp, vap, tep) in WIN17.items():
        F, X, size, _ = load(frame=frame, train=trp); Xn, ret = X.to_numpy(), F.ret.to_numpy(); base = np.sign(F.m30d.to_numpy())
        T = teachers(F, size, span(F, trp)); rep[w] = {}
        for name, p in (("VAL", vap), ("TEST", tep)):
            m = span(F, p); r = ret[m]
            seeds = {s: rollout(PPO.load(OUT / f"{prefix}_{w}_{s}" / f"{TOTAL}.zip", device="cpu"), Xn, size, m, base) for s in SD}
            ens = np.mean(list(seeds.values()), 0); t3 = T["T3_tsmom"][m]
            rep[w][name] = dict(RL=stats(pnl(ens, r)), teachers={k: stats(pnl(v[m], r)) for k, v in T.items()},
                                per_seed={s: stats(pnl(v, r))["sharpe"] for s, v in seeds.items()}, adj_mean_abs=float(np.abs(ens - t3).mean()))
            if name == "TEST":
                P["RL"].append(pnl(ens, r)); [P[k].append(pnl(v[m], r)) for k, v in T.items()]; parts.append((ens - t3, t3, r))
            print(w, name, json.dumps(rep[w][name], ensure_ascii=False, default=float), flush=True)
    pooled = {k: stats(np.concatenate(v)) for k, v in P.items()}; rep["pooled_test"] = pooled
    rng = np.random.default_rng(1); obs = sum(pnl(t3 + a, r).sum() for a, t3, r in parts)
    null = np.array([sum(pnl(t3 + np.roll(a, int(rng.integers(168, len(a) - 168))), r).sum() for a, t3, r in parts) for _ in range(1000)])
    rep["adj_shift_pct"] = float((null < obs).mean() * 100)
    rep["H1"] = bool(pooled["RL"]["sharpe"] > pooled["T3_tsmom"]["sharpe"] and pooled["RL"]["sharpe"] > max(pooled["T1_long"]["sharpe"], pooled["T2_dirmodel"]["sharpe"]))
    rep["H2"] = bool(sum(rep[w]["TEST"]["RL"]["sharpe"] > rep[w]["TEST"]["teachers"]["T3_tsmom"]["sharpe"] for w in WIN17) >= 2)
    rep["H3"] = bool(rep["adj_shift_pct"] > 90)
    rep["verdict"] = "후보(상대 기준)" if rep["H1"] and rep["H2"] and rep["H3"] else "불통과"
    rp.write_text(json.dumps(rep, indent=1, ensure_ascii=False, default=float))
    print(json.dumps({k: rep[k] for k in ("pooled_test", "adj_shift_pct", "H1", "H2", "H3", "verdict")}, ensure_ascii=False, default=float))


def test_v16():
    """v16 관문: G1 RL > T3 그리고 > max(T1,T2) · G2 조정(RL − T3) 원형이동 95분위 · G3 시드 ≥4/5 가 검증·시험 모두 T3 보다 높음."""
    from stable_baselines3 import PPO
    rp = OUT / "test_report_v16.json"; assert not rp.exists(), "판정은 한 번만"
    F, X, size, _ = load(); Xn, ret = X.to_numpy(), F.ret.to_numpy(); tr = span(F, TRAIN); base = np.sign(F.m30d.to_numpy())
    T = teachers(F, size, tr); rep = dict(seeds=SEEDS)
    picks = {s: max(json.loads((OUT / f"r{s}" / "run.json").read_text())["ckpt"], key=lambda c: c["sharpe"])["step"] for s in SEEDS}
    models = {s: PPO.load(OUT / f"r{s}" / f"{st}.zip", device="cpu") for s, st in picks.items()}; rep["picks"] = picks
    for name, p in (("VAL", VAL), ("TEST", TEST)):
        m = span(F, p); r = ret[m]; t3 = T["T3_tsmom"][m]
        seeds = {s: rollout(mo, Xn, size, m, base) for s, mo in models.items()}; ens = np.mean(list(seeds.values()), 0)
        tt = {k: stats(pnl(v[m], r)) for k, v in T.items()}; adj = ens - t3; rng = np.random.default_rng(1); n = len(ens)
        null = np.array([pnl(t3 + np.roll(adj, int(k)), r).sum() for k in rng.integers(168, n - 168, 1000)])
        rep[name] = dict(RL=stats(pnl(ens, r)), teachers=tt, adj_mean_abs=float(np.abs(adj).mean()),
                         shift_pct=float((null < pnl(ens, r).sum()).mean() * 100), per_seed={s: stats(pnl(v, r))["sharpe"] for s, v in seeds.items()})
        print(name, json.dumps(rep[name], ensure_ascii=False, default=float), flush=True)
    t, v = rep["TEST"], rep["VAL"]
    rep["G1"] = bool(t["RL"]["sharpe"] > t["teachers"]["T3_tsmom"]["sharpe"] and t["RL"]["sharpe"] > max(t["teachers"][k]["sharpe"] for k in ("T1_long", "T2_dirmodel")))
    rep["G2"] = bool(t["shift_pct"] > 95)
    rep["G3"] = bool(sum(v["per_seed"][s] > v["teachers"]["T3_tsmom"]["sharpe"] and t["per_seed"][s] > t["teachers"]["T3_tsmom"]["sharpe"] for s in SEEDS) >= 4)
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
    elif a[:1] == ["--plant-r"]: train(SEEDS[0], rho=0.3, tag="plant_r", mirror=True, residual=True)   # v16
    elif a[:1] == ["--train-r"]: assert int(a[1]) in SEEDS; train(int(a[1]), tag=f"r{a[1]}", mirror=True, residual=True)
    elif a[:1] == ["--test-v16"]: test_v16()
    elif a[:1] == ["--build-v17"]: build("2022-11-01", "frame_v17.parquet")
    elif a[:1] == ["--plant-v17"]: train(SEEDS[0], rho=0.3, tag="v17_plant", mirror=True, residual=True, win="R3")
    elif a[:1] == ["--train-v17"]: assert a[1] in WIN17 and int(a[2]) in SEEDS; train(int(a[2]), tag=f"v17_{a[1]}_{a[2]}", mirror=True, residual=True, win=a[1])
    elif a[:1] == ["--test-v17"]: test_v17()
    elif a[:1] == ["--build-v18"]: build("2022-11-01", "frame_v18.parquet", extra=True)
    elif a[:1] == ["--plant-v18"]: train(SEEDS[0], rho=0.3, tag="v18_plant", mirror=True, residual=True, win="R3", frame="frame_v18.parquet")
    elif a[:1] == ["--train-v18"]: assert a[1] in WIN17 and int(a[2]) in SEEDS; train(int(a[2]), tag=f"v18_{a[1]}_{a[2]}", mirror=True, residual=True, win=a[1], frame="frame_v18.parquet")
    elif a[:1] == ["--train-v19"]: assert a[1] in WIN17 and int(a[2]) in SEEDS19; train(int(a[2]), tag=f"v19_{a[1]}_{a[2]}", mirror=True, residual=True, win=a[1], frame="frame_v18.parquet")
    elif a[:1] == ["--test-v19"]: test_v17("v19", "frame_v18.parquet", "test_report_v19.json", SEEDS19)
    elif a[:1] == ["--test-v18"]: test_v17("v18", "frame_v18.parquet", "test_report_v18.json")
    else: print(__doc__, SEEDS)
