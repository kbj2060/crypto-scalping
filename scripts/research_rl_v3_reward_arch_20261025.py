#!/usr/bin/env python3
"""RL v3 — 보상(점수 규칙) 3 × 구조 3 × 시드 2. 사전 등록 docs/prereg/rl_v3_reward_arch_20261025.md (73f325ad).

    python3 scripts/research_rl_v3_reward_arch_20261025.py --selfcheck        # P1·P5·P6 + 보상 크기 점검
    python3 scripts/research_rl_v3_reward_arch_20261025.py --cell 1 --seed 402917   # 한 칸·한 시드 학습(검증 체크포인트 기록)
    python3 scripts/research_rl_v3_reward_arch_20261025.py --collect          # 126 후보 중 검증 최선 1개 → 시험 1회 · G1~G3

환경·슬리브·채점은 research_rl_v2_daily_trend_20261025 그대로(P1 env = SYS.simulate). 실주문 없음. 산출 tmp/rl_v3_reward_arch_20261025/.
"""
import argparse, glob, json, math, pathlib, sys
import numpy as np, pandas as pd
import gymnasium as gym
import torch as th
from torch import nn

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import research_system_trend_ruleb_20261025 as SYS          # noqa: E402
import research_rl_v2_daily_trend_20261025 as V2             # noqa: E402

OUT = SYS.ROOT / "tmp" / "rl_v3_reward_arch_20261025"
TRAIN, VAL, TEST = ("2024-01-01", "2025-12-31"), ("2026-01-01", "2026-05-31"), ("2026-06-01", "2026-10-23")
W, NF, KAPPA, LAM, ETA = 20, V2.NMKT, 10e-4, 2.0, 1 / 60
CELLS = {1: ("Z", "mlp", (402917, 771360)), 2: ("Z", "gru", (158244, 639012)), 3: ("Z", "tcn", (907133, 284476)),
         4: ("DSR", "mlp", (533018, 96241)), 5: ("DSR", "gru", (715590, 348827)), 6: ("DSR", "tcn", (261709, 880453)),
         7: ("DD", "mlp", (47386, 592610)), 8: ("DD", "gru", (823154, 173095)), 9: ("DD", "tcn", (360681, 954228))}
PPO_KW = dict(learning_rate=1e-4, n_steps=256, batch_size=256, n_epochs=10, gamma=0.3, gae_lambda=0.95, clip_range=0.2, ent_coef=0.001)
TOTAL, CKPT, EP_DAYS = 300_000, 50_000, 365


class World(V2.World):
    """v2 World 와 같은 30세계(4코인 + 패널 11코인 × 거울) — 학습 구간만 2024-01~2025-12 로."""
    def __init__(self):
        data, _, inputs, fund = SYS.load_all(); self.data, self.inputs, self.fund = data, inputs, fund
        self.days = pd.date_range("2022-01-01", SYS.END, freq="D"); tr = (self.days >= TRAIN[0]) & (self.days <= TRAIN[1])
        self.W = {s: V2.mkworld(data[s], [fund[s].get(d, ()) for d in self.days]) for s in V2.COINS}
        self.pool = [self.W[s] for s in V2.COINS]
        for s in V2.EXTRA:   # 순서 = v2: 4코인 · (패널, 거울) × 11 · 4코인 거울
            p = pd.read_parquet(SYS.D / f"binance_vision/panel/{s}USDT.parquet", columns=["timestamp", "high", "low", "close"]).set_index("timestamp")
            x = p.rename(columns={"high": "h", "low": "l", "close": "c"}).reindex(pd.date_range("2022-01-01", TRAIN[1] + " 23:55", freq="5min"))
            x["c"] = x.c.ffill(); x["h"] = x.h.fillna(x.c); x["l"] = x.l.fillna(x.c); fl = [()] * (len(x) // V2.NB)
            self.pool += [V2.mkworld(x, fl), V2.mirror(x, fl)]
        self.pool += [V2.mirror(data[s], self.W[s]["fund"]) for s in V2.COINS]
        M = np.concatenate([w["raw"][:len(tr)][tr[:len(w["raw"])]] for w in self.pool]); self.mu, self.sd = np.nanmean(M, 0), np.nanstd(M, 0) + 1e-12
        for w in self.pool + list(self.W.values()):
            w["F"] = ((w["raw"] - self.mu) / self.sd).astype(np.float32)


def window(w, d, o):
    """창 관측: 최근 W일 시장 피쳐(F[d] 는 이미 d−1 종가까지) ⊕ 포지션 3."""
    return np.r_[w["F"][d - W + 1:d + 1].ravel(), o[-3:]].astype(np.float32)


class Rew:
    """보상(점수 규칙). step(e1, e0, tn) → R."""
    def __init__(self, kind):
        self.kind, self.A, self.B, self.peak, self.dd = kind, 0.0, 1.0, None, 0.0

    def step(self, e1, e0, tn):
        x = ((e1 - e0) / e0 - KAPPA * tn / e0) / V2.UNIT
        if self.kind == "Z":
            return x
        if self.kind == "DSR":
            dA, dB = x - self.A, x * x - self.B; den = max(self.B - self.A ** 2, 1e-6) ** 1.5
            r = (self.B * dA - 0.5 * self.A * dB) / den; self.A += ETA * dA; self.B += ETA * dB
            return float(np.clip(r, -10, 10))
        self.peak = e0 if self.peak is None else self.peak; self.peak = max(self.peak, e1); dd = 1 - e1 / self.peak
        r = x - LAM * max(0.0, dd - self.dd) / V2.UNIT; self.dd = dd
        return r


class Env(gym.Env):
    action_space = gym.spaces.Discrete(3)

    def __init__(self, Wd, span, rew, arch, seed):
        self.Wd, self.span_, self.rew, self.arch, self.rng = Wd, span, rew, arch, np.random.default_rng(seed)
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, (NF + 3 if arch == "mlp" else W * NF + 3,), np.float32)

    def _o(self):
        d = min(self.d, self.dend - 1); o = self.sl.obs(d); return o if self.arch == "mlp" else window(self.w, d, o)

    def reset(self, seed=None, options=None):
        a, b = self.span_; self.w = self.Wd.pool[self.rng.integers(len(self.Wd.pool))]
        self.d = int(self.rng.integers(a, max(a + 1, b - EP_DAYS))); self.dend = min(b, self.d + EP_DAYS); self.sl = V2.Sleeve(self.w); self.R = Rew(self.rew)
        return self._o(), {}

    def step(self, a):
        e1, e0, tn = self.sl.day(self.d, int(a) - 1); r = self.R.step(e1, e0, tn); self.d += 1
        return self._o(), float(r), self.d >= self.dend or e1 <= 1e-3, False, {}


def make_extractor():
    from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

    class Seq(BaseFeaturesExtractor):
        """GRU 또는 인과 TCN 으로 W일 창을 요약 ⊕ 포지션 3. 창 정규화 없음."""
        def __init__(self, space, kind="gru", hid=32):
            super().__init__(space, features_dim=hid + 3); self.kind = kind
            if kind == "gru":
                self.rnn = nn.GRU(NF, hid, batch_first=True)
            else:
                self.c1, self.c2 = nn.Conv1d(NF, hid, 3), nn.Conv1d(hid, hid, 3, dilation=2)

        def forward(self, obs):
            seq, pos = obs[:, :W * NF].reshape(-1, W, NF), obs[:, W * NF:]
            if self.kind == "gru":
                z = self.rnn(seq)[1][-1]
            else:   # 인과: 왼쪽만 채움
                x = seq.transpose(1, 2); x = th.relu(self.c1(nn.functional.pad(x, (2, 0)))); x = th.relu(self.c2(nn.functional.pad(x, (4, 0)))); z = x[..., -1]
            return th.cat([z, pos], 1)
    return Seq


def policy_kwargs(arch):
    if arch == "mlp":
        return dict(net_arch=dict(pi=[64, 64], vf=[64, 64]), activation_fn=th.nn.Tanh)
    return dict(features_extractor_class=make_extractor(), features_extractor_kwargs=dict(kind=arch), net_arch=dict(pi=[64], vf=[64]), activation_fn=th.nn.Tanh)


def picker(Wd, m, arch):
    if arch == "mlp":
        return lambda s, d, o: int(m.predict(o, deterministic=True)[0])
    return lambda s, d, o: int(m.predict(window(Wd.W[s], d, o), deterministic=True)[0])


def bc(model, Wd, span, arch):
    X, Y = [], []
    for w in Wd.pool:
        sl = V2.Sleeve(w)
        for d in range(*span):
            a = int(w["D"][d]) + 1; o = sl.obs(d); X.append(o if arch == "mlp" else window(w, d, o)); Y.append(a); sl.day(d, a - 1)
    X, Y = th.tensor(np.array(X)), th.tensor(np.array(Y)); opt = th.optim.Adam(model.policy.parameters(), lr=1e-3); g = th.Generator().manual_seed(0)
    for _ in range(200):
        perm = th.randperm(len(X), generator=g)
        for i in range(0, len(X), 256):
            b = perm[i:i + 256]; loss = th.nn.functional.cross_entropy(model.policy.get_distribution(X[b]).distribution.logits, Y[b])
            opt.zero_grad(); loss.backward(); opt.step()
    with th.no_grad():
        agree = float((model.policy.get_distribution(X).distribution.logits.argmax(1) == Y).float().mean())
    return {"rows": len(X), "train_agree": agree}


def train_one(cell, seed):
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv
    rew, arch, _ = CELLS[cell]; th.set_num_threads(2); th.manual_seed(seed)
    Wd = World(); tr, va = Wd.span(TRAIN, Wd.pool), Wd.span(VAL)
    assert all(not np.isnan(w["F"][tr[0] - W + 1:tr[1]]).any() for w in Wd.pool)   # 창 예열
    venv = DummyVecEnv([lambda i=i: Env(Wd, tr, rew, arch, seed * 10 + i) for i in range(8)])
    m = PPO("MlpPolicy", venv, policy_kwargs=policy_kwargs(arch), seed=seed, verbose=0, device="cpu", **PPO_KW)
    b = bc(m, Wd, tr, arch); assert b["train_agree"] >= 0.95, b   # P7
    out = OUT / f"c{cell}_s{seed}"; out.mkdir(parents=True, exist_ok=True); res = {"cell": cell, "reward": rew, "arch": arch, "seed": seed, "bc": b, "ckpt": []}
    for step in range(0, TOTAL + 1, CKPT):
        if step:
            m.learn(CKPT, reset_num_timesteps=False)
        m.save(out / f"{step}.zip"); r, _, lg = V2.run(Wd, va, picker(Wd, m, arch)); sm = V2.summarize(r, lg); sm["step"] = step; res["ckpt"].append(sm)
        print(f"c{cell} {rew}/{arch} s{seed} step {step:>6} 검증 샤프 {sm['sharpe']:+.2f} MDD {sm['mdd']:+.2f} 교사 일치 {sm['teacher_agree']:.2f} 현금 {sm['flat_share']:.2f}", flush=True)
        (out / "run.json").write_text(json.dumps(res, indent=1, ensure_ascii=False, default=float))


def selfcheck():
    Wd = World(); V2.TRAIN = TRAIN; V2.selfcheck(Wd)   # P1(새 학습 구간)·P2·P3·P4
    d0 = Wd.span(TRAIN)[0] + 100; w = Wd.W["ETH"]; o = V2.Sleeve(w).obs(d0); win = window(w, d0, o)   # P5
    assert np.allclose(win[(W - 1) * NF:W * NF], w["F"][d0]) and np.allclose(win[:NF], w["F"][d0 - W + 1]) and win.shape == (W * NF + 3,)
    print("P5 창 마지막 행 = F[d] · 첫 행 = F[d−19]")
    rng = np.random.default_rng(0); xs = rng.normal(0.001, 0.02, 300); R = Rew("DSR"); A, B = 0.0, 1.0   # P6
    for x in xs:
        e1 = 1.0 + x * V2.UNIT; got = R.step(e1, 1.0, 0.0); X = (e1 - 1.0) / V2.UNIT
        dA, dB = X - A, X * X - B; want = float(np.clip((B * dA - 0.5 * A * dB) / max(B - A * A, 1e-6) ** 1.5, -10, 10)); assert abs(got - want) < 1e-12, (got, want)
        A += ETA * dA; B += ETA * dB
    R = Rew("DD"); eq = 1.0
    for x in np.abs(xs):
        e1 = eq * (1 + x); z = (e1 - eq) / eq / V2.UNIT; assert abs(R.step(e1, eq, 0.0) - z) < 1e-12; eq = e1
    print("P6 DSR = 직접 공식 300걸음 · 오르기만 하면 DD 벌점 0")
    tr = Wd.span(TRAIN, Wd.pool); mags = {}
    for kind in ("Z", "DSR", "DD"):   # 보상 크기 점검: 교사 경로 걸음당 |R| vs 엔트로피 항 0.001·ln3
        vals = []
        for w in Wd.pool[:30:3]:
            sl, Rw = V2.Sleeve(w), Rew(kind)
            for d in range(*tr):
                e1, e0, tn = sl.day(d, int(w["D"][d])); vals.append(abs(Rw.step(e1, e0, tn)))
        mags[kind] = float(np.mean(vals)); print(f"보상 {kind}: 걸음당 |R| {mags[kind]:.3f} · 엔트로피 항 대비 {mags[kind] / (0.001 * math.log(3)):.0f}배")
    (OUT / "selfcheck.json").write_text(json.dumps({"reward_mag": mags}, indent=1))
    print("SELFCHECK PASS")


def collect():
    from stable_baselines3 import PPO
    Wd = World(); va, te = Wd.span(VAL), Wd.span(TEST); runs = [json.loads(open(p).read()) for p in sorted(glob.glob(str(OUT / "c*_s*/run.json")))]
    assert len(runs) == 18 and all(len(r["ckpt"]) == 7 for r in runs), [(r["cell"], r["seed"], len(r["ckpt"])) for r in runs]
    cands = [(c["sharpe"], r["cell"], r["seed"], c["step"]) for r in runs for c in r["ckpt"]]; best = max(cands)
    load = lambda cell, seed, step: PPO.load(OUT / f"c{cell}_s{seed}" / f"{step}.zip", device="cpu")
    _, cell, seed, step = best; arch = CELLS[cell][1]; m = load(cell, seed, step); pk = picker(Wd, m, arch)
    r_va, _, lg_va = V2.run(Wd, va, pk); r_rl, rc_rl, lg_rl = V2.run(Wd, te, pk)
    r_t, _, lg_t = V2.run(Wd, te, V2.teacher(Wd)); r_p, _, lg_p = V2.run(Wd, te, V2.teacher(Wd), stop=False); r_l, _, lg_l = V2.run(Wd, te, lambda s, d, o: 2)
    tdays = Wd.days[te[0]:te[1]]; hold = pd.concat([SYS.daily_close(Wd.data[s]).pct_change() for s in V2.COINS], axis=1).mean(1).loc[tdays]
    nulls = [SYS.sharpe(V2.run(Wd, te, V2.markov(lg_va, sd))[0]) for sd in range(400)]
    per_cell = {}
    for c in sorted(CELLS):   # 보고용: 칸마다 자기 검증 최선
        cc = max(x for x in cands if x[1] == c); rr, _, lg = V2.run(Wd, te, picker(Wd, load(c, cc[2], cc[3]), CELLS[c][1]))
        per_cell[c] = {"reward": CELLS[c][0], "arch": CELLS[c][1], "seed": cc[2], "step": cc[3], "val_sharpe": cc[0], **V2.summarize(rr, lg)}
    eff = {}
    for key, idx in (("reward", 0), ("arch", 1)):   # 주효과: 학습(시드)마다 검증 최선 샤프 평균
        g = {}
        for r in runs:
            g.setdefault(CELLS[r["cell"]][idx], []).append(max(c["sharpe"] for c in r["ckpt"]))
        eff[key] = {k: float(np.mean(v)) for k, v in g.items()}
    diff = r_rl - r_t
    rep = {"prereg": "docs/prereg/rl_v3_reward_arch_20261025.md@73f325ad", "trials": len(cands),
           "chosen": {"cell": cell, "reward": CELLS[cell][0], "arch": arch, "seed": seed, "step": step, "val_sharpe": best[0]},
           "val_chosen": V2.summarize(r_va, lg_va), "main_effects_val": eff, "bc": {f"c{r['cell']}_s{r['seed']}": r["bc"]["train_agree"] for r in runs},
           "test": {"rl": V2.summarize(r_rl, lg_rl), "teacher": V2.summarize(r_t, lg_t), "plain_trend_nostop": V2.summarize(r_p, lg_p), "always_long_env": V2.summarize(r_l, lg_l),
                    "hold_1x": {"sharpe": SYS.sharpe(hold), "ret": float((1 + hold).prod() - 1), "mdd": SYS.mdd(hold)},
                    "rl_per_coin": {s: {"sharpe": SYS.sharpe(v), "ret": float((1 + v).prod() - 1)} for s, v in rc_rl.items()},
                    "rl_minus_teacher_ci": SYS.block_ci(diff), "rl_minus_teacher_mean_bp": float(diff.mean() * 1e4),
                    "random_p9944": float(np.percentile(nulls, 100 - 5 / 9)), "random_median": float(np.median(nulls))},
           "per_cell_test": per_cell}
    T = rep["test"]; rep["gates"] = {"G1": T["rl"]["sharpe"] > 0 and T["rl"]["mdd"] >= -0.20, "G2": T["rl_minus_teacher_ci"][0] > 0, "G3": T["rl"]["sharpe"] > T["random_p9944"]}
    rep["pass"] = all(rep["gates"].values()); (OUT / "report.json").write_text(json.dumps(rep, indent=1, ensure_ascii=False, default=float))
    print(json.dumps({k: rep[k] for k in ("chosen", "main_effects_val", "test", "gates", "pass")}, indent=1, ensure_ascii=False, default=float))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--selfcheck", action="store_true"); ap.add_argument("--cell", type=int); ap.add_argument("--seed", type=int)
    ap.add_argument("--collect", action="store_true"); a = ap.parse_args(); OUT.mkdir(parents=True, exist_ok=True)
    if a.selfcheck:
        return selfcheck()
    if a.collect:
        return collect()
    assert a.seed in CELLS[a.cell][2], "사전 등록된 시드만"
    return train_one(a.cell, a.seed)


if __name__ == "__main__":
    sys.exit(main())
