#!/usr/bin/env python3
"""RL 2단계 — 우리 규칙(교사)으로 시작해 자기 전략을 만드는 딥러닝 RL 에이전트. 사전 등록 docs/prereg/rl_bc_ppo_20261025.md (44e61a5b).

    python3 scripts/research_rl_bc_ppo_20261025.py --selfcheck   # E1 교사 = 독립 루프 · E2 롱 보유 닫힌 식 · E3 피쳐 인과성
    python3 scripts/research_rl_bc_ppo_20261025.py               # BC → PPO 300만 걸음 → 검증 고르기 → 시험 1회 · R1~R3

실주문 없음. 산출 tmp/rl_bc_ppo_20261025/.
"""
import argparse, json, math, pathlib, sys
import numpy as np, pandas as pd
import gymnasium as gym
import torch as th

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import research_system_trend_ruleb_20261025 as SYS   # noqa: E402
import research_user_clone_ml_20261025 as CL          # noqa: E402
from research_rl_info_ceiling_20261025 import FEATS   # noqa: E402

OUT = SYS.ROOT / "tmp" / "rl_bc_ppo_20261025"
TRAIN = ("2022-01-08", "2024-12-31 23:00"); VAL = ("2025-01-01", "2025-12-31 23:00"); TEST = ("2026-01-01", "2026-07-31 23:00")
CAP, ADD, STOP, TAKER, THRU, HB, MAXB = 2.0, 0.0075, 0.05, 4e-4, 1e-4, 12, 2016
TAU = 0.065
WIN_STEPS, TOTAL, CKPT, NENV = 90 * 24, 3_000_000, 250_000, 8


class Data:
    """ETH 5분 배열 + 정시 결정 봉 + 정규화된 시장 피쳐 + 교사 신호(복제 A)."""
    def __init__(self):
        x = SYS.bars5("ETH")[0]; f = CL.mkt(x); self.x = x
        self.c, self.h, self.l = x.c.values, x.h.values, x.l.values; self.r1h, self.rv = f.r1h.values, f.rv24h.values
        self.fund = CL.fund_idx(x, "ETH")
        self.dec = np.flatnonzero((x.index + pd.Timedelta("5min")).minute == 0); self.dec = self.dec[self.dec + HB < len(x)]
        self.t = x.index[self.dec] + pd.Timedelta("5min")
        M = f[FEATS].replace([np.inf, -np.inf], np.nan).values[self.dec]
        tr = (self.t >= TRAIN[0]) & (self.t <= TRAIN[1]); mu, sd = np.nanmean(M[tr], 0), np.nanstd(M[tr], 0) + 1e-12
        self.M = np.nan_to_num((M - mu) / sd).astype(np.float32)
        L = SYS.ledger_trips(); self.ex = CL.Logit(*CL.exit_rows(x, f, L)); _, _, side = CL.fit_entry(x, f, L, CL.FEAT_A)
        self.sig = side[self.dec]

    def span(self, a):
        return int(np.searchsorted(self.t, pd.Timestamp(a[0]))), int(np.searchsorted(self.t, pd.Timestamp(a[1]), side="right"))


class TradeEnv(gym.Env):
    """정시 결정 · 사이 12개 5분봉에서 지정가 체결·손절·펀딩. 보상 = log(순자산 변화)."""
    observation_space = gym.spaces.Box(-np.inf, np.inf, (23,), np.float32)
    action_space = gym.spaces.Discrete(5)

    def __init__(self, D, span, random_window=False, fees=True, funding=True, guard=True, seed=0):
        self.D, self.span_, self.rw, self.fees, self.funding, self.guard = D, span, random_window, fees, funding, guard
        self.rng = np.random.default_rng(seed)

    def reset(self, seed=None, options=None):
        a, b = self.span_
        self.k = int(self.rng.integers(a, max(a + 1, b - WIN_STEPS))) if self.rw else a
        self.kend = min(b, self.k + WIN_STEPS) if self.rw else b
        self.C, self.q, self.pos, self.added, self.P0, self.capE = 1.0, 0.0, 0, True, 0.0, 0.0
        self.mfe = self.mae = 0.0; self.fillb = 0; self.lastflat = self.D.dec[self.k]; self.trades = []; self.eq = []
        return self._obs(), {}

    def _eq(self, i):
        return self.C + self.q * self.D.c[i]

    def _obs(self):
        D = self.D; k = min(self.k, len(D.dec) - 1); i = D.dec[k]
        unr, hold = (self.pos * (D.c[i] / self.P0 - 1), (i - self.fillb) * 5 / 60) if self.pos else (0.0, 0.0)
        pstate = [self.pos, (0 if not self.pos else (1.0 if self.added else 1 / 3)), unr, self.mfe, self.mae, math.log1p(hold), min(48.0, (i - self.lastflat) * 5 / 60)]
        return np.r_[D.M[k], np.array(pstate, np.float32)].astype(np.float32)

    def _close(self, px, i, why):
        self.C += self.q * px - (TAKER * abs(self.q) * px if self.fees else 0.0)
        self.trades.append((self.fillb, i, self.pos, why)); self.q, self.pos, self.added, self.lastflat = 0.0, 0, True, i

    def step(self, a):
        D = self.D; i = D.dec[self.k]; e0 = self._eq(i); pend = None
        if a == 3 and self.pos:
            self._close(D.c[i], i, "agent")
        elif a in (1, 2):
            s = 1 if a == 1 else -1
            if self.pos == -s:
                self._close(D.c[i], i, "flip")
            if not self.pos:
                pend = (s, D.c[i])
        elif a == 4 and self.pos and not self.added:
            dq = self.pos * (2 * self.capE / 3) / D.c[i]; self.C -= dq * D.c[i] + (TAKER * abs(dq) * D.c[i] if self.fees else 0.0)
            self.q += dq; self.added = True
        for j in range(i + 1, i + HB + 1):
            o = D.c[j - 1]
            if self.funding:
                for rate in D.fund.get(j, ()):
                    self.C -= self.q * o * rate
            if pend is not None:
                s, P = pend; hit = (D.l[j] <= P * (1 - THRU)) if s > 0 else (D.h[j] >= P * (1 + THRU))
                if hit or j == i + HB:
                    px = P if hit else D.c[j]; self.capE = CAP * self.C; q1 = s * self.capE / 3 / px
                    self.C -= q1 * px + (0.0 if (hit or not self.fees) else TAKER * abs(q1) * px)
                    self.q, self.pos, self.added, self.P0, self.fillb, self.mfe, self.mae = q1, s, False, px, j, 0.0, 0.0; pend = None
                continue
            if self.pos and self.guard:
                stp = self.P0 * (1 - self.pos * STOP)
                if (self.pos > 0 and D.l[j] <= stp) or (self.pos < 0 and D.h[j] >= stp):
                    self._close(min(o, stp) if self.pos > 0 else max(o, stp), j, "stop"); continue
                if j - self.fillb >= MAXB:
                    self._close(D.c[j], j, "time"); continue
            if self.pos:
                u = self.pos * (D.c[j] / self.P0 - 1); self.mfe, self.mae = max(self.mfe, u), min(self.mae, u)
        e1 = max(self._eq(i + HB), 1e-6); self.eq.append((D.t[self.k] + pd.Timedelta(hours=1), e1))
        self.k += 1; done = self.k >= self.kend or e1 <= 1e-3
        return self._obs(), float(math.log(e1 / max(e0, 1e-6))), done, False, {}

    def teacher(self):
        """우리 규칙: 비었으면 복제 진입 신호 · 들고 있으면 −0.75% 지나면 물타기, 청산 모델 ≥ τ 면 청산."""
        D, i = self.D, self.D.dec[self.k]
        if not self.pos:
            return {1: 1, -1: 2}.get(int(D.sig[self.k]), 0)
        if not self.added and self.pos * (D.c[i] / self.P0 - 1) <= -ADD:
            return 4
        p = D.ex.p(CL.pos_feats(self.P0, self.pos, D.c[i], self.mfe, self.mae, max(1, i - self.fillb), self.pos * D.r1h[i], D.rv[i])[None])[0]
        return 3 if p >= TAU else 0


def rollout(D, span, policy=None, seed=0, freq=None, **kw):
    """policy: None = 교사 · 'random' = freq 빈도 무작위 · 아니면 SB3 모델(결정적). 반환 (일 수익률, env, 행동, 교사 행동)."""
    env = TradeEnv(D, span, seed=seed, **kw); obs, _ = env.reset(); acts, tacts = [], []; rng = np.random.default_rng(seed); done = False
    while not done:
        ta = env.teacher(); tacts.append(ta)
        if policy is None: a = ta
        elif isinstance(policy, str): a = int(rng.choice(5, p=freq))
        else: a = int(policy.predict(obs, deterministic=True)[0])
        acts.append(a); obs, _, done, _, _ = env.step(a)
    eq = pd.Series(dict(env.eq)); day = eq.groupby(eq.index.normalize()).last()
    return day.pct_change().fillna(day.iloc[0] - 1.0), env, np.array(acts), np.array(tacts)


def summarize(r, env, acts, tacts):
    st = CL.stats(r); hold = [(t[1] - t[0]) * 5 / 60 for t in env.trades]
    return {**st, "trades": len(env.trades), "long_share": float(np.mean([t[2] > 0 for t in env.trades])) if env.trades else None,
            "hold_med_h": float(np.median(hold)) if hold else None, "teacher_agree": float(np.mean(acts == tacts)),
            "act_freq": np.bincount(acts, minlength=5).tolist(), "exit_why": pd.Series([t[3] for t in env.trades]).value_counts().to_dict()}


def teacher_loop(D, span):
    """E1 독립 재구현(교사만, env 없이 같은 규칙을 다시 쓴 단순 루프)."""
    a, b = span; C, q, pos, added, P0, capE, mfe, mae, fillb = 1.0, 0.0, 0, True, 0.0, 0.0, 0.0, 0.0, 0; n = 0
    for k in range(a, b):
        i = D.dec[k]; pend = None
        if not pos:
            s = int(D.sig[k]); pend = (s, D.c[i]) if s else None
        elif not added and pos * (D.c[i] / P0 - 1) <= -ADD:
            dq = pos * (2 * capE / 3) / D.c[i]; C -= dq * D.c[i] + TAKER * abs(dq) * D.c[i]; q += dq; added = True
        elif D.ex.p(CL.pos_feats(P0, pos, D.c[i], mfe, mae, max(1, i - fillb), pos * D.r1h[i], D.rv[i])[None])[0] >= TAU:
            C += q * D.c[i] - TAKER * abs(q) * D.c[i]; q, pos, added = 0.0, 0, True; n += 1
        for j in range(i + 1, i + HB + 1):
            o = D.c[j - 1]
            for rate in D.fund.get(j, ()):
                C -= q * o * rate
            if pend is not None:
                s, P = pend; hit = (D.l[j] <= P * (1 - THRU)) if s > 0 else (D.h[j] >= P * (1 + THRU))
                if hit or j == i + HB:
                    px = P if hit else D.c[j]; capE = CAP * C; q1 = s * capE / 3 / px; C -= q1 * px + (0 if hit else TAKER * abs(q1) * px)
                    q, pos, added, P0, fillb, mfe, mae = q1, s, False, px, j, 0.0, 0.0; pend = None
                continue
            if pos:
                stp = P0 * (1 - pos * STOP)
                if (pos > 0 and D.l[j] <= stp) or (pos < 0 and D.h[j] >= stp):
                    px = min(o, stp) if pos > 0 else max(o, stp); C += q * px - TAKER * abs(q) * px; q, pos, added = 0.0, 0, True; n += 1; continue
                if j - fillb >= MAXB:
                    C += q * D.c[j] - TAKER * abs(q) * D.c[j]; q, pos, added = 0.0, 0, True; n += 1; continue
                u = pos * (D.c[j] / P0 - 1); mfe, mae = max(mfe, u), min(mae, u)
    return n, C + q * D.c[D.dec[b - 1] + HB]


def bc_pretrain(model, D, span):
    """교사 (관측, 행동)으로 정책망을 먼저 가르친다(클래스 역빈도 가중, 상한 20)."""
    env = TradeEnv(D, span); obs, _ = env.reset(); X, Y = [], []; done = False
    while not done:
        a = env.teacher(); X.append(obs); Y.append(a); obs, _, done, _, _ = env.step(a)
    X, Y = th.tensor(np.array(X)), th.tensor(np.array(Y)); cnt = np.bincount(Y.numpy(), minlength=5).astype(float)
    w = th.tensor(np.minimum(20.0, cnt.sum() / np.maximum(cnt, 1) / 5), dtype=th.float32)
    opt = th.optim.Adam(model.policy.parameters(), lr=1e-3); g = th.Generator().manual_seed(0)
    for _ in range(30):
        perm = th.randperm(len(X), generator=g)
        for s in range(0, len(X), 1024):
            b = perm[s:s + 1024]; logits = model.policy.get_distribution(X[b]).distribution.logits
            loss = th.nn.functional.cross_entropy(logits, Y[b], weight=w); opt.zero_grad(); loss.backward(); opt.step()
    with th.no_grad():
        pred = model.policy.get_distribution(X).distribution.logits.argmax(1)
    return {"bc_rows": len(X), "teacher_act_freq": cnt.astype(int).tolist(), "bc_train_agree": float((pred == Y).float().mean()),
            "bc_recall_by_action": [float(((pred == k) & (Y == k)).sum() / max(1, (Y == k).sum())) for k in range(5)]}


def selfcheck(D, va):
    r, env, _, _ = rollout(D, va); n2, e2 = teacher_loop(D, va)
    e1 = env._eq(D.dec[env.k - 1] + HB); print(f"E1 교사 env 거래 {len(env.trades)} 순자산 {e1:.12f} · 독립 루프 {n2} {e2:.12f}")
    assert len(env.trades) == n2 and abs(e1 - e2) < 1e-9
    env = TradeEnv(D, va, fees=False, funding=False, guard=False); env.reset(); done = False; n = 0
    while not done:
        _, _, done, _, _ = env.step(1 if n == 0 else (4 if n == 1 else 0)); n += 1
    k0 = va[0]; i0, i1 = D.dec[k0], D.dec[k0 + 1]; P = D.c[i0]   # 닫힌 식: 첫 ⅓ 지정가(관통 시 P, 아니면 1시간 끝 종가) + 다음 정시 ⅔ 시장가
    hit = [j for j in range(i0 + 1, i0 + HB + 1) if D.l[j] <= P * (1 - THRU)]; p1 = P if hit else D.c[i0 + HB]; q1 = CAP / 3 / p1
    p2 = D.c[i1]; q2 = (2 * CAP / 3) / p2; cT = D.c[D.dec[va[1] - 1] + HB]
    want = 1 + q1 * (cT - p1) + q2 * (cT - p2); got = env._eq(D.dec[va[1] - 1] + HB)
    print(f"E2 항상 롱(수수료·펀딩·가드 끔) 순자산 env {got:.12f} · 닫힌 식 {want:.12f}"); assert abs(got - want) < 1e-9 and not env.trades
    rng = np.random.default_rng(3); full = CL.mkt(D.x)
    for i in rng.integers(10000, len(D.x) - 1, 100):
        cut = CL.mkt(D.x.iloc[i - 9000:i + 1]).iloc[-1]
        assert np.allclose(cut[FEATS].values.astype(float), full.iloc[i][FEATS].values.astype(float), rtol=1e-9, equal_nan=True), i
    print("E3 피쳐 인과성 100 표본"); print("SELFCHECK PASS")


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--selfcheck", action="store_true"); a = ap.parse_args(); OUT.mkdir(parents=True, exist_ok=True)
    th.manual_seed(0); D = Data(); tr, va, te = D.span(TRAIN), D.span(VAL), D.span(TEST)
    print("결정 걸음 학습/검증/시험", tr[1] - tr[0], va[1] - va[0], te[1] - te[0], flush=True)
    if a.selfcheck:
        return selfcheck(D, va)
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv
    venv = DummyVecEnv([lambda s=s: TradeEnv(D, tr, random_window=True, seed=s) for s in range(NENV)])
    model = PPO("MlpPolicy", venv, learning_rate=1e-4, n_steps=4096 // NENV, batch_size=512, n_epochs=5, gamma=0.995, gae_lambda=0.95, clip_range=0.1,
                ent_coef=0.001, policy_kwargs=dict(net_arch=dict(pi=[128, 128], vf=[128, 128]), activation_fn=th.nn.Tanh), seed=0, verbose=0, device="cpu")
    rep = {"prereg": "docs/prereg/rl_bc_ppo_20261025.md@44e61a5b", "bc": bc_pretrain(model, D, tr), "ckpt": []}
    print("BC", json.dumps(rep["bc"]), flush=True)
    rep["teacher_val"] = summarize(*rollout(D, va)); print("교사 검증", json.dumps({k: rep["teacher_val"][k] for k in ("sharpe", "mdd", "trades")}), flush=True)
    best = (-1e9, None)
    for step in range(0, TOTAL + 1, CKPT):
        if step:
            model.learn(CKPT, reset_num_timesteps=False)
        path = OUT / f"ckpt_{step}.zip"; model.save(path); s = summarize(*rollout(D, va, model)); s["step"] = step; rep["ckpt"].append(s)
        print(f"step {step:>8} 검증 샤프 {s['sharpe']:+.2f} MDD {s['mdd']:+.2f} 거래 {s['trades']} 교사 일치 {s['teacher_agree']:.2f} 행동 {s['act_freq']}", flush=True)
        (OUT / "progress.json").write_text(json.dumps(rep, indent=1, ensure_ascii=False, default=float))
        if s["sharpe"] > best[0]:
            best = (s["sharpe"], path)
    rep["chosen"] = best[1].name; m = PPO.load(best[1], device="cpu")
    r_rl, env_rl, a_rl, t_rl = rollout(D, te, m); r_t, env_t, a_t, t_t = rollout(D, te)
    va_acts = rollout(D, va, m)[2]; freq = np.bincount(va_acts, minlength=5) / len(va_acts)
    nulls = [CL.stats(rollout(D, te, "random", seed=sd, freq=freq)[0])["sharpe"] for sd in range(50)]
    diff = (r_rl - r_t.reindex(r_rl.index).fillna(0)).dropna()
    rep["test"] = {"rl": summarize(r_rl, env_rl, a_rl, t_rl), "teacher": summarize(r_t, env_t, a_t, t_t), "rl_minus_teacher_ci": SYS.block_ci(diff),
                   "random_p95": float(np.percentile(nulls, 95)), "random_median": float(np.median(nulls)), "val_act_freq": freq.tolist()}
    T = rep["test"]; rep["gates"] = {"R1": T["rl"]["sharpe"] > 0 and T["rl"]["mdd"] >= -0.20, "R2": T["rl_minus_teacher_ci"][0] > 0, "R3": T["rl"]["sharpe"] > T["random_p95"]}
    rep["pass"] = all(rep["gates"].values())
    (OUT / "report.json").write_text(json.dumps(rep, indent=1, ensure_ascii=False, default=float))
    print(json.dumps({k: rep[k] for k in ("chosen", "test", "gates", "pass")}, indent=1, ensure_ascii=False, default=float))


if __name__ == "__main__":
    sys.exit(main())
