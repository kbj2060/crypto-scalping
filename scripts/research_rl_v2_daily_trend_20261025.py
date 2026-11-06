#!/usr/bin/env python3
"""RL v2 — 일봉 결정·추세 규칙 교사(BC) → PPO, 4코인 공유 정책. 사전 등록 docs/prereg/rl_v2_daily_trend_20261025.md.

    python3 scripts/research_rl_v2_daily_trend_20261025.py --selfcheck      # P1 env = SYS.simulate · P2 피쳐 인과성 · P3 미래 누수 트립와이어
    python3 scripts/research_rl_v2_daily_trend_20261025.py --version A      # 시드 3 × 체크포인트 → 2025 고르기 → 2026-01~07 시험 1회 · G1~G3

실주문 없음. 산출 tmp/rl_v2_daily_trend_20261025/<version>/.
"""
import argparse, json, math, pathlib, sys
import numpy as np, pandas as pd
import gymnasium as gym
import torch as th

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import research_system_trend_ruleb_20261025 as SYS   # noqa: E402

OUT = SYS.ROOT / "tmp" / "rl_v2_daily_trend_20261025"
COINS = SYS.COINS
TRAIN, VAL, TEST = ("2022-06-01", "2024-12-31"), ("2025-01-01", "2025-12-31"), ("2026-01-01", "2026-07-31")
CAP, TAU_C, STOP, TAKER, THRU, WIN, NB = 2.0, 0.25, SYS.STOP, SYS.TAKER, SYS.THRU, SYS.WIN, 288
UNIT = TAU_C / math.sqrt(365)          # 보상 단위 = 슬리브 목표 일 변동성
NMKT = 13
EXTRA = ("ADA", "ATOM", "AVAX", "BNB", "DOGE", "DOT", "FIL", "LINK", "LTC", "NEAR", "TRX")
VERSIONS = {
    "A": dict(seeds=(926327, 927731, 200792), total=600_000, ckpt=100_000, gamma=0.3, kappa=10e-4, ent=0.001, lr=1e-4, ep_days=365, pool="4"),
    "B": dict(seeds=(51513, 123516, 263847), total=600_000, ckpt=100_000, gamma=0.3, kappa=10e-4, ent=0.001, lr=1e-4, ep_days=365, pool="15m"),
}


def feats(cl):
    """일 d 결정 피쳐(일 d−1 종가까지). Zhang·Zohren·Roberts 2019 식: 변동성 정규화 수익·MACD 3쌍·RSI30 + 추세 부호 D."""
    lr = np.log(cl).diff(); sig = lr.ewm(span=60, min_periods=30).std(); F = {}
    for k in (1, 7, 14, 21, 28, 63):
        F[f"r{k}"] = np.log(cl / cl.shift(k)) / (sig * math.sqrt(k))
    for S, L in ((8, 24), (16, 48), (32, 96)):
        q = (cl.ewm(span=S).mean() - cl.ewm(span=L).mean()) / cl.rolling(63).std(); F[f"macd{S}"] = q / q.rolling(63).std()   # 원문 252 → 이력 63
    d = cl.diff(); up, dn = d.clip(lower=0).rolling(30).mean(), (-d.clip(upper=0)).rolling(30).mean(); F["rsi"] = up / (up + dn) - 0.5
    F["lsig"] = np.log(sig * math.sqrt(365)); F["vr"] = lr.rolling(20).std() / lr.rolling(60).std() - 1
    F["D"] = np.sign(sum(np.sign(cl / cl.shift(k) - 1) for k in (7, 14, 21, 28)) / 4)
    return pd.DataFrame(F).shift(1), (sig * math.sqrt(365)).shift(1)


def mkworld(x, fl):
    cl = SYS.daily_close(x); F, sg = feats(cl); h, l, c = (x[k].values.reshape(-1, NB) for k in ("h", "l", "c"))
    return dict(h=h, l=l, c=c, P0=cl.shift(1).values, sig=sg.values, D=F.D.fillna(0).values.astype(int), fund=fl, raw=F.values)


def mirror(x, fl):
    """거울(1/가격) 세계: 수익 부호가 뒤집힌 같은 경로. 고가↔저가, 펀딩 부호 반대. 학습 기간 표류(롱 편향)를 지운다."""
    return mkworld(pd.DataFrame({"h": 1 / x.l, "l": 1 / x.h, "c": 1 / x.c}, x.index), [tuple((k, -r) for k, r in f) for f in fl])


class World:
    """4코인 일 단위 배열: 5분 h/l/c (일×288) · P0(전날 종가) · 정규화 피쳐 · 연 변동성 · 교사 D · 펀딩. pool = 학습에 쓰는 세계 목록."""
    def __init__(self, pool="4"):
        data, _, inputs, fund = SYS.load_all(); self.data, self.inputs, self.fund = data, inputs, fund
        self.days = pd.date_range("2022-01-01", SYS.END, freq="D"); tr = (self.days >= TRAIN[0]) & (self.days <= TRAIN[1])
        self.W = {s: mkworld(data[s], [fund[s].get(d, ()) for d in self.days]) for s in COINS}
        self.pool = [self.W[s] for s in COINS]
        if pool == "15m":   # 4코인 + 이력 완전한 패널 11코인(펀딩 0, 학습 구간만) × 거울
            for s in EXTRA:
                p = pd.read_parquet(SYS.D / f"binance_vision/panel/{s}USDT.parquet", columns=["timestamp", "high", "low", "close"]).set_index("timestamp")
                x = p.rename(columns={"high": "h", "low": "l", "close": "c"}).reindex(pd.date_range("2022-01-01", "2024-12-31 23:55", freq="5min"))
                x["c"] = x.c.ffill(); x["h"] = x.h.fillna(x.c); x["l"] = x.l.fillna(x.c); fl = [()] * (len(x) // NB)
                self.pool += [mkworld(x, fl), mirror(x, fl)]
            self.pool += [mirror(data[s], self.W[s]["fund"]) for s in COINS]
        M = np.concatenate([w["raw"][:len(tr)][tr[:len(w["raw"])]] for w in self.pool]); self.mu, self.sd = np.nanmean(M, 0), np.nanstd(M, 0) + 1e-12
        for w in self.pool + list(self.W.values()):
            w["F"] = ((w["raw"] - self.mu) / self.sd).astype(np.float32)

    def span(self, a, ws=None):
        k0, k1 = int(self.days.searchsorted(pd.Timestamp(a[0]))), int(self.days.searchsorted(pd.Timestamp(a[1]), side="right"))
        for w in ws or self.W.values():
            assert not np.isnan(w["F"][k0:k1]).any() and not np.isnan(w["sig"][k0:k1]).any(), a
        return k0, k1


class Sleeve:
    """한 코인 슬리브(시작 순자산 1). 하루 = SYS.simulate(ladder 끔)와 같은 규칙: 전날 종가 지정가 1시간 → 아니면 테이커, −5% 첫 진입가 손절."""
    def __init__(self, w, stop=True):
        self.w, self.stop = w, stop; self.C, self.q, self.dir, self.stp, self.ent, self.held = 1.0, 0.0, 0, 0.0, 0.0, 0

    def eq(self, d):
        return self.C + self.q * self.w["c"][d, -1]

    def day(self, d, Dn):
        w = self.w; h, l, c, P0 = w["h"][d], w["l"][d], w["c"][d], w["P0"][d]; E0 = self.eq(d - 1)
        cap = min(CAP, TAU_C / w["sig"][d]) * E0; tn = 0.0
        exiting, entering, act = self.dir != 0 and Dn != self.dir, Dn != 0 and self.dir != Dn, -1
        for k in range(NB):
            o = c[k - 1] if k else P0
            for kk, rate in w["fund"][d]:
                if kk == k:
                    self.C -= self.q * o * rate
            if self.dir and k > act and self.stop:
                s = self.dir
                if (s > 0 and l[k] <= self.stp) or (s < 0 and h[k] >= self.stp):
                    px = min(o, self.stp) if s > 0 else max(o, self.stp); self.C += self.q * px - TAKER * abs(self.q) * px; tn += abs(self.q) * px
                    self.q, self.dir = 0.0, 0; entering = exiting = False; continue
            if exiting and k < WIN:
                hit = (h[k] >= P0 * (1 + THRU)) if self.q > 0 else (l[k] <= P0 * (1 - THRU))
                if hit or k == WIN - 1:
                    px = P0 if hit else c[k]; self.C += self.q * px - (0 if hit else TAKER * abs(self.q) * px); tn += abs(self.q) * px
                    self.q, self.dir, exiting = 0.0, 0, False
            if entering and not exiting and self.dir == 0 and k < WIN:
                hit = (l[k] <= P0 * (1 - THRU)) if Dn > 0 else (h[k] >= P0 * (1 + THRU))
                if hit or k == WIN - 1:
                    px = P0 if hit else c[k]; q1 = Dn * cap / px; self.C -= q1 * px + (0 if hit else TAKER * abs(q1) * px); tn += abs(q1) * px
                    self.q, self.dir, self.stp, self.ent, self.held = q1, Dn, px * (1 - Dn * STOP), px, 0; entering = False; act = k
        self.held += 1 if self.dir else 0
        return self.eq(d), E0, tn

    def obs(self, d):
        w = self.w; unr = self.dir * (w["P0"][d] / self.ent - 1) / (w["sig"][d] / math.sqrt(365) * math.sqrt(max(1, self.held))) if self.dir else 0.0
        return np.r_[w["F"][d], np.float32([self.dir, math.log1p(self.held), np.clip(unr, -5, 5)])].astype(np.float32)


class Env(gym.Env):
    """학습용: 무작위 코인·무작위 ep_days 창. 행동 Discrete(3) = 목표 방향 숏/없음/롱(Zhang 의 target orders)."""
    observation_space = gym.spaces.Box(-np.inf, np.inf, (NMKT + 3,), np.float32)
    action_space = gym.spaces.Discrete(3)

    def __init__(self, Wd, span, cfg, seed):
        self.Wd, self.span_, self.cfg, self.rng = Wd, span, cfg, np.random.default_rng(seed)

    def reset(self, seed=None, options=None):
        a, b = self.span_; w = self.Wd.pool[self.rng.integers(len(self.Wd.pool))]; n = self.cfg["ep_days"]
        self.d = int(self.rng.integers(a, max(a + 1, b - n))); self.dend = min(b, self.d + n); self.sl = Sleeve(w)
        return self.sl.obs(self.d), {}

    def step(self, a):
        e1, e0, tn = self.sl.day(self.d, int(a) - 1)
        r = ((e1 - e0) / e0 - self.cfg["kappa"] * tn / e0) / UNIT; self.d += 1; done = self.d >= self.dend or e1 <= 1e-3
        return self.sl.obs(min(self.d, self.dend - 1)), float(r), done, False, {}


def run(Wd, span, pick, stop=True):
    """pick(coin, d, obs) → 행동(0..2). 반환: 포트폴리오 일 수익률(4 슬리브 평균), 코인별 행동·교사·방향 기록."""
    a, b = span; rets, log = {}, {}
    for s in COINS:
        sl = Sleeve(Wd.W[s], stop); r, acts, pos = [], [], []
        for d in range(a, b):
            act = pick(s, d, sl.obs(d)); e1, e0, _ = sl.day(d, act - 1); r.append(e1 / e0 - 1); acts.append(act); pos.append(sl.dir)
        rets[s] = pd.Series(r, Wd.days[a:b]); log[s] = (np.array(acts), Wd.W[s]["D"][a:b] + 1, np.array(pos))
    return pd.concat(rets, axis=1).mean(1), rets, log


def summarize(r, log):
    acts = np.concatenate([v[0] for v in log.values()]); tch = np.concatenate([v[1] for v in log.values()])
    pos = np.concatenate([v[2] for v in log.values()]); ent = sum(int(np.sum((v[0][1:] != v[0][:-1]) & (v[0][1:] != 1))) for v in log.values())
    yrs = len(r) / 365
    return {"sharpe": SYS.sharpe(r), "ret": float((1 + r).prod() - 1), "mdd": SYS.mdd(r), "teacher_agree": float(np.mean(acts == tch)),
            "switches_per_coin_yr": ent / len(log) / yrs, "long_share_in_pos": float(np.mean(pos[pos != 0] > 0)) if (pos != 0).any() else None,
            "flat_share": float(np.mean(pos == 0)), "act_freq": np.bincount(acts, minlength=3).tolist()}


def markov(log, seed):
    """RL 검증 행동열의 3×3 전이로 무작위 행동(같은 보유 길이·비율, 시점만 무작위)."""
    T = np.ones((3, 3)) * 1e-9
    for v in log.values():
        np.add.at(T, (v[0][:-1], v[0][1:]), 1)
    T /= T.sum(1, keepdims=True); rng = np.random.default_rng(seed); cur = {s: int(rng.integers(3)) for s in COINS}

    def pick(s, d, o):
        a = cur[s]; cur[s] = int(rng.choice(3, p=T[a])); return a
    return pick


def teacher(Wd):
    return lambda s, d, o: int(Wd.W[s]["D"][d]) + 1


def bc_pretrain(model, Wd, span):
    X, Y = [], []
    for w in Wd.pool:
        sl = Sleeve(w)
        for d in range(*span):
            a = int(w["D"][d]) + 1; X.append(sl.obs(d)); Y.append(a); sl.day(d, a - 1)
    X, Y = th.tensor(np.array(X)), th.tensor(np.array(Y)); opt = th.optim.Adam(model.policy.parameters(), lr=1e-3); g = th.Generator().manual_seed(0)
    for _ in range(200):
        perm = th.randperm(len(X), generator=g)
        for i in range(0, len(X), 256):
            b = perm[i:i + 256]; loss = th.nn.functional.cross_entropy(model.policy.get_distribution(X[b]).distribution.logits, Y[b])
            opt.zero_grad(); loss.backward(); opt.step()
    with th.no_grad():
        agree = float((model.policy.get_distribution(X).distribution.logits.argmax(1) == Y).float().mean())
    return {"rows": len(X), "train_agree": agree}


def selfcheck(Wd):
    va = Wd.span(TRAIN); days = Wd.days[va[0]:va[1]]; rng = np.random.default_rng(5)   # 학습 구간에서만
    for s in COINS:   # P1: 교사·무작위 행동을 env 슬리브와 SYS.simulate(독립 구현, lvl 로 같은 크기)에 넣어 순자산 1e-9
        w = Wd.W[s]
        for name, acts in (("teacher", w["D"][va[0]:va[1]]), ("random", rng.integers(-1, 2, va[1] - va[0]))):
            sl = Sleeve(w); eq = [sl.day(d, int(acts[i]))[0] for i, d in enumerate(range(*va))]
            inp = Wd.inputs[s].copy(); inp["lvl"] = 2.0 / np.minimum(CAP, TAU_C / pd.Series(w["sig"], Wd.days))
            rs = SYS.simulate({s: Wd.data[s]}, {s: inp}, Wd.fund, days, 1, Dover={s: pd.Series(acts, days)}, ladder=False)
            ref = float((1 + rs).prod()); print(f"P1 {s} {name}: env {eq[-1]:.12f} · SYS {ref:.12f}"); assert abs(eq[-1] - ref) < 1e-9
        assert (pd.Series(w["D"], Wd.days).loc[days].values == Wd.inputs[s].D.loc[days].fillna(0).values).all()   # 교사 D = SYS 추세 D
    for s in COINS:   # P2: 일 d 피쳐 = 일 d 종가를 망가뜨리고 그 뒤를 자른 데이터로 다시 계산한 값
        cl = SYS.daily_close(Wd.data[s]); full, fs = feats(cl)
        for d in rng.integers(150, len(cl) - 1, 50):
            cut = cl.iloc[:d + 1].copy(); cut.iloc[-1] *= 7.0; F, sg = feats(cut)
            assert np.allclose(F.iloc[-1].values, full.iloc[d].values, rtol=1e-9, equal_nan=True) and np.isclose(sg.iloc[-1], fs.iloc[d]), (s, d)
    print("P2 피쳐 인과성 4코인 × 50 표본")
    fut = lambda s, d, o: 1 + int(np.sign(Wd.W[s]["c"][d, -1] / Wd.W[s]["P0"][d] - 1))   # P3: 일 d 자신의 수익 부호(미래)
    sh = SYS.sharpe(run(Wd, va, fut)[0]); print(f"P3 미래 누수판 샤프 {sh:.1f} (> 5 이어야)"); assert sh > 5
    tr = Wd.span(TRAIN, Wd.pool); m = Wd.pool[-4]   # P4: 거울 ETH = 교사 부호 반대 · 가격 역수 · 학습 세계 수
    e = Wd.W["ETH"]; assert np.array_equal(m["D"][tr[0]:tr[1]], -e["D"][tr[0]:tr[1]]) and np.allclose(m["c"] * e["c"], 1.0)
    print(f"P4 학습 세계 {len(Wd.pool)}개 · 거울 ETH 교사 = −교사"); assert len(Wd.pool) == 30
    print("SELFCHECK PASS")


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--selfcheck", action="store_true"); ap.add_argument("--version"); a = ap.parse_args()
    th.set_num_threads(2); Wd = World(VERSIONS[a.version]["pool"] if a.version else "15m"); tr, va, te = Wd.span(TRAIN, Wd.pool), Wd.span(VAL), Wd.span(TEST)
    print("일 수 학습/검증/시험", tr[1] - tr[0], va[1] - va[0], te[1] - te[0], flush=True)
    if a.selfcheck:
        return selfcheck(Wd)
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv
    cfg = VERSIONS[a.version]; out = OUT / a.version; out.mkdir(parents=True, exist_ok=True)
    rep = {"prereg": "docs/prereg/rl_v2_daily_trend_20261025.md", "version": a.version, "cfg": cfg, "runs": []}
    pol = lambda m: (lambda s, d, o: int(m.predict(o, deterministic=True)[0]))
    best = (-1e9, None)
    for seed in cfg["seeds"]:
        th.manual_seed(seed)
        venv = DummyVecEnv([lambda i=i: Env(Wd, tr, cfg, seed * 10 + i) for i in range(8)])
        m = PPO("MlpPolicy", venv, learning_rate=cfg["lr"], n_steps=256, batch_size=256, n_epochs=10, gamma=cfg["gamma"], gae_lambda=0.95, clip_range=0.2,
                ent_coef=cfg["ent"], policy_kwargs=dict(net_arch=dict(pi=[64, 64], vf=[64, 64]), activation_fn=th.nn.Tanh), seed=seed, verbose=0, device="cpu")
        run_ = {"seed": seed, "bc": bc_pretrain(m, Wd, tr), "ckpt": []}
        for step in range(0, cfg["total"] + 1, cfg["ckpt"]):
            if step:
                m.learn(cfg["ckpt"], reset_num_timesteps=False)
            p = out / f"s{seed}_{step}.zip"; m.save(p); r, _, lg = run(Wd, va, pol(m)); sm = summarize(r, lg); sm["step"] = step; run_["ckpt"].append(sm)
            print(f"seed {seed} step {step:>7} 검증 샤프 {sm['sharpe']:+.2f} MDD {sm['mdd']:+.2f} 교사 일치 {sm['teacher_agree']:.2f} 전환/코인·년 {sm['switches_per_coin_yr']:.1f}", flush=True)
            if sm["sharpe"] > best[0]:
                best = (sm["sharpe"], p)
            (out / "progress.json").write_text(json.dumps(rep | {"runs": rep["runs"] + [run_]}, indent=1, ensure_ascii=False, default=float))
        rep["runs"].append(run_)
    rep["chosen"] = best[1].name; m = PPO.load(best[1], device="cpu")
    r_va, _, lg_va = run(Wd, va, pol(m))
    r_rl, rc_rl, lg_rl = run(Wd, te, pol(m)); r_t, _, lg_t = run(Wd, te, teacher(Wd)); r_p, _, lg_p = run(Wd, te, teacher(Wd), stop=False)
    r_l, _, lg_l = run(Wd, te, lambda s, d, o: 2)
    tdays = Wd.days[te[0]:te[1]]; hold = pd.concat([SYS.daily_close(Wd.data[s]).pct_change() for s in COINS], axis=1).mean(1).loc[tdays]
    nulls = [SYS.sharpe(run(Wd, te, markov(lg_va, sd))[0]) for sd in range(200)]
    seeds_test = {}
    for run_ in rep["runs"]:   # 보고용: 시드별 자기 검증 최선 체크포인트의 시험(관문에는 고른 1개만)
        k = int(np.argmax([c["sharpe"] for c in run_["ckpt"]])); st = run_["ckpt"][k]["step"]
        rr, _, lg = run(Wd, te, pol(PPO.load(out / f"s{run_['seed']}_{st}.zip", device="cpu"))); seeds_test[run_["seed"]] = {"step": st, **summarize(rr, lg)}
    diff = r_rl - r_t
    rep["test"] = {"rl": summarize(r_rl, lg_rl), "teacher": summarize(r_t, lg_t), "plain_trend_nostop": summarize(r_p, lg_p), "always_long_env": summarize(r_l, lg_l),
                   "hold_1x": {"sharpe": SYS.sharpe(hold), "ret": float((1 + hold).prod() - 1), "mdd": SYS.mdd(hold)},
                   "rl_per_coin": {s: {"sharpe": SYS.sharpe(v), "ret": float((1 + v).prod() - 1)} for s, v in rc_rl.items()},
                   "rl_minus_teacher_ci": SYS.block_ci(diff), "rl_minus_teacher_mean_bp": float(diff.mean() * 1e4),
                   "random_p9833": float(np.percentile(nulls, 100 - 5 / 3)), "random_median": float(np.median(nulls)), "seeds_best_own": seeds_test}
    T = rep["test"]; rep["gates"] = {"G1": T["rl"]["sharpe"] > 0 and T["rl"]["mdd"] >= -0.20, "G2": T["rl_minus_teacher_ci"][0] > 0, "G3": T["rl"]["sharpe"] > T["random_p9833"]}
    rep["pass"] = all(rep["gates"].values())
    (out / "report.json").write_text(json.dumps(rep, indent=1, ensure_ascii=False, default=float))
    print(json.dumps({k: rep[k] for k in ("chosen", "test", "gates", "pass")}, indent=1, ensure_ascii=False, default=float))


if __name__ == "__main__":
    sys.exit(main())
