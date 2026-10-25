#!/usr/bin/env python3
"""RL v5 — 잔차 게이트: 교사(추세) 방향을 따를지(0)·쉴지(1)만 배우는 RL. 사전 등록 docs/prereg/rl_v5_residual_gate_20261025.md.

    python3 scripts/research_rl_v5_residual_gate_20261025.py --selfcheck W1      # S1 따르기 = 교사 · S2 쉬기 = 0
    python3 scripts/research_rl_v5_residual_gate_20261025.py --train W1 222970   # 창·시드 하나 학습(체크포인트 7개)
    python3 scripts/research_rl_v5_residual_gate_20261025.py --collect W1        # 검증 앙상블로 걸음 고르기 → 시험 1회
    python3 scripts/research_rl_v5_residual_gate_20261025.py --pooled            # 3창 합침 V1~V4

나머지(관측·보상 DD·TCN·PPO·30세계)는 research_rl_v3_reward_arch_20261025 그대로. 실주문 없음.
"""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
import gymnasium as gym
import torch as th

sys.path.insert(0, str(Path(__file__).resolve().parent))
import research_rl_v3_reward_arch_20261025 as R3   # noqa: E402
V2, SYS = R3.V2, R3.SYS

BASE = SYS.ROOT / "tmp" / "rl_v5_residual_20261025"
WINDOWS = {"W1": (("2022-06-01", "2023-12-31"), ("2024-01-01", "2024-05-31"), ("2024-06-01", "2024-10-23"), (222970, 213758, 627306, 409020, 596470)),
           "W2": (("2023-01-01", "2024-12-31"), ("2025-01-01", "2025-05-31"), ("2025-06-01", "2025-10-23"), (385252, 634164, 465028, 799203, 299605)),
           "W3": (("2024-01-01", "2025-12-31"), ("2026-01-01", "2026-05-31"), ("2026-06-01", "2026-10-23"), (779065, 423541, 911151, 397112, 521242))}


def set_window(w):
    R3.TRAIN, R3.VAL, R3.TEST, seeds = WINDOWS[w]; out = BASE / w; out.mkdir(parents=True, exist_ok=True)
    return out, seeds


def gate_dir(world, d, a):
    """게이트 0 = 교사 방향 D[d] · 1 = 없음."""
    return int(world["D"][d]) if int(a) == 0 else 0


class Env(R3.Env):
    action_space = gym.spaces.Discrete(2)

    def step(self, a):
        e1, e0, tn = self.sl.day(self.d, gate_dir(self.w, self.d, a)); r = self.R.step(e1, e0, tn); self.d += 1
        return self._o(), float(r), self.d >= self.dend or e1 <= 1e-3, False, {}


def gate_picker(Wd, models, log=None):
    """5시드 다수결 게이트 → V2.run 이 받는 방향 행동(0..2). log 에 게이트열을 남긴다."""
    def pk(s, d, o):
        x = R3.window(Wd.W[s], d, o); g = int(round(np.mean([int(m.predict(x, deterministic=True)[0]) for m in models])))   # 홀수 표 다수결(동률 없음)
        if log is not None:
            log.setdefault(s, []).append(g)
        return gate_dir(Wd.W[s], d, g) + 1
    return pk


def bc_follow(model, Wd, span):
    """정답 = 항상 0. 학습 일치 ≥ .99 첫 에폭에서 멈춤(상수 정답 로짓 포화 방지)."""
    X = []
    for w in Wd.pool:
        sl = V2.Sleeve(w)
        for d in range(*span):
            X.append(R3.window(w, d, sl.obs(d))); sl.day(d, int(w["D"][d]))
    X = th.tensor(np.array(X)); Y = th.zeros(len(X), dtype=th.long); opt = th.optim.Adam(model.policy.parameters(), lr=1e-3); g = th.Generator().manual_seed(0)
    for ep in range(200):
        perm = th.randperm(len(X), generator=g)
        for i in range(0, len(X), 256):
            b = perm[i:i + 256]; loss = th.nn.functional.cross_entropy(model.policy.get_distribution(X[b]).distribution.logits, Y[b])
            opt.zero_grad(); loss.backward(); opt.step()
        with th.no_grad():
            agree = float((model.policy.get_distribution(X).distribution.logits.argmax(1) == 0).float().mean())
        if agree >= 0.99:
            break
    return {"rows": len(X), "train_agree": agree, "epochs": ep + 1}


def train(w, seed):
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv
    out, seeds = set_window(w); assert seed in seeds, "사전 등록된 시드만"; th.set_num_threads(2); th.manual_seed(seed)
    Wd = R3.World(); tr, va = Wd.span(R3.TRAIN, Wd.pool), Wd.span(R3.VAL)
    venv = DummyVecEnv([lambda i=i: Env(Wd, tr, "DD", "tcn", seed * 10 + i) for i in range(8)])
    m = PPO("MlpPolicy", venv, policy_kwargs=R3.policy_kwargs("tcn"), seed=seed, verbose=0, device="cpu", **R3.PPO_KW)
    b = bc_follow(m, Wd, tr); assert b["train_agree"] >= 0.99, b
    od = out / f"s{seed}"; od.mkdir(parents=True, exist_ok=True); res = {"window": w, "seed": seed, "bc": b, "ckpt": []}
    for step in range(0, R3.TOTAL + 1, R3.CKPT):
        if step:
            m.learn(R3.CKPT, reset_num_timesteps=False)
        m.save(od / f"{step}.zip"); gl = {}; r, _, lg = V2.run(Wd, va, gate_picker(Wd, [m], gl)); sm = V2.summarize(r, lg)
        sm.update(step=step, rest_share=float(np.mean(np.concatenate([gl[s] for s in gl])))); res["ckpt"].append(sm)
        print(f"{w} s{seed} 걸음 {step:>6} 검증 샤프 {sm['sharpe']:+.2f} MDD {sm['mdd']:+.2f} 쉬기 {sm['rest_share']:.2f}", flush=True)
        (od / "run.json").write_text(json.dumps(res, indent=1, ensure_ascii=False, default=float))


def selfcheck(w):
    set_window(w); Wd = R3.World(); tr = Wd.span(R3.TRAIN)
    follow, _, _ = V2.run(Wd, tr, lambda s, d, o: gate_dir(Wd.W[s], d, 0) + 1); teach, _, _ = V2.run(Wd, tr, V2.teacher(Wd))
    assert np.max(np.abs(follow.values - teach.values)) < 1e-12; print(f"S1 {w} 따르기 = 교사 {len(follow)}일")
    rest, _, _ = V2.run(Wd, tr, lambda s, d, o: gate_dir(Wd.W[s], d, 1) + 1); assert np.max(np.abs(rest.values)) == 0.0; print("S2 쉬기 = 수익 0")


def markov_gate(gl, seed):
    T = np.ones((2, 2)) * 1e-9
    for v in gl.values():
        v = np.asarray(v); np.add.at(T, (v[:-1], v[1:]), 1)
    T /= T.sum(1, keepdims=True); rng = np.random.default_rng(seed); cur = {}

    def pk(s, d, o, Wd):
        cur[s] = int(rng.integers(2)) if s not in cur else int(rng.choice(2, p=T[cur[s]])); return gate_dir(Wd.W[s], d, cur[s]) + 1
    return pk


def collect(w):
    from stable_baselines3 import PPO
    out, seeds = set_window(w); Wd = R3.World(); va, te = Wd.span(R3.VAL), Wd.span(R3.TEST); rows = []
    for st in range(0, R3.TOTAL + 1, R3.CKPT):
        ms = [PPO.load(out / f"s{sd}" / f"{st}.zip", device="cpu") for sd in seeds]; gl = {}
        r, _, _ = V2.run(Wd, va, gate_picker(Wd, ms, gl)); rows.append((SYS.sharpe(r), st, ms, gl)); print(f"{w} 걸음 {st:>6} 검증 앙상블 샤프 {rows[-1][0]:+.2f}", flush=True)
    vs, st, ms, gl_va = max(rows, key=lambda z: z[0]); gl = {}
    r_e, _, lg_e = V2.run(Wd, te, gate_picker(Wd, ms, gl)); r_t, _, lg_t = V2.run(Wd, te, V2.teacher(Wd))
    nulls = np.array([V2.run(Wd, te, (lambda pk: lambda s, d, o: pk(s, d, o, Wd))(markov_gate(gl_va, sd)))[0].values for sd in range(400)])
    seed_sh = {sd: SYS.sharpe(V2.run(Wd, te, gate_picker(Wd, [m]))[0]) for sd, m in zip(seeds, ms)}
    rep = {"window": w, "step": st, "val_sharpe": vs, "val_grid": [(z[1], z[0]) for z in rows], "ens": V2.summarize(r_e, lg_e), "teacher": V2.summarize(r_t, lg_t),
           "rest_share_test": float(np.mean(np.concatenate([gl[s] for s in gl]))), "seed_test_sharpe": seed_sh}
    (out / "window_report.json").write_text(json.dumps(rep, indent=1, ensure_ascii=False, default=float))
    pd.DataFrame({"ens": r_e, "teacher": r_t}).to_parquet(out / "window_daily.parquet"); np.save(out / "window_nulls.npy", nulls)
    print(json.dumps({k: rep[k] for k in ("window", "step", "val_sharpe", "ens", "teacher", "rest_share_test", "seed_test_sharpe")}, indent=1, ensure_ascii=False, default=float))


def pooled():
    D = {w: pd.read_parquet(BASE / w / "window_daily.parquet") for w in WINDOWS}; N = {w: np.load(BASE / w / "window_nulls.npy") for w in WINDOWS}
    Rp = {w: json.loads((BASE / w / "window_report.json").read_text()) for w in WINDOWS}
    ens, tch = pd.concat([D[w].ens for w in WINDOWS]), pd.concat([D[w].teacher for w in WINDOWS]); diff = ens - tch
    nul = np.concatenate([N[w] for w in WINDOWS], axis=1); nsh = nul.mean(1) / nul.std(1) * np.sqrt(365)
    out = {"prereg": "docs/prereg/rl_v5_residual_gate_20261025.md@aae47224", "pooled_sharpe": SYS.sharpe(ens), "pooled_teacher_sharpe": SYS.sharpe(tch), "days": len(ens),
           "ens_minus_teacher_ci": SYS.block_ci(diff), "ens_minus_teacher_bp": float(diff.mean() * 1e4), "null_p95": float(np.percentile(nsh, 95)),
           "per_window": {w: {"ens": Rp[w]["ens"]["sharpe"], "teacher": Rp[w]["teacher"]["sharpe"], "ens_mdd": Rp[w]["ens"]["mdd"], "teacher_mdd": Rp[w]["teacher"]["mdd"],
                              "rest": Rp[w]["rest_share_test"], "step": Rp[w]["step"], "seeds_ge_teacher": sum(v >= Rp[w]["teacher"]["sharpe"] for v in Rp[w]["seed_test_sharpe"].values())} for w in WINDOWS}}
    pw = out["per_window"]
    out["gates"] = {"V1": out["ens_minus_teacher_ci"][0] > 0, "V2": out["pooled_sharpe"] > out["pooled_teacher_sharpe"] and sum(v["ens_mdd"] >= v["teacher_mdd"] for v in pw.values()) >= 2,
                    "V3": out["pooled_sharpe"] > out["null_p95"], "V4": sum(v["seeds_ge_teacher"] for v in pw.values()) >= 9}
    out["pass"] = all(out["gates"].values()); (BASE / "pooled_report.json").write_text(json.dumps(out, indent=1, ensure_ascii=False, default=float)); print(json.dumps(out, indent=1, ensure_ascii=False, default=float))


if __name__ == "__main__":
    a = sys.argv[1:]
    if "--selfcheck" in a:
        selfcheck(a[a.index("--selfcheck") + 1])
    elif "--train" in a:
        i = a.index("--train"); train(a[i + 1], int(a[i + 2]))
    elif "--collect" in a:
        collect(a[a.index("--collect") + 1])
    elif "--pooled" in a:
        pooled()
    else:
        print(__doc__)
