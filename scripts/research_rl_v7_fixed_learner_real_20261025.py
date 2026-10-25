#!/usr/bin/env python3
"""RL v7 — v6 장치 검증으로 고친 장치(BC 없음 · Z 보상 · 검증 조기 멈춤)를 실데이터 굴림 3창에 1회.
사전 등록 docs/prereg/rl_v7_fixed_learner_real_20261025.md.

    python3 scripts/research_rl_v7_fixed_learner_real_20261025.py --train W1 764717
    python3 scripts/research_rl_v7_fixed_learner_real_20261025.py --collect W1
    python3 scripts/research_rl_v7_fixed_learner_real_20261025.py --pooled

나머지는 research_rl_v3_reward_arch_20261025(v3.1) 그대로. 실주문 없음.
"""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
import torch as th

sys.path.insert(0, str(Path(__file__).resolve().parent))
import research_rl_v3_reward_arch_20261025 as R3   # noqa: E402
V2, SYS = R3.V2, R3.SYS

BASE = SYS.ROOT / "tmp" / "rl_v7_fixed_20261025"
WINDOWS = {"W1": (("2022-06-01", "2023-12-31"), ("2024-01-01", "2024-05-31"), ("2024-06-01", "2024-10-23"), (764717, 299354, 936204, 927367, 128266)),
           "W2": (("2023-01-01", "2024-12-31"), ("2025-01-01", "2025-05-31"), ("2025-06-01", "2025-10-23"), (288931, 454084, 510346, 650879, 600505)),
           "W3": (("2024-01-01", "2025-12-31"), ("2026-01-01", "2026-05-31"), ("2026-06-01", "2026-10-23"), (852983, 775630, 494870, 835715, 150614))}


def set_window(w):
    R3.TRAIN, R3.VAL, R3.TEST, seeds = WINDOWS[w]; out = BASE / w; out.mkdir(parents=True, exist_ok=True)
    return out, seeds


def train(w, seed):
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv
    out, seeds = set_window(w); assert seed in seeds, "사전 등록된 시드만"; th.set_num_threads(2); th.manual_seed(seed)
    Wd = R3.World(); tr, va = Wd.span(R3.TRAIN, Wd.pool), Wd.span(R3.VAL)
    venv = DummyVecEnv([lambda i=i: R3.Env(Wd, tr, "Z", "tcn", seed * 10 + i) for i in range(8)])
    m = PPO("MlpPolicy", venv, policy_kwargs=R3.policy_kwargs("tcn"), seed=seed, verbose=0, device="cpu", **R3.PPO_KW)   # BC 없음
    od = out / f"s{seed}"; od.mkdir(parents=True, exist_ok=True); res = {"window": w, "seed": seed, "ckpt": []}
    for step in range(0, R3.TOTAL + 1, R3.CKPT):
        if step:
            m.learn(R3.CKPT, reset_num_timesteps=False)
        m.save(od / f"{step}.zip"); r, _, lg = V2.run(Wd, va, R3.picker(Wd, m, "tcn")); sm = V2.summarize(r, lg); sm["step"] = step; res["ckpt"].append(sm)
        print(f"{w} s{seed} 걸음 {step:>6} 검증 샤프 {sm['sharpe']:+.2f} MDD {sm['mdd']:+.2f} 교사 일치 {sm['teacher_agree']:.2f} 현금 {sm['flat_share']:.2f} 롱 {sm['long_share_in_pos']}", flush=True)
        (od / "run.json").write_text(json.dumps(res, indent=1, ensure_ascii=False, default=float))


def collect(w):
    from stable_baselines3 import PPO
    out, seeds = set_window(w); Wd = R3.World(); va, te = Wd.span(R3.VAL), Wd.span(R3.TEST); rows = []
    for st in range(0, R3.TOTAL + 1, R3.CKPT):
        ms = [PPO.load(out / f"s{sd}" / f"{st}.zip", device="cpu") for sd in seeds]
        r, _, _ = V2.run(Wd, va, R3.ens_picker(Wd, ms, "tcn")); rows.append((SYS.sharpe(r), st, ms)); print(f"{w} 걸음 {st:>6} 검증 앙상블 샤프 {rows[-1][0]:+.2f}", flush=True)
    vs, st, ms = max(rows, key=lambda z: z[0]); pk = R3.ens_picker(Wd, ms, "tcn")
    _, _, lg_va = V2.run(Wd, va, pk); r_e, _, lg_e = V2.run(Wd, te, pk); r_t, _, lg_t = V2.run(Wd, te, V2.teacher(Wd))
    nulls = np.array([V2.run(Wd, te, V2.markov(lg_va, sd))[0].values for sd in range(400)])
    seed_sh = {sd: SYS.sharpe(V2.run(Wd, te, R3.picker(Wd, m, "tcn"))[0]) for sd, m in zip(seeds, ms)}
    tdays = Wd.days[te[0]:te[1]]; hold = pd.concat([SYS.daily_close(Wd.data[s]).pct_change() for s in V2.COINS], axis=1).mean(1).loc[tdays]
    rep = {"window": w, "step": st, "val_sharpe": vs, "val_grid": [(z[1], z[0]) for z in rows], "ens": V2.summarize(r_e, lg_e), "teacher": V2.summarize(r_t, lg_t),
           "hold_1x": {"sharpe": SYS.sharpe(hold), "ret": float((1 + hold).prod() - 1)}, "seed_test_sharpe": seed_sh}
    (out / "window_report.json").write_text(json.dumps(rep, indent=1, ensure_ascii=False, default=float))
    pd.DataFrame({"ens": r_e, "teacher": r_t}).to_parquet(out / "window_daily.parquet"); np.save(out / "window_nulls.npy", nulls)
    print(json.dumps({k: rep[k] for k in ("window", "step", "val_sharpe", "ens", "teacher", "hold_1x", "seed_test_sharpe")}, indent=1, ensure_ascii=False, default=float))


def pooled():
    D = {w: pd.read_parquet(BASE / w / "window_daily.parquet") for w in WINDOWS}; N = {w: np.load(BASE / w / "window_nulls.npy") for w in WINDOWS}
    Rp = {w: json.loads((BASE / w / "window_report.json").read_text()) for w in WINDOWS}
    ens = pd.concat([D[w].ens for w in WINDOWS]); diff = pd.concat([D[w].ens - D[w].teacher for w in WINDOWS])
    nul = np.concatenate([N[w] for w in WINDOWS], axis=1); nsh = nul.mean(1) / nul.std(1) * np.sqrt(365)
    out = {"prereg": "docs/prereg/rl_v7_fixed_learner_real_20261025.md@d3f54342", "pooled_sharpe": SYS.sharpe(ens), "days": len(ens),
           "pooled_teacher_sharpe": SYS.sharpe(pd.concat([D[w].teacher for w in WINDOWS])),
           "ens_minus_teacher_ci": SYS.block_ci(diff), "ens_minus_teacher_bp": float(diff.mean() * 1e4), "null_p95": float(np.percentile(nsh, 95)),
           "per_window": {w: {"ens": Rp[w]["ens"]["sharpe"], "ens_mdd": Rp[w]["ens"]["mdd"], "teacher": Rp[w]["teacher"]["sharpe"], "hold": Rp[w]["hold_1x"]["sharpe"],
                              "step": Rp[w]["step"], "flat": Rp[w]["ens"]["flat_share"], "long_in_pos": Rp[w]["ens"]["long_share_in_pos"],
                              "seeds_pos": sum(v > 0 for v in Rp[w]["seed_test_sharpe"].values())} for w in WINDOWS}}
    pw = out["per_window"]
    out["gates"] = {"R1": out["pooled_sharpe"] > 0 and all(v["ens_mdd"] >= -0.20 for v in pw.values()), "R2": out["ens_minus_teacher_ci"][0] > 0,
                    "R3": out["pooled_sharpe"] > out["null_p95"], "R4": sum(v["ens"] > 0 for v in pw.values()) >= 2}
    out["pass"] = all(out["gates"].values()); (BASE / "pooled_report.json").write_text(json.dumps(out, indent=1, ensure_ascii=False, default=float))
    print(json.dumps(out, indent=1, ensure_ascii=False, default=float))


if __name__ == "__main__":
    a = sys.argv[1:]
    if "--train" in a:
        i = a.index("--train"); train(a[i + 1], int(a[i + 2]))
    elif "--collect" in a:
        collect(a[a.index("--collect") + 1])
    elif "--pooled" in a:
        pooled()
    else:
        print(__doc__)
