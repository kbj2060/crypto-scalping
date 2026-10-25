#!/usr/bin/env python3
"""RL v6 — 학습 장치 검증: 관측 «D» 열에 정답 힌트 s_d = ρ·z_d + √(1−ρ²)·ε 를 심은 세계에서 RL 이 배우는가.
사전 등록 docs/prereg/rl_v6_planted_signal_20261025.md.

    python3 scripts/research_rl_v6_planted_signal_20261025.py --selfcheck            # 심은 신호 일 IC ≈ ρ · 교사 열 불변
    python3 scripts/research_rl_v6_planted_signal_20261025.py --train 0.12 Z 331161  # 한 학습
    python3 scripts/research_rl_v6_planted_signal_20261025.py --report               # 24학습 + 교사·오라클 → L1~L3

학습 레시피 = v3.1(research_rl_v3_reward_arch_20261025). 학습 W3(2024~25) 30세계 · 평가 4코인 2022-06-01~2023-12-31. 실주문 없음.
"""
import json, math, sys
from pathlib import Path
import numpy as np, pandas as pd
import torch as th

sys.path.insert(0, str(Path(__file__).resolve().parent))
import research_rl_v3_reward_arch_20261025 as R3   # noqa: E402
V2, SYS = R3.V2, R3.SYS

OUT = SYS.ROOT / "tmp" / "rl_v6_planted_20261025"
EVAL = ("2022-06-01", "2023-12-31")
COL = 12                                  # feats() 열 순서의 마지막 = D(추세 부호)
RHOS = (0.0, 0.03, 0.06, 0.12)
SEEDS = {"Z": (331161, 783382, 713244), "DD": (835240, 832343, 218323)}


def planted_world(rho):
    """R3.World 를 만든 뒤 모든 세계의 F[:, COL] 을 심은 신호로 바꾼다. 세계마다 ε 시드 고정(순서 = pool, 그다음 실제 4코인)."""
    Wd = R3.World(); seen = set()
    for k, w in enumerate(Wd.pool + [Wd.W[s] for s in V2.COINS]):
        if id(w) in seen:
            continue
        seen.add(id(w))
        z = np.log(w["c"][:, -1] / w["P0"]) / (w["sig"] / math.sqrt(365)); z = np.clip(np.nan_to_num(z), -5, 5)
        eps = np.random.default_rng(10_000 + k).standard_normal(len(z))
        w["F"][: len(z), COL] = np.clip(rho * z + math.sqrt(1 - rho ** 2) * eps, -5, 5).astype(np.float32)
        w["S"] = np.sign(w["F"][:, COL]).astype(int)      # 오라클 방향
    return Wd


def evaluate(Wd, pick):
    ev = Wd.span(EVAL); r, _, lg = V2.run(Wd, ev, pick); sm = V2.summarize(r, lg)
    a0, a1 = ev; acts = np.concatenate([lg[s][0] - 1 for s in V2.COINS]); ora = np.concatenate([Wd.W[s]["S"][a0:a1] for s in V2.COINS])
    sm["oracle_agree"] = float(np.mean(acts == ora)); sm["long_share"] = float(np.mean(acts == 1))
    return sm, r


def train(rho, rew, seed):
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv
    assert rho in RHOS and seed in SEEDS[rew]; th.set_num_threads(2); th.manual_seed(seed)
    Wd = planted_world(rho); tr = Wd.span(R3.TRAIN, Wd.pool)
    venv = DummyVecEnv([lambda i=i: R3.Env(Wd, tr, rew, "tcn", seed * 10 + i) for i in range(8)])
    m = PPO("MlpPolicy", venv, policy_kwargs=R3.policy_kwargs("tcn"), seed=seed, verbose=0, device="cpu", **R3.PPO_KW)
    b = R3.bc(m, Wd, tr, "tcn"); assert b["train_agree"] >= 0.95, b
    od = OUT / f"r{rho}_{rew}_s{seed}"; od.mkdir(parents=True, exist_ok=True); res = {"rho": rho, "reward": rew, "seed": seed, "bc": b, "ckpt": []}
    for step in range(0, R3.TOTAL + 1, R3.CKPT):
        if step:
            m.learn(R3.CKPT, reset_num_timesteps=False)
        sm, _ = evaluate(Wd, R3.picker(Wd, m, "tcn")); sm["step"] = step; res["ckpt"].append(sm)
        print(f"ρ{rho} {rew} s{seed} 걸음 {step:>6} 평가 샤프 {sm['sharpe']:+.2f} 오라클 일치 {sm['oracle_agree']:.2f} 현금 {sm['flat_share']:.2f} 롱 {sm['long_share']:.2f}", flush=True)
        (od / "run.json").write_text(json.dumps(res, indent=1, ensure_ascii=False, default=float))


def selfcheck():
    for rho in (0.0, 0.12):
        Wd = planted_world(rho); a0, a1 = Wd.span(EVAL); ics = []
        for s in V2.COINS:
            w = Wd.W[s]; z = np.log(w["c"][a0:a1, -1] / w["P0"][a0:a1]); ics.append(pd.Series(w["F"][a0:a1, COL]).corr(pd.Series(z), method="spearman"))
        print(f"ρ {rho}: 평가 구간 심은 신호 일 IC(스피어만) 코인별 {np.round(ics, 3)}")
    print("SELFCHECK OK")


def report():
    rows, ref = [], {}
    for rho in RHOS:
        Wd = planted_world(rho)
        t, _ = evaluate(Wd, V2.teacher(Wd)); o, _ = evaluate(Wd, lambda s, d, o_: int(Wd.W[s]["S"][d]) + 1); ref[rho] = {"teacher": t["sharpe"], "oracle": o["sharpe"]}
        gap = o["sharpe"] - t["sharpe"]
        for rew, seeds in SEEDS.items():
            for sd in seeds:
                r = json.loads((OUT / f"r{rho}_{rew}_s{sd}" / "run.json").read_text()); last = r["ckpt"][-1]; best = max(r["ckpt"], key=lambda c: c["sharpe"])
                rows.append({"rho": rho, "reward": rew, "seed": sd, "final": last["sharpe"], "best": best["sharpe"], "best_step": best["step"],
                             "uplift_frac_final": (last["sharpe"] - t["sharpe"]) / gap if gap > 0 else None, "oracle_agree_final": last["oracle_agree"],
                             "flat_final": last["flat_share"], "long_final": last["long_share"], "curve": [round(c["sharpe"], 2) for c in r["ckpt"]]})
        print(f"ρ {rho}: 교사 {t['sharpe']:+.2f} · 오라클 {o['sharpe']:+.2f}", flush=True)
    D = pd.DataFrame(rows)
    z12, z0 = D[(D.rho == 0.12) & (D.reward == "Z")], D[(D.rho == 0.0) & (D.reward == "Z")]
    out = {"prereg": "docs/prereg/rl_v6_planted_signal_20261025.md@ebca1879", "ref": ref, "runs": rows,
           "L1": int((z12.uplift_frac_final >= 0.5).sum()) >= 2, "L2": int((z0.final <= ref[0.0]["teacher"] + 0.3).sum()) >= 2,
           "L3_dd_vs_z_final_median": {str(r): [float(D[(D.rho == r) & (D.reward == "DD")].final.median()), float(D[(D.rho == r) & (D.reward == "Z")].final.median())] for r in RHOS}}
    (OUT / "report.json").write_text(json.dumps(out, indent=1, ensure_ascii=False, default=float))
    print(D.drop(columns=["curve"]).round(3).to_string()); print(json.dumps({k: out[k] for k in ("ref", "L1", "L2", "L3_dd_vs_z_final_median")}, ensure_ascii=False))


# ---------- 2차 장치 진단(합성만): D1 오라클 BC · D2 BC 없음 · D3 BC 없음+lr3e-4·ent.01 · D4 BC 조기정지 .90+lr3e-4·ent.01 ----------
VARIANTS = {"D1": dict(bc="oracle", ppo=False), "D2": dict(bc=None), "D3": dict(bc=None, lr=3e-4, ent=0.01), "D4": dict(bc="early", lr=3e-4, ent=0.01)}


def bc_labels(model, Wd, span, label, stop=None, epochs=200):
    """label(w, d) → 행동(0..2). 슬리브는 그 정답대로 굴려 포지션 관측을 만든다. stop = 학습 일치 문턱(첫 도달 에폭에서 멈춤)."""
    X, Y = [], []
    for w in Wd.pool:
        sl = V2.Sleeve(w)
        for d in range(*span):
            a = label(w, d); X.append(R3.window(w, d, sl.obs(d))); Y.append(a); sl.day(d, a - 1)
    X, Y = th.tensor(np.array(X)), th.tensor(np.array(Y)); opt = th.optim.Adam(model.policy.parameters(), lr=1e-3); g = th.Generator().manual_seed(0)
    for ep in range(epochs):
        perm = th.randperm(len(X), generator=g)
        for i in range(0, len(X), 256):
            b = perm[i:i + 256]; loss = th.nn.functional.cross_entropy(model.policy.get_distribution(X[b]).distribution.logits, Y[b])
            opt.zero_grad(); loss.backward(); opt.step()
        with th.no_grad():
            agree = float((model.policy.get_distribution(X).distribution.logits.argmax(1) == Y).float().mean())
        if stop is not None and agree >= stop:
            break
    return {"rows": len(X), "train_agree": agree, "epochs": ep + 1}


def train_variant(var, seed, rho=0.12):
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv
    cfg = VARIANTS[var]; th.set_num_threads(2); th.manual_seed(seed)
    Wd = planted_world(rho); tr = Wd.span(R3.TRAIN, Wd.pool)
    kw = dict(R3.PPO_KW, learning_rate=cfg.get("lr", R3.PPO_KW["learning_rate"]), ent_coef=cfg.get("ent", R3.PPO_KW["ent_coef"]))
    venv = DummyVecEnv([lambda i=i: R3.Env(Wd, tr, "Z", "tcn", seed * 10 + i) for i in range(8)])
    m = PPO("MlpPolicy", venv, policy_kwargs=R3.policy_kwargs("tcn"), seed=seed, verbose=0, device="cpu", **kw)
    b = None
    if cfg["bc"] == "oracle":
        b = bc_labels(m, Wd, tr, lambda w, d: int(w["S"][d]) + 1)
    elif cfg["bc"] == "early":
        b = bc_labels(m, Wd, tr, lambda w, d: int(w["D"][d]) + 1, stop=0.90)
    od = OUT / f"diag_{var}_r{rho}_s{seed}"; od.mkdir(parents=True, exist_ok=True); res = {"variant": var, "cfg": cfg, "rho": rho, "seed": seed, "bc": b, "ckpt": []}
    for step in range(0, (R3.TOTAL if cfg.get("ppo", True) else 0) + 1, R3.CKPT):
        if step:
            m.learn(R3.CKPT, reset_num_timesteps=False)
        sm, _ = evaluate(Wd, R3.picker(Wd, m, "tcn")); sm["step"] = step; res["ckpt"].append(sm)
        print(f"{var} ρ{rho} s{seed} 걸음 {step:>6} 평가 샤프 {sm['sharpe']:+.2f} 힌트 일치 {sm['oracle_agree']:.2f} 교사 일치 {sm['teacher_agree']:.2f} 현금 {sm['flat_share']:.2f}", flush=True)
        (od / "run.json").write_text(json.dumps(res, indent=1, ensure_ascii=False, default=float))


if __name__ == "__main__":
    a = sys.argv[1:]
    if "--diag" in a:
        i = a.index("--diag"); train_variant(a[i + 1], int(a[i + 2]), float(a[i + 3]) if len(a) > i + 3 else 0.12)
    elif "--selfcheck" in a:
        selfcheck()
    elif "--train" in a:
        i = a.index("--train"); train(float(a[i + 1]), a[i + 2], int(a[i + 3]))
    elif "--report" in a:
        report()
    else:
        print(__doc__)
