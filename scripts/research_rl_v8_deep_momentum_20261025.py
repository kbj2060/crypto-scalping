#!/usr/bin/env python3
"""RL v8 — 딥 모멘텀 네트워크(Lim·Zohren·Roberts 2019, arXiv 1904.04912): TCN → tanh 포지션, 음의 샤프(식 16·35) 직접 학습.
사전 등록 docs/prereg/rl_v8_deep_momentum_20261025.md.

    python3 scripts/research_rl_v8_deep_momentum_20261025.py --selfcheck
    python3 scripts/research_rl_v8_deep_momentum_20261025.py --a 0.12 604829     # Stage A 한 학습(심은 신호)
    python3 scripts/research_rl_v8_deep_momentum_20261025.py --areport          # Stage A 판정
    python3 scripts/research_rl_v8_deep_momentum_20261025.py --b W1             # Stage B 창 하나: 5시드 학습 → 앙상블 시험 1회
    python3 scripts/research_rl_v8_deep_momentum_20261025.py --pooled           # Stage B R1~R4

실행기·세계·관문은 v3/v6/v7 그대로 재사용. 실주문 없음.
"""
import copy, json, math, sys
from pathlib import Path
import numpy as np, pandas as pd
import torch as th
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parent))
import research_rl_v3_reward_arch_20261025 as R3            # noqa: E402
import research_rl_v6_planted_signal_20261025 as R6         # noqa: E402
import research_rl_v7_fixed_learner_real_20261025 as R7     # noqa: E402
V2, SYS = R3.V2, R3.SYS

OUT = SYS.ROOT / "tmp" / "rl_v8_dmn_20261025"
TRAIN_A, VAL_A = ("2024-01-01", "2025-08-31"), ("2025-09-01", "2025-12-31")
COST, THR, CHUNK, BATCH, LR, CLIP, EPOCHS, PATIENCE, DROP = 10e-4, 0.1, 63, 8, 1e-3, 1.0, 100, 25, 0.3
SEEDS_A = (604829, 815177)
SEEDS_B = {"W1": (915220, 741300, 367705, 866748, 846067), "W2": (515822, 762299, 652083, 883627, 644436), "W3": (857383, 818728, 972233, 968506, 205479)}
W, NF = R3.W, R3.NF


class DMN(nn.Module):
    """R3 Seq 의 인과 TCN(포지션 입력 없음) → 드롭아웃 → Linear → tanh."""
    def __init__(self, hid=32):
        super().__init__(); self.c1, self.c2 = nn.Conv1d(NF, hid, 3), nn.Conv1d(hid, hid, 3, dilation=2); self.drop, self.head = nn.Dropout(DROP), nn.Linear(hid, 1)

    def forward(self, x):   # x (..., NF, W)
        sh = x.shape[:-2]; x = x.reshape(-1, NF, W)
        x = th.relu(self.c1(nn.functional.pad(x, (2, 0)))); x = th.relu(self.c2(nn.functional.pad(x, (4, 0))))
        return th.tanh(self.head(self.drop(x[..., -1]))).reshape(sh)


def wins(w):
    """일 d 의 창 = F[d−W+1 .. d] (채널 먼저). 반환 (일, NF, W), d < W−1 은 NaN."""
    v = np.lib.stride_tricks.sliding_window_view(w["F"], W, axis=0)
    return np.concatenate([np.full((W - 1, NF, W), np.nan, np.float32), v]).astype(np.float32)


def target(w):
    """r_d = c_d / c_{d−1} − 1 (결정일 d 수익) · L_d = min(2, 0.25/σ_d) (슬리브 크기와 같음)."""
    return (w["c"][:, -1] / w["P0"] - 1).astype(np.float32), np.minimum(V2.CAP, V2.TAU_C / w["sig"]).astype(np.float32)


def neg_sharpe(z, r, L):
    """식 (35)·(16): 궤적(행)마다 첫 열은 전날 포지션용. R = X·L·r − c·|ΔX·L| → −평균/표준편차·√365."""
    p = z * L; R = (p[:, 1:] * r[:, 1:] - COST * (p[:, 1:] - p[:, :-1]).abs()).reshape(-1)
    return -R.mean() / (R.std(unbiased=False) + 1e-12) * math.sqrt(365)


def seq(w, a, b):
    """일 a−1 .. b−1 (전날 포함) 의 창·목표 텐서."""
    X, (r, L) = wins(w)[a - 1:b], (x[a - 1:b] for x in target(w)); assert not np.isnan(X).any() and not np.isnan(r).any() and not np.isnan(L).any(), (a, b)
    return th.tensor(X), th.tensor(r), th.tensor(L)


def fit(Wd, tr, va, seed):
    th.manual_seed(seed); rng = np.random.default_rng(seed); m = DMN(); opt = th.optim.Adam(m.parameters(), lr=LR)
    Xs, rs, Ls = [], [], []
    for w in Wd.pool:   # 63일 궤적 고정 타일(+전날 1)
        X, r, L = seq(w, tr[0] + 1, tr[1])
        for i in range(0, len(r) - CHUNK, CHUNK):
            Xs.append(X[i:i + CHUNK + 1]); rs.append(r[i:i + CHUNK + 1]); Ls.append(L[i:i + CHUNK + 1])
    Xs, rs, Ls = th.stack(Xs), th.stack(rs), th.stack(Ls)
    V = [seq(Wd.W[s], *va) for s in V2.COINS]; VX, Vr, VL = (th.stack([v[k] for v in V]) for k in range(3))
    best, best_ep, hist = math.inf, 0, []
    for ep in range(EPOCHS):
        m.train(); perm = rng.permutation(len(Xs))
        for i in range(0, len(perm), BATCH):
            b = perm[i:i + BATCH]; loss = neg_sharpe(m(Xs[b]), rs[b], Ls[b]); opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(m.parameters(), CLIP); opt.step()
        m.eval()
        with th.no_grad():
            vl = float(neg_sharpe(m(VX), Vr, VL)); tl = float(neg_sharpe(m(Xs), rs, Ls))
        hist.append((ep, tl, vl))
        if vl < best:
            best, best_ep, state = vl, ep, copy.deepcopy(m.state_dict())
        elif ep - best_ep >= PATIENCE:
            break
    m.load_state_dict(state); m.eval()
    return m, {"seed": seed, "best_epoch": best_ep, "epochs_run": len(hist), "val_sharpe_loss_form": -best, "hist": hist}


def zmap(Wd, models):
    """코인별 전체 일 z̄(모델 평균). 창 NaN 인 앞쪽 일은 NaN."""
    out = {}
    with th.no_grad():
        for s in V2.COINS:
            X = th.tensor(np.nan_to_num(wins(Wd.W[s]))); bad = np.isnan(wins(Wd.W[s])).any((1, 2))
            z = np.mean([m(X).numpy() for m in models], 0); z[bad] = np.nan; out[s] = z
    return out


def picker(Z):
    def pick(s, d, o):
        z = Z[s][d]; assert not np.isnan(z), (s, d)
        return 1 + int(np.sign(z)) if abs(z) > THR else 1
    return pick


# ---------- Stage A ----------
def stage_a(rho, seed):
    assert rho in (0.0, 0.06, 0.12) and seed in SEEDS_A; th.set_num_threads(2)
    R3.TRAIN = TRAIN_A; Wd = R6.planted_world(rho); tr, va = Wd.span(TRAIN_A, Wd.pool), Wd.span(VAL_A)
    m, info = fit(Wd, tr, va, seed); sm, _ = R6.evaluate(Wd, picker(zmap(Wd, [m])))
    od = OUT / "stageA"; od.mkdir(parents=True, exist_ok=True); th.save(m.state_dict(), od / f"r{rho}_s{seed}.pt")
    res = {"rho": rho, **info, "eval": sm}; (od / f"r{rho}_s{seed}.json").write_text(json.dumps(res, indent=1, ensure_ascii=False, default=float))
    print(f"ρ{rho} s{seed} 멈춘 에폭 {info['best_epoch']}/{info['epochs_run']} 검증(손실식) {info['val_sharpe_loss_form']:+.2f} · 평가 샤프 {sm['sharpe']:+.2f} 힌트 일치 {sm['oracle_agree']:.2f} 현금 {sm['flat_share']:.2f} 롱 {sm['long_share']:.2f}", flush=True)


def a_report():
    R3.TRAIN = TRAIN_A; rows, ref = [], {}
    for rho in (0.0, 0.06, 0.12):
        Wd = R6.planted_world(rho); t, _ = R6.evaluate(Wd, V2.teacher(Wd)); o, _ = R6.evaluate(Wd, lambda s, d, o_: int(Wd.W[s]["S"][d]) + 1)
        ref[rho] = {"teacher": t["sharpe"], "oracle": o["sharpe"]}; gap = o["sharpe"] - t["sharpe"]
        for sd in SEEDS_A:
            r = json.loads((OUT / "stageA" / f"r{rho}_s{sd}.json").read_text()); e = r["eval"]
            rows.append({"rho": rho, "seed": sd, "eval_sharpe": e["sharpe"], "uplift_frac": (e["sharpe"] - t["sharpe"]) / gap, "hint_agree": e["oracle_agree"],
                         "flat": e["flat_share"], "long": e["long_share"], "mdd": e["mdd"], "best_epoch": r["best_epoch"], "epochs_run": r["epochs_run"], "val_loss_sharpe": r["val_sharpe_loss_form"]})
    D = pd.DataFrame(rows); a12, a0 = D[D.rho == 0.12], D[D.rho == 0.0]
    out = {"prereg": "docs/prereg/rl_v8_deep_momentum_20261025.md@7d953722", "ref": ref, "runs": rows,
           "A_learn": bool((a12.uplift_frac >= 0.5).all()), "A_no_halluc": bool((a0.eval_sharpe <= ref[0.0]["teacher"] + 0.3).all())}
    out["pass"] = out["A_learn"] and out["A_no_halluc"]; (OUT / "stageA_report.json").write_text(json.dumps(out, indent=1, ensure_ascii=False, default=float))
    print(D.round(3).to_string()); print(json.dumps({k: out[k] for k in ("ref", "A_learn", "A_no_halluc", "pass")}, ensure_ascii=False, default=float))


def a_diag():
    """사후 진단(관문 아님, 합성만): 저장 모델의 |z| 분포 · 문턱 0(sign z) 실행기 샤프 · 비례(손실식) 평가 샤프."""
    global THR
    R3.TRAIN = TRAIN_A; rows = []
    for rho in (0.0, 0.06, 0.12):
        Wd = R6.planted_world(rho); a, b = Wd.span(R6.EVAL)
        for sd in SEEDS_A:
            m = DMN(); m.load_state_dict(th.load(OUT / "stageA" / f"r{rho}_s{sd}.pt")); m.eval(); Z = zmap(Wd, [m]); az = np.abs(np.concatenate([Z[s][a:b] for s in V2.COINS]))
            THR = 0.0; sm, _ = R6.evaluate(Wd, picker(Z)); THR = 0.1
            V = [seq(Wd.W[s], a, b) for s in V2.COINS]
            with th.no_grad():
                prop = -float(neg_sharpe(m(th.stack([v[0] for v in V])), th.stack([v[1] for v in V]), th.stack([v[2] for v in V])))
            rows.append({"rho": rho, "seed": sd, "absz_p50": np.median(az), "absz_p90": np.percentile(az, 90), "absz_max": az.max(),
                         "sign_exec_sharpe": sm["sharpe"], "sign_hint_agree": sm["oracle_agree"], "sign_long": sm["long_share"], "prop_lossform_sharpe": prop})
    D = pd.DataFrame(rows); (OUT / "stageA_diag.json").write_text(D.to_json(orient="records", indent=1)); print(D.round(3).to_string())


# ---------- Stage B (v7 collect 와 같은 산출 모양 → R7.pooled 재사용) ----------
def stage_b(w):
    assert (OUT / "stageA_report.json").exists() and json.loads((OUT / "stageA_report.json").read_text())["pass"], "Stage A 통과 전 실데이터 금지"
    th.set_num_threads(2); R3.TRAIN, R3.VAL, R3.TEST, _ = R7.WINDOWS[w]; od = OUT / "stageB" / w; od.mkdir(parents=True, exist_ok=True)
    Wd = R3.World(); tr, va, te = Wd.span(R3.TRAIN, Wd.pool), Wd.span(R3.VAL), Wd.span(R3.TEST); ms, infos = [], []
    for sd in SEEDS_B[w]:
        m, info = fit(Wd, tr, va, sd); ms.append(m); infos.append(info); th.save(m.state_dict(), od / f"s{sd}.pt")
        print(f"{w} s{sd} 멈춘 에폭 {info['best_epoch']}/{info['epochs_run']} 검증(손실식) {info['val_sharpe_loss_form']:+.2f}", flush=True)
    pk = picker(zmap(Wd, ms)); r_va, _, lg_va = V2.run(Wd, va, pk); r_e, _, lg_e = V2.run(Wd, te, pk); r_t, _, lg_t = V2.run(Wd, te, V2.teacher(Wd))
    nulls = np.array([V2.run(Wd, te, V2.markov(lg_va, sd))[0].values for sd in range(400)])
    seed_sh = {sd: SYS.sharpe(V2.run(Wd, te, picker(zmap(Wd, [m])))[0]) for sd, m in zip(SEEDS_B[w], ms)}
    tdays = Wd.days[te[0]:te[1]]; hold = pd.concat([SYS.daily_close(Wd.data[s]).pct_change() for s in V2.COINS], axis=1).mean(1).loc[tdays]
    rep = {"window": w, "step": [i["best_epoch"] for i in infos], "val_sharpe": SYS.sharpe(r_va), "val": V2.summarize(r_va, lg_va), "ens": V2.summarize(r_e, lg_e),
           "teacher": V2.summarize(r_t, lg_t), "hold_1x": {"sharpe": SYS.sharpe(hold), "ret": float((1 + hold).prod() - 1)}, "seed_test_sharpe": seed_sh,
           "fit": [{k: v for k, v in i.items() if k != "hist"} for i in infos]}
    (od / "window_report.json").write_text(json.dumps(rep, indent=1, ensure_ascii=False, default=float))
    pd.DataFrame({"ens": r_e, "teacher": r_t}).to_parquet(od / "window_daily.parquet"); np.save(od / "window_nulls.npy", nulls)
    print(json.dumps({k: rep[k] for k in ("window", "step", "val_sharpe", "ens", "teacher", "hold_1x", "seed_test_sharpe")}, indent=1, ensure_ascii=False, default=float))


def pooled():
    R7.BASE = OUT / "stageB"; R7.pooled(); p = R7.BASE / "pooled_report.json"; o = json.loads(p.read_text())
    o["prereg"] = "docs/prereg/rl_v8_deep_momentum_20261025.md@7d953722"; p.write_text(json.dumps(o, indent=1, ensure_ascii=False, default=float))


def selfcheck():
    R3.TRAIN = TRAIN_A; Wd = R6.planted_world(0.12); w = Wd.W["ETH"]; d = Wd.span(VAL_A)[0] + 5
    assert np.allclose(wins(w)[d].T.ravel(), R3.window(w, d, np.zeros(3))[:-3]); print("창 = R3.window 시장 부분")
    a, b = Wd.span(R6.EVAL); r, _ = target(w); ic = pd.Series(w["F"][a:b, R6.COL]).corr(pd.Series(r[a:b]), method="spearman")
    print(f"심은 힌트 vs 목표 r 일 IC {ic:.3f} (≈ .12 이어야)"); assert 0.05 < ic < 0.2
    rr = th.tensor(np.random.default_rng(0).normal(0, .03, (4, 64)), dtype=th.float32); L = th.ones_like(rr)
    assert neg_sharpe(th.sign(rr), rr, L) < -5 and neg_sharpe(-th.sign(rr), rr, L) > 5   # 정답 포지션 = 큰 샤프
    z = th.full_like(rr, 0.5); R = (0.5 * rr[:, 1:]).reshape(-1); assert abs(float(neg_sharpe(z, rr, L)) + float(R.mean() / R.std(unbiased=False)) * math.sqrt(365)) < 1e-4   # 상수 포지션 = 회전 0
    Z = {"ETH": np.array([0.05, -0.2, 0.3, np.nan])}; pk = picker(Z); assert [pk("ETH", i, None) for i in range(3)] == [1, 0, 2]
    print("SELFCHECK OK")


if __name__ == "__main__":
    a = sys.argv[1:]
    if "--selfcheck" in a:
        selfcheck()
    elif "--a" in a:
        i = a.index("--a"); stage_a(float(a[i + 1]), int(a[i + 2]))
    elif "--areport" in a:
        a_report()
    elif "--adiag" in a:
        a_diag()
    elif "--b" in a:
        stage_b(a[a.index("--b") + 1])
    elif "--pooled" in a:
        pooled()
    else:
        print(__doc__)
