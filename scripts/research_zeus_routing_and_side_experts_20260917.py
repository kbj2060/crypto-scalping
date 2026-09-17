#!/usr/bin/env python3
"""Zeus — **라우팅이 있는 것 vs 없는 것**, 그리고 **롱/숏 전문가 분리** (2026-09-17)

## 왜 학습이 필요한가
추론 절제(`--routing`)는 **이미 라우팅으로 특화되어 학습된** 전문가를 다르게 쓰는 것뿐이라
「애초에 라우팅 없이 하나로 배웠다면?」에 답하지 못한다. 그건 균등 가중으로 새로 학습해야 한다.
(그 절제 결과: 하드 라우팅이 **무작위 배정 5시드 전부보다 낮다** — 건수맞춤 후 −4.68bp.)

## 팔 (사전 지정)
  N0  현행 — 3 레짐 전문가 · 가중 `balanced × route_prob` · 하드 라우팅        모델 3
  N1  **라우팅 없음** — 1 모델 · 가중 `balanced` 만 · 전 봉 동일 모델          모델 1
  N1b 라우팅 없음 · N1 의 3시드 **앙상블** ← ⭐**용량 대조**(추가 학습 0)       모델 3
  N2  **측면 분리** — 롱 전문가(LONG+CASH 행만) + 숏 전문가(SHORT+CASH 행만)   모델 2
      ⚠️측면은 «예측 대상»이라 라우팅 키가 못 된다. 둘 다 전 봉을 채점하고 확률을 비교한다.

🔴**용량 대조가 필수다.** 전문가를 나누면 파라미터가 늘어 「분리가 좋다」와 「모델이 커서
좋다」가 섞인다. N1b(같은 용량·라우팅만 없음)가 그 둘을 가른다.
🔴**건수를 맞춘다.** 확신도가 높을수록 p 가 높으므로(stageN: D1 40.51%→D10 47.53%)
게이트 통과 수가 다르면 「서열」과 「선별성」이 뒤섞인다.

라벨은 배포와 동일한 zigzag(품질=방향, `same_as_direction`). 청산은 Zeus 더블 배리어.
"""
from __future__ import annotations
import importlib.util, json, sys
from pathlib import Path
import numpy as np, pandas as pd, torch
from sklearn.utils.class_weight import compute_sample_weight

ROOT = Path.home() / "crypto-scalping"
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import train_eval_omega1_2_tabm_3head_20260603 as tabm  # noqa: E402
import train_eval_omega461_parent_zig075_longwindow_20260917 as E  # noqa: E402
import research_omega461_side_skill_decomposition_20260917 as K  # noqa: E402

SEEDS = [613042, 27851, 904377]
FOLDS = [f for f in K.FOLDS if f[0] in ("F1", "F2", "F3", "CAND")]
TARGET_N = int(next((a.split("=", 1)[1] for a in sys.argv if a.startswith("--match=")), 3700))
ARMS = (next((a.split("=", 1)[1].split(",") for a in sys.argv if a.startswith("--arms=")), None)
        or ["N0", "N1", "N2"])
OUTJ = E.OUT / "stageP_routing_side_experts.json"
CACHE = E.OUT / "stageP_probs.npz"
TPB, SLB, COST = K.BASE_TP * 1e4, K.BASE_SL * 1e4, 1.02


def log(*a): print(*a, flush=True)


def gate_score(D, Q):
    da = D.argmax(1)
    qf = np.where(da > 0, Q[np.arange(len(Q)), da], Q[:, 0])
    return da, qf


def realize(te, side, idx):
    hi = pd.to_numeric(te["high"]).to_numpy(np.float64)
    lo = pd.to_numeric(te["low"]).to_numpy(np.float64)
    cl = pd.to_numeric(te["close"]).to_numpy(np.float64)
    r, h, _rs, _rn, _m = K._first_touch_open(idx, side, hi, lo, cl, K.BASE_TP, K.BASE_SL, K.MAXBARS)
    return r * 1e4 - COST, h.astype(float), te.timestamp.dt.floor("D").to_numpy()[idx]


def main() -> int:
    E.OUT.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"device={device} · seeds={SEEDS} · arms={ARMS} · 건수맞춤 {TARGET_N:,} · "
        f"더블배리어 TP{K.BASE_TP*100:g}%/SL{K.BASE_SL*100:g}%")
    df, base_cols = E.load()
    cache = dict(np.load(CACHE, allow_pickle=True)) if CACHE.exists() else {}
    store = {a: [] for a in ("N0", "N1", "N1b", "N2")}

    for name, t0, t1, v0, v1 in FOLDS:
        tm = (df.timestamp >= t0) & (df.timestamp <= t1 + " 23:59:59")
        vm = (df.timestamp >= v0) & (df.timestamp <= v1 + " 23:59:59")
        tr, te = df[tm].reset_index(drop=True), df[vm].reset_index(drop=True)
        assert tr.timestamp.max() < te.timestamp.min(), f"{name} TRAIN 이 TEST 를 침범"
        yt = pd.to_numeric(tr["zigzag_action"]).to_numpy(np.int64)
        rt = tabm._route_probs(tr)
        ev = tabm._route_probs(te).argmax(1)
        xs, scaler = tabm._standardize_fit(tabm._base_input(tr, base_cols))
        xv = tabm._standardize_apply(tabm._base_input(te, base_cols), scaler)
        n = len(tr); split = max(int(n * 0.85), min(n - 1, 512))
        bal = compute_sample_weight("balanced", y=yt).astype(np.float32)
        log(f"\n{'='*92}\n=== {name} TRAIN {t0}~{t1} {n:,} · TEST {v0}~{v1} {len(te):,}\n{'='*92}")

        def fit_get(key, ytr, wtr, rows=None):
            """캐시된 (D,Q) 를 주거나 학습한다. rows 가 있으면 그 부분집합만 학습한다."""
            ck = f"{name}|{key}"
            if ck in cache:
                z = dict(cache[ck].item()); return z["D"], z["Q"]
            sel = np.ones(n, bool) if rows is None else rows
            si = np.where(sel[:split])[0]; sv = np.where(sel[split:])[0]
            # ⭐부분집합 학습이면 클래스 가중치를 «그 부분집합에서» 다시 계산한다.
            # 전체 3클래스 기준 가중치를 쓰면 롱 전문가(SHORT 를 안 봄)의 균형이 어긋난다.
            w_ = wtr if rows is None else compute_sample_weight("balanced", y=ytr[sel]).astype(np.float32)
            wi = w_[:len(si)] if rows is not None else w_[:split][si]
            wv = w_[len(si):] if rows is not None else w_[split:][sv]
            assert len(wi) == len(si) and len(wv) == len(sv), "부분집합 가중치 길이 불일치"
            m, _ = E.fit_expert(xs[:split][si], ytr[:split][si], wi,
                                xs[split:][sv], ytr[split:][sv], wv,
                                seed=int(key.split("s")[-1]), ei=0, device=device)
            D, Q = E.heads(m, xv, device)
            cache[ck] = np.array({"D": D, "Q": Q}, dtype=object); np.savez(CACHE, **cache)
            return D, Q

        # ── N0: 3 레짐 전문가 · 라우팅 가중 · 하드 라우팅 ──
        if "N0" in ARMS:
            for sd in SEEDS:
                D = np.zeros((len(te), 3)); Q = np.zeros((len(te), 3))
                for ei, en in enumerate(("bull", "bear", "chop")):
                    ck = f"{name}|N0{en}s{sd}"
                    if ck in cache:
                        z = dict(cache[ck].item()); Dd, Qq = z["D"], z["Q"]
                    else:
                        w = bal * rt[:, ei].astype(np.float32)
                        m, _ = E.fit_expert(xs[:split], yt[:split], w[:split],
                                            xs[split:], yt[split:], w[split:],
                                            seed=sd, ei=ei, device=device)
                        Dd, Qq = E.heads(m, xv, device)
                        cache[ck] = np.array({"D": Dd, "Q": Qq}, dtype=object); np.savez(CACHE, **cache)
                    s_ = ev == ei
                    D[s_], Q[s_] = Dd[s_], Qq[s_]
                store["N0"].append((name, te, D, Q)); log(f"  N0/seed {sd} 완료")

        # ── N1: 라우팅 없음, 단일 모델 ──
        if "N1" in ARMS:
            Ds, Qs = [], []
            for sd in SEEDS:
                D, Q = fit_get(f"N1s{sd}", yt, bal)
                Ds.append(D); Qs.append(Q)
                store["N1"].append((name, te, D, Q)); log(f"  N1/seed {sd} 완료")
            store["N1b"].append((name, te, np.mean(Ds, 0), np.mean(Qs, 0)))   # 용량 대조(공짜)

        # ── N2: 측면 분리 (LONG+CASH 행 / SHORT+CASH 행) ──
        if "N2" in ARMS:
            for sd in SEEDS:
                DL, QL = fit_get(f"N2Ls{sd}", yt, bal, rows=(yt != 2))
                DS, QS = fit_get(f"N2Ss{sd}", yt, bal, rows=(yt != 1))
                # 둘 다 전 봉 채점 -> 확률 비교. 측면은 예측 대상이라 라우팅 키가 못 된다.
                D = np.stack([np.minimum(DL[:, 0], DS[:, 0]), DL[:, 1], DS[:, 2]], 1)
                D = D / np.maximum(D.sum(1, keepdims=True), 1e-12)
                Q = np.stack([np.minimum(QL[:, 0], QS[:, 0]), QL[:, 1], QS[:, 2]], 1)
                Q = Q / np.maximum(Q.sum(1, keepdims=True), 1e-12)
                store["N2"].append((name, te, D, Q)); log(f"  N2/seed {sd} 완료")

    # ── 건수 맞춘 평가 ──
    rows = []
    for arm, segs in store.items():
        if not segs:
            continue
        by_seed = {}
        for i, (name, te, D, Q) in enumerate(segs):
            by_seed.setdefault(i % max(len(segs) // len(FOLDS), 1), []).append((name, te, D, Q))
        for si, group in by_seed.items():
            allq = np.concatenate([gate_score(D, Q)[1][gate_score(D, Q)[0] != 0] for _n, _t, D, Q in group])
            thr = float(np.sort(allq)[::-1][min(TARGET_N, len(allq)) - 1])
            pnl, hold, days = [], [], []
            for _n, te, D, Q in group:
                da, qf = gate_score(D, Q)
                side = np.where((da == 1) & (qf >= thr), 1.0, np.where((da == 2) & (qf >= thr), -1.0, 0.0))
                idx = np.where(side != 0)[0]
                if len(idx) < 20:
                    continue
                a, b, c = realize(te, side, idx); pnl.append(a); hold.append(b); days.append(c)
            pnl = np.concatenate(pnl); hold = np.concatenate(hold); days = np.concatenate(days)
            lo_, hi_, nd = E.block_ci(pnl, days)
            g = float(pnl.mean()); pdy = 288.0 / max(hold.mean(), 1e-9)
            rows.append({"arm": arm, "seed_slot": si, "q": thr, "n": int(len(pnl)), "indep_days": nd,
                         "gross_bp": g, "p": (g + COST + SLB) / (TPB + SLB), "ci95": [lo_, hi_],
                         "median_hold": float(np.median(hold)), "per_day": pdy, "net_day": g * pdy})
    D_ = pd.DataFrame(rows)
    log(f"\n{'='*104}\n■ 라우팅 유무 · 측면 분리 (건수맞춤 {TARGET_N:,} · 더블배리어 · 4폴드)")
    log(f"{'팔':<6}{'모델수':>7}{'건수':>8}{'건당bp':>9}{'함축p':>8}{'CI':>20}{'건/일':>7}{'순/일':>8}")
    NM = {"N0": 3, "N1": 1, "N1b": 3, "N2": 2}
    for arm, g in D_.groupby("arm", sort=False):
        log(f"{arm:<6}{NM[arm]:>7}{int(g.n.mean()):>8,}{g.gross_bp.mean():>+9.2f}{g.p.mean()*100:>7.2f}%"
            f"  [{g.ci95.apply(lambda x: x[0]).mean():+7.2f},{g.ci95.apply(lambda x: x[1]).mean():+7.2f}]"
            f"{g.per_day.mean():>7.2f}{g.net_day.mean():>8.1f}"
            f"   시드폭 {g.gross_bp.max()-g.gross_bp.min():.2f}")
    OUTJ.write_text(json.dumps(rows, indent=2, default=float))
    log(f"저장: {OUTJ}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
