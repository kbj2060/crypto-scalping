#!/usr/bin/env python3
"""F단계 — ① 품질 머리 보정으로 통과율을 낮춰본다 ② 세 모델의 **포지션 편향**을 잰다.

## ① 보정
배포본은 같은 q=0.75 에서 통과율 **3.5%**, 내 판은 **7.7~8.3%** 다. 두 가지 중 하나다:
  (A) 순위는 멀쩡한데 **보정만 헐겁다** -> 임계값만 올리면 배포본을 따라잡는다
  (B) **순위 자체가 나쁘다** -> 통과율을 맞춰도 못 따라잡는다
그래서 **통과율을 맞춰놓고** 비교한다. 두 방식을 같이 낸다:
  · 임계값 스윕 전체 곡선(투명하게 전부 보고 — 점 하나 고르면 그게 선택편향이다)
  · **온도 보정**: TRAIN 꼬리(내부검증)에서 온도 T 와 임계값 q* 를 정하고 **VAL 에 그대로
    적용**한다. VAL 을 보고 고르지 않으므로 이쪽이 정직한 단일 숫자다.

## ② 포지션 편향
[[direction_4h_top5_gross_edge_candidate_20260915]]에서 «모델 실력이 아니라 고변동 꼬리의
롱 편향»이었던 전례가 있다. 그래서 **모델−롱 / 모델−숏 초과**를 같은 게이트 통과봉에서
날짜블록 CI 와 함께 낸다. 절대 수익만 보면 시장 드리프트를 실력으로 읽는다.
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np, pandas as pd, torch
from sklearn.utils.class_weight import compute_sample_weight

ROOT = Path.home() / "crypto-scalping"
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import train_eval_omega1_2_tabm_3head_20260603 as tabm  # noqa: E402
import train_eval_omega461_parent_zig075_longwindow_20260917 as E  # noqa: E402

OUT = E.OUT
GRID = np.unique(np.round(np.r_[np.arange(0.34, 1.00, 0.02), 0.75], 4))
TARGET_RATE = 0.035          # 배포본의 VAL 통과율


def log(*a, **kw): print(*a, flush=True)


def temp_scale(logit_like_prob, y, grid=np.arange(0.5, 6.01, 0.05)):
    """확률에 온도를 걸어 NLL 최소화. p^(1/T) 재정규화 = 로짓 /T 와 동치."""
    p = np.clip(logit_like_prob, 1e-9, 1.0)
    best, bT = np.inf, 1.0
    for T in grid:
        q = p ** (1.0 / T)
        q /= q.sum(1, keepdims=True)
        nll = -np.log(np.clip(q[np.arange(len(y)), y], 1e-12, None)).mean()
        if nll < best:
            best, bT = nll, float(T)
    return bT, float(best)


def gate(D, Q, q_thresh):
    da = D.argmax(1)
    qf = np.where(da > 0, Q[np.arange(len(Q)), da], Q[:, 0])
    final = np.where((da != 0) & (qf >= q_thresh), da, 0)
    return final, np.where(final == 1, 1.0, np.where(final == 2, -1.0, 0.0))


def econ(side, fwd, days, hname="1h"):
    ok = (side != 0) & np.isfinite(fwd)
    if ok.sum() < 100:
        return {"n": int(ok.sum()), "gross_bp": float("nan"), "ci95": [np.nan, np.nan]}
    pnl = side[ok] * fwd[ok]
    lo, hi, nd = E.block_ci(pnl, days[ok])
    return {"n": int(ok.sum()), "indep_days": nd, "gross_bp": float(pnl.mean()), "ci95": [lo, hi]}


def bias_report(name, side, fwd, yv, days):
    """모델 − 항상롱 / 모델 − 항상숏 초과. 같은 봉에서 짝지어 잰다."""
    ok = (side != 0) & np.isfinite(fwd)
    n = int(ok.sum())
    nl, ns = int((side[ok] == 1).sum()), int((side[ok] == -1).sum())
    pnl = side[ok] * fwd[ok]
    lo_l, hi_l, _ = E.block_ci(pnl - fwd[ok], days[ok])        # 모델 − 롱
    lo_s, hi_s, nd = E.block_ci(pnl + fwd[ok], days[ok])       # 모델 − 숏
    per_side = {}
    for lbl, s in (("롱", 1.0), ("숏", -1.0)):
        m = ok & (side == s)
        per_side[lbl] = float((s * fwd[m]).mean()) if m.sum() > 30 else float("nan")
    return {"name": name, "n": n, "long": nl, "short": ns,
            "long_share": nl / n if n else float("nan"),
            "gross_bp": float(pnl.mean()),
            "per_side_bp": per_side,
            "excess_vs_long": {"bp": float((pnl - fwd[ok]).mean()), "ci95": [lo_l, hi_l]},
            "excess_vs_short": {"bp": float((pnl + fwd[ok]).mean()), "ci95": [lo_s, hi_s]},
            "indep_days": nd}


def main() -> int:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"device={device}")
    df, base_cols = E.load()

    vm = (df.timestamp >= E.VAL[0]) & (df.timestamp <= E.VAL[1] + " 23:59:59")
    val = df[vm].reset_index(drop=True)
    yv = pd.to_numeric(val["zigzag_action"]).to_numpy(np.int64)
    xv_raw = tabm._base_input(val, base_cols)
    ev_expert = tabm._route_probs(val).argmax(1)
    vdays = val.timestamp.dt.floor("D").to_numpy()
    fwd1 = val["fwd_1h_bp"].to_numpy(np.float64)
    fwd4 = val["fwd_4h_bp"].to_numpy(np.float64)

    # ── 시장 드리프트 기준선 (실력 아닌 것을 실력이라 부르지 않기 위해) ──
    fin = np.isfinite(fwd1)
    lo, hi, nd = E.block_ci(fwd1[fin], vdays[fin])
    log(f"\n=== VAL 시장 드리프트 (전 봉 항상롱) ===")
    log(f"  1h {np.nanmean(fwd1):+.3f}bp CI[{lo:+.3f},{hi:+.3f}] · 독립일 {nd} · "
        f"라벨 롱/숏 {(yv==1).mean():.4f}/{(yv==2).mean():.4f}")

    store, bias_rows, sweep_rows = {}, [], []
    CACHE = OUT / "stageF_probs.npz"
    if CACHE.exists():
        z = np.load(CACHE, allow_pickle=True)
        store = {k: dict(v.item()) for k, v in z.items()}
        log(f"⚡확률 캐시 재사용: {CACHE} ({len(store)}개 모델) -- 재학습 건너뜀")

    # ── 배포본 ──
    b = None if store else torch.load(E.BUNDLE, map_location="cpu", weights_only=False)
    if b is not None:
        D = np.zeros((len(val), 3)); Q = np.zeros((len(val), 3))
        for ei, ename in enumerate(("bull", "bear", "chop")):
            pay = dict(b["models"][ename])
            m = tabm.ThreeHeadTabM(int(pay["n_features"]),
                                   cfg=tabm.ThreeHeadConfig(**dict(pay["config"]))).to(device)
            m.load_state_dict(pay["state_dict"]); m.eval()
            sel = ev_expert == ei
            if sel.any():
                d, q = E.heads(m, tabm._standardize_apply(xv_raw[sel], dict(pay["scaler"])), device)
                D[sel], Q[sel] = d, q
        store["deployed/-"] = {"D": D, "Q": Q, "tailD": None, "tailQ": None, "taily": None}

    # ── 내 두 판 ──
    for arm, (t0, t1) in (E.ARMS.items() if b is not None else []):
        tm = (df.timestamp >= t0) & (df.timestamp <= t1 + " 23:59:59")
        tr = df[tm].reset_index(drop=True)
        yt = pd.to_numeric(tr["zigzag_action"]).to_numpy(np.int64)
        rt = tabm._route_probs(tr)
        n = len(tr); split = max(int(n * 0.85), min(n - 1, 512))
        xs, scaler = tabm._standardize_fit(tabm._base_input(tr, base_cols))
        xv_std = tabm._standardize_apply(xv_raw, scaler)
        tail_expert = rt[split:].argmax(1)
        log(f"\n=== {arm} TRAIN {t0}~{t1} {n:,}행 · 보정용 꼬리 {n-split:,}행", flush=True)
        for seed in E.SEEDS:
            D = np.zeros((len(val), 3)); Q = np.zeros((len(val), 3))
            tD = np.zeros((n - split, 3)); tQ = np.zeros((n - split, 3))
            for ei in range(3):
                w = compute_sample_weight("balanced", y=yt).astype(np.float32) * rt[:, ei].astype(np.float32)
                m, _ = E.fit_expert(xs[:split], yt[:split], w[:split],
                                    xs[split:], yt[split:], w[split:],
                                    seed=seed, ei=ei, device=device)
                sel = ev_expert == ei
                if sel.any():
                    D[sel], Q[sel] = E.heads(m, xv_std[sel], device)
                tsel = tail_expert == ei
                if tsel.any():
                    tD[tsel], tQ[tsel] = E.heads(m, xs[split:][tsel], device)
            store[f"{arm}/{seed}"] = {"D": D, "Q": Q, "tailD": tD, "tailQ": tQ,
                                      "taily": yt[split:]}
            log(f"  seed {seed} 확률 저장 완료")
    if b is not None:
        np.savez(CACHE, **{k: np.array(v, dtype=object) for k, v in store.items()})
        log(f"확률 캐시 저장: {CACHE}")

    # ── ① 임계값 스윕 + 온도 보정 ──
    log(f"\n{'='*92}\n=== ① 품질 보정: 통과율을 낮추면 따라잡히는가 ===\n{'='*92}")
    for key, s in store.items():
        D, Q = s["D"], s["Q"]
        curve = []
        for q in GRID:
            _, side = gate(D, Q, q)
            r = econ(side, fwd1, vdays)
            curve.append({"q": float(q), "rate": float((side != 0).mean()), **r})
        sweep_rows.append({"model": key, "curve": curve})
        # 통과율을 배포본에 맞추는 지점 (VAL 에서 맞춤 -- 참고용, 선택편향 있음)
        near = min(curve, key=lambda c: abs(c["rate"] - TARGET_RATE))
        base = next(c for c in curve if abs(c["q"] - E.Q_THRESH) < 1e-9)
        log(f"\n{key}")
        log(f"  q=0.75 기본     : 통과율 {base['rate']:.3f} · 1h {base['gross_bp']:+6.2f}bp "
            f"CI[{base['ci95'][0]:+.2f},{base['ci95'][1]:+.2f}] · n {base['n']:,}")
        log(f"  통과율 {TARGET_RATE:.3f} 맞춤: q={near['q']:.2f} · 통과율 {near['rate']:.3f} · "
            f"1h {near['gross_bp']:+6.2f}bp CI[{near['ci95'][0]:+.2f},{near['ci95'][1]:+.2f}] "
            f"· n {near['n']:,}  ⚠️VAL 에서 고른 지점")
        if s["tailD"] is not None:
            T, _ = temp_scale(s["tailQ"], s["taily"])
            tQc = s["tailQ"] ** (1.0 / T); tQc /= tQc.sum(1, keepdims=True)
            _, tside = gate(s["tailD"], tQc, 0.0)          # 방향만
            tda = s["tailD"].argmax(1)
            tqf = np.where(tda > 0, tQc[np.arange(len(tQc)), tda], tQc[:, 0])
            qstar = float(np.quantile(tqf[tda != 0], 1.0 - TARGET_RATE / max((tda != 0).mean(), 1e-9)))
            qstar = min(max(qstar, 0.34), 0.999)
            Qc = Q ** (1.0 / T); Qc /= Qc.sum(1, keepdims=True)
            _, side = gate(D, Qc, qstar)
            r = econ(side, fwd1, vdays)
            log(f"  ⭐온도보정(TRAIN꼬리): T={T:.2f} q*={qstar:.3f} → 통과율 {(side!=0).mean():.3f} · "
                f"1h {r['gross_bp']:+6.2f}bp CI[{r['ci95'][0]:+.2f},{r['ci95'][1]:+.2f}] · n {r['n']:,}"
                f"   (VAL 미사용)")
            store[key]["calibrated_side"] = side

    # ── ② 포지션 편향 ──
    log(f"\n{'='*92}\n=== ② 포지션 편향 (같은 통과봉에서 짝지은 초과) ===\n{'='*92}")
    log(f"{'모델':<14}{'통과':>7}{'롱%':>7}{'롱bp':>8}{'숏bp':>8}{'총bp':>8}"
        f"{'모델−롱':>10}{'CI(모델−롱)':>20}{'모델−숏':>10}")
    for key, s in store.items():
        _, side = gate(s["D"], s["Q"], E.Q_THRESH)
        r = bias_report(key, side, fwd1, yv, vdays)
        bias_rows.append(r)
        el, es = r["excess_vs_long"], r["excess_vs_short"]
        log(f"{key:<14}{r['n']:>7,}{r['long_share']*100:>6.1f}%"
            f"{r['per_side_bp']['롱']:>8.2f}{r['per_side_bp']['숏']:>8.2f}{r['gross_bp']:>8.2f}"
            f"{el['bp']:>+10.2f}  [{el['ci95'][0]:+7.2f},{el['ci95'][1]:+7.2f}]{es['bp']:>+10.2f}")

    (OUT / "stageF_sweep.json").write_text(json.dumps(sweep_rows, indent=2, default=float))
    (OUT / "stageF_bias.json").write_text(json.dumps(bias_rows, indent=2, default=float))
    log(f"\n저장: {OUT}/stageF_sweep.json · stageF_bias.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
