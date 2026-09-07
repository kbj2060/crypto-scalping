#!/usr/bin/env python3
"""V자 급등락 라벨 분해 — **크기인가 방향인가** (2026-09-07).

사용자: *"V자 급등락이 곧 되돌림 아니야?"*

## 질문
V자 급등락 라벨은 `move >= 1.5*atr and confirmed` 다.
`move` 는 **되돌림 방향** 움직임이므로 사용자 지적대로 방향은 되돌림이 맞다.
그런데 `>= 1.5*atr` 는 **크기 조건**이다. 그리고 이번 세션에서
  크기 축 AUC 0.77 (raw atr_pct 만으로 0.74~0.79) -- 예측된다
  방향 축 AUC 0.54 -- 안 된다
가 확정됐다. 그렇다면 V자의 AUC 0.67 은 **크기 성분**일 수 있다.
(방증: 이 저장소의 V자반등 최종 판정이 "매매 엣지 없음 / 재량 지표로만 유효" 였다 --
 AUC 는 높은데 거래 엣지가 없는 것은 크기만 맞히는 모델의 서명이다.)

## 분해 — 같은 모집단·같은 피쳐 위에서 라벨만 셋
스윕 봉의 극점(하락스윕=저가, 상승스윕=고가)을 기준으로 **양방향**을 같은 방식으로 잰다.
  하락스윕: reb = fast_high_max - low  (위)   · cont = low - fast_low_min  (아래)
  상승스윕: reb = high - fast_low_min  (아래) · cont = fast_high_max - high (위)

  L0 원본     `reb >= 1.5*atr and confirmed`            -- 배포 라벨
  S  크기     `max(reb, cont) >= 1.5*atr`                -- "뭐라도 크게 움직였나"(방향 무관)
  D  방향     S==1 인 사건에서만 `reb > cont`            -- 크기 통제, **극점 기준**
  D2 방향'    같은 것을 **종가 기준**으로                 -- ⚠️D 의 구조적 비대칭 제거

⚠️D 의 양성률이 **0.8352** 로 나왔다. 스윕 정의가 "저가가 레벨 아래로 갔다가 **종가는 위에서
마감**" 이라 사건 자체에 이미 되돌림이 들어 있고, 그 봉의 **극점**에서 재면 reb 가 구조적으로
커진다. 그래서 중립 기준점(스윕 봉 종가)에서 양방향을 대칭으로 재는 D2 를 함께 낸다.

D 가 0.5 근처로 떨어지면 L0 의 0.67 은 크기였다는 뜻이다.
D 가 0.6 대를 지키면 앵커 방향 축 실패는 모집단·지평 탓이라는 뜻이고 그쪽이 더 중요하다.

## 파리티 (필수)
내가 재계산한 `reb` 가 저장된 `rebound_move` 와 일치해야 한다. 안 맞으면 분해가 무의미하다.
"""
from __future__ import annotations

import importlib.util
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
LBL = ROOT / "scripts/build_eth_5m_liquidity_sweep_v_rebound_labels_20260829.py"
FEAT = ROOT / "data/labels/eth_5m_sweep_v_rebound_20260829/eth_5m_sweep_v_rebound_features_tier0.csv"
LABELS = ROOT / "data/labels/eth_5m_sweep_v_rebound_20260829/eth_5m_sweep_v_rebound_labels.csv"
OUT = ROOT / "tmp/eth_v_rebound_decomp_20260907"
SPLITS = [("TRAIN", None, "2025-09-01"), ("VAL", "2025-09-01", "2026-01-01"),
          ("OOS", "2026-01-01", "2026-04-01"), ("HOLDOUT_SPENT", "2026-04-01", None)]
SEED, N_EST, BOOT = 20260907, 4, 1200


def tznaive(x):
    """⚠️이 저장소는 소스마다 tz-aware/naive 가 섞여 있다(V자 라벨 CSV=UTC, 신호 프레임=naive).
    조인 전에 반드시 통일한다 -- 안 하면 merge 가 ValueError 로 죽거나(운 좋을 때)
    조용히 0행을 낸다(MASHT 스왑 1차에서 창 유효 0개가 정확히 이것이었다)."""
    x = pd.to_datetime(x, utc=True)
    return x.dt.tz_localize(None) if hasattr(x, "dt") else x.tz_localize(None)


def _load(n, p):
    s = importlib.util.spec_from_file_location(n, p)
    m = importlib.util.module_from_spec(s); s.loader.exec_module(m); return m


def day_ci(y, p, d, rng, B=BOOT):
    u = np.unique(d)
    if len(u) < 5: return (np.nan, np.nan)
    idx = {x: np.flatnonzero(d == x) for x in u}; o = []
    for _ in range(B):
        i = np.concatenate([idx[x] for x in rng.choice(u, len(u), replace=True)])
        if len(np.unique(y[i])) > 1: o.append(roc_auc_score(y[i], p[i]))
    return (float(np.percentile(o, 2.5)), float(np.percentile(o, 97.5))) if len(o) > B // 3 else (np.nan, np.nan)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    from tabpfn import TabPFNClassifier
    import torch
    LB = _load("vrl", LBL)
    impl = LB.load_impl()
    frame = impl.add_causal_columns(impl.load_5m(LB.SOURCE))
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[1/5] 소스 프레임 {len(frame):,} · device {dev}", flush=True)

    ts = frame["timestamp"].to_numpy()
    atr_s = frame["atr"].to_numpy()
    hi = frame["high"].to_numpy(float); lo = frame["low"].to_numpy(float)
    FB, LA = LB.V_REBOUND_FAST_BARS, LB.LOOKAHEAD_BARS
    rows = []
    for i in range(impl.SWEEP_LOOKBACK_BARS, len(frame) - LA):
        r = frame.iloc[i]
        atr = atr_s[i - 1]
        if not np.isfinite(atr) or atr <= 0: continue
        fh = hi[i + 1:i + 1 + FB]; fl = lo[i + 1:i + 1 + FB]
        fut_c = frame["close"].to_numpy(float)[i + 1:i + 1 + LA]
        if len(fh) < FB: continue
        L = r["sweep_level_low"]
        if np.isfinite(L) and r["low"] < L and r["close"] > L:
            reb = float(fh.max() - r["low"]); cont = float(r["low"] - fl.min())
            c0 = float(r["close"])
            rows.append({"i": i, "timestamp": pd.Timestamp(ts[i]), "side": "downside",
                         "atr": float(atr), "reb": reb, "cont": cont,
                         "reb_c": float(fh.max() - c0), "cont_c": float(c0 - fl.min()),
                         "confirmed": bool((fut_c > L).all())})
        L = r["sweep_level_high"]
        if np.isfinite(L) and r["high"] > L and r["close"] < L:
            reb = float(r["high"] - fl.min()); cont = float(fh.max() - r["high"])
            c0 = float(r["close"])
            rows.append({"i": i, "timestamp": pd.Timestamp(ts[i]), "side": "upside",
                         "atr": float(atr), "reb": reb, "cont": cont,
                         "reb_c": float(c0 - fl.min()), "cont_c": float(fh.max() - c0),
                         "confirmed": bool((fut_c < L).all())})
    R = pd.DataFrame(rows)
    R["timestamp"] = tznaive(R["timestamp"])
    print(f"[2/5] 스윕 사건 재계산 {len(R):,} · {R.timestamp.min()} ~ {R.timestamp.max()}", flush=True)

    # ── 파리티: 저장된 rebound_move 와 내 reb 가 같은가
    S0 = pd.read_csv(LABELS)
    S0["timestamp"] = tznaive(S0["timestamp"])
    M = R.merge(S0[["timestamp", "side", "rebound_move", "label", "atr"]],
                on=["timestamp", "side"], suffixes=("", "_ref"))
    d_reb = np.abs(M["reb"] - M["rebound_move"]); d_atr = np.abs(M["atr"] - M["atr_ref"])
    par = {"n_matched": int(len(M)), "n_stored": int(len(S0)),
           "max_abs_diff_reb": float(d_reb.max()), "max_abs_diff_atr": float(d_atr.max()),
           "PASS": bool(len(M) >= 0.98 * len(S0) and d_reb.max() < 1e-6 and d_atr.max() < 1e-6)}
    print(f"[3/5] 파리티: 매칭 {len(M):,}/{len(S0):,} · reb 최대오차 {d_reb.max():.2e} "
          f"· atr 최대오차 {d_atr.max():.2e} → {'PASS' if par['PASS'] else '🔴FAIL'}", flush=True)
    if not par["PASS"]:
        print("   🔴재현 실패 -- 분해 중단"); (OUT / "parity.json").write_text(json.dumps(par, indent=1)); return 1
    # 원본 라벨도 재현되는지
    l0 = ((M["reb"] >= 1.5 * M["atr"]) & M["confirmed"]).astype(int)
    print(f"      원본 label 재현 일치율 {(l0 == M['label']).mean():.4f}", flush=True)

    # ── 세 라벨
    M["L0"] = M["label"].astype(int)
    M["S"] = ((np.maximum(M["reb"], M["cont"])) >= 1.5 * M["atr"]).astype(int)
    M["D"] = np.where(M["S"] == 1, (M["reb"] > M["cont"]).astype(int), np.nan)
    M["S2"] = ((np.maximum(M["reb_c"], M["cont_c"])) >= 1.5 * M["atr"]).astype(int)
    M["D2"] = np.where(M["S2"] == 1, (M["reb_c"] > M["cont_c"]).astype(int), np.nan)
    print(f"[4/5] 라벨 양성률 · L0 {M.L0.mean():.4f} · S {M.S.mean():.4f} "
          f"· D {np.nanmean(M['D']):.4f} (S==1 {int(M.S.sum()):,}건) "
          f"· D2 {np.nanmean(M['D2']):.4f} (S2==1 {int(M.S2.sum()):,}건, 종가기준)", flush=True)

    F = pd.read_csv(FEAT)
    F["timestamp"] = tznaive(F["timestamp"])
    VR = _load("vr", ROOT / "scripts/live_eth_sweep_v_rebound_signal_20260829.py")
    feats = [c for c in VR.FEATURES if c in F.columns]
    print(f"      배포 피쳐 {len(feats)}/{len(VR.FEATURES)}", flush=True)
    M2 = M.drop(columns=[c for c in feats if c in M.columns], errors="ignore")   # 컬럼 충돌 방지
    J = M2.merge(F[["timestamp", "side"] + feats], on=["timestamp", "side"], how="inner")
    print(f"      피쳐 조인 {len(J):,}", flush=True)

    sp = np.array(["?"] * len(J), dtype=object)
    for nm, a, b in SPLITS:
        m = np.ones(len(J), bool)
        if a: m &= (J.timestamp >= a).to_numpy()
        if b: m &= (J.timestamp < b).to_numpy()
        sp[m] = nm
    day = J.timestamp.dt.floor("D").to_numpy()
    X = J[feats].to_numpy(np.float64)

    print("\n[5/5] 같은 모집단·같은 피쳐, 라벨만 교체", flush=True)
    print("=" * 96, flush=True)
    res = []
    for lab, desc in [("L0", "원본 (되돌림 ≥1.5ATR & confirmed)"),
                      ("S", "크기 (양방향 중 하나라도 ≥1.5ATR)"),
                      ("D", "방향-극점기준 (크기 통제)"),
                      ("D2", "⭐방향-종가기준 (크기 통제·대칭)")]:
        y = J[lab].to_numpy(float)
        ok = np.isfinite(y) & np.isfinite(X).all(axis=1)
        tr = ok & (sp == "TRAIN")
        if tr.sum() < 200 or len(np.unique(y[tr])) < 2:
            print(f"  {lab}: TRAIN 부족"); continue
        clf = TabPFNClassifier(device=dev, n_estimators=N_EST, random_state=SEED,
                               ignore_pretraining_limits=True, memory_saving_mode=True)
        clf.fit(np.nan_to_num(X[tr]).astype(np.float32), y[tr].astype(int))
        rec = {"label": lab, "desc": desc, "n_train": int(tr.sum()), "pos_rate": float(y[ok].mean())}
        line = f"  {lab:<3}{desc:<36}"
        for w, _, _ in SPLITS[1:]:
            te = ok & (sp == w)
            if te.sum() < 40 or len(np.unique(y[te])) < 2:
                line += f"{w[:3]} -           "; continue
            p = clf.predict_proba(np.nan_to_num(X[te]).astype(np.float32))[:, 1]
            a = roc_auc_score(y[te].astype(int), p); l, h = day_ci(y[te].astype(int), p, day[te], rng)
            rec[f"{w}_auc"], rec[f"{w}_lo"], rec[f"{w}_n"] = a, l, int(te.sum())
            line += f"{w[:3]} {a:.4f}[{l:.3f}] "
        rec["mean3"] = np.nanmean([rec.get(f"{w}_auc", np.nan) for w, _, _ in SPLITS[1:]])
        res.append(rec)
        print(line + f" mean3 {rec['mean3']:.4f} (n_tr {rec['n_train']:,}, 양성 {rec['pos_rate']:.3f})", flush=True)

    A = pd.DataFrame(res); A.to_csv(OUT / "decomp.csv", index=False)
    (OUT / "parity.json").write_text(json.dumps(par, indent=1, ensure_ascii=False))
    if len(A) >= 4:
        g = lambda k: float(A[A.label == k].mean3.iloc[0])
        print("\n" + "=" * 96, flush=True)
        print(f"  원본 {g('L0'):.4f} · 크기 {g('S'):.4f} · 방향(극점) {g('D'):.4f} "
              f"· **방향(종가) {g('D2'):.4f}**", flush=True)
        di = g("D2")
        print(f"  ⇒ {'크기 성분이 대부분 -- V자 0.67 은 되돌림 예측이 아니라 크기 예측이다' if di < 0.56 else '방향 성분이 살아 있다 -- 앵커 축 실패는 모집단/지평 탓일 수 있다'}",
              flush=True)
    print(f"\n저장: {OUT}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
