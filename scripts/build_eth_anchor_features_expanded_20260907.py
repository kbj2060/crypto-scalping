#!/usr/bin/env python3
"""앵커 방향 예측 — **피쳐 확장** (2026-09-07).

사용자: *"피쳐를 늘려서 테스트해줘"*

기존 39피쳐의 가장 큰 여백은 **시간척도**였다 -- 창이 [t−2, t] = ±10분 **하나뿐**이었다.
동시 세션 `build_eth_anchor_features_20260907` 의 `build_context`/`window_features` 를 그대로
쓰되 **창 W 를 5단계**로 늘리고, 앵커 고유 정보와 시간/간격 맥락을 더한다.

## 확장 내역
  A 다중 시간척도  W in {2, 6, 12, 24, 48}봉 = ±10분/30분/1h/2h/4h -> 36 x 5 = 180
  B 신호 원핫      8종 발동 여부 (조합 정체는 앵커 '정확도'에선 정보 0이었으나 **방향 기준 미검정**)
  C 시간 맥락      KST 시각 sin/cos · 요일 · 아시아/유럽/미국 세션
  D 앵커 간격      직전 앵커 이후 경과 봉수(log) · 직전 앵커가 같은 측면인가
  E 기존 스칼라    votes · side_is_bottom · atr_pct
=> 약 200피쳐. TRAIN 2,655행이므로 **과적합이 주된 위험**이고, 그것 자체가 검정 대상이다.

## 반드시 함께 도는 것
  L3 누수검사  단일 피쳐 AUC >= 0.95 면 FAIL (부록 L 에서 `w_quality` 를 이 검사로 잡았다)
  화이트리스트 피쳐는 이 빌더가 만든 컬럼만 -- 외부 파일에서 접두어로 긁지 않는다(부록 L2 교훈)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT, ROOT / "scripts"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import build_eth_anchor_features_20260907 as PF  # noqa: E402  (동시 세션 빌더: build_context/window_features)
import build_eth_anchor_label_dataset_20260907 as B  # noqa: E402

SRC_LAB = ROOT / "tmp/eth_anchor_training_set_20260907/train_set_P1_H48.parquet"
OUT = ROOT / "tmp/eth_anchor_features_expanded_20260907"
W_GRID = (2, 6, 12, 24, 48)
SIGNALS8 = ["taker", "strz", "sweep", "orth", "smt", "fib", "dem", "kal"]


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    D = pd.read_parquet(SRC_LAB)
    kl = pd.read_csv(B.ETH_KL, usecols=["timestamp", "open", "high", "low", "close", "volume",
                                        "quote_volume", "trades", "taker_buy_base"],
                     parse_dates=["timestamp"]).drop_duplicates("timestamp").sort_values("timestamp").reset_index(drop=True)
    seg = kl[kl["timestamp"] <= PF.FEAT_END].reset_index(drop=True)
    pos = {t: i for i, t in enumerate(seg["timestamp"].to_numpy())}
    D = D[D["timestamp"].isin(pos.keys())].reset_index(drop=True)
    D["kp"] = D["timestamp"].map(pos).astype(int)
    D = D[(D["timestamp"] >= PF.FEAT_START) & (D["kp"] > max(W_GRID) + 20)].reset_index(drop=True)
    print(f"[1/4] 앵커 {len(D):,}행 · split {D.groupby('split').size().to_dict()}", flush=True)

    print("[2/4] 컨텍스트 1회 + 창 5단계 ...", flush=True)
    X = PF.build_context(seg)
    kp = D["kp"].to_numpy()
    sgn = np.where(D["side"].to_numpy() == "bottom", 1.0, -1.0)
    parts = []
    for W in W_GRID:
        PF.W = W                                   # 모듈 상수 교체 -> window_features 가 [kp−W, kp] 를 본다
        f = PF.window_features(X, kp, sgn, 0)
        f.columns = [f"{c}_w{W}" for c in f.columns]
        parts.append(f)
        print(f"      W={W:>2}봉 · {f.shape[1]}개", flush=True)
    F = pd.concat(parts, axis=1).replace([np.inf, -np.inf], np.nan)

    print("[3/4] 앵커·시간·간격 맥락 ...", flush=True)
    sig = D["signals"].fillna("").to_numpy()
    for s in SIGNALS8:
        F[f"sig_{s}"] = np.array([1.0 if s in x.split("+") else 0.0 for x in sig])
    kst = D["timestamp"] + pd.Timedelta(hours=9)
    h = kst.dt.hour + kst.dt.minute / 60.0
    F["t_sin"] = np.sin(2 * np.pi * h / 24); F["t_cos"] = np.cos(2 * np.pi * h / 24)
    F["t_dow"] = kst.dt.dayofweek.astype(float)
    F["t_asia"] = ((h >= 8) & (h < 16)).astype(float)
    F["t_eu"] = ((h >= 16) & (h < 24)).astype(float)
    F["t_us"] = ((h >= 22) | (h < 6)).astype(float)
    gap = np.full(len(D), np.nan); same = np.full(len(D), np.nan)
    order = np.argsort(kp); prev_k = None; prev_s = None
    for o in order:
        if prev_k is not None:
            gap[o] = np.log1p(kp[o] - prev_k); same[o] = float(D["side"].to_numpy()[o] == prev_s)
        prev_k, prev_s = kp[o], D["side"].to_numpy()[o]
    F["a_gap_log"] = gap; F["a_same_side"] = same
    F["votes"] = D["n_signals"].to_numpy(float)
    F["side_is_bottom"] = (D["side"].to_numpy() == "bottom").astype(float)
    F["atr_pct_feat"] = D["atr_pct"].to_numpy(float)   # D 의 atr_pct 와 이름 충돌 회피
    print(f"      총 {F.shape[1]}개 피쳐", flush=True)

    print("[4/4] L3 누수검사 (단일 피쳐 AUC >= 0.95 면 FAIL) ...", flush=True)
    sp = D["split"].to_numpy()
    checks = {"방향(y_bin)": D["y_bin"].to_numpy(), "혼재(y3==1)": (D["y3"].to_numpy() == 1).astype(float)}
    worst = []
    for lab, y in checks.items():
        mx, arg = 0.0, None
        for w in ("VAL", "OOS"):
            m = (sp == w) & np.isfinite(y)
            for c in F.columns:
                v = F[c].to_numpy(float)[m]
                k = np.isfinite(v)
                if k.sum() < 50 or len(np.unique(y[m][k])) < 2:
                    continue
                a = roc_auc_score(y[m][k], v[k]); a = max(a, 1 - a)
                if a > mx:
                    mx, arg = a, (w, c)
        worst.append({"label": lab, "max_auc": mx, "where": arg})
        print(f"      {lab:<14} 최대 {mx:.4f} ({arg[0]}, {arg[1]}) → {'⚠️FAIL' if mx >= 0.95 else 'OK'}", flush=True)

    dup = [c for c in F.columns if c in D.columns]
    if dup:
        print(f"      D 와 겹치는 컬럼 {len(dup)}개는 F 쪽을 버린다: {dup}", flush=True)
        F = F.drop(columns=dup)
    out = pd.concat([D.reset_index(drop=True), F.reset_index(drop=True)], axis=1)
    out.to_parquet(OUT / "features_expanded.parquet", index=False)
    (OUT / "meta.json").write_text(json.dumps({
        "n_rows": len(out), "n_features": int(F.shape[1]), "feature_cols": list(F.columns),
        "W_grid": list(W_GRID), "leak_check": worst,
        "note": "피쳐는 이 빌더가 만든 컬럼만(화이트리스트). 외부 파일 접두어 수집 금지(부록 L2)."},
        indent=2, ensure_ascii=False, default=str))
    print(f"\n저장: {OUT} · {out.shape}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
