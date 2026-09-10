#!/usr/bin/env python3
"""사이징이 실제로 먹는 양의 **보정 검증** — 전방 최대불리이탈(MAE) (2026-09-11).

사용자: *"크기에 대한 정확도가 정확히 어떤 의미지?"*

## 왜 기존 숫자로는 답이 안 되나
- 「크기 AUC 0.82」의 라벨은 `cont_pnl > +50bp` — **이진 분류**다. AUC 는 *순위* 능력이지
  "크기를 몇 % 오차로 맞힌다"가 아니다. 게다가 `atr_pct` 단독 0.8216 이고 57피쳐 기여 +0.001.
- 「변동성 전망 0.836」의 라벨은 *다음 24h RV ≥ 현재의 1.3배* — **상대 확대**다.
  「위험」 구간의 **절대** 전방 변동성은 오히려 안정 등급보다 낮다(2026-09-11 실측).
⇒ 둘 다 **사이징 입력이 아니다.** 사이징이 먹는 건 *보유 동안 예상되는 불리한 최대 이탈(MAE)*,
  그것도 **가격 단위의 보정된 추정치**다. 저장소는 그걸 검증한 적이 없다.

## 무엇을 재는가 — AUC 가 아니라 **보정**
추정 `MAE_hat = c · atr_pct · sqrt(H)`(역변동성 사이징이 암묵적으로 쓰는 형태).
계수 `c` 는 **TRAIN 에서만** 적합하고 VAL/OOS 는 건드리지 않는다.
보고 지표:
 · **적중률**: 실현 MAE ≤ 추정 의 비율(목표: 분위 q 로 잡으면 q 에 근접해야 보정된 것)
 · **상대오차 중앙값** |실현−추정|/실현
 · **분위 보정**: q ∈ {0.5,0.8,0.9,0.95} 각각에서 실제 초과율이 1−q 에 얼마나 가까운가
 · **대조군**: 상수 추정(전 구간 평균 MAE) — atr_pct 가 상수보다 나은가
출력 tmp/mae_calib_20260911/report.json
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("/home/kbj20/crypto-scalping")
OUT = Path(__file__).resolve().parents[1] / "tmp/mae_calib_20260911"
KL = ROOT / "binance_data/klines/ETHUSDT/ETHUSDT-5m-api.csv"
HOLDS = (12, 48, 288)              # 1시간 · 4시간 · 24시간 (5분봉)
QS = (0.5, 0.8, 0.9, 0.95)
TRAIN_END = pd.Timestamp("2025-08-31 23:59")
OOS_A = pd.Timestamp("2025-09-01")


def log(m):
    print(f"[mae {time.strftime('%H:%M:%S')}] {m}", flush=True)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    d = pd.read_csv(KL, usecols=["timestamp", "high", "low", "close"], parse_dates=["timestamp"])
    d = d.dropna(subset=["timestamp"]).sort_values("timestamp").set_index("timestamp")
    c = d["close"].to_numpy(float); hi = d["high"].to_numpy(float); lo = d["low"].to_numpy(float)
    n = len(c)
    log(f"5분봉 {n:,} · {d.index[0]} → {d.index[-1]}")

    tr = np.abs(np.diff(c, prepend=c[0]))
    atr = pd.Series(tr).rolling(288, min_periods=200).mean().to_numpy()
    atr_pct = atr / np.maximum(c, 1e-9)

    rep = {"note": "AUC 가 아니라 보정을 잰다. 추정=c·atr_pct·sqrt(H), c 는 TRAIN 에서만 적합",
           "holds": HOLDS, "qs": QS, "cells": {}}
    idx = np.arange(600, n - max(HOLDS) - 1, 12)          # 1시간 간격 표본
    is_tr = d.index[idx] <= TRAIN_END
    is_oos = d.index[idx] >= OOS_A
    log(f"표본 {len(idx):,} · TRAIN {int(is_tr.sum()):,} · OOS {int(is_oos.sum()):,}")

    for H in HOLDS:
        # 실현 MAE: 롱 기준 진입 후 H봉 동안 최저가까지의 하락폭(양수 비율)
        mae_l = np.array([(c[i] - lo[i + 1:i + 1 + H].min()) / c[i] for i in idx])
        mae_s = np.array([(hi[i + 1:i + 1 + H].max() - c[i]) / c[i] for i in idx])
        base = atr_pct[idx] * np.sqrt(H)
        ok = np.isfinite(base) & np.isfinite(mae_l) & np.isfinite(mae_s) & (base > 0)
        for side, mae in (("롱", mae_l), ("숏", mae_s)):
            m_tr = ok & is_tr; m_oos = ok & is_oos
            if m_tr.sum() < 500 or m_oos.sum() < 200:
                continue
            cell = {"n_train": int(m_tr.sum()), "n_oos": int(m_oos.sum()),
                    "realized_mae_median_bp": float(np.median(mae[m_oos]) * 1e4), "q": {}}
            for q in QS:
                # 계수는 TRAIN 에서만: 실현/기저 비율의 q분위
                k = float(np.quantile(mae[m_tr] / base[m_tr], q))
                est = k * base
                exceed_tr = float((mae[m_tr] > est[m_tr]).mean())
                exceed_oos = float((mae[m_oos] > est[m_oos]).mean())
                # 대조군: 상수(TRAIN q분위 실현 MAE)
                kc = float(np.quantile(mae[m_tr], q))
                exceed_const = float((mae[m_oos] > kc).mean())
                relerr = float(np.median(np.abs(mae[m_oos] - est[m_oos])
                                         / np.maximum(mae[m_oos], 1e-9)))
                cell["q"][str(q)] = {
                    "k_train": k, "target_exceed": round(1 - q, 4),
                    "exceed_train": exceed_tr, "exceed_oos": exceed_oos,
                    "exceed_oos_const_ctrl": exceed_const,
                    "calib_gap_oos": abs(exceed_oos - (1 - q)),
                    "calib_gap_const": abs(exceed_const - (1 - q)),
                    "beats_const": bool(abs(exceed_oos - (1 - q)) < abs(exceed_const - (1 - q))),
                    "median_rel_err_oos": relerr}
            rep["cells"][f"H{H//12}h|{side}"] = cell

    log("=" * 104)
    log(f"{'셀':>10} {'분위':>5} {'목표초과':>7} {'OOS초과':>8} {'상수대조':>8} "
        f"{'보정격차':>8} {'상수격차':>8} {'상대오차중앙':>11} {'상수보다':>7}")
    for k, cell in rep["cells"].items():
        for q, v in cell["q"].items():
            log(f"{k:>10} {q:>5} {v['target_exceed']:>7.2f} {v['exceed_oos']:>8.3f} "
                f"{v['exceed_oos_const_ctrl']:>8.3f} {v['calib_gap_oos']:>8.3f} "
                f"{v['calib_gap_const']:>8.3f} {v['median_rel_err_oos']:>11.2f} "
                f"{'✅' if v['beats_const'] else '—':>7}")
    tot = [(v["beats_const"], v["calib_gap_oos"]) for c in rep["cells"].values() for v in c["q"].values()]
    log("=" * 104)
    log(f"⭐상수 대조군보다 잘 보정된 셀: {sum(b for b, _ in tot)}/{len(tot)} · "
        f"OOS 보정격차 중앙 {np.median([g for _, g in tot]):.3f}")
    rep["summary"] = {"cells": len(tot), "beats_const": int(sum(b for b, _ in tot)),
                      "median_calib_gap": float(np.median([g for _, g in tot]))}
    (OUT / "report.json").write_text(json.dumps(rep, ensure_ascii=False, indent=1, default=float))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
