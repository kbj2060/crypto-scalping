#!/usr/bin/env python3
"""K — h48qual 품질 라벨의 **비용 가정을 실측으로 정정**하고 발화율을 되살린다.

## 왜 h48qual 이 하루 0.7건만 쏘는가
라벨은 timeout 이 아니라 **경제 라벨**이다:
    long_q  = long_ret  − fee_cost − 0.20·max(−MAE,0) − 0.003·(손절로 끝남)
    action  = 1 if long_q>0 and long_q≥short_q ; 2 if short_q>0 ; else 0(CASH)
그리고 `fee_cost = (FEE 0.0005 + SLIP 0.0002) × 2다리 × 3배 = **0.0042 = 42bp**` 다.
`min_tp` 가 0.6%(60bp)이므로 TP 를 맞아도 순 18bp 고, timeout 수익은 거의 전부 CASH 가 된다.

이 프로젝트에서 실측한 **USDC 메이커 왕복 비용은 1.02bp**(메이커 0% + AS 0.51bp/다리).
라벨이 **실제보다 41배 비싼 세계**를 가정하고 있다. 임의 튜닝이 아니라 **정정**이다.

## 절차
⭐배리어 시뮬레이션은 한 번만 하고(`tb_*_ret/mae/reason` 이 전부 저장된다) fee_cost 는 사후에
쓸어본다. 그리고 **원래 값(0.0042)으로 기존 ETH 라벨을 재현하는 관문**을 먼저 통과해야
다음으로 간다 -- ETH 생성기가 커밋돼 있지 않아 BTC 판 함수를 쓰기 때문이다(로직 동일, 경로만 다름).
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path.home() / "crypto-scalping"
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
import build_omega1_2_triple_barrier_labels_btc_20260708 as TB  # noqa: E402

OUT = ROOT / "tmp/omega461_longwindow_20260917/h48_retune"
FRAME = ROOT / "tmp/omega461_longwindow_20260917/features_136_2022_2026_realfunding.parquet"
GOLD_DIR = ROOT / "tmp/causal_regen_20260516/omega1_2_triple_barrier_labels_20260619"
CFG = next(c for c in TB.CONFIGS if c.name == "h48_conservative")
ORIG_FEE = (TB.FEE_RATE + TB.SLIP_RATE) * 2.0 * 3.0          # 0.0042
FEES = {"orig_taker3x_42bp": ORIG_FEE,
        "taker1x_14bp": (TB.FEE_RATE + TB.SLIP_RATE) * 2.0,
        "usdc3x_3.1bp": 0.0000102 * 100 * 3.0 / 100 * 1.0,   # 1.02bp × 3 = 3.06bp
        "usdc1x_1.0bp": 0.000102}
FEES["usdc3x_3.1bp"] = 0.000306
MAE_PEN, SL_PEN = 0.20, 0.003


def action_at(df: pd.DataFrame, fee: float, *, mae_pen=MAE_PEN, sl_pen=SL_PEN):
    lq = df.tb_long_ret - fee - mae_pen * np.maximum(-df.tb_long_mae, 0.0) - sl_pen * (df.tb_long_reason == "sl")
    sq = df.tb_short_ret - fee - mae_pen * np.maximum(-df.tb_short_mae, 0.0) - sl_pen * (df.tb_short_reason == "sl")
    a = np.where((lq > 0) & (lq >= sq), 1, np.where(sq > 0, 2, 0))
    return a.astype(np.int64), np.asarray(np.maximum(lq, sq))


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    f = pd.read_parquet(FRAME, columns=["timestamp", "open", "high", "low", "close"])
    f["timestamp"] = pd.to_datetime(f["timestamp"])
    f = f[f.timestamp < "2026-09-01"].sort_values("timestamp").reset_index(drop=True)
    print(f"프레임 {len(f):,}봉 {f.timestamp.min()} ~ {f.timestamp.max()}", flush=True)

    cache = OUT / "tb_h48cons_sim.parquet"
    if cache.exists():
        sim = pd.read_parquet(cache)
        print(f"⚡시뮬 캐시 재사용 {len(sim):,}행", flush=True)
    else:
        print("배리어 시뮬 중(한 번만)...", flush=True)
        sim = TB._build_config_labels(f, CFG, fee_cost=0.0)   # fee 는 사후 적용
        sim.to_parquet(cache, index=False)
        print(f"시뮬 완료 {len(sim):,}행 → {cache}", flush=True)
    sim["timestamp"] = pd.to_datetime(sim["timestamp"])

    # ── 관문: 원래 fee 로 기존 ETH 라벨을 재현하는가 ──
    gold = pd.concat([pd.read_csv(GOLD_DIR / n, usecols=["timestamp", "tb_action_h48_conservative"])
                      for n in ("train_triple_barrier_labels.csv",
                                "validation_triple_barrier_labels.csv",
                                "oos_triple_barrier_labels.csv")], ignore_index=True)
    gold["timestamp"] = pd.to_datetime(gold["timestamp"])
    gold = gold.drop_duplicates("timestamp").sort_values("timestamp")
    print(f"정본 {len(gold):,}행 {gold.timestamp.min()} ~ {gold.timestamp.max()}", flush=True)

    a_orig, _ = action_at(sim, ORIG_FEE)
    chk = sim[["timestamp"]].copy(); chk["mine"] = a_orig
    j = gold.merge(chk, on="timestamp", how="inner")
    agree = float((j.tb_action_h48_conservative.to_numpy() == j.mine.to_numpy()).mean())
    print(f"\n=== 관문: 원래 fee({ORIG_FEE:.4f}) 재현 ===")
    print(f"  겹치는 {len(j):,}행 · 일치율 {agree:.6f}")
    print(f"  정본 counts {dict(j.tb_action_h48_conservative.value_counts().sort_index())}")
    print(f"  재현 counts {dict(pd.Series(j.mine).value_counts().sort_index())}")
    if agree < 0.98:
        print("🔴관문 실패 -- 원시 시세 원천이 다르거나 로직이 다르다. 여기서 멈춘다.", flush=True)
        return 1
    print("🟢관문 통과", flush=True)

    # ── fee_cost 스윕 ──
    print(f"\n=== fee_cost 스윕 (배리어는 동일, 비용 가정만 교체) ===")
    print(f"{'이름':<22}{'fee':>10}{'CASH%':>8}{'LONG%':>8}{'SHORT%':>8}{'발화율':>8}")
    rows = []
    for name, fee in FEES.items():
        a, q = action_at(sim, fee)
        sh = np.bincount(a, minlength=3) / len(a)
        rows.append({"name": name, "fee_cost": fee, "shares": sh.tolist(),
                     "active_rate": float(1 - sh[0])})
        print(f"{name:<22}{fee:>10.6f}{sh[0]*100:>7.1f}%{sh[1]*100:>7.1f}%{sh[2]*100:>7.1f}%"
              f"{(1-sh[0])*100:>7.1f}%")
        out = sim[["timestamp"]].copy()
        out["tb_action"] = a; out["tb_quality"] = q
        out.to_parquet(OUT / f"h48cons_{name}.parquet", index=False)
    (OUT / "sweep.json").write_text(json.dumps(rows, indent=2, default=float))
    print(f"\n저장: {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
