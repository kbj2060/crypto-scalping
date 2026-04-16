#!/usr/bin/env python3
"""
두 진입/청산 임계값 세트를 2026 OOS 데이터로 비교

A) eval_specialists_2026.py 방식: DSACRouter 내부 로직
   - 진입: abs(raw) > 0.12
   - 청산1: abs(raw) < 0.03  (신호 약해짐)
   - 청산2: 반대방향 > 0.12

B) trading_bot.py 현재 방식
   - 진입: abs(raw) > 0.08
   - 청산: 반대방향 > 0.05 (청산1 없음)
"""
from __future__ import annotations
import sys, os, copy, math
from pathlib import Path
import numpy as np
import pandas as pd
import torch

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from ensemble.train_rl_dsac_agent import (
    DSAC_STATE_DIM, GaussianActor, DSACRouter,
)
from ensemble.train_rl_agent import OnlineHMMDetector, MultiTimeframeFeatures

_CSV   = str(_ROOT / "data/splits/year_oos/rl_meta_2026.csv")
_CKPT  = str(_ROOT / "data/ensemble/ckpt/best_dsac_agents.pth")
FEE    = 0.0004
SLIP   = 0.0002
ANNUAL = math.sqrt(365 * 24 * 12)   # 5분봉 기준


# ── 공통 지표 ─────────────────────────────────────────────────────
def _metrics(trades: list[float], eq: list[float]) -> dict:
    if not trades:
        return {"pnl": 0.0, "wr": 0.0, "trades": 0, "mdd": 0.0, "sharpe": 0.0}
    t = np.array(trades)
    a = np.array(eq, dtype=np.float64)
    peak = np.maximum.accumulate(a)
    mdd  = float(np.min(a / np.maximum(peak, 1e-12) - 1.0)) * 100
    rets = np.diff(a) / np.maximum(a[:-1], 1e-12)
    sharpe = float(rets.mean() / rets.std() * ANNUAL) if len(rets) > 2 and rets.std() > 1e-12 else 0.0
    return {
        "pnl":    round(float(t.sum()) * 100, 2),
        "wr":     round(float((t > 0).mean()) * 100, 1),
        "trades": len(t),
        "mdd":    round(mdd, 2),
        "sharpe": round(sharpe, 3),
    }


# ── 시뮬레이션 ────────────────────────────────────────────────────
def simulate(raw_actions: np.ndarray, closes: np.ndarray,
             pos_th: float, close_th: float | None) -> tuple[list, list]:
    """
    pos_th   : 진입 및 반대방향 청산 임계값
    close_th : 신호 약해짐 청산 임계값 (None이면 적용 안 함)
    """
    pos = 0        # 0=없음, 1=LONG, 2=SHORT
    entry = 0.0
    trades, eq = [], [1.0]
    balance = 1.0

    for i, (raw, price) in enumerate(zip(raw_actions, closes)):
        if i == 0:
            eq.append(1.0)
            continue

        abs_raw = abs(raw)

        # ── 청산 판단 ─────────────────────────────────
        if pos == 1:  # LONG 보유
            exit_now = False
            if close_th is not None and abs_raw < close_th:
                exit_now = True          # 신호 약해짐
            elif raw < -pos_th:
                exit_now = True          # 반대방향 강함
            if exit_now:
                pnl = (price - entry) / entry - 2 * FEE
                trades.append(pnl)
                balance *= (1 + pnl)
                pos = 0; entry = 0.0

        elif pos == 2:  # SHORT 보유
            exit_now = False
            if close_th is not None and abs_raw < close_th:
                exit_now = True
            elif raw > pos_th:
                exit_now = True
            if exit_now:
                pnl = (entry - price) / entry - 2 * FEE
                trades.append(pnl)
                balance *= (1 + pnl)
                pos = 0; entry = 0.0

        # ── 신규 진입 ─────────────────────────────────
        if pos == 0:
            if raw > pos_th:
                pos = 1; entry = price
            elif raw < -pos_th:
                pos = 2; entry = price

        eq.append(balance)

    # 미청산 포지션 마감
    if pos != 0 and entry > 0:
        price = closes[-1]
        if pos == 1:
            pnl = (price - entry) / entry - 2 * FEE
        else:
            pnl = (entry - price) / entry - 2 * FEE
        trades.append(pnl)
        balance *= (1 + pnl)

    return trades, eq


def main():
    # ── 데이터 로드 ───────────────────────────────────────────────
    df = pd.read_csv(_CSV)
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["close"]).reset_index(drop=True)
    closes = df["close"].values.astype(np.float64)
    print(f"데이터: {len(df):,} 행  ({df['timestamp'].min()} ~ {df['timestamp'].max()})")

    # ── Primary DSAC 추론 ─────────────────────────────────────────
    ckpt = torch.load(_CKPT, map_location="cpu", weights_only=False)
    state_dim = int(ckpt.get("state_dim", DSAC_STATE_DIM) or DSAC_STATE_DIM)
    actor = GaussianActor(state_dim=state_dim)
    actor.load_state_dict(ckpt["actor"])
    actor.eval()

    hmm = OnlineHMMDetector()
    mtf = MultiTimeframeFeatures(df["close"].values.astype(np.float32))
    router = DSACRouter(actor, device="cpu", hmm_detector=hmm, mtf_features=mtf)

    print("Raw action 배치 추론 중...")
    raw_actions = np.zeros(len(df), dtype=np.float32)
    pos_dict = {}
    for i, row in enumerate(df.itertuples(index=False)):
        features = {col: getattr(row, col) for col in df.columns
                    if col != "timestamp" and isinstance(getattr(row, col), (int, float, np.floating, np.integer))}
        try:
            _, _, info = router.decide(features, pos_dict)
            raw_actions[i] = float(info.get("raw_action", 0.0))
        except Exception:
            raw_actions[i] = 0.0
        if (i + 1) % 10000 == 0:
            print(f"  {i+1:,} / {len(df):,}")

    print(f"추론 완료. raw_action: mean={raw_actions.mean():.4f} std={raw_actions.std():.4f}")

    # ── A) eval_specialists_2026.py 방식 (POS_THRESH=0.12, CLOSE_THRESH=0.03) ──
    trades_a, eq_a = simulate(raw_actions, closes, pos_th=0.12, close_th=0.03)
    m_a = _metrics(trades_a, eq_a)

    # ── B) trading_bot.py 현재 방식 (enter=0.08, exit=0.05, close_th 없음) ──
    trades_b, eq_b = simulate(raw_actions, closes, pos_th=0.08, close_th=None)
    m_b = _metrics(trades_b, eq_b)

    # ── 추가 후보: B2) enter=0.08, close_th=0.03 (반대방향 exit는 0.08) ──
    trades_b2, eq_b2 = simulate(raw_actions, closes, pos_th=0.08, close_th=0.03)
    m_b2 = _metrics(trades_b2, eq_b2)

    # ── 결과 출력 ─────────────────────────────────────────────────
    print(f"\n{'='*65}")
    print(f"  2026 OOS 진입/청산 임계값 비교")
    print(f"{'='*65}")
    print(f"{'전략':<38} {'PnL%':>7} {'WR%':>6} {'거래':>6} {'MDD%':>7} {'Sharpe':>8}")
    print(f"{'─'*38} {'─'*7} {'─'*6} {'─'*6} {'─'*7} {'─'*8}")

    rows = [
        ("A) eval 방식  pos=0.12 close=0.03", m_a),
        ("B) bot 방식   pos=0.08 close=None", m_b),
        ("B2) 혼합      pos=0.08 close=0.03", m_b2),
    ]
    for label, m in rows:
        print(f"{label:<38} {m['pnl']:>+7.2f} {m['wr']:>6.1f} {m['trades']:>6} {m['mdd']:>7.2f} {m['sharpe']:>8.3f}")

    print(f"{'='*65}")

    # ── 추천 ──────────────────────────────────────────────────────
    best_label, best_m = max(rows, key=lambda x: x[1]["sharpe"])
    print(f"\n✅ 추천: {best_label.split(')')[0].strip()[0]}  (Sharpe {best_m['sharpe']:.3f})")

    # 세부 비교를 위해 그리드 서치도 출력
    print(f"\n{'─'*65}")
    print("  그리드 서치 (pos_th × close_th)")
    print(f"{'pos_th':<8} {'close_th':<10} {'PnL%':>7} {'WR%':>6} {'거래':>6} {'Sharpe':>8}")
    print(f"{'─'*8} {'─'*10} {'─'*7} {'─'*6} {'─'*6} {'─'*8}")
    best_sharpe, best_cfg = -999, None
    for pt in [0.06, 0.08, 0.10, 0.12, 0.15]:
        for ct in [None, 0.02, 0.03, 0.04, 0.05]:
            t, e = simulate(raw_actions, closes, pos_th=pt, close_th=ct)
            m = _metrics(t, e)
            ct_str = f"{ct:.2f}" if ct is not None else "None"
            print(f"{pt:<8.2f} {ct_str:<10} {m['pnl']:>+7.2f} {m['wr']:>6.1f} {m['trades']:>6} {m['sharpe']:>8.3f}")
            if m["sharpe"] > best_sharpe:
                best_sharpe = m["sharpe"]
                best_cfg = (pt, ct, m)
    print(f"{'─'*65}")
    print(f"최적: pos_th={best_cfg[0]:.2f} close_th={best_cfg[1]}  "
          f"→ Sharpe {best_cfg[2]['sharpe']:.3f}  PnL {best_cfg[2]['pnl']:+.2f}%")


if __name__ == "__main__":
    main()
