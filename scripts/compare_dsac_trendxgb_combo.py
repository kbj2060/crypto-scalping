#!/usr/bin/env python3
"""
Primary DSAC × TrendXGB 조합 비교 (2026 OOS)

전략:
  A) DSAC 단독               (pos_th=0.15, close_th=None)
  B) DSAC + XGB 방향 필터     진입 시 TrendXGB 방향이 DSAC 와 일치해야
  C) DSAC + XGB 방향 필터     + 청산 시 TrendXGB 가 반대 방향 확인
  D) DSAC × XGB 가중 결합     raw = dsac_raw * (p_up - p_dn)
  E) 그리드 서치: (pos_th) × (xgb_agree_th)
"""
from __future__ import annotations
import sys, os, math
from pathlib import Path
import numpy as np
import pandas as pd
import torch

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from ensemble.train_rl_dsac_agent import DSAC_STATE_DIM, GaussianActor, DSACRouter
from ensemble.train_rl_agent import OnlineHMMDetector, MultiTimeframeFeatures

_CSV  = str(_ROOT / "data/splits/year_oos/rl_meta_2026.csv")
_CKPT = str(_ROOT / "data/ensemble/ckpt/best_dsac_agents.pth")
FEE   = 0.0004
ANNUAL = math.sqrt(365 * 24 * 12)


# ── 지표 ──────────────────────────────────────────────────────────
def _metrics(trades: list[float], eq: list[float], label="") -> dict:
    if not trades:
        return {"label": label, "pnl": 0.0, "wr": 0.0, "trades": 0, "mdd": 0.0, "sharpe": 0.0}
    t = np.array(trades)
    a = np.array(eq, dtype=np.float64)
    peak = np.maximum.accumulate(a)
    mdd  = float(np.min(a / np.maximum(peak, 1e-12) - 1.0)) * 100
    rets = np.diff(a) / np.maximum(a[:-1], 1e-12)
    sharpe = float(rets.mean() / rets.std() * ANNUAL) if len(rets) > 2 and rets.std() > 1e-12 else 0.0
    return {
        "label":  label,
        "pnl":    round(float(t.sum()) * 100, 2),
        "wr":     round(float((t > 0).mean()) * 100, 1),
        "trades": len(t),
        "mdd":    round(mdd, 2),
        "sharpe": round(sharpe, 3),
    }

def _print(m: dict):
    print(f"  {m['label']:<45} {m['pnl']:>+7.2f}%  WR {m['wr']:>5.1f}%  "
          f"거래 {m['trades']:>5}  MDD {m['mdd']:>7.2f}%  Sharpe {m['sharpe']:>7.3f}")


# ── 시뮬레이션 엔진 ───────────────────────────────────────────────
def _sim(
    closes: np.ndarray,
    dsac_raw: np.ndarray,
    xgb_net: np.ndarray,          # p_up - p_dn  ∈ [-1, 1]
    pos_th: float    = 0.15,
    xgb_entry_th: float = 0.0,    # 진입 시 XGB net > th (롱) 또는 < -th (숏)
    xgb_exit_th:  float = 0.0,    # 청산 시 XGB 반대 방향 net 절대값 > th
    use_combined: bool  = False,   # D) dsac_raw × xgb_net 가중 결합
) -> tuple[list, list]:
    pos, entry = 0, 0.0
    trades, eq, balance = [], [1.0], 1.0

    for i in range(1, len(closes)):
        raw   = float(dsac_raw[i])
        xnet  = float(xgb_net[i])
        price = float(closes[i])

        if use_combined:
            # 결합 신호: DSAC 방향 × XGB 확신도
            eff_raw = raw * max(abs(xnet), 0.0)
        else:
            eff_raw = raw

        abs_raw = abs(eff_raw)

        # ── 청산 ─────────────────────────────────────────
        if pos == 1:  # LONG
            close_now = eff_raw < -pos_th
            if xgb_exit_th > 0:
                close_now = close_now or xnet < -xgb_exit_th
            if close_now:
                pnl = (price - entry) / entry - 2 * FEE
                trades.append(pnl); balance *= (1 + pnl)
                pos = 0; entry = 0.0

        elif pos == 2:  # SHORT
            close_now = eff_raw > pos_th
            if xgb_exit_th > 0:
                close_now = close_now or xnet > xgb_exit_th
            if close_now:
                pnl = (entry - price) / entry - 2 * FEE
                trades.append(pnl); balance *= (1 + pnl)
                pos = 0; entry = 0.0

        # ── 진입 ─────────────────────────────────────────
        if pos == 0:
            want_long  = eff_raw > pos_th and (xgb_entry_th == 0 or xnet > xgb_entry_th)
            want_short = eff_raw < -pos_th and (xgb_entry_th == 0 or xnet < -xgb_entry_th)
            if want_long:
                pos = 1; entry = price
            elif want_short:
                pos = 2; entry = price

        eq.append(balance)

    # 미청산 마감
    if pos != 0 and entry > 0:
        price = closes[-1]
        pnl = (price - entry) / entry * (1 if pos == 1 else -1) - 2 * FEE
        trades.append(pnl)

    return trades, eq


def main():
    # ── 데이터 로드 ─────────────────────────────────────────────
    df = pd.read_csv(_CSV)
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["close"]).reset_index(drop=True)
    closes  = df["close"].values.astype(np.float64)
    xgb_net = (df["m7_trend_xgb_up"] - df["m7_trend_xgb_dn"]).values.astype(np.float32)
    print(f"데이터: {len(df):,} 행  ({df['timestamp'].min()} ~ {df['timestamp'].max()})")
    print(f"XGB net (p_up-p_dn): mean={xgb_net.mean():.4f}  std={xgb_net.std():.4f}  "
          f"up비율={(xgb_net>0).mean()*100:.1f}%  dn비율={(xgb_net<0).mean()*100:.1f}%")

    # ── Primary DSAC 배치 추론 ────────────────────────────────
    ckpt = torch.load(_CKPT, map_location="cpu", weights_only=False)
    state_dim = int(ckpt.get("state_dim", DSAC_STATE_DIM) or DSAC_STATE_DIM)
    actor = GaussianActor(state_dim=state_dim)
    actor.load_state_dict(ckpt["actor"])
    actor.eval()

    hmm = OnlineHMMDetector()
    mtf = MultiTimeframeFeatures(df["close"].values.astype(np.float32))
    router = DSACRouter(actor, device="cpu", hmm_detector=hmm, mtf_features=mtf)

    print("DSAC 배치 추론 중...")
    dsac_raw = np.zeros(len(df), dtype=np.float32)
    for i, row in enumerate(df.itertuples(index=False)):
        features = {col: getattr(row, col) for col in df.columns
                    if col != "timestamp"
                    and isinstance(getattr(row, col), (int, float, np.floating, np.integer))}
        try:
            _, _, info = router.decide(features, {})
            dsac_raw[i] = float(info.get("raw_action", 0.0))
        except Exception:
            dsac_raw[i] = 0.0
        if (i + 1) % 10000 == 0:
            print(f"  {i+1:,}/{len(df):,}")

    print(f"DSAC raw: mean={dsac_raw.mean():.4f}  std={dsac_raw.std():.4f}")

    # ── 전략 비교 ────────────────────────────────────────────────
    PT = 0.15   # 기준 pos_th

    results = []

    # A) DSAC 단독
    t, e = _sim(closes, dsac_raw, xgb_net, pos_th=PT)
    results.append(_metrics(t, e, f"A) DSAC 단독            (pos_th={PT})"))

    # B) DSAC + XGB 방향 필터 (진입만)
    for xth in [0.05, 0.10, 0.15, 0.20, 0.30]:
        t, e = _sim(closes, dsac_raw, xgb_net, pos_th=PT, xgb_entry_th=xth)
        results.append(_metrics(t, e, f"B) +XGB 진입필터        (xgb_entry_th={xth:.2f})"))

    # C) DSAC + XGB 청산 필터 (청산만)
    for xth in [0.10, 0.20, 0.30]:
        t, e = _sim(closes, dsac_raw, xgb_net, pos_th=PT, xgb_exit_th=xth)
        results.append(_metrics(t, e, f"C) +XGB 청산필터        (xgb_exit_th={xth:.2f})"))

    # D) DSAC × XGB 가중 결합
    t, e = _sim(closes, dsac_raw, xgb_net, pos_th=PT, use_combined=True)
    results.append(_metrics(t, e, f"D) DSAC×XGB 결합        (pos_th={PT})"))

    # D2) 결합 + pos_th 조정
    for pt in [0.05, 0.08, 0.10, 0.12]:
        t, e = _sim(closes, dsac_raw, xgb_net, pos_th=pt, use_combined=True)
        results.append(_metrics(t, e, f"D) DSAC×XGB 결합        (pos_th={pt:.2f})"))

    # ── 출력 ─────────────────────────────────────────────────────
    print(f"\n{'='*85}")
    print(f"  Primary DSAC × TrendXGB 조합 비교 (2026 OOS)")
    print(f"{'='*85}")
    print(f"  {'전략':<45} {'PnL%':>8}  {'WR%':>7}  {'거래':>6}  {'MDD%':>8}  {'Sharpe':>8}")
    print(f"  {'─'*45} {'─'*8}  {'─'*7}  {'─'*6}  {'─'*8}  {'─'*8}")
    for m in results:
        _print(m)
    print(f"{'='*85}")

    best = max(results, key=lambda x: x["sharpe"])
    print(f"\n✅ 최적: {best['label'].strip()}")
    print(f"   Sharpe {best['sharpe']:.3f}  PnL {best['pnl']:+.2f}%  거래 {best['trades']}")

    # ── 그리드 서치: pos_th × xgb_entry_th ───────────────────────
    print(f"\n{'─'*75}")
    print("  그리드 서치: pos_th × xgb_entry_th (진입 필터)")
    print(f"  {'pos_th':<8} {'xgb_th':<8} {'PnL%':>8}  {'WR%':>6}  {'거래':>6}  {'Sharpe':>8}")
    print(f"  {'─'*8} {'─'*8} {'─'*8}  {'─'*6}  {'─'*6}  {'─'*8}")
    best_s, best_g = -999, None
    for pt in [0.10, 0.12, 0.15, 0.18, 0.20]:
        for xth in [0.0, 0.05, 0.10, 0.15, 0.20, 0.25]:
            t, e = _sim(closes, dsac_raw, xgb_net, pos_th=pt, xgb_entry_th=xth)
            m = _metrics(t, e)
            print(f"  {pt:<8.2f} {xth:<8.2f} {m['pnl']:>+8.2f}  {m['wr']:>6.1f}  {m['trades']:>6}  {m['sharpe']:>8.3f}")
            if m["sharpe"] > best_s:
                best_s = m["sharpe"]
                best_g = (pt, xth, m)
    print(f"{'─'*75}")
    print(f"  최적: pos_th={best_g[0]:.2f}  xgb_entry_th={best_g[1]:.2f}  "
          f"→ Sharpe {best_g[2]['sharpe']:.3f}  PnL {best_g[2]['pnl']:+.2f}%  거래 {best_g[2]['trades']}")


if __name__ == "__main__":
    main()
