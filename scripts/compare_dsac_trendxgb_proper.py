#!/usr/bin/env python3
"""
Primary DSAC × TrendXGB 조합 비교 — 올바른 방법
DSACCompactTradingEnv 로 매 스텝 state 를 정확히 추적한 뒤
actor 의 raw action 을 XGB 로 변조해 env.step() 에 전달.

전략:
  A) DSAC 단독                          (baseline, eval_specialists_2026 와 동일)
  B) DSAC × |XGB net|                   (raw *= |p_up - p_dn|)
  C) DSAC + XGB 방향 필터               (방향 불일치 시 action→0)
  D) DSAC × XGB net (부호 포함)         (raw *= (p_up - p_dn))
  E) 그리드: B 방식 + pos_thresh 변경
"""
from __future__ import annotations
import sys, copy, math
from pathlib import Path
import numpy as np
import pandas as pd
import torch

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from ensemble.train_rl_dsac_agent import (
    DSAC_STATE_DIM, GaussianActor, DSACCompactTradingEnv,
)
from ensemble.rl_runtime_primitives import OnlineHMMDetector, MultiTimeframeFeatures

_CSV  = str(_ROOT / "data/splits/year_oos/rl_meta_2026.csv")
_CKPT = str(_ROOT / "data/ensemble/ckpt/best_dsac_agents.pth")
FEE   = 0.0004
SLIP  = 0.0002
ANNUAL = math.sqrt(365 * 24 * 12)


# ── 지표 ──────────────────────────────────────────────────────────
def _mdd(eq):
    a = np.array(eq, dtype=np.float64)
    peak = np.maximum.accumulate(a)
    return float(np.min(a / np.maximum(peak, 1e-12) - 1.0)) * 100

def _sharpe(eq):
    a = np.array(eq, dtype=np.float64)
    rets = np.diff(a) / np.maximum(a[:-1], 1e-12)
    if len(rets) < 3 or rets.std() < 1e-12:
        return 0.0
    return float(rets.mean() / rets.std() * ANNUAL)

def _metrics(label, env, eq):
    pnl = (env.balance / env.initial_balance - 1.0) * 100.0
    wr  = env.win_rate * 100.0
    return {
        "label":  label,
        "pnl":    round(pnl, 2),
        "wr":     round(wr, 1),
        "trades": env.total_trades,
        "mdd":    round(_mdd(eq), 2),
        "sharpe": round(_sharpe(eq), 3),
    }

def _print(m):
    print(f"  {m['label']:<45} {m['pnl']:>+7.2f}%  WR {m['wr']:>5.1f}%  "
          f"거래 {m['trades']:>5}  MDD {m['mdd']:>7.2f}%  Sharpe {m['sharpe']:>7.3f}")


# ── 공통: actor + env 로드 ─────────────────────────────────────────
def _load_actor():
    ckpt = torch.load(_CKPT, map_location="cpu", weights_only=False)
    state_dim = int(ckpt.get("state_dim", DSAC_STATE_DIM) or DSAC_STATE_DIM)
    actor = GaussianActor(state_dim=state_dim)
    actor.load_state_dict(ckpt["actor"])
    actor.eval()
    return actor

def _make_env(df, hmm, mtf, pos_thresh=None, close_thresh=None):
    kwargs = dict(
        initial_balance=10_000.0,
        fee=FEE, slip=SLIP, phase="val",
        hmm_detector=copy.deepcopy(hmm),
        mtf_features=mtf,
    )
    env = DSACCompactTradingEnv(df, **kwargs)
    # 기본값: pos_thresh=0.12, close_thresh=0.03 (학습 때와 동일)
    if pos_thresh is not None:
        env.pos_thresh = float(pos_thresh)
    if close_thresh is not None:
        env.close_thresh = float(close_thresh)
    return env


# ── 시뮬레이션 ─────────────────────────────────────────────────────
def run_sim(df, actor, hmm, mtf, xgb_net,
            mode="A", xgb_agree_th=0.0,
            pos_thresh=None, close_thresh=None):
    """
    mode:
      A  — DSAC 단독
      B  — raw *= |xgb_net|
      C  — 방향 불일치 시 action=0
      D  — raw *= xgb_net (부호 포함)
    """
    env = _make_env(df, hmm, mtf, pos_thresh, close_thresh)
    state = env.reset()
    eq = [env.initial_balance]
    done = False
    i = 0

    with torch.no_grad():
        while not done:
            s_t = torch.FloatTensor(state).unsqueeze(0)
            raw = float(torch.tanh(actor.forward(s_t)[0]).item())
            xnet = float(xgb_net[i]) if i < len(xgb_net) else 0.0

            if mode == "A":
                action = raw
            elif mode == "B":
                action = raw * abs(xnet)
            elif mode == "C":
                # 방향 불일치(합의 안 됨) → action 0
                if xgb_agree_th > 0:
                    if raw > 0 and xnet < xgb_agree_th:
                        action = 0.0
                    elif raw < 0 and xnet > -xgb_agree_th:
                        action = 0.0
                    else:
                        action = raw
                else:
                    # 단순 부호 일치
                    if (raw > 0 and xnet < 0) or (raw < 0 and xnet > 0):
                        action = 0.0
                    else:
                        action = raw
            elif mode == "D":
                action = raw * xnet  # 부호 반전 가능
            else:
                action = raw

            state, _, done, _ = env.step(float(np.clip(action, -1.0, 1.0)))
            bal = env.balance * (1.0 + (env.unrealized_pnl if env.pos is not None else 0.0))
            eq.append(max(bal, 1e-8))
            i += 1

    return _metrics("", env, eq), eq


def main():
    # ── 데이터 로드 ─────────────────────────────────────────────
    df = pd.read_csv(_CSV)
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["close"]).reset_index(drop=True)

    xgb_net = (df["m7_trend_xgb_up"] - df["m7_trend_xgb_dn"]).values.astype(np.float32)
    print(f"데이터: {len(df):,}행  ({df['timestamp'].min()} ~ {df['timestamp'].max()})")
    print(f"XGB net: mean={xgb_net.mean():.4f}  std={xgb_net.std():.4f}  "
          f"|net|>0.2 비율={(np.abs(xgb_net)>0.2).mean()*100:.1f}%")

    actor = _load_actor()
    hmm = OnlineHMMDetector()
    mtf = MultiTimeframeFeatures(df["close"].values.astype(np.float32))

    results = []

    print("\n추론 중...", flush=True)

    # A) DSAC 단독 (eval_specialists_2026 와 동일 조건)
    m, _ = run_sim(df, actor, hmm, mtf, xgb_net, mode="A")
    m["label"] = "A) DSAC 단독           (pos=0.12 close=0.03)"
    results.append(m); _print(m)

    # B) DSAC × |XGB net|
    m, _ = run_sim(df, actor, hmm, mtf, xgb_net, mode="B")
    m["label"] = "B) DSAC × |XGBnet|     (pos=0.12 close=0.03)"
    results.append(m); _print(m)

    # C) XGB 방향 필터 — 부호 불일치 시 0
    m, _ = run_sim(df, actor, hmm, mtf, xgb_net, mode="C", xgb_agree_th=0.0)
    m["label"] = "C) XGB 방향필터        (부호불일치→0)"
    results.append(m); _print(m)

    # C2) XGB 방향 필터 — XGB 확신 낮으면 0
    for xth in [0.10, 0.20, 0.30]:
        m, _ = run_sim(df, actor, hmm, mtf, xgb_net, mode="C", xgb_agree_th=xth)
        m["label"] = f"C) XGB 방향필터        (xgb_agree_th={xth:.2f})"
        results.append(m); _print(m)

    # D) DSAC × XGB net (부호 포함 → 반전 가능)
    m, _ = run_sim(df, actor, hmm, mtf, xgb_net, mode="D")
    m["label"] = "D) DSAC × XGBnet(부호) (pos=0.12 close=0.03)"
    results.append(m); _print(m)

    # ── 그리드: B 방식 × pos_thresh ──────────────────────────────
    print("\n그리드 서치 (B 방식 × pos_thresh)...")
    grid_results = []
    for pt in [0.08, 0.10, 0.12, 0.15, 0.18, 0.20]:
        m, _ = run_sim(df, actor, hmm, mtf, xgb_net, mode="B", pos_thresh=pt, close_thresh=0.03)
        m["label"] = f"B) DSAC×|XGBnet|       (pos={pt:.2f} close=0.03)"
        grid_results.append(m); _print(m)

    # ── 결과 출력 ────────────────────────────────────────────────
    all_r = results + grid_results
    print(f"\n{'='*85}")
    print(f"  Primary DSAC × TrendXGB 조합 비교 (2026 OOS) — 올바른 환경 추론")
    print(f"{'='*85}")
    print(f"  {'전략':<45} {'PnL%':>8}  {'WR%':>6}  {'거래':>6}  {'MDD%':>8}  {'Sharpe':>8}")
    print(f"  {'─'*45} {'─'*8}  {'─'*6}  {'─'*6}  {'─'*8}  {'─'*8}")
    for m in all_r:
        _print(m)
    print(f"{'='*85}")

    best = max(all_r, key=lambda x: x["sharpe"])
    print(f"\n✅ 최적: {best['label'].strip()}")
    print(f"   Sharpe {best['sharpe']:.3f}  PnL {best['pnl']:+.2f}%  WR {best['wr']:.1f}%  거래 {best['trades']}")


if __name__ == "__main__":
    main()
