#!/usr/bin/env python3
"""
Primary-First + Specialist-Fill  전략  —  2026 OOS 평가

우선순위 로직 (튜닝 파라미터 2개):
  1순위: Primary env 포지션 있음  →  Primary가 직접 관리 (exit 포함)
  2순위: Primary env 포지션 없음 + long_signal >= S_ENTER_TH  →  Long 진입
         Primary env 포지션 없음 + short_signal >= S_ENTER_TH →  Short 진입
  3순위: 모두 관망  →  무포지션

스페셜리스트 신호: meta_long_raw / meta_short_raw (pre-computed, 포지션 독립)
Primary 신호: Primary 환경(DSACCompactTradingEnv) 직접 실행

그리드 서치 모드 (--grid):
  S_ENTER_TH를 전수 탐색하여 최적값 출력

사용법:
  python scripts/eval_primary_fill_2026.py
  python scripts/eval_primary_fill_2026.py --s-th 0.35
  python scripts/eval_primary_fill_2026.py --grid
"""
from __future__ import annotations

import argparse
import copy
import itertools
import logging
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from ensemble.train_rl_dsac_agent import DSAC_STATE_DIM, GaussianActor, DSACCompactTradingEnv
from ensemble.rl_runtime_primitives import OnlineHMMDetector, MultiTimeframeFeatures

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s")
log = logging.getLogger("primary_fill")

_DEFAULT_CSV = str(_ROOT / "data/splits/year_oos/rl_meta_2026.csv")
_CKPT_P = str(_ROOT / "data/ensemble/ckpt/best_dsac_agents.pth")

FEE_RATE  = 0.0004
MAX_KELLY = 0.35
ANNUAL    = math.sqrt(365 * 24 * 12)


# ─── 유틸 ─────────────────────────────────────────────────────────────────────

def _mdd(eq: np.ndarray) -> float:
    peak = np.maximum.accumulate(eq)
    return float((eq / np.maximum(peak, 1e-12) - 1.0).min()) * 100.0


def _sharpe(eq: np.ndarray) -> float:
    r = np.diff(eq) / np.maximum(eq[:-1], 1e-12)
    return float(r.mean() / r.std() * ANNUAL) if r.std() > 1e-12 else 0.0


def _sortino(pnls: np.ndarray) -> float:
    if len(pnls) == 0:
        return 0.0
    neg = pnls[pnls < 0]
    dd = float(np.sqrt((neg ** 2).mean())) if len(neg) > 0 else 1e-8
    return float(pnls.mean() / (dd + 1e-8))


# ─── Primary 모델 로드 ─────────────────────────────────────────────────────────

def load_primary(device: torch.device) -> GaussianActor:
    actor = GaussianActor(state_dim=DSAC_STATE_DIM).to(device)
    actor.load_state_dict(
        torch.load(_CKPT_P, map_location=device, weights_only=False)["actor"]
    )
    actor.eval()
    log.info("Primary DSAC 로드 완료 (state_dim=%d)", DSAC_STATE_DIM)
    return actor


# ─── 시뮬레이션 ───────────────────────────────────────────────────────────────

def run_once(
    df: pd.DataFrame,
    actor_p: GaussianActor,
    hmm: OnlineHMMDetector,
    s_enter_th: float,
    device: torch.device,
    verbose: bool = False,
) -> dict:
    """Primary env 직접 실행 + 관망 구간에서 스페셜리스트 Fill.

    Primary env는 자체 포지션 관리 (trailing stop, max hold 등) 그대로 사용.
    Primary가 flat인 구간에서만 스페셜리스트 진입 허용.

    Args:
        s_enter_th: 스페셜리스트 진입 임계값 (meta_long_raw / meta_short_raw 기준)
    """
    for col in ("meta_long_raw", "meta_short_raw"):
        if col not in df.columns:
            raise ValueError(f"CSV에 '{col}' 컬럼 없음. generate_specialist_inference.py 먼저 실행.")

    l_vals = df["meta_long_raw"].values.astype(np.float32)
    s_vals = df["meta_short_raw"].values.astype(np.float32)
    closes = df["close"].values.astype(np.float64)
    n = len(closes)

    # ── Primary env 설정 ──────────────────────────────────────────
    mtf   = MultiTimeframeFeatures(df["close"].values.astype(np.float32))
    env_p = DSACCompactTradingEnv(
        df,
        initial_balance=10_000.0,
        fee=FEE_RATE,
        slip=0.0002,
        phase="val",
        hmm_detector=copy.deepcopy(hmm),
        mtf_features=mtf,
    )
    state_p = env_p.reset()

    # ── 스페셜리스트 마스터 계좌 (Primary와 별개 운용) ────────────
    sp_pos         = 0       # -1 / 0 / +1
    sp_entry_price = 0.0
    sp_kelly       = 0.0
    sp_hold_bars   = 0
    sp_balance     = 1.0
    sp_unrealized  = 0.0
    sp_trades: list[dict] = []

    # ── 통합 Equity 곡선 추적 ──────────────────────────────────────
    # Primary: env.balance 기준
    # Specialist: sp_balance + sp_unrealized
    # 두 계좌는 독립 운용 (Primary 자본 + Specialist 자본 동시 투자)
    # 합산 Equity = Primary equity + Specialist equity (각각 1.0 시작)

    eq_p  = [1.0]  # Primary Equity
    eq_sp = [1.0]  # Specialist Equity

    done = False
    step = 0
    src_long = src_short = 0

    while not done:
        cur_close  = closes[step] if step < n else closes[-1]
        next_close = closes[step + 1] if step + 1 < n else cur_close

        # ── Primary 스텝 ──────────────────────────────────────────
        with torch.no_grad():
            ts_p   = torch.FloatTensor(state_p).unsqueeze(0).to(device)
            action = float(torch.tanh(actor_p.forward(ts_p)[0]).item())

        state_p, _, done, _ = env_p.step(action)

        # Primary 포지션 상태 확인
        primary_in_pos = (env_p.pos is not None)  # 'LONG' / 'SHORT' / None

        # Primary Equity
        bal_p = env_p.balance * (1.0 + (env_p.unrealized_pnl if primary_in_pos else 0.0))
        eq_p.append(max(bal_p / env_p.initial_balance, 1e-8))

        # ── Specialist 스텝 (Primary 관망 구간에서만 진입) ─────────
        l_val = float(l_vals[step]) if step < n else 0.0
        s_val = float(s_vals[step]) if step < n else 0.0

        # 청산: 반대 방향 신호 or Primary가 해당 방향과 반대 포지션 진입 시
        if sp_pos == 1 and (s_val >= s_enter_th or (primary_in_pos and env_p.pos == "SHORT")):
            ret      = (cur_close - sp_entry_price) / max(sp_entry_price, 1e-8)
            realized = sp_pos * ret * sp_kelly - sp_kelly * FEE_RATE * 2
            sp_trades.append({"pnl": realized, "bars": sp_hold_bars})
            sp_balance += realized
            sp_pos = 0; sp_entry_price = 0.0; sp_kelly = 0.0; sp_hold_bars = 0; sp_unrealized = 0.0

        elif sp_pos == -1 and (l_val >= s_enter_th or (primary_in_pos and env_p.pos == "LONG")):
            ret      = (cur_close - sp_entry_price) / max(sp_entry_price, 1e-8)
            realized = sp_pos * ret * sp_kelly - sp_kelly * FEE_RATE * 2
            sp_trades.append({"pnl": realized, "bars": sp_hold_bars})
            sp_balance += realized
            sp_pos = 0; sp_entry_price = 0.0; sp_kelly = 0.0; sp_hold_bars = 0; sp_unrealized = 0.0

        # 진입: Primary가 flat인 구간에서만
        if sp_pos == 0 and not primary_in_pos:
            if l_val >= s_enter_th and s_val < s_enter_th:
                sp_pos, sp_kelly, sp_entry_price = 1,  float(np.clip(l_val, 0.0, MAX_KELLY)), cur_close
                sp_balance -= sp_kelly * FEE_RATE; sp_hold_bars = 0; sp_unrealized = 0.0
                src_long += 1
            elif s_val >= s_enter_th and l_val < s_enter_th:
                sp_pos, sp_kelly, sp_entry_price = -1, float(np.clip(s_val, 0.0, MAX_KELLY)), cur_close
                sp_balance -= sp_kelly * FEE_RATE; sp_hold_bars = 0; sp_unrealized = 0.0
                src_short += 1
            elif l_val >= s_enter_th and s_val >= s_enter_th:
                if l_val >= s_val:
                    sp_pos, sp_kelly, sp_entry_price = 1,  float(np.clip(l_val, 0.0, MAX_KELLY)), cur_close
                    src_long += 1
                else:
                    sp_pos, sp_kelly, sp_entry_price = -1, float(np.clip(s_val, 0.0, MAX_KELLY)), cur_close
                    src_short += 1
                sp_balance -= sp_kelly * FEE_RATE; sp_hold_bars = 0; sp_unrealized = 0.0

        # 스페셜리스트 미실현 갱신
        if sp_pos != 0 and sp_entry_price > 0:
            ret           = (next_close - sp_entry_price) / max(sp_entry_price, 1e-8)
            sp_unrealized = sp_pos * ret * sp_kelly
            sp_hold_bars  += 1

        eq_sp.append(max(sp_balance + sp_unrealized, 1e-8))
        step += 1

    # ── 결과 집계 ──────────────────────────────────────────────────
    # Primary 최종 PnL
    primary_pnl = (env_p.balance / env_p.initial_balance - 1.0) * 100.0
    primary_wr  = env_p.win_rate * 100.0

    # Specialist 결과
    sp_pnl_arr = np.array([t["pnl"] for t in sp_trades], dtype=np.float32) if sp_trades else np.array([0.0])
    sp_pnl     = float((eq_sp[-1] - 1.0) * 100.0)
    sp_wr      = float((sp_pnl_arr > 0).mean() * 100) if len(sp_trades) else 0.0

    # 통합 Equity (동일 가중 평균)
    eq_p_arr  = np.array(eq_p,  dtype=np.float64)
    eq_sp_arr = np.array(eq_sp, dtype=np.float64)
    min_len   = min(len(eq_p_arr), len(eq_sp_arr))
    eq_comb   = (eq_p_arr[:min_len] + eq_sp_arr[:min_len]) / 2.0

    comb_pnl    = float((eq_comb[-1] - 1.0) * 100.0)
    comb_sharpe = _sharpe(eq_comb)
    comb_mdd    = _mdd(eq_comb)

    result = {
        "s_th":            s_enter_th,
        # Primary
        "p_pnl":           round(primary_pnl, 3),
        "p_wr":            round(primary_wr, 2),
        "p_trades":        env_p.total_trades,
        # Specialist
        "sp_pnl":          round(sp_pnl, 3),
        "sp_wr":           round(sp_wr, 2),
        "sp_trades":       len(sp_trades),
        "sp_long":         src_long,
        "sp_short":        src_short,
        # Combined
        "comb_pnl":        round(comb_pnl, 3),
        "comb_sharpe":     round(comb_sharpe, 3),
        "comb_mdd":        round(comb_mdd, 3),
    }

    if verbose:
        print(
            f"\n{'─'*60}\n"
            f"  [S_TH={s_enter_th:.2f}]\n"
            f"\n"
            f"  [Primary]     PnL: {primary_pnl:+.2f}%  WR: {primary_wr:.1f}%  Trades: {env_p.total_trades}\n"
            f"  [Specialist]  PnL: {sp_pnl:+.2f}%  WR: {sp_wr:.1f}%  "
            f"Trades: {len(sp_trades)} (L={src_long} S={src_short})\n"
            f"\n"
            f"  [Combined 50:50]\n"
            f"    PnL:    {comb_pnl:+.2f}%\n"
            f"    Sharpe: {comb_sharpe:.3f}\n"
            f"    MDD:    {comb_mdd:.2f}%\n"
            f"{'─'*60}"
        )

    return result


# ─── 그리드 서치 ──────────────────────────────────────────────────────────────

def run_grid(df, actor_p, hmm, device):
    candidates = [0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50, 0.60, 0.70]
    log.info("그리드 서치: S_TH %d 후보", len(candidates))

    results = []
    for i, s_th in enumerate(candidates, 1):
        r = run_once(df, actor_p, hmm, s_th, device, verbose=False)
        results.append(r)
        log.info("[%d/%d] S_TH=%.2f  Specialist PnL=%+.2f%%  Trades=%d  Combined Sharpe=%.3f",
                 i, len(candidates), s_th, r["sp_pnl"], r["sp_trades"], r["comb_sharpe"])

    results.sort(key=lambda x: x["comb_sharpe"], reverse=True)
    best = results[0]

    print(f"\n{'='*65}")
    print("  [그리드 서치 결과 — Combined Sharpe 기준]")
    print(f"{'='*65}")
    print(f"{'S_TH':>6}  {'Sp PnL%':>8}  {'Sp Tr':>6}  {'Comb PnL%':>10}  {'Comb Sharpe':>12}  {'MDD%':>7}")
    print(f"{'─'*6}  {'─'*8}  {'─'*6}  {'─'*10}  {'─'*12}  {'─'*7}")
    for r in results:
        print(f"{r['s_th']:>6.2f}  {r['sp_pnl']:>+8.2f}  {r['sp_trades']:>6}  "
              f"{r['comb_pnl']:>+10.2f}  {r['comb_sharpe']:>12.3f}  {r['comb_mdd']:>7.2f}")
    print(f"{'='*65}")
    print(f"\n  ★ 최적 S_TH={best['s_th']:.2f}  Combined PnL={best['comb_pnl']:+.2f}%  "
          f"Sharpe={best['comb_sharpe']:.3f}")

    return best


# ─── 메인 ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="Primary-First + Specialist-Fill Evaluator")
    ap.add_argument("--csv",    default=_DEFAULT_CSV)
    ap.add_argument("--s-th",   type=float, default=0.35,
                    help="스페셜리스트 진입 임계값 (기본 0.35)")
    ap.add_argument("--grid",   action="store_true",
                    help="S_TH 전수 그리드 서치")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--rl-csv", default=str(_ROOT / "data/rl_training_data_full.csv"))
    args = ap.parse_args()

    device = torch.device(
        "cuda" if (args.device == "auto" and torch.cuda.is_available()) else "cpu"
    )
    log.info("Device: %s", device)

    # 데이터 로드
    df = pd.read_csv(args.csv)
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["close"]).reset_index(drop=True)
    log.info("2026 OOS: %d rows  (%s ~ %s)",
             len(df), df["timestamp"].iloc[0], df["timestamp"].iloc[-1])

    # HMM fit
    hmm = OnlineHMMDetector()
    try:
        df_full = pd.read_csv(
            args.rl_csv,
            usecols=["timestamp", "log_return", "garch_vol_z", "oi_change_rate"],
        )
        df_full["timestamp"] = pd.to_datetime(df_full["timestamp"], errors="coerce")
        df_2024 = df_full[df_full["timestamp"].dt.year < 2025]
        if len(df_2024) > 100:
            hmm.fit(df_2024, n_iter=30)
            log.info("HMM fit: %d rows (2024)", len(df_2024))
    except Exception as e:
        log.warning("HMM fit 생략: %s", e)

    # Primary 모델 로드
    actor_p = load_primary(device)

    print("\n[참조] 개별 모델 성능")
    print("  Primary DSAC    : PnL +307.74%  WR 55.0%  Trades  814  Sharpe 23.197")
    print("  Long Specialist : PnL +246.27%  WR 57.2%  Trades  731  Sharpe 17.668")
    print("  Short Specialist: PnL +243.87%  WR 49.9%  Trades  373  Sharpe 13.873")

    if args.grid:
        run_grid(df, actor_p, hmm, device)
    else:
        run_once(df, actor_p, hmm, args.s_th, device, verbose=True)


if __name__ == "__main__":
    main()
