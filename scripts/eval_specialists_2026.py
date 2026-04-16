#!/usr/bin/env python3
"""
2026 OOS 데이터로 롱/숏 스페셜리스트 + 프라이머리 DSAC 단독 성능 평가

출력:
  - 각 에이전트별 PnL, WR, Trades, MDD, Sharpe
  - 롱/숏 스페셜리스트 비교 (진입 횟수, 평균 홀딩 등)

사용법:
  python scripts/eval_specialists_2026.py
  python scripts/eval_specialists_2026.py --csv data/splits/year_oos/rl_meta_2026.csv
"""
from __future__ import annotations

import argparse
import copy
import math
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from ensemble.train_rl_dsac_agent import (
    DSAC_STATE_DIM,
    GaussianActor,
    DSACCompactTradingEnv,
)
from ensemble.train_rl_dsac_long_agent import (
    SigmoidActor as LongActor,
    LongSpecialistEnv,
    STATE_DIM as LONG_STATE_DIM,
)
from ensemble.train_rl_dsac_short_agent import (
    SigmoidActor as ShortActor,
    ShortSpecialistEnv,
    STATE_DIM as SHORT_STATE_DIM,
)
from ensemble.train_rl_agent import OnlineHMMDetector, MultiTimeframeFeatures

_DEFAULT_CSV = str(_ROOT / "data/splits/year_oos/rl_meta_2026.csv")
_PRIMARY_CKPT = str(_ROOT / "data/ensemble/ckpt/best_dsac_agents.pth")
_LONG_CKPT    = str(_ROOT / "data/ensemble/ckpt/best_dsac_long_agents.pth")
_SHORT_CKPT   = str(_ROOT / "data/ensemble/ckpt/best_dsac_short_agents.pth")

FEE  = 0.0004
SLIP = 0.0002
ANNUAL_FACTOR = math.sqrt(365 * 24 * 12)  # 5분봉 기준


# ─── 공통 유틸 ────────────────────────────────────────────────────────────────

def _mdd(eq: list[float]) -> float:
    a = np.array(eq, dtype=np.float64)
    peak = np.maximum.accumulate(a)
    dd = a / np.maximum(peak, 1e-12) - 1.0
    return float(np.min(dd)) * 100.0


def _sharpe(eq: list[float]) -> float:
    a = np.array(eq, dtype=np.float64)
    rets = np.diff(a) / np.maximum(a[:-1], 1e-12)
    if len(rets) < 3 or rets.std() < 1e-12:
        return 0.0
    return float(rets.mean() / rets.std() * ANNUAL_FACTOR)


def _load_ckpt(path: str, weights_only: bool = False):
    return torch.load(path, map_location="cpu", weights_only=weights_only)


def _fit_hmm(df_full: pd.DataFrame) -> OnlineHMMDetector:
    """2024년 데이터로 HMM fit"""
    hmm = OnlineHMMDetector()
    df_2024 = df_full[df_full["timestamp"].dt.year < 2025].copy()
    if len(df_2024) > 100:
        hmm.fit(df_2024, n_iter=30)
        print(f"  HMM fit: {len(df_2024):,} rows (2024)")
    else:
        print("  HMM fit: 2024 데이터 없음, 기본값 사용")
    return hmm


def _print_result(name: str, r: dict) -> None:
    print(
        f"\n{'─'*55}\n"
        f"  [{name}]\n"
        f"  PnL:    {r['pnl_pct']:+.2f}%\n"
        f"  WR:     {r['wr_pct']:.1f}%\n"
        f"  Trades: {r['trades']}\n"
        f"  MDD:    {r['mdd_pct']:.2f}%\n"
        f"  Sharpe: {r['sharpe']:.3f}\n"
        f"  Long entries:  {r.get('long_entries', 'N/A')}\n"
        f"  Short entries: {r.get('short_entries', 'N/A')}\n"
        f"{'─'*55}"
    )


# ─── PRIMARY DSAC ─────────────────────────────────────────────────────────────

def eval_primary(df26: pd.DataFrame, hmm: OnlineHMMDetector, device: str) -> dict:
    print("\n[PRIMARY DSAC] 평가 시작...")
    ckpt = _load_ckpt(_PRIMARY_CKPT)
    state_dim = ckpt.get("state_dim", DSAC_STATE_DIM)

    actor = GaussianActor(state_dim=state_dim)
    actor.load_state_dict(ckpt["actor"])
    actor.eval()

    mtf = MultiTimeframeFeatures(df26["close"].values.astype(np.float32))
    env = DSACCompactTradingEnv(
        df26,
        initial_balance=10_000.0,
        fee=FEE,
        slip=SLIP,
        phase="val",
        hmm_detector=copy.deepcopy(hmm),
        mtf_features=mtf,
    )

    state = env.reset()
    done = False
    eq = [env.initial_balance]

    with torch.no_grad():
        while not done:
            s_t = torch.FloatTensor(state).unsqueeze(0)
            action = float(torch.tanh(actor.forward(s_t)[0]).item())
            state, _, done, _ = env.step(action)
            bal = env.balance * (1.0 + (env.unrealized_pnl if env.pos is not None else 0.0))
            eq.append(max(bal, 1e-8))

    pnl = (env.balance / env.initial_balance - 1.0) * 100.0
    wr  = env.win_rate * 100.0

    # 롱/숏 진입 수 추출
    long_e = getattr(env, "long_entries", 0)
    short_e = getattr(env, "short_entries", 0)

    return {
        "pnl_pct": round(pnl, 4),
        "wr_pct":  round(wr, 2),
        "trades":  env.total_trades,
        "mdd_pct": round(_mdd(eq), 4),
        "sharpe":  round(_sharpe(eq), 4),
        "long_entries":  long_e,
        "short_entries": short_e,
    }


# ─── LONG SPECIALIST ──────────────────────────────────────────────────────────

def eval_long(df26: pd.DataFrame, hmm: OnlineHMMDetector, device: str) -> dict:
    print("\n[LONG SPECIALIST] 평가 시작...")
    ckpt = _load_ckpt(_LONG_CKPT)

    actor = LongActor(state_dim=LONG_STATE_DIM)
    actor.load_state_dict(ckpt["actor"])
    actor.eval()

    mtf = MultiTimeframeFeatures(df26["close"].values.astype(np.float32))
    env = LongSpecialistEnv(
        df26,
        initial_balance=10_000.0,
        fee=FEE,
        slip=SLIP,
        phase="val",
        hmm_detector=copy.deepcopy(hmm),
        mtf_features=mtf,
    )

    state = env.reset()
    done = False
    eq = [env.initial_balance]

    with torch.no_grad():
        while not done:
            s_t = torch.FloatTensor(state).unsqueeze(0)
            action = float(torch.sigmoid(actor.forward_logits(s_t)[1]).item())
            state, _, done, _ = env.step(action)
            bal = env.balance * (1.0 + (env.unrealized_pnl if env.pos is not None else 0.0))
            eq.append(max(bal, 1e-8))

    pnl = (env.balance / env.initial_balance - 1.0) * 100.0
    wr  = env.win_rate * 100.0

    return {
        "pnl_pct": round(pnl, 4),
        "wr_pct":  round(wr, 2),
        "trades":  env.total_trades,
        "mdd_pct": round(_mdd(eq), 4),
        "sharpe":  round(_sharpe(eq), 4),
        "long_entries":  env.total_trades,
        "short_entries": 0,
    }


# ─── SHORT SPECIALIST ─────────────────────────────────────────────────────────

def eval_short(df26: pd.DataFrame, hmm: OnlineHMMDetector, device: str) -> dict:
    print("\n[SHORT SPECIALIST] 평가 시작...")
    ckpt = _load_ckpt(_SHORT_CKPT)

    actor = ShortActor(state_dim=SHORT_STATE_DIM)
    actor.load_state_dict(ckpt["actor"])
    actor.eval()

    mtf = MultiTimeframeFeatures(df26["close"].values.astype(np.float32))
    env = ShortSpecialistEnv(
        df26,
        initial_balance=10_000.0,
        fee=FEE,
        slip=SLIP,
        phase="val",
        hmm_detector=copy.deepcopy(hmm),
        mtf_features=mtf,
    )

    state = env.reset()
    done = False
    eq = [env.initial_balance]

    with torch.no_grad():
        while not done:
            s_t = torch.FloatTensor(state).unsqueeze(0)
            action = float(torch.sigmoid(actor.forward_logits(s_t)[1]).item())
            state, _, done, _ = env.step(action)
            bal = env.balance * (1.0 + (env.unrealized_pnl if env.pos is not None else 0.0))
            eq.append(max(bal, 1e-8))

    pnl = (env.balance / env.initial_balance - 1.0) * 100.0
    wr  = env.win_rate * 100.0

    return {
        "pnl_pct": round(pnl, 4),
        "wr_pct":  round(wr, 2),
        "trades":  env.total_trades,
        "mdd_pct": round(_mdd(eq), 4),
        "sharpe":  round(_sharpe(eq), 4),
        "long_entries":  0,
        "short_entries": env.total_trades,
    }


# ─── MAIN ─────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv",    default=_DEFAULT_CSV)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--rl-csv", default=str(_ROOT / "data/rl_training_data_full.csv"),
                    help="HMM fit용 전체 데이터 (2024년 컬럼 필요)")
    ap.add_argument("--skip-hmm", action="store_true", help="HMM fit 생략 (빠른 테스트)")
    args = ap.parse_args()

    print(f"CSV: {args.csv}")
    df26 = pd.read_csv(args.csv)
    df26["timestamp"] = pd.to_datetime(df26["timestamp"], errors="coerce")
    df26 = df26.dropna(subset=["close"]).reset_index(drop=True)
    print(f"2026 rows: {len(df26):,}  ({df26['timestamp'].min()} ~ {df26['timestamp'].max()})")

    # HMM fit
    if args.skip_hmm:
        hmm = OnlineHMMDetector()
        print("  HMM fit 생략")
    else:
        try:
            df_full = pd.read_csv(
                args.rl_csv,
                usecols=["timestamp", "log_return", "garch_vol_z", "oi_change_rate"],
            )
            df_full["timestamp"] = pd.to_datetime(df_full["timestamp"], errors="coerce")
            hmm = _fit_hmm(df_full)
        except Exception as e:
            print(f"  HMM fit 실패 ({e}), 기본값 사용")
            hmm = OnlineHMMDetector()

    results = {}

    # ── Primary
    try:
        results["primary"] = eval_primary(df26, hmm, args.device)
        _print_result("PRIMARY DSAC", results["primary"])
    except Exception as e:
        print(f"  [ERROR] Primary: {e}")
        import traceback; traceback.print_exc()

    # ── Long
    try:
        results["long"] = eval_long(df26, hmm, args.device)
        _print_result("LONG SPECIALIST", results["long"])
    except Exception as e:
        print(f"  [ERROR] Long: {e}")
        import traceback; traceback.print_exc()

    # ── Short
    try:
        results["short"] = eval_short(df26, hmm, args.device)
        _print_result("SHORT SPECIALIST", results["short"])
    except Exception as e:
        print(f"  [ERROR] Short: {e}")
        import traceback; traceback.print_exc()

    # ── 요약 비교표
    if results:
        print("\n\n{'='*55}")
        print("  [요약] 2026 OOS 성능 비교")
        print(f"{'='*55}")
        print(f"{'모델':<20} {'PnL%':>8} {'WR%':>7} {'거래수':>7} {'MDD%':>8} {'Sharpe':>8}")
        print(f"{'─'*20} {'─'*8} {'─'*7} {'─'*7} {'─'*8} {'─'*8}")
        name_map = {"primary": "Primary DSAC", "long": "Long Specialist", "short": "Short Specialist"}
        for key in ["primary", "long", "short"]:
            if key not in results:
                continue
            r = results[key]
            print(
                f"{name_map[key]:<20} "
                f"{r['pnl_pct']:>+8.2f} "
                f"{r['wr_pct']:>7.1f} "
                f"{r['trades']:>7} "
                f"{r['mdd_pct']:>8.2f} "
                f"{r['sharpe']:>8.3f}"
            )
        print(f"{'='*55}")


if __name__ == "__main__":
    main()
