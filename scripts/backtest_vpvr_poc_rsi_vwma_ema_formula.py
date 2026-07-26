#!/usr/bin/env python3
"""
VPVR/POC + RSI + VWMA100 + EMA 룰기반 공식 백테스트 & 파라미터 최적화
================================================================================
`ensemble/vpvr_poc_rsi_vwma_ema_formula.py`의 FormulaEngine이 생성하는
composite_score 기반으로 롱/숏 진입 → TP/SL/트레일링/시그널청산을 시뮬레이션하고,
**총 PnL(pnl_pct) 최대화**를 목표로 파라미터를 탐색한다(Optuna 사용 가능 시 TPE,
아니면 random search로 폴백 — 저장소가 optuna에 의존하지만 `scripts/backtest_msaf_formula.py`
류의 "공식" 백테스트들은 numpy random search 관례를 쓰므로 둘 다 지원).

데이터: `features/engineering.py`(RSI) + `core/cvp.py`(VPVR/POC, add_cvp_features)로
생성된 5분봉 피처 CSV (기본 `data/training_features_5m.csv`). 최소 필요 컬럼:
open, high, low, close, volume, rsi, cvp_poc_dist, cvp_volume_imbalance, cvp_regime.

사용 예:
    python scripts/backtest_vpvr_poc_rsi_vwma_ema_formula.py \\
        --data data/training_features_5m.csv --trials 300 --leverage 10
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ensemble.vpvr_poc_rsi_vwma_ema_formula import (
    REQUIRED_INPUT_COLS,
    FormulaConfig,
    FormulaEngine,
)

# FormulaConfig 필드 중 파라미터 탐색 대상(ema/vwma 기간은 요청대로 고정값 사용).
TUNABLE_FIELDS = [
    "w_ema", "w_vwma", "w_poc", "w_rsi",
    "k_ema", "k_vwma", "k_poc1", "k_poc2", "w_regime_gate", "rsi_regime_power",
    "theta_entry", "theta_exit", "theta_full",
    "rsi_hot", "rsi_cold", "base_size",
    "tp_pct", "sl_pct", "trail_pct", "max_hold_bars", "cooldown_bars",
]


@dataclass
class SimResult:
    pnl_pct: float
    mdd_pct: float
    sharpe: float
    trades: int
    win_rate_pct: float
    equity_final: float
    params: dict


def calc_mdd(eq: np.ndarray) -> float:
    peak = np.maximum.accumulate(eq)
    dd = eq / np.maximum(peak, 1e-12) - 1.0
    return float(dd.min()) * 100.0


def calc_sharpe(eq: np.ndarray, bars_per_year: int = 365 * 24 * 12) -> float:
    if len(eq) < 10:
        return 0.0
    r = np.diff(eq) / np.maximum(eq[:-1], 1e-12)
    s = float(np.std(r))
    if s < 1e-12:
        return 0.0
    return float(np.mean(r) / s * math.sqrt(bars_per_year))


# ════════════════════════════════════════════════════════════════
# 데이터 로드
# ════════════════════════════════════════════════════════════════
def load_data(path: str, timestamp_col: str = "timestamp") -> pd.DataFrame:
    df = pd.read_csv(path)

    missing = [c for c in REQUIRED_INPUT_COLS if c not in df.columns]
    if missing:
        raise KeyError(
            f"{path} 에 필수 컬럼 누락: {missing}. "
            "features/engineering.py(RSI) + core/cvp.py(add_cvp_features, VPVR/POC)로 "
            "생성된 5분봉 피처 CSV(예: data/training_features_5m.csv)를 사용하세요."
        )

    if timestamp_col in df.columns:
        df["ts"] = pd.to_datetime(df[timestamp_col], errors="coerce")
        df = df.dropna(subset=["ts"]).sort_values("ts").reset_index(drop=True)

    for c in REQUIRED_INPUT_COLS:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.dropna(subset=["open", "high", "low", "close", "volume"]).reset_index(drop=True)
    return df


def split_train_test(df: pd.DataFrame, train_ratio: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    n = len(df)
    cut = int(max(200, min(n - 50, round(n * train_ratio))))
    return df.iloc[:cut].reset_index(drop=True), df.iloc[cut:].reset_index(drop=True)


# ════════════════════════════════════════════════════════════════
# 파라미터 샘플링
# ════════════════════════════════════════════════════════════════
def _default_params() -> dict:
    cfg = FormulaConfig()
    return {f: getattr(cfg, f) for f in TUNABLE_FIELDS}


def _sample_params(rng: np.random.Generator) -> dict:
    theta_entry = float(rng.uniform(0.15, 0.65))
    theta_exit = float(rng.uniform(0.02, theta_entry * 0.85))
    theta_full = float(theta_entry + rng.uniform(0.05, 0.9))
    return {
        "w_ema": float(rng.uniform(0.0, 2.0)),
        "w_vwma": float(rng.uniform(0.0, 2.0)),
        "w_poc": float(rng.uniform(0.0, 2.0)),
        "w_rsi": float(rng.uniform(0.0, 2.0)),
        "k_ema": float(rng.uniform(2.0, 25.0)),
        "k_vwma": float(rng.uniform(2.0, 25.0)),
        "k_poc1": float(rng.uniform(0.5, 6.0)),
        "k_poc2": float(rng.uniform(0.5, 6.0)),
        "w_regime_gate": float(rng.uniform(0.0, 1.0)),
        "rsi_regime_power": float(rng.uniform(0.5, 2.5)),
        "theta_entry": theta_entry,
        "theta_exit": theta_exit,
        "theta_full": theta_full,
        # rsi_hot/cold는 방향성 게이트가 아닌 극단 안전판이므로 넓게 탐색
        "rsi_hot": float(rng.uniform(75.0, 95.0)),
        "rsi_cold": float(rng.uniform(5.0, 25.0)),
        "base_size": float(rng.uniform(0.3, 1.5)),
        "tp_pct": float(rng.uniform(0.002, 0.03)),
        "sl_pct": float(rng.uniform(0.001, 0.015)),
        "trail_pct": float(rng.uniform(0.0, 0.02)),
        "max_hold_bars": int(rng.integers(6, 200)),
        "cooldown_bars": int(rng.integers(0, 12)),
    }


def _cfg_from_params(p: dict) -> FormulaConfig:
    return FormulaConfig(**{k: p[k] for k in TUNABLE_FIELDS if k in p})


# ════════════════════════════════════════════════════════════════
# 백테스트 시뮬레이션
# (진입: composite_score 임계값 돌파 + 매크로필터 + RSI게이트 / 청산: TP·SL·
#  트레일링·시그널청산·최대보유 — scripts/backtest_param_ensemble.py의
#  _ensemble_backtest 상태머신 관례를 따름)
# ════════════════════════════════════════════════════════════════
def run_formula_sim(
    df: pd.DataFrame,
    p: dict,
    fee_bps: float,
    slip_bps: float,
    leverage: float,
) -> SimResult:
    cfg = _cfg_from_params(p)
    m = FormulaEngine(cfg).compute(df.copy())

    close = m["close"].to_numpy(np.float64)
    high = m["high"].to_numpy(np.float64)
    low = m["low"].to_numpy(np.float64)
    comp = m["composite_score"].to_numpy(np.float64)
    macro = m["macro_filter"].to_numpy(np.float64)
    rsi_long_ok = m["rsi_long_ok"].to_numpy(bool)
    rsi_short_ok = m["rsi_short_ok"].to_numpy(bool)

    fee = float(fee_bps) / 10_000.0
    slip = float(slip_bps) / 10_000.0
    lev = float(max(leverage, 0.0))

    pos = 0
    size = 0.0
    entry = 0.0
    peak_px = 0.0
    trough_px = 0.0
    bars_in_pos = 0
    cooldown = 0

    eq = 1.0
    eq_curve = [eq]
    trades = 0
    wins = 0

    n = len(m)
    for i in range(1, n):
        if pos == 0:
            if cooldown > 0:
                cooldown -= 1
            else:
                want_long = bool(comp[i] > cfg.theta_entry and macro[i] >= 0 and rsi_long_ok[i])
                want_short = bool(comp[i] < -cfg.theta_entry and macro[i] <= 0 and rsi_short_ok[i])
                if want_long or want_short:
                    pos = 1 if want_long else -1
                    strength = min(1.0, abs(comp[i]) / max(cfg.theta_full, 1e-6))
                    size = float(np.clip(cfg.base_size * strength, 0.0, 1.0))
                    entry = close[i] * (1 + slip if pos == 1 else 1 - slip)
                    eq *= (1.0 - fee * size * lev)
                    trades += 1
                    peak_px = entry
                    trough_px = entry
                    bars_in_pos = 0
        else:
            bars_in_pos += 1
            if pos == 1:
                peak_px = max(peak_px, high[i])
            else:
                trough_px = min(trough_px, low[i])

            rr_m = (close[i] - entry) / max(entry, 1e-12)
            if pos == -1:
                rr_m = -rr_m

            hit_tp = rr_m >= cfg.tp_pct
            hit_sl = rr_m <= -cfg.sl_pct

            hit_trail = False
            if cfg.trail_pct > 0:
                if pos == 1:
                    hit_trail = close[i] <= peak_px * (1 - cfg.trail_pct)
                else:
                    hit_trail = close[i] >= trough_px * (1 + cfg.trail_pct)

            signal_fade = abs(comp[i]) < cfg.theta_exit
            signal_flip = (pos == 1 and comp[i] < -cfg.theta_entry) or (
                pos == -1 and comp[i] > cfg.theta_entry
            )
            hold_exceeded = cfg.max_hold_bars > 0 and bars_in_pos >= cfg.max_hold_bars

            should_exit = (
                hit_tp or hit_sl or hit_trail or signal_fade or signal_flip or hold_exceeded
            )
            if should_exit:
                exit_px = close[i] * (1 - slip if pos == 1 else 1 + slip)
                rr = (exit_px - entry) / max(entry, 1e-12)
                if pos == -1:
                    rr = -rr
                pnl = rr * size * lev
                eq *= (1.0 + pnl)
                eq *= (1.0 - fee * size * lev)
                wins += int(pnl > 0)
                pos = 0
                size = 0.0
                entry = 0.0
                cooldown = cfg.cooldown_bars
                bars_in_pos = 0

        eq_curve.append(eq)

    eq_arr = np.asarray(eq_curve, dtype=np.float64)
    pnl_pct = float((eq_arr[-1] - 1.0) * 100.0)
    mdd_pct = calc_mdd(eq_arr)
    sharpe = calc_sharpe(eq_arr)
    win_rate = float(wins / trades * 100.0) if trades > 0 else 0.0

    return SimResult(
        pnl_pct=pnl_pct,
        mdd_pct=mdd_pct,
        sharpe=sharpe,
        trades=int(trades),
        win_rate_pct=win_rate,
        equity_final=float(eq_arr[-1]),
        params=dict(p),
    )


# ════════════════════════════════════════════════════════════════
# 파라미터 탐색 (목표: 총 PnL(pnl_pct) 최대화)
# ════════════════════════════════════════════════════════════════
def tune_random_search(
    df: pd.DataFrame, trials: int, fee_bps: float, slip_bps: float, leverage: float, seed: int
) -> SimResult:
    rng = np.random.default_rng(seed)
    best = run_formula_sim(df, _default_params(), fee_bps, slip_bps, leverage)
    for _ in range(max(1, trials)):
        p = _sample_params(rng)
        r = run_formula_sim(df, p, fee_bps, slip_bps, leverage)
        if r.pnl_pct > best.pnl_pct:
            best = r
    return best


def tune_optuna(
    df: pd.DataFrame, trials: int, fee_bps: float, slip_bps: float, leverage: float, seed: int
) -> SimResult:
    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def objective(trial: "optuna.Trial") -> float:
        theta_entry = trial.suggest_float("theta_entry", 0.15, 0.65)
        theta_exit = trial.suggest_float("theta_exit_ratio", 0.05, 0.85) * theta_entry
        theta_full = theta_entry + trial.suggest_float("theta_full_extra", 0.05, 0.9)
        p = {
            "w_ema": trial.suggest_float("w_ema", 0.0, 2.0),
            "w_vwma": trial.suggest_float("w_vwma", 0.0, 2.0),
            "w_poc": trial.suggest_float("w_poc", 0.0, 2.0),
            "w_rsi": trial.suggest_float("w_rsi", 0.0, 2.0),
            "k_ema": trial.suggest_float("k_ema", 2.0, 25.0),
            "k_vwma": trial.suggest_float("k_vwma", 2.0, 25.0),
            "k_poc1": trial.suggest_float("k_poc1", 0.5, 6.0),
            "k_poc2": trial.suggest_float("k_poc2", 0.5, 6.0),
            "w_regime_gate": trial.suggest_float("w_regime_gate", 0.0, 1.0),
            "rsi_regime_power": trial.suggest_float("rsi_regime_power", 0.5, 2.5),
            "theta_entry": theta_entry,
            "theta_exit": theta_exit,
            "theta_full": theta_full,
            # rsi_hot/cold는 방향성 게이트가 아닌 극단 안전판이므로 넓게 탐색
            "rsi_hot": trial.suggest_float("rsi_hot", 75.0, 95.0),
            "rsi_cold": trial.suggest_float("rsi_cold", 5.0, 25.0),
            "base_size": trial.suggest_float("base_size", 0.3, 1.5),
            "tp_pct": trial.suggest_float("tp_pct", 0.002, 0.03),
            "sl_pct": trial.suggest_float("sl_pct", 0.001, 0.015),
            "trail_pct": trial.suggest_float("trail_pct", 0.0, 0.02),
            "max_hold_bars": trial.suggest_int("max_hold_bars", 6, 200),
            "cooldown_bars": trial.suggest_int("cooldown_bars", 0, 12),
        }
        r = run_formula_sim(df, p, fee_bps, slip_bps, leverage)
        trial.set_user_attr("params", r.params)
        trial.set_user_attr("mdd_pct", r.mdd_pct)
        trial.set_user_attr("sharpe", r.sharpe)
        trial.set_user_attr("trades", r.trades)
        trial.set_user_attr("win_rate_pct", r.win_rate_pct)
        # 목표 함수: 총 PnL(pnl_pct) 최대화 (사용자 선택 그대로, MDD/Sharpe로 패널티 주지 않음)
        return r.pnl_pct

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(objective, n_trials=max(1, trials), show_progress_bar=False)

    best_params = dict(study.best_trial.user_attrs["params"])
    return run_formula_sim(df, best_params, fee_bps, slip_bps, leverage)


# ════════════════════════════════════════════════════════════════
# CLI
# ════════════════════════════════════════════════════════════════
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", default="data/training_features_5m.csv")
    ap.add_argument("--train-ratio", type=float, default=0.7)
    ap.add_argument("--trials", type=int, default=300)
    ap.add_argument(
        "--optimizer", choices=["auto", "optuna", "random"], default="auto",
        help="auto: optuna 있으면 사용, 없으면 random search로 폴백",
    )
    ap.add_argument("--fee-bps", type=float, default=5.0, help="taker 0.0005 = 5bps 기본값")
    ap.add_argument("--slip-bps", type=float, default=2.0)
    ap.add_argument("--leverage", type=float, default=10.0, help="README 기본 LEVERAGE=10")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--out", default="data/ensemble/metrics/vpvr_poc_rsi_vwma_ema_formula_result.json"
    )
    args = ap.parse_args()

    df = load_data(args.data)
    if len(df) < 300:
        raise RuntimeError(f"데이터 행 수 부족: {len(df)}행 (최소 300행 필요)")

    train_df, test_df = split_train_test(df, float(args.train_ratio))

    default_p = _default_params()
    default_train = run_formula_sim(train_df, default_p, args.fee_bps, args.slip_bps, args.leverage)
    default_test = run_formula_sim(test_df, default_p, args.fee_bps, args.slip_bps, args.leverage)

    optimizer_used = args.optimizer
    tuned_train: SimResult | None = None
    if args.optimizer in ("auto", "optuna"):
        try:
            tuned_train = tune_optuna(
                train_df, args.trials, args.fee_bps, args.slip_bps, args.leverage, args.seed
            )
            optimizer_used = "optuna"
        except ImportError:
            if args.optimizer == "optuna":
                raise
            tuned_train = None
    if tuned_train is None:
        tuned_train = tune_random_search(
            train_df, args.trials, args.fee_bps, args.slip_bps, args.leverage, args.seed
        )
        optimizer_used = "random"

    tuned_test = run_formula_sim(
        test_df, tuned_train.params, args.fee_bps, args.slip_bps, args.leverage
    )

    result = {
        "meta": {
            "data": args.data,
            "rows_total": len(df),
            "rows_train": len(train_df),
            "rows_test": len(test_df),
            "train_ratio": float(args.train_ratio),
            "fee_bps": float(args.fee_bps),
            "slip_bps": float(args.slip_bps),
            "leverage": float(args.leverage),
            "trials": int(args.trials),
            "optimizer": optimizer_used,
            "objective": "total_pnl_pct (train split)",
            "seed": int(args.seed),
        },
        "default": {
            "train": asdict(default_train),
            "test": asdict(default_test),
        },
        "tuned": {
            "train": asdict(tuned_train),
            "test": asdict(tuned_test),
        },
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"\nSaved: {out_path}")
    print(
        "\n[참고] tuned.test 는 목표함수(총 PnL)에 포함되지 않은 홀드아웃 구간 결과입니다. "
        "과최적화 여부를 가늠하는 참고 지표일 뿐, 탐색 제약으로 쓰이지 않았습니다."
    )


if __name__ == "__main__":
    main()
