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
# 목적함수 (기본: 총 PnL(pnl_pct) 최대화. sharpe / pnl_mdd 로 과최적화 억제 가능)
# ════════════════════════════════════════════════════════════════
def compute_objective(r: SimResult, objective: str, mdd_penalty: float) -> float:
    if objective == "sharpe":
        return r.sharpe
    if objective == "pnl_mdd":
        # mdd_pct는 항상 <= 0 이므로 그대로 더하면 낙폭이 클수록 목적함수가 깎인다.
        return r.pnl_pct + mdd_penalty * r.mdd_pct
    return r.pnl_pct


def tune_random_search(
    df: pd.DataFrame,
    trials: int,
    fee_bps: float,
    slip_bps: float,
    leverage: float,
    seed: int,
    objective: str = "pnl",
    mdd_penalty: float = 1.0,
) -> SimResult:
    rng = np.random.default_rng(seed)
    best = run_formula_sim(df, _default_params(), fee_bps, slip_bps, leverage)
    best_score = compute_objective(best, objective, mdd_penalty)
    for _ in range(max(1, trials)):
        p = _sample_params(rng)
        r = run_formula_sim(df, p, fee_bps, slip_bps, leverage)
        score = compute_objective(r, objective, mdd_penalty)
        if score > best_score:
            best = r
            best_score = score
    return best


def tune_optuna(
    df: pd.DataFrame,
    trials: int,
    fee_bps: float,
    slip_bps: float,
    leverage: float,
    seed: int,
    objective: str = "pnl",
    mdd_penalty: float = 1.0,
) -> SimResult:
    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def _objective_fn(trial: "optuna.Trial") -> float:
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
        return compute_objective(r, objective, mdd_penalty)

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(_objective_fn, n_trials=max(1, trials), show_progress_bar=False)

    best_params = dict(study.best_trial.user_attrs["params"])
    return run_formula_sim(df, best_params, fee_bps, slip_bps, leverage)


def tune(
    train_df: pd.DataFrame,
    optimizer: str,
    trials: int,
    fee_bps: float,
    slip_bps: float,
    leverage: float,
    seed: int,
    objective: str,
    mdd_penalty: float,
) -> tuple[SimResult, str]:
    """optimizer in {"auto","optuna","random"} 에 따라 튜닝 실행, (결과, 실제 사용된 optimizer) 반환."""
    if optimizer in ("auto", "optuna"):
        try:
            r = tune_optuna(
                train_df, trials, fee_bps, slip_bps, leverage, seed, objective, mdd_penalty
            )
            return r, "optuna"
        except ImportError:
            if optimizer == "optuna":
                raise
    r = tune_random_search(
        train_df, trials, fee_bps, slip_bps, leverage, seed, objective, mdd_penalty
    )
    return r, "random"


# ════════════════════════════════════════════════════════════════
# Walk-forward 검증 (expanding window): 데이터를 n_folds+1개 연속 블록으로 나눠
# fold k는 블록[0..k]를 train, 블록[k+1]을 test로 사용. 매 fold마다 test는
# train 이후 시점만 포함하므로(시간 순서 유지) 단일 70/30 split보다 강건한
# generalization 신호를 준다.
# ════════════════════════════════════════════════════════════════
def run_walk_forward(
    df: pd.DataFrame,
    n_folds: int,
    optimizer: str,
    trials: int,
    fee_bps: float,
    slip_bps: float,
    leverage: float,
    seed: int,
    objective: str,
    mdd_penalty: float,
) -> dict:
    n_blocks = n_folds + 1
    n = len(df)
    block_size = n // n_blocks
    if block_size < 200:
        raise RuntimeError(
            f"fold당 블록 크기가 너무 작습니다({block_size}행). n_folds를 줄이세요."
        )
    bounds = [i * block_size for i in range(n_blocks)] + [n]

    folds = []
    for k in range(1, n_blocks):
        train_df = df.iloc[bounds[0]:bounds[k]].reset_index(drop=True)
        test_df = df.iloc[bounds[k]:bounds[k + 1]].reset_index(drop=True)
        tuned_train, optimizer_used = tune(
            train_df, optimizer, trials, fee_bps, slip_bps, leverage, seed, objective, mdd_penalty
        )
        tuned_test = run_formula_sim(test_df, tuned_train.params, fee_bps, slip_bps, leverage)
        folds.append({
            "fold": k,
            "rows_train": len(train_df),
            "rows_test": len(test_df),
            "optimizer": optimizer_used,
            "train": asdict(tuned_train),
            "test": asdict(tuned_test),
        })

    test_pnls = [f["test"]["pnl_pct"] for f in folds]
    test_mdds = [f["test"]["mdd_pct"] for f in folds]
    test_sharpes = [f["test"]["sharpe"] for f in folds]
    n_positive = sum(1 for p in test_pnls if p > 0)

    summary = {
        "n_folds": n_folds,
        "test_pnl_pct_mean": float(np.mean(test_pnls)),
        "test_pnl_pct_median": float(np.median(test_pnls)),
        "test_mdd_pct_mean": float(np.mean(test_mdds)),
        "test_sharpe_mean": float(np.mean(test_sharpes)),
        "folds_with_positive_test_pnl": int(n_positive),
        "folds_total": len(folds),
    }

    return {"folds": folds, "summary": summary}


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
        "--objective", choices=["pnl", "sharpe", "pnl_mdd"], default="pnl",
        help="탐색 목적함수: pnl=총PnL(기본), sharpe=Sharpe, pnl_mdd=PnL+mdd_penalty*MDD",
    )
    ap.add_argument(
        "--mdd-penalty", type=float, default=1.0,
        help="--objective pnl_mdd 일 때 낙폭 패널티 가중치",
    )
    ap.add_argument(
        "--mode", choices=["single", "walk-forward"], default="single",
        help="single=70/30 1회 분할(기본), walk-forward=expanding-window 다중 fold 검증",
    )
    ap.add_argument("--n-folds", type=int, default=5, help="--mode walk-forward 일 때 fold 수")
    ap.add_argument(
        "--out", default="data/ensemble/metrics/vpvr_poc_rsi_vwma_ema_formula_result.json"
    )
    args = ap.parse_args()

    df = load_data(args.data)
    if len(df) < 300:
        raise RuntimeError(f"데이터 행 수 부족: {len(df)}행 (최소 300행 필요)")

    if args.mode == "walk-forward":
        wf = run_walk_forward(
            df, args.n_folds, args.optimizer, args.trials,
            args.fee_bps, args.slip_bps, args.leverage, args.seed,
            args.objective, args.mdd_penalty,
        )
        result = {
            "meta": {
                "mode": "walk-forward",
                "data": args.data,
                "rows_total": len(df),
                "n_folds": int(args.n_folds),
                "fee_bps": float(args.fee_bps),
                "slip_bps": float(args.slip_bps),
                "leverage": float(args.leverage),
                "trials": int(args.trials),
                "objective": args.objective,
                "mdd_penalty": float(args.mdd_penalty),
                "seed": int(args.seed),
            },
            **wf,
        }
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(result, ensure_ascii=False, indent=2))
        print(f"\nSaved: {out_path}")
        return

    train_df, test_df = split_train_test(df, float(args.train_ratio))

    default_p = _default_params()
    default_train = run_formula_sim(train_df, default_p, args.fee_bps, args.slip_bps, args.leverage)
    default_test = run_formula_sim(test_df, default_p, args.fee_bps, args.slip_bps, args.leverage)

    tuned_train, optimizer_used = tune(
        train_df, args.optimizer, args.trials, args.fee_bps, args.slip_bps, args.leverage,
        args.seed, args.objective, args.mdd_penalty,
    )

    tuned_test = run_formula_sim(
        test_df, tuned_train.params, args.fee_bps, args.slip_bps, args.leverage
    )

    result = {
        "meta": {
            "mode": "single",
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
            "objective": f"{args.objective} (train split)",
            "mdd_penalty": float(args.mdd_penalty),
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
        "\n[참고] tuned.test 는 목표함수 탐색에 포함되지 않은 홀드아웃 구간 결과입니다. "
        "과최적화 여부를 가늠하는 참고 지표일 뿐, 탐색 제약으로 쓰이지 않았습니다."
    )


if __name__ == "__main__":
    main()
