#!/usr/bin/env python3
"""일리아스 — 레짐 직결 노출 정책(Regime-Direct Exposure) 백테스트 (2026-08-22).

배경: 일리아스 3라벨(zigzag/h48qual/cusum) TabM은 방향 정보량이 정보이론적으로 0이고
(BCE=절편하한), 시드검증 실패의 원인은 시드별 우연한 롱/숏 편향 × 구간 추세임이 확정됐다
(long_frac↔PnL 상관 0.888~0.918). 이 프로젝트에서 시드 다양성 게이트(N=5)를 실제로 통과한
유일한 자산은 wide24 HMM 레짐 분류기(states=24/sticky=0.90, OOS std≈0.0001)다.

가설: PnL을 결정하는 방향 판단을 시드-불안정한 TabM이 아니라 시드-안정한 HMM 레짐 확률에
직결하면(bull→LONG, bear→SHORT, 신뢰 저하→청산), 시드검증에서 부호 일치 + 양수 PnL이
구조적으로 가능해진다. HMM의 `filter_proba`는 인과적 forward 알고리즘(과거+현재 bar만
사용)이므로 fresh-forward bar-by-bar 컨벤션과 호환된다.

정책(자유변수 2개뿐): Schmitt 트리거 —
  flat→LONG  : bull_prob > p_hi
  flat→SHORT : bear_prob > p_hi
  LONG→flat  : bull_prob < p_lo   /   SHORT→flat : bear_prob < p_lo
신호는 bar t 종가 기준, 포지션은 bar t+1 수익률에 적용(1-bar lag). 수수료+슬리피지
(0.0005+0.0002)/side, cost_mult 1x/2x/3x 보고.

선정: p_hi×p_lo 소격자(9개)를 VAL(2026-04-01~06-30)에서 5시드 평균 PnL로 1개 선택 →
그 1개만 OOS(2026-07-01~08-19)에서 5시드 평가(single-touch).

⚠️ 정직성 주석:
- VAL(2026Q2)은 HMM 파라미터 fit 구간(2024-01~2026-06-30) 안이라 in-sample성 있음 —
  단 HMM은 PnL로 학습된 적 없음(비지도 likelihood+ADX라벨 캘리브레이션). OOS(07-01~)만
  파라미터 기준 진짜 미학습 구간.
- 이 OOS 캘린더 창은 다른 축들이 이미 조회한 창 — 이 결과는 research/dev score이며
  단독으로 promotion 근거가 아니다.
- fresh_forward_bar_by_bar=true, trade_ledgers_used_as_input=false,
  saved_parent_exit_timestamps_used=false, future_rows_used_for_entry=false.
"""
from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

# mamba_ssm은 미설치 — 경험 스크립트가 transitively import하지만 인스턴스화는 안 함
# (tmp/eth_hmm_wide24_resweep_train2026h1_20260821/run_wide24.py와 동일한 기존 워크어라운드)
stub = types.ModuleType("mamba_ssm")
stub.Mamba = object
sys.modules["mamba_ssm"] = stub

import joblib  # noqa: E402

from experiment_regime3_current_hmm_wide24_20260529 import _transform  # noqa: E402

FEE_RATE = 0.0005   # scripts/train_eval_omega1_2_tabm_diffusion_risk_20260603.py와 동일
SLIP_RATE = 0.0002

DATA_2026 = ROOT / "data/splits/year_oos/training_features_2026_rebuilt.csv"
RESWEEP = ROOT / "tmp/eth_hmm_wide24_resweep_train2026h1_20260821"
SEED_MODELS = {
    7529: RESWEEP / "states24_sticky0.90/models/regime3_current_sensitive_v2_hmm_wide24_2024.joblib",
    534964: RESWEEP / "seedcheck_states24_sticky0.90_seed534964/models/regime3_current_sensitive_v2_hmm_wide24_2024.joblib",
    116595: RESWEEP / "seedcheck_states24_sticky0.90_seed116595/models/regime3_current_sensitive_v2_hmm_wide24_2024.joblib",
    666940: RESWEEP / "seedcheck_states24_sticky0.90_seed666940/models/regime3_current_sensitive_v2_hmm_wide24_2024.joblib",
    505456: RESWEEP / "seedcheck_states24_sticky0.90_seed505456/models/regime3_current_sensitive_v2_hmm_wide24_2024.joblib",
}

VAL_START, VAL_END = "2026-04-01", "2026-07-01"   # [start, end)
OOS_START, OOS_END = "2026-07-01", "2026-08-20"

GRID_P_HI = [0.50, 0.60, 0.70]
GRID_P_LO = [0.30, 0.40, 0.45]

OUT_DIR = ROOT / "tmp/ilias_regime_direct_exposure_20260822"


def run_policy(bull: np.ndarray, bear: np.ndarray, close: np.ndarray,
               p_hi: float, p_lo: float, cost_mult: float = 1.0) -> dict:
    """신호 bar t → 포지션은 bar t+1 수익률에 적용. 반환: equity curve 통계."""
    cost_side = (FEE_RATE + SLIP_RATE) * cost_mult
    n = len(close)
    ret = np.zeros(n)
    ret[1:] = close[1:] / close[:-1] - 1.0

    pos = 0  # -1/0/+1, bar t 종가 시점에 결정된 "다음 bar 동안 보유할" 포지션
    equity = 1.0
    peak = 1.0
    mdd = 0.0
    trades = 0
    long_bars = 0
    short_bars = 0
    curve = np.empty(n)
    curve[0] = equity
    for t in range(1, n):
        # bar t 수익률은 직전에 결정된 pos로 실현
        equity *= 1.0 + pos * ret[t]
        # bar t 종가에서 재결정
        new_pos = pos
        if pos == 1 and bull[t] < p_lo:
            new_pos = 0
        elif pos == -1 and bear[t] < p_lo:
            new_pos = 0
        if new_pos == 0:
            if bull[t] > p_hi:
                new_pos = 1
            elif bear[t] > p_hi:
                new_pos = -1
        if new_pos != pos:
            # 사이드 변화량만큼 비용(청산+진입이 겹치면 2회분)
            legs = abs(new_pos - pos)
            equity *= (1.0 - cost_side) ** legs
            trades += 1
        pos = new_pos
        if pos == 1:
            long_bars += 1
        elif pos == -1:
            short_bars += 1
        peak = max(peak, equity)
        mdd = min(mdd, equity / peak - 1.0)
        curve[t] = equity
    exposed = long_bars + short_bars
    return {
        "pnl_pct": (equity - 1.0) * 100.0,
        "mdd_pct": mdd * 100.0,
        "trades": trades,
        "exposure_frac": exposed / max(n - 1, 1),
        "long_frac_of_exposed": long_bars / max(exposed, 1),
    }


def benchmarks(close: np.ndarray, cost_mult: float = 1.0) -> dict:
    cost = (FEE_RATE + SLIP_RATE) * cost_mult
    hold = close[-1] / close[0] - 1.0
    return {
        "always_long_pnl_pct": ((1.0 + hold) * (1.0 - cost) ** 2 - 1.0) * 100.0,
        "always_short_pnl_pct": ((1.0 - hold) * (1.0 - cost) ** 2 - 1.0) * 100.0,
    }


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(DATA_2026)
    frame["timestamp"] = pd.to_datetime(frame["timestamp"])
    frame = frame.sort_values("timestamp").reset_index(drop=True)

    # 시드별 인과 filtered 확률: 2026-01-01부터 연속 필터링(워밍업 1분기 확보)
    per_seed = {}
    for seed, path in SEED_MODELS.items():
        payload = joblib.load(path)
        overlay, _diag = _transform(payload, frame)
        prefix = f"{payload['prefix_stem']}_{payload['feature_set']}_"
        merged = pd.DataFrame({
            "timestamp": overlay["timestamp"],
            "bull": overlay[f"{prefix}bull_prob"].to_numpy(),
            "bear": overlay[f"{prefix}bear_prob"].to_numpy(),
        })
        merged["close"] = frame["close"].to_numpy()
        per_seed[seed] = merged
        print(f"seed={seed} overlay rows={len(merged)}", flush=True)

    def window(df: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
        m = (df["timestamp"] >= start) & (df["timestamp"] < end)
        return df.loc[m].reset_index(drop=True)

    # 1) VAL 격자 — 5시드 평균 PnL로 단일 config 선택 (OOS 미접촉)
    val_grid = {}
    for p_hi in GRID_P_HI:
        for p_lo in GRID_P_LO:
            pnls = []
            for seed, df in per_seed.items():
                w = window(df, VAL_START, VAL_END)
                r = run_policy(w["bull"].to_numpy(), w["bear"].to_numpy(), w["close"].to_numpy(), p_hi, p_lo)
                pnls.append(r["pnl_pct"])
            val_grid[f"hi{p_hi}_lo{p_lo}"] = {
                "p_hi": p_hi, "p_lo": p_lo,
                "val_pnl_mean": float(np.mean(pnls)),
                "val_pnl_min": float(np.min(pnls)),
                "val_pnl_per_seed": {str(s): float(p) for s, p in zip(per_seed, pnls)},
            }
    chosen_key = max(val_grid, key=lambda k: val_grid[k]["val_pnl_mean"])
    chosen = val_grid[chosen_key]
    print(f"chosen config: {chosen_key} val_pnl_mean={chosen['val_pnl_mean']:.2f}%", flush=True)

    # 2) 선택된 1개 config만 VAL/OOS 상세 + cost stress (single-touch)
    detail = {}
    for wname, (ws, we) in {"VAL": (VAL_START, VAL_END), "OOS": (OOS_START, OOS_END)}.items():
        rows = {}
        for seed, df in per_seed.items():
            w = window(df, ws, we)
            rows[str(seed)] = {
                f"cost{m}x": run_policy(w["bull"].to_numpy(), w["bear"].to_numpy(),
                                        w["close"].to_numpy(), chosen["p_hi"], chosen["p_lo"], cost_mult=m)
                for m in (1.0, 2.0, 3.0)
            }
        anyseed = next(iter(per_seed.values()))
        w = window(anyseed, ws, we)
        detail[wname] = {
            "window": [ws, we],
            "bars": int(len(w)),
            "benchmarks_cost1x": benchmarks(w["close"].to_numpy()),
            "per_seed": rows,
            "pnl_cost1x_all_seeds": [rows[str(s)]["cost1.0x"]["pnl_pct"] for s in per_seed],
        }

    def sign_summary(vals):
        pos = sum(1 for v in vals if v > 0)
        return f"{pos}/{len(vals)} positive"

    report = {
        "experiment": "ilias_regime_direct_exposure_backtest_20260822",
        "hypothesis": "seed-stable wide24 HMM regime probs as the sole direction driver -> seed-consistent positive PnL",
        "policy": {"type": "schmitt_trigger", "p_hi": chosen["p_hi"], "p_lo": chosen["p_lo"],
                   "signal_lag_bars": 1, "fee_per_side": FEE_RATE, "slip_per_side": SLIP_RATE},
        "selection": {"grid": val_grid, "chosen": chosen_key,
                      "selection_metric": "mean VAL pnl over 5 seeds, cost 1x"},
        "seeds": list(map(str, SEED_MODELS)),
        "results": detail,
        "seed_sign_summary": {
            "VAL_cost1x": sign_summary(detail["VAL"]["pnl_cost1x_all_seeds"]),
            "OOS_cost1x": sign_summary(detail["OOS"]["pnl_cost1x_all_seeds"]),
        },
        "fresh_forward_bar_by_bar": True,
        "trade_ledgers_used_as_input": False,
        "saved_parent_exit_timestamps_used": False,
        "future_rows_used_for_entry": False,
        "honesty_caveats": [
            "VAL(2026Q2) is inside the HMM parameter fit period (2024-01~2026-06-30) — in-sample-ish; HMM was never fit on PnL though.",
            "OOS calendar window (2026-07-01~08-19) already consumed by other research axes — research/dev score only, not promotion evidence.",
            "HMM filtered probs are causal (forward algorithm); overlay generated with warmup from 2026-01-01.",
        ],
    }
    out = OUT_DIR / "report.json"
    out.write_text(json.dumps(report, indent=2))
    print(json.dumps({"chosen": chosen_key, "sign_summary": report["seed_sign_summary"],
                      "VAL": detail["VAL"]["pnl_cost1x_all_seeds"],
                      "OOS": detail["OOS"]["pnl_cost1x_all_seeds"],
                      "benchmarks_VAL": detail["VAL"]["benchmarks_cost1x"],
                      "benchmarks_OOS": detail["OOS"]["benchmarks_cost1x"]}, indent=2))
    print(f"report -> {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
