#!/usr/bin/env python3
"""V_REBOUND 라벨 공식 재설계 -- 변형 후보 진단 스크린 (Step 1, screening-only).

docs/homer/v_rebound_open_issues_20260901.md #6/#8이 남긴 "라벨 공식 자체의 재설계" 착수 단계.
현행 라벨(label_side(), research_btc_v_rebound_gridscreen_20260901.py)의 held_up 얽힘이 수식의
어느 부분에서 오는지 특정하고, 그것을 끊는 변형 후보들이 실제로 얽힘을 줄이는지 CPU로 싸게 측정한다.

## 현행 수식의 얽힘 메커니즘 (bottom 기준)

  fast_move = max(close[i+1 : i+7]) - low[i]        <- low[i]가 FIXED ANCHOR
  fast_mult = fast_move / atr[i-1]
  peak      = max(high[i+1 : i+13])
  giveback  = (peak - close[i+12]) / (peak - low[i])
  is_v      = fast_mult >= 1.5 AND giveback <= 0.20
  is_chop   = fast_mult < 1.0

`low[i]`를 고정 바닥으로 쓰기 때문에, "이후 6봉 동안 low[i] 밑으로 안 내려감"(held_up)은
fast_mult가 커지기 위한 사실상의 선행조건이 된다 -- 계속 하락하면 max(close[i+1..i+6])가
low[i] 근처/아래라 fast_mult가 작아져 chop으로 떨어진다. local_extreme은 정의상
low[i]==min(low[i-6:i+7])이라 held_up을 100% 보장하므로, 이 트리거는 라벨의 선행조건을
미리 만족시킨 후보만 공급한다.

## 이 스크립트가 검증하는 두 가지

A) **타이밍 동시성 주장**: local_extreme의 확정 시점(bar i+6)과 fast_move 창의 마지막 봉(i+6)이
   같다 -> 라이브에서 local_extreme 신호가 표시되는 순간 fast_mult는 이미 확정되어 있고
   원리적으로 계산 가능하다(모델이 "예측"할 대상이 아니다). 인덱스 최대치를 직접 비교해 확인.

B) **라벨 변형별 얽힘 잔존도**: 각 변형에 대해 "local_extreme을 제외한 8트리거 후보"를 held_up
   True/False로 나눈 label rate 비율을 측정한다. 현행은 약 4.2~4.4배(= 얽힘 큼). 이 비율이
   1.0에 가까워지는 변형이 얽힘을 끊은 것.

변형 축은 두 개뿐(최소 변경 원칙):
  - anchor: 'wick'(현행, low[i]/high[i]) vs 'close'(close[i])
  - shift S: 평가창 전체를 S봉 뒤로 미룸. S=6은 local_extreme 확정 시점, S=7은 경제성
    재계산(project_v_rebound_local_extreme_entry_timing_realism_20260901)이 쓴 현실적 진입시점.
    shift는 anchor/ATR/fast창/full창/end_price 전부를 함께 이동시킨다(라이브에서 그 시점에
    실제로 보이는 것과 일치).

⚠️ 이 스크립트는 진단 전용이다: TabPFN 학습 없음, 라이브 코드 변경 없음, OOS/HOLDOUT 미터치
(TRAIN+VAL, timestamp < 2026-01-01만 로드). 라벨 변형의 채택 여부는 사용자 판단.

Run with the quant_ai conda env:
  ~/anaconda3/envs/quant_ai/bin/python3 scripts/research_eth_v_rebound_label_redesign_variant_screen_20260901.py
"""
from __future__ import annotations

import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("/home/kbj20/crypto-scalping")
for _p in (ROOT, ROOT / "scripts"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from live_evidence_signal_dashboard_20260823 import compute_signals  # noqa: E402

ETH_CSV = ROOT / "binance_data/klines/ETHUSDT/ETHUSDT-5m-api.csv"
BTC_CSV = ROOT / "binance_data/klines/BTCUSDT/BTCUSDT-5m-api.csv"
IMPL_SCRIPT = ROOT / "scripts/build_eth_5m_sweep_followthrough_v2_labels_20260829.py"
OUT_JSON = ROOT / "data/labels/eth_5m_v_rebound_multitrigger_20260831/label_redesign_variant_screen_report.json"

VAL_END = pd.Timestamp("2026-01-01", tz="UTC")

# label constants, verbatim from research_btc_v_rebound_gridscreen_20260901.py
FAST_BARS = 6
FULL_BARS = 12
ATR_MULT = 1.5
CHOP_MULT = 1.0
T_SUSTAIN = 0.20

W = 6  # LOCAL_EXTREME_W
NAMED8 = ["liquidity_sweep", "taker_delta_z_climax", "short_term_return_z", "orthogonal_combo",
          "smt_divergence", "fib_extension_exhaustion", "demarker_extreme", "kalman_deviation_meanrev"]
ALL9 = NAMED8 + ["local_extreme"]

# (name, anchor_mode, shift_bars)
VARIANTS = [
    ("V0_baseline_wick_shift0", "wick", 0),   # == current live label, exact reproduction
    ("V1_close_shift0",         "close", 0),  # anchor only
    ("V2_wick_shift6",          "wick", 6),   # window moved past local_extreme's confirmation bar
    ("V3_close_shift6",         "close", 6),
    ("V4_close_shift7",         "close", 7),  # matches the realistic-entry correction (+7 bars)
]


def log(msg: str) -> None:
    print(f"[label_redesign_screen] {msg}", flush=True)


def load_klines(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, usecols=["timestamp", "open", "high", "low", "close", "volume", "taker_buy_base"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    return df.sort_values("timestamp").drop_duplicates("timestamp").reset_index(drop=True)


def load_impl():
    spec = importlib.util.spec_from_file_location("sweep_impl_label_redesign_20260901", IMPL_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def build_base() -> pd.DataFrame:
    eth = load_klines(ETH_CSV)
    btc = load_klines(BTC_CSV)
    impl = load_impl()
    causal = impl.add_causal_columns(eth[["timestamp", "open", "high", "low", "close"]].copy())
    sig = compute_signals(eth, btc_df=btc, funding_df=None)
    sig["atr"] = causal["atr"].to_numpy()
    sig = sig.loc[sig["timestamp"] < VAL_END].reset_index(drop=True)
    log(f"TRAIN+VAL population: {len(sig)} rows, {sig['timestamp'].iloc[0]} .. {sig['timestamp'].iloc[-1]}")

    n = len(sig)
    low, high = sig["low"].to_numpy(), sig["high"].to_numpy()
    lo_flag = np.zeros(n, dtype=bool)
    hi_flag = np.zeros(n, dtype=bool)
    for i in range(W, n - W):
        if low[i] == low[i - W:i + W + 1].min():
            lo_flag[i] = True
        if high[i] == high[i - W:i + W + 1].max():
            hi_flag[i] = True
    sig["bottom_local_extreme"] = lo_flag
    sig["top_local_extreme"] = hi_flag
    return sig


def fwd_window(arr: np.ndarray, start_off: int, length: int, agg: str) -> np.ndarray:
    """agg over arr[i+start_off : i+start_off+length], NaN when the window runs past the end.
    Direct loop (not rolling) so the offsets are unambiguous and auditable."""
    n = len(arr)
    out = np.full(n, np.nan)
    for i in range(n):
        a = i + start_off
        b = a + length
        if a < 0 or b > n:
            continue
        seg = arr[a:b]
        out[i] = seg.max() if agg == "max" else seg.min()
    return out


def shifted_at(arr: np.ndarray, off: int) -> np.ndarray:
    """arr[i+off], NaN out of range."""
    n = len(arr)
    out = np.full(n, np.nan)
    lo_i = max(0, -off)
    hi_i = min(n, n - off)
    if hi_i > lo_i:
        out[lo_i:hi_i] = arr[lo_i + off:hi_i + off]
    return out


def label_variant(sig: pd.DataFrame, is_down: bool, anchor_mode: str, shift: int) -> np.ndarray:
    """Returns a status array: 'invalid' | 'v_rebound' | 'chop' | 'ambiguous'.

    Reproduces label_side()'s arithmetic exactly at shift=0/anchor='wick'; the shift moves the
    anchor bar, its ATR reference, the fast window, the full window and end_price together by S
    bars (i.e. the whole outcome measurement starts from bar i+S instead of bar i)."""
    close = sig["close"].to_numpy()
    high = sig["high"].to_numpy()
    low = sig["low"].to_numpy()
    atr = sig["atr"].to_numpy()

    S = shift
    if anchor_mode == "wick":
        extreme = shifted_at(low if is_down else high, S)
    else:
        extreme = shifted_at(close, S)
    pre_atr = shifted_at(atr, S - 1)

    fast_close_max = fwd_window(close, S + 1, FAST_BARS, "max")
    fast_close_min = fwd_window(close, S + 1, FAST_BARS, "min")
    full_high_max = fwd_window(high, S + 1, FULL_BARS, "max")
    full_low_min = fwd_window(low, S + 1, FULL_BARS, "min")
    end_price = shifted_at(close, S + FULL_BARS)

    if is_down:
        fast_move = fast_close_max - extreme
        peak = full_high_max
    else:
        fast_move = extreme - fast_close_min
        peak = full_low_min

    with np.errstate(invalid="ignore", divide="ignore"):
        valid = (np.isfinite(pre_atr) & (pre_atr > 0) & np.isfinite(full_high_max)
                 & np.isfinite(full_low_min) & np.isfinite(end_price) & np.isfinite(extreme))
        fast_mult = fast_move / pre_atr
        denom = (peak - extreme) if is_down else (extreme - peak)
        giveback = np.where(np.abs(denom) >= 1e-12,
                            (peak - end_price) / denom if is_down else (end_price - peak) / denom,
                            np.nan)
        is_v = (fast_mult >= ATR_MULT) & np.isfinite(giveback) & (giveback <= T_SUSTAIN)
        is_chop = fast_mult < CHOP_MULT

    return np.where(~valid, "invalid", np.where(is_v, "v_rebound", np.where(is_chop, "chop", "ambiguous")))


def rate(status: np.ndarray, mask: np.ndarray) -> dict:
    pool = status[mask]
    denom = int((pool != "invalid").sum())
    n_v = int((pool == "v_rebound").sum())
    return {"n": int(mask.sum()), "n_labeled": denom, "n_v": n_v,
            "rate": round(n_v / denom, 4) if denom else None}


def verify_timing_claim(sig: pd.DataFrame) -> dict:
    """Claim A: local_extreme[i]'s confirmation and the fast_move window both complete at bar i+6,
    so fast_mult is fully determined the moment the live chip can first show a local_extreme fire.
    Verified by comparing the maximum forward index each one reads."""
    le_max_fwd = W                    # low[i-W .. i+W] -> furthest forward index is i+W
    fast_max_fwd = 1 + FAST_BARS - 1  # close[i+1 .. i+FAST_BARS] -> i+FAST_BARS
    full_max_fwd = 1 + FULL_BARS - 1  # high[i+1 .. i+FULL_BARS]
    end_max_fwd = FULL_BARS

    # empirical cross-check: recompute fast_mult using ONLY bars <= i+W and confirm identity
    close = sig["close"].to_numpy()
    n = len(sig)
    fast_from_full = fwd_window(close, 1, FAST_BARS, "max")
    fast_truncated = np.full(n, np.nan)
    for i in range(n):
        b = i + W  # last bar available at local_extreme confirmation time
        if b >= n or i + 1 > b:
            continue
        fast_truncated[i] = close[i + 1:b + 1].max()
    both = np.isfinite(fast_from_full) & np.isfinite(fast_truncated)
    n_mismatch = int((fast_from_full[both] != fast_truncated[both]).sum())

    return {
        "local_extreme_max_forward_index_offset": le_max_fwd,
        "fast_move_window_max_forward_index_offset": fast_max_fwd,
        "full_window_max_forward_index_offset": full_max_fwd,
        "end_price_forward_index_offset": end_max_fwd,
        "fast_move_fully_determined_at_local_extreme_confirmation": bool(fast_max_fwd <= le_max_fwd),
        "giveback_still_unknown_at_confirmation": bool(full_max_fwd > le_max_fwd),
        "empirical_recompute_using_only_bars_up_to_i_plus_W": {
            "n_compared": int(both.sum()), "n_mismatches": n_mismatch,
        },
        "interpretation": (
            "local_extreme confirms at i+6 and the fast_move window ends at i+6 -> the 'fast leg' "
            "half of the label is already history (and directly computable) at the instant the live "
            "chip can first display a local_extreme fire; only giveback (needs i+7..i+12) is a "
            "genuine forward prediction at that moment."
        ),
    }


def main() -> int:
    t0 = time.time()
    sig = build_base()
    n = len(sig)
    low, high = sig["low"].to_numpy(), sig["high"].to_numpy()

    log("verifying timing-simultaneity claim (A)...")
    timing = verify_timing_claim(sig)
    log(f"  local_extreme confirms at i+{timing['local_extreme_max_forward_index_offset']}, "
        f"fast window ends at i+{timing['fast_move_window_max_forward_index_offset']} -> "
        f"fast_move already determined: {timing['fast_move_fully_determined_at_local_extreme_confirmation']}")
    log(f"  empirical recompute (bars <= i+W only): {timing['empirical_recompute_using_only_bars_up_to_i_plus_W']['n_mismatches']} mismatches "
        f"of {timing['empirical_recompute_using_only_bars_up_to_i_plus_W']['n_compared']}")

    # held_up (the property local_extreme guarantees) -- defined on the ORIGINAL bar i, unchanged
    # across variants: it is what we are testing the variants' independence FROM.
    fwd_low_min = fwd_window(low, 1, W, "min")
    fwd_high_max = fwd_window(high, 1, W, "max")
    held_up_bottom = fwd_low_min >= low
    held_up_top = fwd_high_max <= high
    valid_fwd_b = np.isfinite(fwd_low_min)
    valid_fwd_t = np.isfinite(fwd_high_max)

    others_b = np.any([sig[f"bottom_{nm}"].fillna(False).to_numpy() for nm in NAMED8], axis=0)
    others_t = np.any([sig[f"top_{nm}"].fillna(False).to_numpy() for nm in NAMED8], axis=0)

    results = {}
    for vname, anchor_mode, shift in VARIANTS:
        log(f"=== variant {vname} (anchor={anchor_mode}, shift={shift}) ===")
        st_b = label_variant(sig, is_down=True, anchor_mode=anchor_mode, shift=shift)
        st_t = label_variant(sig, is_down=False, anchor_mode=anchor_mode, shift=shift)

        entangle = {}
        for side, st, held_up, valid_fwd, others in (
            ("bottom", st_b, held_up_bottom, valid_fwd_b, others_b),
            ("top", st_t, held_up_top, valid_fwd_t, others_t),
        ):
            base = others & valid_fwd
            hu_t = rate(st, base & held_up)
            hu_f = rate(st, base & ~held_up)
            ratio = (round(hu_t["rate"] / hu_f["rate"], 3)
                     if hu_t["rate"] and hu_f["rate"] else None)
            entangle[side] = {"held_up_true": hu_t, "held_up_false": hu_f, "ratio": ratio}
            log(f"  [{side}] other-8 candidates: held_up=True {hu_t['rate']}(n={hu_t['n_labeled']}) "
                f"vs held_up=False {hu_f['rate']}(n={hu_f['n_labeled']})  RATIO={ratio}")

        per_trigger = {}
        for name in ALL9:
            mb = sig[f"bottom_{name}"].fillna(False).to_numpy()
            mt = sig[f"top_{name}"].fillna(False).to_numpy()
            rb, rt = rate(st_b, mb), rate(st_t, mt)
            nd = rb["n_labeled"] + rt["n_labeled"]
            nv = rb["n_v"] + rt["n_v"]
            per_trigger[name] = {"n_labeled": nd, "n_v": nv,
                                 "rate": round(nv / nd, 4) if nd else None}

        dist = {}
        for side, st in (("bottom", st_b), ("top", st_t)):
            u, c = np.unique(st, return_counts=True)
            dist[side] = {str(k): int(v) for k, v in zip(u, c)}

        le_rate = per_trigger["local_extreme"]["rate"]
        others_rates = [per_trigger[nm]["rate"] for nm in NAMED8 if per_trigger[nm]["rate"]]
        mean_other = round(float(np.mean(others_rates)), 4) if others_rates else None
        dominance = round(le_rate / mean_other, 3) if le_rate and mean_other else None
        log(f"  local_extreme rate={le_rate} vs mean(other8)={mean_other} -> DOMINANCE={dominance}x")

        results[vname] = {
            "anchor_mode": anchor_mode, "shift_bars": shift,
            "held_up_entanglement_among_other8": entangle,
            "per_trigger_rate": per_trigger,
            "local_extreme_dominance_vs_mean_other8": dominance,
            "status_distribution": dist,
        }

    report = {
        "signal": "v_rebound_label_redesign_variant_screen", "asset": "ETHUSDT",
        "fresh_forward_bar_by_bar": True, "trade_ledgers_used_as_input": False,
        "saved_parent_exit_timestamps_used": False, "future_rows_used_for_entry": False,
        "scope": {
            "screening_only": True, "tabpfn_training_done": False, "economic_cost_gate_done": False,
            "live_code_changed": False, "holdout_touched": False, "oos_touched": False,
            "population": "TRAIN+VAL only (timestamp < 2026-01-01)",
            "purpose": ("Step 1 of the V_REBOUND label-formula redesign flagged in "
                        "docs/homer/v_rebound_open_issues_20260901.md #6/#8: locate the held_up "
                        "entanglement inside label_side() and measure which label variants break it."),
        },
        "label_constants": {"FAST_BARS": FAST_BARS, "FULL_BARS": FULL_BARS,
                            "ATR_MULT": ATR_MULT, "CHOP_MULT": CHOP_MULT, "T_SUSTAIN": T_SUSTAIN},
        "claim_A_timing_simultaneity": timing,
        "claim_B_variant_entanglement": results,
        "runtime_sec": round(time.time() - t0, 1),
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str))
    log(f"report saved -> {OUT_JSON}")
    log(f"total runtime: {time.time()-t0:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
