#!/usr/bin/env python3
"""BTC로 오디세이4 섀도우(entry-veto+exit-guard) 이식이 유망한지 저렴한 사전진단 (2026-08-20).

=== 배경 ===
ETH에서 오디세이4 섀도우(SustainedUptrendDetector로 "상승추세 중 zig075 SHORT 진입"을 규칙기반
스킵)가 h48qual/zig075 인코더를 진짜로 재시드해도 6/6창 포트폴리오 부호를 안정시킨다는 게 확인됐다
(docs/experiments/eth_odyssey4_shadow_full_reseed_causal_isolation_20260820.md). 유력 메커니즘
가설: "상승추세 중 SHORT 진입"이 시드마다 확신도/캘리브레이션이 가장 크게 갈리는 트레이드 유형이라,
그걸 규칙으로 제거하면 시드 분산의 핵심 원천이 사라진다는 것 -- 단, 트레이드 단위로 직접 검증된
적은 없다(그 문서의 "한계" 섹션 1번).

이 스크립트는 그 메커니즘 가설을 BTC 자신의 데이터로 직접 검증하는 게 아니라, "새 메커니즘을 만들기
전에" 저렴하게 사전진단만 한다 -- 신규 학습 없음, 신규 백테스트 없음. 2026-08-19에 이미 완료된 BTC
h48qual+swingtransition 라이브 N=5 시드 검증
(scripts/btc_live_promotion_seed_robustness_eval_5seed_20260819.py, seed-diversity ensemble
promotion gate 대상이 아닌 진단용 5-seed 페어링)이 서버에만 갖고 있던 trade ledger 30개
(5시드 x 6창)를 pull해서, 그 ledger의 trade_return을 그대로 재집계만 한다.

=== 방법 ===
1. BTC용 "상승추세" 프록시: ETH의 SustainedUptrendDetector와 동일한 정의
   (rolling(2016bar=1주, min_periods=2016).mean(dual_momentum>0) > threshold, 완전 causal,
   BASE_2025/BASE_2026 파일별로 독립적으로 rolling -- ETH의 build_detector와 동일 컨벤션)을
   BTC의 data/splits/year_oos/btc_features_{2025,2026}_swingtransition.csv dual_momentum
   컬럼에 적용한다. threshold는 ETH의 고정값(0.8025793650793651)을 재사용하지 않고 BTC 자신의
   2025년 전체 스코어 분포 p90에서 새로 계산한다 -- 이건 진단용 placeholder이지 프로덕션
   캘리브레이션이 아니다(ETH는 2025 Q1+Q2만 캘리브레이션 샘플로 썼지만, 여기서는 사용자 지시대로
   BTC 2025년 전체 분포를 쓴다 -- Q3까지 포함되므로 ETH보다 느슨한 컨벤션이라는 점을 명시한다).
2. 5개 시드 x 6창 ledger에서 SHORT(side==-1) 트레이드 중 entry_timestamp가 이 마스크에서 active인
   것을 식별한다.
3. 각 (시드, 창)에서 baseline(전체 트레이드 compound PnL, report.json과 대조용)과
   "상승추세중 SHORT 제외" compound PnL을 계산한다 -- 둘 다 저장된 ledger의 trade_return을
   그대로 순서대로 복리 재계산하는 것뿐, 새 백테스트가 전혀 아니다(_compound_metrics는
   apply_final_scale_map_btc_freshforward_ext_swingtransition_20260806.py에서 그대로 복사).
4. 대조군 2종(둘 다 같은 시드/창에서 실제 제거 개수 k와 동일한 개수만 무작위 제거, R_REPEATS회
   재추출):
   C1 = 그 창의 전체 트레이드 중 무작위 k개 제외 (순수 "트레이드 수 감소" 효과 대조군)
   C2 = "상승추세 아님 구간의 SHORT" 풀에서 무작위 k개 제외 (풀이 k보다 작으면 전부 제외,
        분산 없음-단일값으로 기록)
5. 각 창에서 5시드 벡터의 표준편차/부호일치를 baseline vs treatment vs C1분포 vs C2분포로 비교.
   treatment의 std가 C1/C2 분포에서 유독 낮은 백분위(percentile_rank_in_random)에 위치하면
   "상승추세중 SHORT를 특정 제거하는 게 유독 효과적"이라는 신호, C1/C2와 비슷한 위치면 트리비얼한
   "트레이드 수 감소=분산 감소" 효과와 구분 불가.

fresh_forward_bar_by_bar=true(원본 ledger가 그렇게 만들어짐, 여기서는 그 산출물의 사후 재집계),
trade_ledgers_used_as_input=true로 명시(이 스크립트 자체가 diagnostic 재집계이지 promotion 근거
아님 -- CLAUDE.md Fresh-Forward 규칙에 따라 이 결과는 promotion/test 근거로 쓰지 않는다).
새 모델/메커니즘 구축 아님, 순수 사후분석.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
LEDGER_DIR = ROOT / "tmp/causal_regen_20260516/btc_live_promotion_seed_robustness_20260819_eval"
REPORT_JSON = LEDGER_DIR / "report.json"
OUT_DIR = ROOT / "tmp/causal_regen_20260516/btc_odyssey4_shadow_uptrend_short_variance_diagnostic_20260820"

SEED_LABELS = ["260620_original", "750703416", "160125165", "626578270", "179796523"]
WINDOW_DEFS = {
    "2025q1": {"start": "2025-01-01", "end": "2025-03-31 23:59:59", "year_file": "2025"},
    "2025q2": {"start": "2025-04-01", "end": "2025-06-30 23:59:59", "year_file": "2025"},
    "2025q3": {"start": "2025-07-01", "end": "2025-09-30 23:59:59", "year_file": "2025"},
    "val":    {"start": "2025-10-01", "end": "2025-12-31", "year_file": "2025"},
    "oos_q1": {"start": "2026-01-01", "end": "2026-03-31", "year_file": "2026"},
    "oos_q2": {"start": "2026-04-01", "end": "2026-06-30", "year_file": "2026"},
}
WEEK_BARS = 2016  # 1 week of 5m bars -- identical convention to ETH's SustainedUptrendDetector
CALIB_PERCENTILE = 0.90  # ETH's primary percentile (p90); BTC's own threshold VALUE is recomputed, not reused
R_REPEATS = 2000
RNG_SEED = 20260820
PNL_SIGN_TOL = 0.0  # sign bucket: pnl >= 0 vs pnl < 0, matches eval_5seed's own `p >= 0` convention


def _rolling_score(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, usecols=["timestamp", "dual_momentum"])
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    dm_pos = (df["dual_momentum"] > 0.0).astype(float)
    df["sustained_uptrend_score"] = dm_pos.rolling(WEEK_BARS, min_periods=WEEK_BARS).mean()
    return df[["timestamp", "sustained_uptrend_score"]]


def _compound_metrics(returns: np.ndarray) -> dict:
    """Verbatim logic copy of apply_final_scale_map_btc_freshforward_ext_swingtransition_20260806.py
    ::_compound_metrics, operating on a raw trade_return array instead of a DataFrame (identical
    arithmetic, no pandas overhead -- needed since this loop runs thousands of times per window)."""
    if len(returns) == 0:
        return {"pnl": 0.0, "mdd": 0.0, "trades": 0, "wr": 0.0}
    cash = 1.0
    peak = 1.0
    mdd = 0.0
    wins = 0
    for ret in returns:
        cash *= 1.0 + float(ret)
        wins += int(ret > 0.0)
        peak = max(peak, cash)
        mdd = min(mdd, cash / max(peak, 1e-12) - 1.0)
    return {"pnl": float((cash - 1.0) * 100.0), "mdd": float(mdd * 100.0), "trades": int(len(returns)), "wr": float(wins / len(returns))}


def _pnl_only(returns: np.ndarray) -> float:
    """Vectorized pnl-only shortcut, mathematically identical to _compound_metrics(...)['pnl']
    (same cash *= (1+ret) product, just via np.prod instead of a python loop) -- used inside the
    R_REPEATS resampling loops where mdd/wr are not needed and python-loop overhead would dominate."""
    if len(returns) == 0:
        return 0.0
    return float((np.prod(1.0 + returns.astype(np.float64)) - 1.0) * 100.0)


def _sign_consistent(pnls: list[float]) -> bool:
    return len({p >= PNL_SIGN_TOL for p in pnls}) == 1


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    score_2025 = _rolling_score(ROOT / "data/splits/year_oos/btc_features_2025_swingtransition.csv")
    score_2026 = _rolling_score(ROOT / "data/splits/year_oos/btc_features_2026_swingtransition.csv")
    score_by_year = {"2025": score_2025, "2026": score_2026}

    calib = score_2025["sustained_uptrend_score"].dropna()
    threshold = float(calib.quantile(CALIB_PERCENTILE))
    print(f"[calibration] BTC own full-2025 rolling-1wk dual_momentum>0 score p{int(CALIB_PERCENTILE * 100)} = "
          f"{threshold:.6f} (n_calib_bars={len(calib)}/{len(score_2025)}, diagnostic placeholder, "
          f"NOT ETH's locked 0.8025793650793651, NOT production)", flush=True)

    # Sanity context (not a formal test): correlation between the score and trailing-week BTC
    # price return, to confirm the proxy is not pure noise for this asset before using it below.
    px = pd.read_csv(ROOT / "data/splits/year_oos/btc_features_2025_swingtransition.csv", usecols=["timestamp", "close"])
    px["timestamp"] = pd.to_datetime(px["timestamp"])
    trail_ret = px["close"].pct_change(WEEK_BARS)
    corr = float(pd.Series(score_2025["sustained_uptrend_score"].to_numpy()).corr(trail_ret))
    print(f"[sanity] corr(rolling-1wk uptrend score, trailing-1wk BTC price return), 2025 = {corr:.3f} "
          f"(context only, not part of the verdict)", flush=True)

    rng = np.random.default_rng(RNG_SEED)

    report_ledger_sanity: list[str] = []
    windows_out: dict = {}

    for wname, wd in WINDOW_DEFS.items():
        score_df = score_by_year[wd["year_file"]]
        seed_ledgers: dict[str, dict] = {}
        for seed_label in SEED_LABELS:
            ledger = pd.read_csv(LEDGER_DIR / f"ledger_{seed_label}_{wname}.csv")
            ledger["entry_timestamp"] = pd.to_datetime(ledger["entry_timestamp"])
            merged = ledger.merge(score_df, left_on="entry_timestamp", right_on="timestamp", how="left", validate="many_to_one")
            if len(merged) != len(ledger):
                raise RuntimeError(f"{seed_label}/{wname}: merge row count mismatch ({len(merged)} vs {len(ledger)})")
            uptrend_active = (merged["sustained_uptrend_score"] > threshold).fillna(False).to_numpy(dtype=bool)
            side = ledger["side"].to_numpy(dtype=np.int64)
            returns = ledger["trade_return"].to_numpy(dtype=np.float64)
            is_short = side == -1
            is_uptrend_short = is_short & uptrend_active
            is_nonuptrend_short = is_short & ~uptrend_active
            seed_ledgers[seed_label] = {
                "returns": returns, "is_short": is_short, "is_uptrend_short": is_uptrend_short,
                "is_nonuptrend_short": is_nonuptrend_short, "n": len(ledger), "k": int(is_uptrend_short.sum()),
            }

        # actual baseline / treatment vectors across the 5 seeds
        pnl_all, pnl_excl, ks, ns, n_shorts = [], [], [], [], []
        for seed_label in SEED_LABELS:
            d = seed_ledgers[seed_label]
            full_metrics = _compound_metrics(d["returns"])
            pnl_all.append(full_metrics["pnl"])
            pnl_excl.append(_pnl_only(d["returns"][~d["is_uptrend_short"]]))
            ks.append(d["k"]); ns.append(d["n"]); n_shorts.append(int(d["is_short"].sum()))

        std_all, std_excl = float(np.std(pnl_all)), float(np.std(pnl_excl))
        sign_all, sign_excl = _sign_consistent(pnl_all), _sign_consistent(pnl_excl)

        # controls: shared repeat index across the 5 seeds so we get a std-across-seeds NULL distribution
        std_c1_repeats, sign_c1_repeats = [], []
        std_c2_repeats, sign_c2_repeats = [], []
        c2_pool_exhausted_any = False
        for r in range(R_REPEATS):
            vals_c1, vals_c2 = [], []
            for seed_label in SEED_LABELS:
                d = seed_ledgers[seed_label]
                n, k = d["n"], d["k"]
                if k == 0:
                    vals_c1.append(_pnl_only(d["returns"]))
                    vals_c2.append(_pnl_only(d["returns"]))
                    continue
                drop1 = rng.choice(n, size=k, replace=False)
                keep1 = np.ones(n, dtype=bool); keep1[drop1] = False
                vals_c1.append(_pnl_only(d["returns"][keep1]))

                pool = np.where(d["is_nonuptrend_short"])[0]
                if len(pool) >= k:
                    drop2 = rng.choice(pool, size=k, replace=False)
                else:
                    drop2 = pool  # pool exhausted -- deterministic, same every repeat
                    c2_pool_exhausted_any = True
                keep2 = np.ones(n, dtype=bool); keep2[drop2] = False
                vals_c2.append(_pnl_only(d["returns"][keep2]))
            std_c1_repeats.append(float(np.std(vals_c1))); sign_c1_repeats.append(_sign_consistent(vals_c1))
            std_c2_repeats.append(float(np.std(vals_c2))); sign_c2_repeats.append(_sign_consistent(vals_c2))

        std_c1_repeats = np.asarray(std_c1_repeats); std_c2_repeats = np.asarray(std_c2_repeats)
        pctile_excl_in_c1 = float(np.mean(std_c1_repeats <= std_excl)) if sum(ks) > 0 else None
        pctile_excl_in_c2 = float(np.mean(std_c2_repeats <= std_excl)) if sum(ks) > 0 else None

        windows_out[wname] = {
            "threshold": threshold,
            "per_seed": {s: {"n_trades": seed_ledgers[s]["n"], "n_short": int(seed_ledgers[s]["is_short"].sum()),
                              "n_uptrend_short": seed_ledgers[s]["k"],
                              "n_nonuptrend_short": int(seed_ledgers[s]["is_nonuptrend_short"].sum())}
                         for s in SEED_LABELS},
            "pnl_all": dict(zip(SEED_LABELS, pnl_all)),
            "pnl_excl_uptrend_short": dict(zip(SEED_LABELS, pnl_excl)),
            "std_all": std_all, "std_excl_uptrend_short": std_excl,
            "sign_consistent_all": sign_all, "sign_consistent_excl_uptrend_short": sign_excl,
            "total_k_removed_across_seeds": int(sum(ks)),
            "control_c1_random_matched_count": {
                "std_mean": float(std_c1_repeats.mean()), "std_p10": float(np.quantile(std_c1_repeats, 0.10)),
                "std_p50": float(np.quantile(std_c1_repeats, 0.50)),
                "sign_consistent_rate": float(np.mean(sign_c1_repeats)),
                "percentile_rank_of_actual_excl_std": pctile_excl_in_c1,
            },
            "control_c2_nonuptrend_short_matched_count": {
                "std_mean": float(std_c2_repeats.mean()), "std_p10": float(np.quantile(std_c2_repeats, 0.10)),
                "std_p50": float(np.quantile(std_c2_repeats, 0.50)),
                "sign_consistent_rate": float(np.mean(sign_c2_repeats)),
                "percentile_rank_of_actual_excl_std": pctile_excl_in_c2,
                "pool_exhausted_in_some_repeat": c2_pool_exhausted_any,
            },
        }

        print(f"\n=== window={wname} (n5seed trades={ns}, shorts={n_shorts}, uptrend_shorts={ks}) ===")
        print(f"  pnl_all          = {[round(p, 2) for p in pnl_all]}  std={std_all:.2f}  sign_consistent={sign_all}")
        print(f"  pnl_excl_up_short= {[round(p, 2) for p in pnl_excl]}  std={std_excl:.2f}  sign_consistent={sign_excl}")
        print(f"  C1(random, matched-k)      : std_mean={std_c1_repeats.mean():.2f} p10={np.quantile(std_c1_repeats,0.10):.2f} "
              f"sign_rate={np.mean(sign_c1_repeats):.3f} pctile_rank(actual<=?)={pctile_excl_in_c1}")
        print(f"  C2(nonuptrend-short,matched): std_mean={std_c2_repeats.mean():.2f} p10={np.quantile(std_c2_repeats,0.10):.2f} "
              f"sign_rate={np.mean(sign_c2_repeats):.3f} pctile_rank(actual<=?)={pctile_excl_in_c2}")

    # sanity check against the already-known report.json ("pnl_all" here must reproduce it exactly)
    known = json.loads(REPORT_JSON.read_text())["windows"]
    max_abs_diff = 0.0
    for wname in WINDOW_DEFS:
        for seed_label in SEED_LABELS:
            known_pnl = known[seed_label][wname]["pnl"]
            got_pnl = windows_out[wname]["pnl_all"][seed_label]
            max_abs_diff = max(max_abs_diff, abs(known_pnl - got_pnl))
    print(f"\n[sanity] max abs diff between recomputed pnl_all and report.json pnl = {max_abs_diff:.8f} (expect ~0)")

    report = {
        "trade_ledgers_used_as_input": True,
        "note": "diagnostic post-hoc re-aggregation of an existing fresh-forward ledger, NOT a new backtest/training; "
                "per CLAUDE.md Fresh-Forward rule this output is NOT promotion/test evidence.",
        "threshold_source": "BTC-own full-2025 rolling(2016,min_periods=2016).mean(dual_momentum>0) distribution, p90 "
                             "-- diagnostic placeholder, NOT ETH's locked 0.8025793650793651, NOT production calibration.",
        "threshold_value": threshold,
        "sanity_corr_score_vs_trailing_week_return_2025": corr,
        "r_repeats_control": R_REPEATS,
        "seed_labels": SEED_LABELS,
        "sanity_max_abs_pnl_diff_vs_known_report_json": max_abs_diff,
        "windows": windows_out,
    }
    (OUT_DIR / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"\nreport={OUT_DIR / 'report.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
