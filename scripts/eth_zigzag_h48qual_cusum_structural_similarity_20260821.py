#!/usr/bin/env python3
"""zigzag/h48qual/cusum 3개 라벨이 왜 시각적으로 비슷해 보이는지(chart_zigzag_h48qual_cusum_
label_comparison_20260821.py) 구조적으로 검증. DC-vs-CUSUM 유사도 통계분석(이 서브프로젝트
메모리 기록, Pearson daily-count 0.125 + vol_at_entry KS=0.220 + 매칭점 방향일치율 92.9~100%)과
동일 방법론을 3개 라벨쌍(zigzag-h48qual/zigzag-cusum/h48qual-cusum) 전체로 확장.

공통 윈도우: 2024-01-01~2026-02-28 (h48qual의 실제 5-way 학습용 quality label 소스,
sltp_h48_conservative_padded_to_zigzag_timestamps가 2026-02-28까지만 실제 평가값을 갖고
있어 -- 그 이후는 quality 미평가로 CASH 디폴트 처리되는 것으로 추정 -- 이 윈도우로 제한해야
h48qual쪽이 "진짜 신호"인 구간만 비교하게 됨. zigzag/cusum은 2026-06-30까지 커버하지만
이 분석에서는 h48qual 커버리지에 맞춰 자른다).

4가지 체크:
  1. 같은-bar 매칭률 + 매칭점 방향일치율 (DC-vs-CUSUM 정확 재현)
  2. 일별 활성이벤트수 Pearson 상관
  3. 활성bar의 시장변동성(zigzag_atr_pct) 분포 KS-test
  4. circular-shift 순열귀무(200회) -- 매칭률이 각 라벨 고유의 시간적 자기상관(run-length
     구조)을 보존한 채 무작위 정렬해도 나올 법한 수준인지, 아니면 진짜 시간정렬된 구조인지
  5. zigzag_segment_id 단위 포함관계 -- zigzag 세그먼트(고점-저점 스윙) 내부 bar들 중 몇 %가
     h48qual/cusum도 같은 방향으로 활성인지(핵심 가설: "같은 스윙을 다른 강도로 게이팅"을
     세그먼트 단위로 직접 검증)"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path("/home/kbj20/crypto-scalping")
WINDOW_START, WINDOW_END = "2024-01-01", "2026-02-28"

ZIGZAG_DIR = ROOT / "tmp/causal_regen_20260516/omega_current_only_all_label_candidate_parent_screen_20260629/label_contracts/zigzag_action_labels_20260531"
H48_DIR = ROOT / "tmp/causal_regen_20260516/omega_zigzag_fix_all_solutions_20260630/label_contracts/sltp_h48_conservative_padded_to_zigzag_timestamps"
CUSUM_DIR = ROOT / "tmp/eth_cusum_triple_barrier_labels_dense_cashfill_20260820"
OUT_JSON = Path("/tmp/claude-1000/-home-kbj20-crypto-scalping/7445be14-7df6-4085-bc4a-6a5de4e4597d/scratchpad/structural_similarity_20260821.json")

RNG = np.random.default_rng(20260821)
N_PERM = 200


def _load_year_concat(dir_: Path, years: list[int], cols: list[str]) -> pd.DataFrame:
    parts = []
    for y in years:
        p = dir_ / f"zigzag_action_labels_{y}.csv"
        if not p.exists():
            continue
        parts.append(pd.read_csv(p, usecols=[c for c in cols if c in pd.read_csv(p, nrows=0).columns], parse_dates=["timestamp"]))
    return pd.concat(parts, ignore_index=True).drop_duplicates(subset="timestamp").sort_values("timestamp").reset_index(drop=True)


def _window(df: pd.DataFrame) -> pd.DataFrame:
    return df[(df["timestamp"] >= WINDOW_START) & (df["timestamp"] <= WINDOW_END)].reset_index(drop=True)


def _same_bar_match(a: pd.Series, b: pd.Series) -> dict:
    """a, b: aligned zigzag_action arrays (0/1/2), same index/length."""
    a_active = a != 0
    b_active = b != 0
    both_active = a_active & b_active
    match_rate_given_a_active = float(both_active.sum() / max(1, a_active.sum()))
    dir_agree = float((a[both_active] == b[both_active]).mean()) if both_active.sum() > 0 else float("nan")
    return {
        "a_active_n": int(a_active.sum()), "b_active_n": int(b_active.sum()),
        "both_active_n": int(both_active.sum()),
        "match_rate_given_a_active": match_rate_given_a_active,
        "direction_agreement_at_match": dir_agree,
        "independence_expected_match_rate": float(b_active.mean()),
    }


def _tolerance_match(a: np.ndarray, b: np.ndarray, tol_bars: int) -> dict:
    """DC-vs-CUSUM 정확재현: a가 active인 bar t에서 b가 [t-tol,t+tol] 구간 어딘가에서도
    active인지(같은 방향인지는 그 구간 내 최근접 매칭점 기준). searchsorted로 벡터화(zigzag처럼
    active bar가 20만개대라 순수파이썬 루프는 비현실적)."""
    a_idx = np.flatnonzero(a != 0)
    b_idx = np.flatnonzero(b != 0)
    if len(b_idx) == 0 or len(a_idx) == 0:
        return {"a_active_n": int(len(a_idx)), "matched_within_tol": 0, "match_rate": 0.0, "direction_agreement_at_match": float("nan")}
    pos = np.searchsorted(b_idx, a_idx)
    left = np.clip(pos - 1, 0, len(b_idx) - 1)
    right = np.clip(pos, 0, len(b_idx) - 1)
    dist_left = np.abs(b_idx[left] - a_idx)
    dist_right = np.abs(b_idx[right] - a_idx)
    use_right = dist_right <= dist_left
    nearest_b_idx = np.where(use_right, b_idx[right], b_idx[left])
    nearest_dist = np.where(use_right, dist_right, dist_left)
    within = nearest_dist <= tol_bars
    matched = int(within.sum())
    agree = int((b[nearest_b_idx[within]] == a[a_idx[within]]).sum())
    return {"a_active_n": int(len(a_idx)), "matched_within_tol": matched,
            "match_rate": float(matched / len(a_idx)), "direction_agreement_at_match": float(agree / matched) if matched else float("nan")}


def _circular_perm_null(a: np.ndarray, b: np.ndarray, n_perm: int) -> dict:
    n = len(a)
    a_active = a != 0
    observed = float(((a != 0) & (b != 0)).sum())
    null_vals = np.empty(n_perm)
    for i in range(n_perm):
        shift = int(RNG.integers(1, n - 1))
        b_shift = np.roll(b, shift)
        null_vals[i] = float(((a != 0) & (b_shift != 0)).sum())
    p = float((null_vals >= observed).mean())
    return {"observed_both_active_n": observed, "null_mean": float(null_vals.mean()), "null_std": float(null_vals.std()),
            "empirical_p_ge_observed": p}


def _flip_frequency(action: np.ndarray) -> dict:
    """CASH(0) bar 제외한 LONG/SHORT 결정만 남겨 연속 방향전환 빈도+연속유지길이(run length,
    bar단위/결정단위 둘다) 계산. 사용자 관찰("cusum이 너무 자주 롱숏을 바꾸는거 아니냐")을
    직접 정량화."""
    nonzero_mask = action != 0
    seq = action[nonzero_mask]
    if len(seq) < 2:
        return {"n_decisions": int(len(seq)), "flip_rate": float("nan"), "run_len_decisions_mean": float("nan")}
    flips = seq[1:] != seq[:-1]
    flip_rate = float(flips.mean())
    # run lengths in decision-count units (consecutive equal values)
    change_idx = np.flatnonzero(flips) + 1
    bounds = np.concatenate(([0], change_idx, [len(seq)]))
    run_lens = np.diff(bounds)
    # run lengths in bar-count units (consecutive equal values including CASH gaps between them)
    idx_all = np.flatnonzero(nonzero_mask)
    run_starts_idx = idx_all[np.concatenate(([0], change_idx))]
    run_ends_idx = idx_all[np.concatenate((change_idx, [len(seq)])) - 1]
    run_len_bars = run_ends_idx - run_starts_idx + 1
    return {
        "n_decisions": int(len(seq)),
        "n_flips": int(flips.sum()),
        "flip_rate": flip_rate,
        "run_len_decisions_mean": float(run_lens.mean()), "run_len_decisions_median": float(np.median(run_lens)),
        "run_len_bars_mean": float(run_len_bars.mean()), "run_len_bars_median": float(np.median(run_len_bars)),
        "run_len_bars_p90": float(np.percentile(run_len_bars, 90)),
    }


def main() -> None:
    print("loading...", flush=True)
    zigzag = _window(_load_year_concat(ZIGZAG_DIR, [2024, 2025, 2026], ["timestamp", "zigzag_action", "zigzag_atr_pct", "zigzag_segment_id"]))
    h48 = _window(_load_year_concat(H48_DIR, [2025, 2026], ["timestamp", "zigzag_action"]))
    cusum = _window(_load_year_concat(CUSUM_DIR, [2024, 2025, 2026], ["timestamp", "zigzag_action", "vol_at_entry"]))
    print(f"zigzag={len(zigzag)} h48={len(h48)} cusum={len(cusum)}", flush=True)

    base = zigzag[["timestamp", "zigzag_action", "zigzag_atr_pct", "zigzag_segment_id"]].rename(columns={"zigzag_action": "zigzag"})
    base = base.merge(h48[["timestamp", "zigzag_action"]].rename(columns={"zigzag_action": "h48qual"}), on="timestamp", how="inner")
    base = base.merge(cusum[["timestamp", "zigzag_action", "vol_at_entry"]].rename(columns={"zigzag_action": "cusum"}), on="timestamp", how="inner")
    base = base.sort_values("timestamp").reset_index(drop=True)
    print(f"joined common rows: {len(base)} [{base['timestamp'].min()}..{base['timestamp'].max()}]", flush=True)

    results: dict = {"window": [WINDOW_START, WINDOW_END], "rows": len(base), "pairs": {}, "flip_frequency": {}}

    print("\n--- flip-frequency / run-length (사용자 관찰 검증: cusum이 더 자주 뒤집히나?) ---", flush=True)
    for name in ("zigzag", "h48qual", "cusum"):
        vc = base[name].value_counts().to_dict()
        ff = _flip_frequency(base[name].to_numpy())
        results["flip_frequency"][name] = ff
        print(f"  {name}: active_ratio={(base[name] != 0).mean():.4f} counts={vc} | "
              f"n_decisions={ff['n_decisions']} flip_rate={ff['flip_rate']:.4f} "
              f"run_len_bars(mean/median/p90)={ff['run_len_bars_mean']:.1f}/{ff['run_len_bars_median']:.1f}/{ff['run_len_bars_p90']:.1f}", flush=True)

    pairs = [("zigzag", "h48qual"), ("zigzag", "cusum"), ("h48qual", "cusum")]

    for a_name, b_name in pairs:
        a = base[a_name].to_numpy()
        b = base[b_name].to_numpy()
        match = _same_bar_match(base[a_name], base[b_name])
        match_rev = _same_bar_match(base[b_name], base[a_name])
        tol1 = _tolerance_match(a, b, 1)   # +-5min
        tol3 = _tolerance_match(a, b, 3)   # +-15min
        daily = base.assign(date=base["timestamp"].dt.date)
        daily_counts = daily.groupby("date").apply(lambda g: pd.Series({
            a_name: (g[a_name] != 0).sum(), b_name: (g[b_name] != 0).sum()}), include_groups=False)
        pearson_r, pearson_p = stats.pearsonr(daily_counts[a_name], daily_counts[b_name])
        perm = _circular_perm_null(a, b, N_PERM)

        a_vol = base.loc[base[a_name] != 0, "zigzag_atr_pct"].dropna()
        b_vol = base.loc[base[b_name] != 0, "zigzag_atr_pct"].dropna()
        ks_stat, ks_p = stats.ks_2samp(a_vol, b_vol)

        results["pairs"][f"{a_name}_vs_{b_name}"] = {
            f"match_given_{a_name}_active": match,
            f"match_given_{b_name}_active": match_rev,
            "tolerance_5min_given_a_active": tol1, "tolerance_15min_given_a_active": tol3,
            "daily_count_pearson_r": float(pearson_r), "daily_count_pearson_p": float(pearson_p),
            "circular_shift_permutation_null": perm,
            "atr_pct_at_active_ks": {"statistic": float(ks_stat), "p_value": float(ks_p),
                                      f"{a_name}_median_atr_pct": float(a_vol.median()), f"{b_name}_median_atr_pct": float(b_vol.median())},
        }
        print(f"\n=== {a_name} vs {b_name} ===", flush=True)
        print(f"  same-bar match_rate(given {a_name} active)={match['match_rate_given_a_active']:.4f} "
              f"(independence-expected={match['independence_expected_match_rate']:.4f}) "
              f"dir_agreement_at_match={match['direction_agreement_at_match']:.4f}", flush=True)
        print(f"  +-5min  match_rate={tol1['match_rate']:.4f} dir_agreement={tol1['direction_agreement_at_match']:.4f}", flush=True)
        print(f"  +-15min match_rate={tol3['match_rate']:.4f} dir_agreement={tol3['direction_agreement_at_match']:.4f}", flush=True)
        print(f"  daily_count Pearson r={pearson_r:.4f} (p={pearson_p:.2e})", flush=True)
        print(f"  circular-shift null: observed_both_active={perm['observed_both_active_n']:.0f} "
              f"null_mean={perm['null_mean']:.1f}+-{perm['null_std']:.1f} empirical_p={perm['empirical_p_ge_observed']:.4f}", flush=True)
        print(f"  ATR%% KS: stat={ks_stat:.4f} p={ks_p:.2e} median({a_name})={a_vol.median():.4f} median({b_name})={b_vol.median():.4f}", flush=True)

    # segment-level containment: for each zigzag segment, what fraction of its bars are also
    # h48qual/cusum-active in the SAME direction as the segment's own zigzag_action?
    seg = base.groupby("zigzag_segment_id").apply(lambda g: pd.Series({
        "n_bars": len(g),
        "zigzag_dir": g["zigzag"].mode().iat[0] if len(g["zigzag"].mode()) else 0,
        "h48qual_same_dir_frac": float((g["h48qual"] == g["zigzag"]).mean()),
        "cusum_same_dir_frac": float((g["cusum"] == g["zigzag"]).mean()),
    }), include_groups=False)
    seg = seg[seg["zigzag_dir"] != 0]
    results["segment_containment"] = {
        "n_segments": int(len(seg)),
        "h48qual_same_dir_frac_mean": float(seg["h48qual_same_dir_frac"].mean()),
        "h48qual_same_dir_frac_median": float(seg["h48qual_same_dir_frac"].median()),
        "cusum_same_dir_frac_mean": float(seg["cusum_same_dir_frac"].mean()),
        "cusum_same_dir_frac_median": float(seg["cusum_same_dir_frac"].median()),
    }
    print(f"\n=== zigzag segment containment (n_segments={len(seg)}) ===", flush=True)
    print(f"  h48qual same-direction-as-zigzag-segment: mean={seg['h48qual_same_dir_frac'].mean():.4f} median={seg['h48qual_same_dir_frac'].median():.4f}", flush=True)
    print(f"  cusum   same-direction-as-zigzag-segment: mean={seg['cusum_same_dir_frac'].mean():.4f} median={seg['cusum_same_dir_frac'].median():.4f}", flush=True)

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")
    print(f"\nsaved {OUT_JSON}", flush=True)


if __name__ == "__main__":
    main()
