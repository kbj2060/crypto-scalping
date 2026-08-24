#!/usr/bin/env python3
"""CUSUM 정보량 검증 -- 2026(eval)년만 독립적으로. pooled(2025+2026) 검증은 2025가
지배적(38,430/55,851=69%)이라 유의성이 대부분 2025에서 왔을 수 있고, 앞선 top5-피쳐만의
연도분할 체크는 pooled에서 이미 뽑힌 5개로 범위를 좁혀서 봤다는 점에서 은근한 선택편향이
있다(그 5개가 애초에 2026 정보까지 포함해 뽑힌 것이므로). 이번엔 158개 base 피쳐 전체를
2026 데이터에만 독립적으로 다시 돌려 어떤 피쳐가 최고인지부터 새로 뽑고, 그 최고값이
2026 자체의 permutation null(라벨셔플 200회) 대비 유의한지 판정한다 -- DC θ=0.004/0.015,
CUSUM pooled와 동일 방법론, 모집단만 2026 단독으로 축소."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
SCRATCH = Path("/tmp/claude-1000/-home-kbj20-crypto-scalping/7445be14-7df6-4085-bc4a-6a5de4e4597d/scratchpad")
BASE_158 = json.loads((SCRATCH / "dc_base_158_cols.json").read_text())

sys.path.insert(0, str(ROOT / "scripts"))
import eth_directional_change_tabm_training_canonicaldata_20260819 as canon  # noqa: E402
omega = canon.omega

CUSUM_LABEL_DIR = ROOT / "tmp/eth_cusum_triple_barrier_labels_20260819"
RNG = np.random.default_rng(20260820)
N_PERM = 200
REF = {
    "dc_theta_0.004_pooled": {"n_events": 12193, "max_auc": 0.5141, "empirical_p": 0.380},
    "dc_theta_0.015_pooled": {"n_events": 1817, "max_auc": 0.5293, "empirical_p": 0.825},
    "cusum_pooled_2025+2026": {"n_events": 55851, "max_auc": 0.5105, "empirical_p": 0.000},
}


def auc_dir_agnostic(y: np.ndarray, x: np.ndarray) -> float:
    valid = ~np.isnan(x)
    if valid.sum() < 30 or len(np.unique(y[valid])) < 2:
        return float("nan")
    auc = roc_auc_score(y[valid], x[valid])
    return max(auc, 1.0 - auc)


def main() -> None:
    _train, eval_df = omega._load_omega_frames()[:2]  # eval_df = 2026
    feat = eval_df[["timestamp", *BASE_158]].copy()
    feat["timestamp"] = pd.to_datetime(feat["timestamp"])

    lbl = pd.read_csv(CUSUM_LABEL_DIR / "zigzag_action_labels_2026.csv", usecols=["timestamp", "zigzag_action"], parse_dates=["timestamp"])
    data = feat.merge(lbl, on="timestamp", how="inner")
    events = data[data["zigzag_action"] != 0].reset_index(drop=True)
    y = (events["zigzag_action"] == 1).to_numpy().astype(np.int64)
    n_long, n_short = int(y.sum()), int((1 - y).sum())
    print(f"CUSUM 2026 단독: 이벤트 {len(events):,}개 (LONG={n_long} SHORT={n_short})\n", flush=True)
    for k, v in REF.items():
        print(f"참고 -- {k}: n={v['n_events']:,} max_auc={v['max_auc']:.4f} empirical_p={v['empirical_p']:.3f}", flush=True)
    print(flush=True)

    X = {}
    real_auc = {}
    for c in BASE_158:
        x = pd.to_numeric(events[c], errors="coerce").to_numpy(dtype=np.float64)
        X[c] = x
        real_auc[c] = auc_dir_agnostic(y, x)
    vals = {k: v for k, v in real_auc.items() if not np.isnan(v)}
    arr = np.array(list(vals.values()))
    top10 = sorted(vals.items(), key=lambda kv: kv[1], reverse=True)[:10]
    mean_auc, max_auc = float(arr.mean()), float(arr.max())
    print(f"2026단독(158개 전체 새로 순위): mean_auc={mean_auc:.4f} max_auc={max_auc:.4f}", flush=True)
    for feat_name, a in top10:
        flag = " <- pooled top5" if feat_name in {"trades", "volume_btc", "quote_volume", "sum_open_interest_value", "quote_volume_btc"} else ""
        print(f"  top: {feat_name:50s} auc={a:.4f}{flag}", flush=True)

    Xmat = np.column_stack([X[c] for c in BASE_158])
    null_max = []
    for _ in range(N_PERM):
        y_perm = RNG.permutation(y)
        aucs = []
        for j in range(Xmat.shape[1]):
            xj = Xmat[:, j]
            valid = ~np.isnan(xj)
            if valid.sum() < 30:
                continue
            a = roc_auc_score(y_perm[valid], xj[valid])
            aucs.append(max(a, 1 - a))
        null_max.append(max(aucs) if aucs else float("nan"))
    null_arr = np.array(null_max)
    p95 = float(np.nanpercentile(null_arr, 95))
    empirical_p = float((null_arr >= max_auc).mean())
    print(f"\n2026단독 permutation null: 95th={p95:.4f} empirical_p={empirical_p:.3f} "
          f"({'유의' if empirical_p < 0.05 else '비유의(chance)'})", flush=True)

    # pooled top5가 2026에서 몇 위인지도 별도로 보고
    pooled_top5 = ["trades", "volume_btc", "quote_volume", "sum_open_interest_value", "quote_volume_btc"]
    ranked = sorted(vals.items(), key=lambda kv: kv[1], reverse=True)
    rank_of = {f: i + 1 for i, (f, _) in enumerate(ranked)}
    print("\npooled top5가 2026단독 158개 중 몇 위인지:", flush=True)
    for f in pooled_top5:
        print(f"  {f:30s} 2026단독 순위={rank_of.get(f, '?')}/158 auc={vals.get(f, float('nan')):.4f}", flush=True)

    out = {"n_events": int(len(events)), "n_long": n_long, "n_short": n_short,
           "mean_auc": mean_auc, "max_auc": max_auc, "top10": [(f, float(a)) for f, a in top10],
           "null_p95": p95, "empirical_p": empirical_p,
           "pooled_top5_rank_in_2026": {f: rank_of.get(f) for f in pooled_top5}, "reference": REF}
    out_path = ROOT / "tmp/eth_cusum_2026only_feature_information_content_20260820.json"
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    print(f"\n[report] {out_path}")


if __name__ == "__main__":
    main()
