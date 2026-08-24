#!/usr/bin/env python3
"""DC 방향라벨(LONG/SHORT) 대상 피쳐셋 정보량 점검 -- 원본 102개(h48qual/zig075 프로덕션
번들, `research_eth_omega461_exit_sweep_20260721.COMPONENTS["h48qual"]["bundle"]["base_cols"]`)
서브셋과 이번 DC 학습이 실제로 쓴 158개 base 피쳐(102개+신규56개, canonical 파이프라인 자연
진화분 -- docs/experiments/eth_odyssey4_exit_head_liveatr_barrier_and_label_reaudit_20260818.md
후속세션6/7 참고) 사이에 정보량 차이가 있는지 직접 비교한다.

⚠️ 이건 그 문서의 "pinned102 vs posfix" Fresh-Forward PnL 비교와 다르다 -- 거기는 피쳐셋+
sidecar+threshold 3개가 동시에 바뀐 단일시드 결과라 피쳐셋 하나만의 순효과를 볼 수 없었다.
여기서는 risk sidecar/threshold/모델 자체를 완전히 빼고, "이 피쳐가 실제 DC 방향라벨과
단독으로 얼마나 상관이 있는가"만 순수하게 잰다(모델 학습 없음, 초 단위로 끝남).

방법: 이벤트 bar(zigzag_action != CASH)만 놓고 LONG=1/SHORT=0 이진타겟에 대해 각 피쳐 단독
AUC-ROC(방향무관, max(auc,1-auc))를 계산. permutation null(라벨 셔플 200회)로 "158개/102개/
56개 중 순전히 우연으로 나올 수 있는 최고 AUC" 분포를 만들어 실제 최고 AUC가 유의한지 판단
-- 158개처럼 피쳐가 많으면 우연히도 하나쯤은 AUC가 높게 나올 수 있어(다중비교) raw AUC를
그냥 눈대중으로 보면 안 된다."""
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
ORIG_102 = json.loads((SCRATCH / "original_102_cols.json").read_text())
NEW_56 = sorted(set(BASE_158) - set(ORIG_102))
assert len(ORIG_102) == 102 and len(NEW_56) == 56 and len(BASE_158) == 158

LABEL_DIR = ROOT / "tmp/eth_directional_change_triple_barrier_labels_dense_cashfill_20260819"

sys.path.insert(0, str(ROOT / "scripts"))
import eth_directional_change_tabm_training_canonicaldata_20260819 as canon  # noqa: E402 (TRAIN_CSV/EVAL_CSV + overlay 오버라이드 부작용)
omega = canon.omega

RNG = np.random.default_rng(20260820)
N_PERM = 200


def auc_dir_agnostic(y: np.ndarray, x: np.ndarray) -> float:
    valid = ~np.isnan(x)
    if valid.sum() < 30 or len(np.unique(y[valid])) < 2:
        return float("nan")
    auc = roc_auc_score(y[valid], x[valid])
    return max(auc, 1.0 - auc)


def main() -> None:
    train, eval_df = omega._load_omega_frames()[:2]  # 우리 DC TabM 학습이 실제로 쓴 것과 동일한 프레임(2025=train, 2026=eval, 오버레이 전부 포함)
    frames = []
    for year, feat in ((2025, train), (2026, eval_df)):
        feat = feat[["timestamp", *BASE_158]].copy()
        feat["timestamp"] = pd.to_datetime(feat["timestamp"])
        lbl = pd.read_csv(LABEL_DIR / f"zigzag_action_labels_{year}.csv", usecols=["timestamp", "zigzag_action"], parse_dates=["timestamp"])
        m = feat.merge(lbl, on="timestamp", how="inner")
        frames.append(m)
    data = pd.concat(frames, ignore_index=True)
    events = data[data["zigzag_action"] != 0].reset_index(drop=True)
    y = (events["zigzag_action"] == 1).to_numpy().astype(np.int64)  # 1=LONG, 0=SHORT
    print(f"전체 병합 {len(data):,}행, 이벤트bar {len(events):,}개 (LONG={int(y.sum())} SHORT={int((1-y).sum())})", flush=True)

    real_auc: dict[str, float] = {}
    X: dict[str, np.ndarray] = {}
    for c in BASE_158:
        x = pd.to_numeric(events[c], errors="coerce").to_numpy(dtype=np.float64)
        X[c] = x
        real_auc[c] = auc_dir_agnostic(y, x)

    def summarize(name: str, cols: list[str]) -> dict:
        vals = {c: real_auc[c] for c in cols if not np.isnan(real_auc[c])}
        arr = np.array(list(vals.values()))
        top5 = sorted(vals.items(), key=lambda kv: kv[1], reverse=True)[:5]
        return {"name": name, "n_features": len(cols), "n_valid": len(arr),
                "mean_auc": float(arr.mean()), "median_auc": float(np.median(arr)),
                "max_auc": float(arr.max()), "top5": top5}

    groups = {"전체158": BASE_158, "원본102": ORIG_102, "신규56": NEW_56}
    summaries = {k: summarize(k, v) for k, v in groups.items()}
    for k, s in summaries.items():
        print(f"\n[{k}] n={s['n_features']}(valid={s['n_valid']}) mean_auc={s['mean_auc']:.4f} "
              f"median={s['median_auc']:.4f} max_auc={s['max_auc']:.4f}", flush=True)
        for feat, a in s["top5"]:
            print(f"    top: {feat:50s} auc={a:.4f}", flush=True)

    # --- permutation null: 라벨을 셔플해서 "우연히 나올 수 있는 최고 AUC" 분포 (그룹별 크기 반영) ---
    # 158개 전체에 대해 퍼뮤테이션당 1회만 AUC를 계산하고, 원본102/신규56은 그 결과에서 부분집합만
    # 취한다(102+56=158 컬럼이 세 그룹에 중복 소속되므로 재계산 방지 -- 158회/perm이면 충분).
    print(f"\npermutation null 계산 중 (N={N_PERM})...", flush=True)
    Xmat_all = np.column_stack([X[c] for c in BASE_158])
    idx_102 = [BASE_158.index(c) for c in ORIG_102]
    idx_56 = [BASE_158.index(c) for c in NEW_56]
    group_idx = {"전체158": list(range(len(BASE_158))), "원본102": idx_102, "신규56": idx_56}
    null_max = {k: [] for k in groups}
    for i in range(N_PERM):
        y_perm = RNG.permutation(y)
        aucs_all = np.full(len(BASE_158), np.nan)
        for j in range(len(BASE_158)):
            xj = Xmat_all[:, j]
            valid = ~np.isnan(xj)
            if valid.sum() < 30:
                continue
            a = roc_auc_score(y_perm[valid], xj[valid])
            aucs_all[j] = max(a, 1 - a)
        for k, idxs in group_idx.items():
            sub = aucs_all[idxs]
            sub = sub[~np.isnan(sub)]
            null_max[k].append(float(sub.max()) if len(sub) else float("nan"))

    report = {}
    for k in groups:
        null_arr = np.array(null_max[k])
        real_max = summaries[k]["max_auc"]
        p95 = float(np.nanpercentile(null_arr, 95))
        p99 = float(np.nanpercentile(null_arr, 99))
        empirical_p = float((null_arr >= real_max).mean())
        print(f"\n[{k}] 실제 최고AUC={real_max:.4f} vs null(라벨셔플) 95th={p95:.4f} 99th={p99:.4f} "
              f"empirical_p={empirical_p:.3f} (이 그룹 크기={len(groups[k])}개 피쳐 기준)", flush=True)
        report[k] = {**summaries[k], "top5": [(f, float(a)) for f, a in summaries[k]["top5"]],
                      "null_p95": p95, "null_p99": p99, "empirical_p_of_max": empirical_p}

    out_path = ROOT / "tmp/eth_dc_feature_set_information_content_20260820.json"
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    print(f"\n[report] {out_path}", flush=True)


if __name__ == "__main__":
    main()
