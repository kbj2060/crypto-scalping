#!/usr/bin/env python3
"""`eth_tabm_label_logic_5way_seed_variant_20260820.py`의 앵커드 walk-forward 변형 -- 로직은
동일(단일모델 monkeypatch+exit-label-mode=independent_entry_hold_offsets+epochs=2), 유일한
차이는 `eth_dc_engineered_features_canonicaldata_20260820` 대신 그 위에 TRAIN/EVAL/레짐오버레이
4개 경로만 앵커드 walk-forward로 갈아끼운 `eth_directional_change_tabm_training_ilias_anchored_20260821`을
씀. zigzag/h48qual/cusum 3개만 지원(dc는 이 이관 축에서 이미 제외됨, 위 계약서 참고)."""
from __future__ import annotations

import argparse
import copy
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np  # noqa: E402
import torch  # noqa: E402

import eth_directional_change_tabm_training_ilias_anchored_20260821 as feat154  # noqa: E402
import train_omega1_regime3_expert_direction_head_volpca_20260602 as hard  # noqa: E402

parent_script = feat154.parent_script


def _uniform_route_probs(frame):
    return np.ones((len(frame), len(hard.EXPERT_NAMES)), dtype=np.float64)


parent_script._route_probs = _uniform_route_probs

_cache: dict[str, object] = {}


def _fit_expert_omega4_unified(*args, **kwargs):
    expert_idx = kwargs.get("expert_idx")
    model_path = kwargs.get("model_path")
    orig = _fit_expert_omega4_unified._orig
    if expert_idx == 0:
        t0 = time.time()
        print(f"  expert_idx=0 (bull) -- ACTUALLY TRAINING (uniform regime weight) start", flush=True)
        payload = orig(*args, **kwargs)
        print(f"  expert_idx=0 done epochs_ran={payload.get('epochs_ran')} "
              f"best_validation_loss={payload.get('best_validation_loss')} elapsed={time.time() - t0:.1f}s", flush=True)
        _cache["payload"] = payload
        return payload
    cached = copy.deepcopy(_cache["payload"])
    cached["expert"] = hard.EXPERT_NAMES[int(expert_idx)]
    Path(model_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(cached, model_path)
    print(f"  expert_idx={expert_idx} ({hard.EXPERT_NAMES[int(expert_idx)]}) -- SKIPPED, reused bull's weights "
          f"(same best_validation_loss={cached.get('best_validation_loss')})", flush=True)
    return cached


_fit_expert_omega4_unified._orig = parent_script._fit_expert_omega4
parent_script._fit_expert_omega4 = _fit_expert_omega4_unified

# ⚠️ 2026-08-21 재정정: `_prepare_frames()`(train_eval_omega4_3head_parent72_loose_entry_
# quality_20260620.py:319)가 `_read_labels(direction_label_dir, 2025, ...)`로 **연도를
# 하드코딩**해서, omega.TRAIN_CSV를 2024까지 넓혀도 2024는 `_align()`에서 조용히 버려졌었다
# (실측 확인: train_predictions가 2025-01~09만 있었음). 이 공유 스크립트(BTC/SOL 형제+라이브
# quality_threshold 선정에도 쓰임, [[eth_omega4_quality_threshold_alpha67_pipeline_irreproducible_20260815]])는
# 수정하지 않는다 -- 대신 `_read_labels`가 파일명(`zigzag_action_labels_2025.csv`)만 보고
# 내용의 실제 연도는 검증 안 한다는 점을 이용해, **"2025"라는 이름의 파일 안에 2024+2025
# 병합 데이터를 넣은 새 디렉토리**를 가리키는 것으로 우회(로컬 오버라이드, 원본 무수정).
ZIGZAG_ACTION_DIR = ROOT / "tmp/ilias_labellogic_recheck_20260821/direction_labels_2024merged/zigzag"
# h48qual quality: original source (alpha6/7-lineage price data) is permanently irreproducible
# ([[eth_omega4_quality_threshold_alpha67_pipeline_irreproducible_20260815]]) -- recomputed the
# SAME barrier recipe/params (imported, not reimplemented, from
# build_omega1_2_triple_barrier_labels_20260619.py) on canonical OHLC instead
# (build_h48_conservative_barrier_canonicaldata_20260821.py), then padded onto the SAME rebuilt
# zigzag timestamps used above (pad_h48_conservative_canonicaldata_to_zigzag_timestamps_20260821.py)
# -- so h48qual's direction can now use the rebuilt zigzag too (row counts match by construction).
H48_CONSERVATIVE_QUALITY_DIR = ROOT / "tmp/ilias_labellogic_recheck_20260821/quality_labels_2024merged/h48_conservative"
CUSUM_DENSE_LABEL_DIR = ROOT / "tmp/ilias_labellogic_recheck_20260821/direction_labels_2024merged/cusum"

LABEL_CONFIGS = {
    "zigzag": {
        "direction_label_dir": str(ZIGZAG_ACTION_DIR),
        "quality_mode": "same_as_direction",
        "quality_label_dir": None,
    },
    "h48qual": {
        "direction_label_dir": str(ZIGZAG_ACTION_DIR),
        "quality_mode": "quality_label_action",
        "quality_label_dir": str(H48_CONSERVATIVE_QUALITY_DIR),
    },
    "cusum": {
        "direction_label_dir": str(CUSUM_DENSE_LABEL_DIR),
        "quality_mode": "same_as_direction",
        "quality_label_dir": None,
    },
}

COMMON_ARGS = [
    "--exit-label-mode", "independent_entry_hold_offsets",
    "--epochs", "2",
    "--device", "cpu",
]

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", required=True, choices=sorted(LABEL_CONFIGS.keys()))
    ap.add_argument("--seed", type=int, required=True)
    known, _ = ap.parse_known_args()

    cfg = LABEL_CONFIGS[known.label]
    args = ["--direction-label-dir", cfg["direction_label_dir"], "--quality-mode", cfg["quality_mode"]]
    if cfg["quality_label_dir"]:
        args += ["--quality-label-dir", cfg["quality_label_dir"]]
    args += COMMON_ARGS

    out_suffix = f"label5way_{known.label}_154feat_ilias_anchored_seed{known.seed}_20260821"
    sys.argv = [sys.argv[0], *args, "--seed", str(known.seed), "--out-suffix", out_suffix]
    print(f"stage=start label={known.label} seed={known.seed} out_suffix={out_suffix}", flush=True)
    t0 = time.time()
    result = parent_script.main()
    print(f"stage=done label={known.label} seed={known.seed} elapsed={time.time() - t0:.1f}s", flush=True)
    raise SystemExit(result)
