#!/usr/bin/env python3
"""`eth_tabm_label_logic_5way_seed_variant_ilias_anchored_20260821.py`와 로직 동일(단일모델
monkeypatch+exit-label-mode=independent_entry_hold_offsets+epochs=2) -- 유일한 차이는
`eth_directional_change_tabm_training_ilias_anchored_20260821` 대신 계약서 확정 split 규약을
쓰는 `eth_directional_change_tabm_training_ilias_anchored_canonical_20260822`을 씀."""
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

import eth_directional_change_tabm_training_ilias_anchored_canonical_20260822 as feat154  # noqa: E402
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

# ⚠️ 2026-08-22 재확장: 원본 `_2024merged` 라벨은 "2025.csv"라는 이름 아래 2024+2025만
# 병합해뒀다 -- 이 스크립트의 TRAIN_CSV는 2026 Q2(06-30)까지 담기는데, `_prepare_frames()`가
# `_read_labels(dir, 2025)`로 그 이름의 파일만 읽어 TRAIN 정렬에 쓰므로, 2026 Q1/Q2 라벨이
# 없으면 그 구간이 `_align()`에서 통째로 조용히 사라진다(실측: val_raw가 비어 "need at least
# one array to concatenate"로 크래시). `_2024_2026q2merged`는 "2025.csv" 안에 2024+2025+
# 2026-06-30까지 재병합해 이 문제를 해소한다("2026.csv"는 원본 그대로, OOS 정렬엔 영향 없음).
ZIGZAG_ACTION_DIR = ROOT / "tmp/ilias_labellogic_recheck_20260821/direction_labels_2024_2026q2merged/zigzag"
H48_CONSERVATIVE_QUALITY_DIR = ROOT / "tmp/ilias_labellogic_recheck_20260821/quality_labels_2024_2026q2merged/h48_conservative"
CUSUM_DENSE_LABEL_DIR = ROOT / "tmp/ilias_labellogic_recheck_20260821/direction_labels_2024_2026q2merged/cusum"

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

import os  # noqa: E402

COMMON_ARGS = [
    "--exit-label-mode", "independent_entry_hold_offsets",
    "--epochs", os.environ.get("TABM_EPOCHS", "2"),   # 2026-08-22: epoch어블레이션용 env override
    "--device", "auto",
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

    epochs_tag = f"_ep{os.environ.get('TABM_EPOCHS', '2')}" if os.environ.get("TABM_EPOCHS") else ""
    out_suffix = f"label5way_{known.label}_154feat_ilias_anchored_canonical_seed{known.seed}_20260822{epochs_tag}"
    sys.argv = [sys.argv[0], *args, "--seed", str(known.seed), "--out-suffix", out_suffix]
    print(f"stage=start label={known.label} seed={known.seed} out_suffix={out_suffix}", flush=True)
    t0 = time.time()
    result = parent_script.main()
    print(f"stage=done label={known.label} seed={known.seed} elapsed={time.time() - t0:.1f}s", flush=True)
    raise SystemExit(result)
