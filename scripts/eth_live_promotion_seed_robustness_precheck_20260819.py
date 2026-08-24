#!/usr/bin/env python3
"""본 학습(수 분 소요) 돌리기 전에, canonical 데이터에서 h48qual/zig075 원본 102 base_cols가
전부 존재하는지만 가볍게 확인 (모델 학습 없음, _load_omega_frames + _numeric_feature_cols만
호출)."""
from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import eth_live_promotion_seed_robustness_canonicaldata_20260819 as canon_wrap  # noqa: E402
import research_eth_omega461_exit_sweep_20260721 as sweep  # noqa: E402

parent_script = canon_wrap.parent_script
omega = parent_script.omega

train_all, eval_df, overlay_report = omega._load_omega_frames()
full = omega._numeric_feature_cols(train_all, eval_df)
full_set = set(full)
print(f"auto-derived full feature count: {len(full)}", flush=True)

for name in ("h48qual", "zig075"):
    orig_cols = list(torch.load(sweep.COMPONENTS[name]["bundle"], map_location="cpu", weights_only=False)["base_cols"])
    missing = sorted(set(orig_cols) - full_set)
    print(f"{name}: original_102_count={len(orig_cols)} missing={len(missing)} {missing[:20]}", flush=True)

print("PRECHECK_DONE", flush=True)
