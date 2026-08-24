#!/usr/bin/env python3
"""train_eth_ilias1_zig075_ssl_trunk_pretrain_20260819.py가 저장한 사전학습 trunk 가중치를
새로 생성되는 모든 ThreeHeadTabM 인스턴스에 초기화값으로 주입한 뒤, trial12(Optuna 우승
레시피: lr=9.98e-4, wd=1.32e-4, direction_focal_gamma=7.0, patience=10, epochs cap=40)와
완전히 동일한 설정으로 seed=260620 하나만 파인튜닝 -- trial12/seed260620의 이미 알려진
val_loss=1.5339(사전학습 없음, 랜덤초기화)를 대조군으로 그대로 재사용해서 재실행 없이 비교.

패치 지점: parent.ThreeHeadTabM.__init__ 직후 -- head(direction/quality/exit) 가중치는
사전학습 때 forward 경로에 없어서 무의미(랜덤 그대로)하므로 로드 대상에서 명시적으로 제외,
trunk(input_scale/input_bias/in_proj/blocks/expert_scale/norms)만 주입. 이후 3-head 학습은
_fit_expert_omega4가 정상적으로 head를 처음부터 학습시킴(fine-tuning, trunk는 초기값만
바뀌고 계속 학습됨 -- freeze 아님)."""
from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import torch  # noqa: E402

import train_eval_omega4_3head_parent72_eth_canonicaldata_pinned102_20260818 as canon  # noqa: E402
import train_eval_omega1_2_tabm_3head_20260603 as parent  # noqa: E402

parent_script = canon.parent_script

TRUNK_PATH = ROOT / "tmp/causal_regen_20260516/ssl_zig075_trunk_pretrain_20260819/trunk_state.pt"
HEAD_KEY_PREFIXES = ("direction_head.", "quality_head.", "exit_head.")

SEED = 260620
LR = 9.98e-4
WEIGHT_DECAY = 1.32e-4
FOCAL_GAMMA = 7.0
PATIENCE = 10
EPOCHS_CAP = 40

_BASE_CFG = parent.CFG
_trunk_payload = torch.load(TRUNK_PATH, map_location="cpu", weights_only=False)
_TRUNK_STATE = {k: v for k, v in _trunk_payload["trunk_state_dict"].items() if not k.startswith(HEAD_KEY_PREFIXES)}
print(f"loaded pretrained trunk: {len(_TRUNK_STATE)} tensors, pretrain best_val_recon_mse={_trunk_payload['best_val_recon_mse']:.4f}", flush=True)

_orig_init = parent.ThreeHeadTabM.__init__


def _patched_init(self, n_features, *, cfg=parent.CFG):
    _orig_init(self, n_features, cfg=cfg)
    sd = self.state_dict()
    applied = 0
    for k, v in _TRUNK_STATE.items():
        if k in sd and sd[k].shape == v.shape:
            sd[k] = v
            applied += 1
    missing = len(_TRUNK_STATE) - applied
    if missing:
        raise RuntimeError(f"SSL trunk injection: {missing}/{len(_TRUNK_STATE)} tensors did not match by key+shape -- refusing partial injection")
    self.load_state_dict(sd)


parent.ThreeHeadTabM.__init__ = _patched_init
parent.CFG = replace(_BASE_CFG, lr=LR, weight_decay=WEIGHT_DECAY, patience=PATIENCE)

BASE_ARGS = [
    "--direction-label-dir", str(ROOT / "tmp/causal_regen_20260516/omega_current_only_all_label_candidate_parent_screen_20260629/label_contracts/zigzag_action_labels_20260531"),
    "--quality-mode", "same_as_direction",
    "--exit-label-mode", "entry_label_terminal_giveback",
    "--max-exit-samples", "30000",
    "--epochs", str(EPOCHS_CAP),
    "--quality-thresholds", "0.55,0.60,0.65,0.70,0.75,0.80,0.85,0.90,0.95",
    "--device", "cpu",
    "--seed", str(SEED),
    "--out-suffix", "ssl_pretrained_finetune_20260819",
    "--direction-focal-gamma", str(FOCAL_GAMMA),
]

if __name__ == "__main__":
    sys.argv = [sys.argv[0], *BASE_ARGS]
    print(f"stage=start seed={SEED} lr={LR} weight_decay={WEIGHT_DECAY} focal_gamma={FOCAL_GAMMA} (SSL-pretrained trunk init)", flush=True)
    result = parent_script.main()
    print("=== STUDY DONE ===", flush=True)
    raise SystemExit(result)
