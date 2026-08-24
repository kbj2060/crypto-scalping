#!/usr/bin/env python3
"""zig075의 direction head 재설계 축(라벨 없이 진행 가능한지) 중 자기지도 사전학습 첫 테스트.
`reference_direction_quality_exit_label_methodology_20260819.md`에서 확인한 "RL(완전
라벨프리)은 이미 소진, 자기지도 사전학습은 미개척"이라는 결론에 따라, ThreeHeadTabM의
공유 trunk(input_scale/input_bias/in_proj/blocks/expert_scale/norms)를 masked-feature-
reconstruction(VIME 스타일 tabular SSL, "Denoised Labels for Financial Time-Series Data
via Self-Supervised Learning" arXiv:2112.10139 계열)으로 먼저 사전학습한다.

방법: 각 bar의 표준화된 102피쳐 중 30%를 랜덤으로 0(=피쳐평균)으로 마스킹, trunk로 인코딩한
뒤(k=8 앙상블 멤버 평균pool) 원래 값을 MSE로 복원 -- 마스킹된 위치에서만 loss 계산(전체
identity mapping을 배우지 못하게). 데이터는 direction/quality/exit 라벨을 전혀 쓰지 않고
zig075 학습에 실제 쓰이는 것과 정확히 같은 TRAIN기간 전체 bar(zigzag 라벨 유무 무관, 모든 bar)
-- 데이터준비 중복구현 대신 _fit_expert_omega4를 첫 호출에서 가로채 raw x_dir DataFrame만
캡처하고 즉시 중단(실제 3-head 학습은 전혀 안 돌림, 데이터 준비 로직만 100% 재사용).

trunk 학습 하이퍼파라미터(lr/wd)는 trial12 우승값(optuna_eth_ilias1_zig075_lr_wd_
focalgamma_20260819.py)을 그대로 재사용 -- pretraining objective는 다르지만 이 아키텍처에
대해 검증된 유일한 lr/wd 기준점이므로 fine-tuning 단계와의 일관성을 위해 채택. 첫 테스트는
단일시드(260620, trial12 baseline과 동일)로 저렴하게 확인 후 신호 있으면 N>=5 확장."""
from __future__ import annotations

import copy
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
from torch.utils.data import DataLoader, TensorDataset  # noqa: E402

import train_eval_omega4_3head_parent72_eth_canonicaldata_pinned102_20260818 as canon  # noqa: E402
import train_eval_omega1_2_tabm_3head_20260603 as parent  # noqa: E402

parent_script = canon.parent_script

SEED = 260620
MASK_PROB = 0.30
LR = 9.98e-4
WEIGHT_DECAY = 1.32e-4
EPOCHS_CAP = 40
PATIENCE = 10
OUT_PATH = ROOT / "tmp/causal_regen_20260516/ssl_zig075_trunk_pretrain_20260819/trunk_state.pt"


class _StopAfterCapture(Exception):
    pass


def _capture_x_dir() -> "np.ndarray":
    captured: dict[str, object] = {}

    def _capture_and_stop(x_dir, *args, expert_idx, **kwargs):
        if expert_idx == 0 and "x_dir" not in captured:
            captured["x_dir"] = x_dir.copy()
        raise _StopAfterCapture()

    orig_fit = parent_script._fit_expert_omega4
    parent_script._fit_expert_omega4 = _capture_and_stop
    base_args = [
        "--direction-label-dir", str(ROOT / "tmp/causal_regen_20260516/omega_current_only_all_label_candidate_parent_screen_20260629/label_contracts/zigzag_action_labels_20260531"),
        "--quality-mode", "same_as_direction",
        "--exit-label-mode", "entry_label_terminal_giveback",
        "--max-exit-samples", "30000",
        "--epochs", "1",
        "--quality-thresholds", "0.80",
        "--device", "cpu",
        "--seed", str(SEED),
        "--out-suffix", "ssl_pretrain_datacapture_scratch",
    ]
    sys.argv = [sys.argv[0], *base_args]
    try:
        parent_script.main()
    except _StopAfterCapture:
        pass
    finally:
        parent_script._fit_expert_omega4 = orig_fit

    if "x_dir" not in captured:
        raise RuntimeError("failed to capture x_dir -- _fit_expert_omega4 never called with expert_idx=0")
    x_dir_raw = captured["x_dir"]
    x_np, scaler = parent._standardize_fit(x_dir_raw)
    return x_np


class ReconTrunk(nn.Module):
    def __init__(self, n_features: int, cfg=parent.CFG) -> None:
        super().__init__()
        self.backbone = parent.ThreeHeadTabM(n_features, cfg=cfg)
        self.recon_head = nn.Linear(int(cfg.hidden), n_features)

    def forward(self, x_masked: torch.Tensor) -> torch.Tensor:
        h = self.backbone.encode(x_masked)  # (batch, k, hidden)
        h_pooled = h.mean(dim=1)  # (batch, hidden)
        return self.recon_head(h_pooled)  # (batch, n_features)


def main() -> int:
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    device = torch.device("cpu")

    print("stage=capture_data", flush=True)
    t0 = time.time()
    x_np = _capture_x_dir()
    print(f"captured x_dir: shape={x_np.shape} elapsed={time.time() - t0:.1f}s", flush=True)

    n = x_np.shape[0]
    n_features = x_np.shape[1]
    split = int(n * 0.85)
    perm = np.random.RandomState(SEED).permutation(n)
    train_idx, val_idx = perm[:split], perm[split:]

    x_all = torch.from_numpy(x_np)
    ds_train = TensorDataset(x_all[train_idx])
    ds_val = TensorDataset(x_all[val_idx])
    dl_train = DataLoader(ds_train, batch_size=2048, shuffle=True)
    dl_val = DataLoader(ds_val, batch_size=4096, shuffle=False)

    model = ReconTrunk(n_features).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    def _masked_mse(xb: torch.Tensor) -> torch.Tensor:
        mask = (torch.rand_like(xb) < MASK_PROB).float()
        x_masked = xb * (1.0 - mask)
        recon = model(x_masked)
        se = (recon - xb) ** 2 * mask
        return se.sum() / mask.sum().clamp(min=1.0)

    best_val = float("inf")
    best_state = None
    stale = 0
    for epoch in range(1, EPOCHS_CAP + 1):
        model.train()
        train_losses = []
        for (xb,) in dl_train:
            xb = xb.to(device)
            loss = _masked_mse(xb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            train_losses.append(float(loss.item()))
        model.eval()
        val_losses = []
        with torch.no_grad():
            for (xb,) in dl_val:
                xb = xb.to(device)
                val_losses.append(float(_masked_mse(xb).item()))
        train_mean = sum(train_losses) / len(train_losses)
        val_mean = sum(val_losses) / len(val_losses)
        marker = ""
        if val_mean < best_val - 1e-6:
            best_val = val_mean
            best_state = copy.deepcopy(model.backbone.state_dict())
            stale = 0
            marker = " *BEST*"
        else:
            stale += 1
        print(f"epoch={epoch:02d} train_recon_mse={train_mean:.4f} val_recon_mse={val_mean:.4f} stale={stale}/{PATIENCE}{marker}", flush=True)
        if stale >= PATIENCE:
            break

    assert best_state is not None
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"trunk_state_dict": best_state, "best_val_recon_mse": best_val, "n_features": n_features,
                "seed": SEED, "mask_prob": MASK_PROB, "lr": LR, "weight_decay": WEIGHT_DECAY}, OUT_PATH)
    print(f"best_val_recon_mse={best_val:.4f}", flush=True)
    print(f"saved trunk to {OUT_PATH}", flush=True)
    print("=== STUDY DONE ===", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
