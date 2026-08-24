#!/usr/bin/env python3
"""사용자가 처음 제안한 4개 기법 중 딥앙상블(이미 구현+2배치 검증완료)을 뺀 나머지 3개를
실제로 구현한다: (1)하이퍼파라미터/정규화 재조정, (2)ASWA류 가중치평균화, (3)배깅.

`train_eval_omega4_3head_parent72_loose_entry_quality_20260620.py::_fit_expert_omega4`
(line 423-589)를 거의 그대로 복제하고, 각 변형이 실제로 건드리는 지점만 최소 수정했다
(공유모듈 소스 자체는 안 건드림 -- 이번 세션 내내 쓴 monkeypatch 패턴 그대로). 3개 변형을
하나의 함수로 묶은 이유는 셋 다 원본 로직의 95%를 그대로 공유해서 별도 파일 3개로 쪼개면
같은 100줄을 세 번 복붙하게 되기 때문 -- `variant` 인자로 분기한다.

- **hp**: lr 2e-3->2e-4(레포 기존 학습체크리스트[[feedback_modern_dl_training_checklist]]
  근거 -- 임의 값 아님), weight_decay 2e-4->1e-3(5배, "안정성 방향" 정규화 강화), 첫 에폭
  동안 선형 warmup(0->목표lr). 나머지(patience/batch_size/k 등)는 원본과 동일 -- 이 변형만의
  순효과를 보려는 것이지 다른 축과 섞을 이유가 없다.
- **aswa**: ThreeHeadTabM은 LayerNorm만 쓰고 BatchNorm이 없어(확인함) SWA 특유의 "평균 후
  BN재보정" 문제가 없다. burn-in(epoch>=2 -- epoch30 테스트에서 실측된 수렴 시점 근방) 이후
  매 에폭 체크포인트를 전부 누적평균("aggressive" 취지 -- 드문드문이 아니라 burn-in 이후
  전부). 최종 가중치는 이 누적평균이지 best-validation-loss 단일 체크포인트가 아니다.
  lr/weight_decay/patience는 원본과 동일(격리).
- **bag**: `train_idx`를 복원추출(bootstrap resample, 같은 크기)로 바꿔친다. `val_idx`
  (조기종료 기준)는 안 건드림 -- 정직한 early stopping 유지가 배깅의 전제(각 모델이 자기
  bootstrap 표본에 과적합해도 실제 홀드아웃으로 멈춘다). exit_head용 `exit_train_idx`는
  손대지 않음(이번 조사의 초점은 direction 서브태스크 -- 범위 확대 안 함). lr/weight_decay는
  원본과 동일.

세 변형 다 seed에 따른 결과(개별 조건부 방향정확도, 5시드 앙상블 시 안정성)를 원본 스크리닝
배치·1차/2차 딥앙상블과 직접 비교하는 게 목적이다."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.utils.class_weight import compute_sample_weight
from torch.utils.data import DataLoader, TensorDataset

import train_eval_omega4_3head_parent72_loose_entry_quality_20260620 as parent_script
import train_omega1_regime3_expert_direction_head_volpca_20260602 as hard

parent = parent_script.parent  # train_eval_omega1_2_tabm_3head_20260603 (CFG/ThreeHeadTabM 등)

ASWA_BURNIN_EPOCH = 2


def fit_expert_omega4_variant(
    x_dir: pd.DataFrame, y_dir: np.ndarray, y_qual: np.ndarray, route_frame: pd.DataFrame,
    x_exit: pd.DataFrame, y_exit: np.ndarray, exit_route_frame: pd.DataFrame,
    *, expert_idx: int, seed: int, epochs: int, device: torch.device, model_path: Path,
    direction_class_weights: dict[int, float], quality_class_weights: dict[int, float],
    direction_focal_gamma: float = 0.0, hard_regime_filter: bool = False,
    variant: str = "hp",
) -> dict[str, Any]:
    if variant not in ("hp", "aswa", "bag"):
        raise ValueError(f"unknown variant: {variant}")
    torch.manual_seed(int(seed) + int(expert_idx))
    np.random.seed(int(seed) + int(expert_idx))
    model_path.parent.mkdir(parents=True, exist_ok=True)
    x_all = pd.concat([x_dir, x_exit], ignore_index=True)
    _x_np, scaler = parent._standardize_fit(x_all)
    x_dir_np = parent._standardize_apply(x_dir, scaler)
    x_exit_np = parent._standardize_apply(x_exit, scaler)
    y_dir_np = np.asarray(y_dir, dtype=np.int64)
    y_qual_np = np.asarray(y_qual, dtype=np.int64)
    y_exit_np = np.asarray(y_exit, dtype=np.int64)
    route_probs = parent._route_probs(route_frame)
    exit_route_probs = parent._route_probs(exit_route_frame)
    if hard_regime_filter:
        route_w = (route_probs.argmax(axis=1) == int(expert_idx)).astype(np.float32)
        exit_w = (exit_route_probs.argmax(axis=1) == int(expert_idx)).astype(np.float32)
    else:
        route_w = route_probs[:, int(expert_idx)].astype(np.float32)
        exit_w = exit_route_probs[:, int(expert_idx)].astype(np.float32)
    dir_w = compute_sample_weight(class_weight="balanced", y=y_dir_np).astype(np.float32) * route_w
    qual_w = compute_sample_weight(class_weight="balanced", y=y_qual_np).astype(np.float32) * route_w
    dir_w *= np.asarray([float(direction_class_weights.get(int(y), 1.0)) for y in y_dir_np], dtype=np.float32)
    qual_w *= np.asarray([float(quality_class_weights.get(int(y), 1.0)) for y in y_qual_np], dtype=np.float32)
    ex_w = compute_sample_weight(class_weight="balanced", y=y_exit_np).astype(np.float32) * exit_w
    if float(dir_w.sum()) <= 0.0 or float(qual_w.sum()) <= 0.0 or float(ex_w.sum()) <= 0.0:
        raise RuntimeError(f"{hard.EXPERT_NAMES[expert_idx]} invalid Omega4 sample weights")

    n = len(y_dir_np)
    split = max(int(n * 0.85), min(n - 1, 512))
    train_idx = np.arange(split)
    if variant == "bag":
        bag_rng = np.random.default_rng(int(seed) + int(expert_idx) + 999)
        train_idx = bag_rng.choice(train_idx, size=len(train_idx), replace=True)
    val_idx = np.arange(split, n)
    exit_n = len(y_exit_np)
    exit_split = max(int(exit_n * 0.85), min(exit_n - 1, 256))
    exit_train_idx = np.arange(exit_split)
    exit_val_idx = np.arange(exit_split, exit_n)

    lr = float(parent.CFG.lr)
    weight_decay = float(parent.CFG.weight_decay)
    if variant == "hp":
        lr = 2.0e-4
        weight_decay = 1.0e-3

    model = parent.ThreeHeadTabM(x_dir_np.shape[1], cfg=parent.CFG).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    ds_dir = TensorDataset(
        torch.from_numpy(x_dir_np[train_idx]), torch.from_numpy(y_dir_np[train_idx]),
        torch.from_numpy(y_qual_np[train_idx]), torch.from_numpy(dir_w[train_idx]), torch.from_numpy(qual_w[train_idx]),
    )
    ds_exit = TensorDataset(torch.from_numpy(x_exit_np[exit_train_idx]), torch.from_numpy(y_exit_np[exit_train_idx]), torch.from_numpy(ex_w[exit_train_idx]))
    dl_dir = DataLoader(ds_dir, batch_size=int(parent.CFG.batch_size), shuffle=True, drop_last=False)
    dl_exit = DataLoader(ds_exit, batch_size=int(parent.CFG.batch_size), shuffle=True, drop_last=False)

    warmup_steps = max(1, len(dl_dir)) if variant == "hp" else 0
    global_step = 0
    best_state: dict[str, torch.Tensor] | None = None
    best_loss = float("inf")
    stale = 0
    last_epoch = 0
    aswa_sum: dict[str, torch.Tensor] | None = None
    aswa_count = 0

    for epoch in range(int(epochs)):
        last_epoch = epoch + 1
        model.train()
        exit_iter = iter(dl_exit)
        for xb, yb, yqb, wb, qwb in dl_dir:
            try:
                xe, ye, we = next(exit_iter)
            except StopIteration:
                exit_iter = iter(dl_exit)
                xe, ye, we = next(exit_iter)
            xb = xb.to(device, non_blocking=True); yb = yb.to(device, non_blocking=True)
            yqb = yqb.to(device, non_blocking=True); wb = wb.to(device, non_blocking=True); qwb = qwb.to(device, non_blocking=True)
            xe = xe.to(device, non_blocking=True); ye = ye.to(device, non_blocking=True); we = we.to(device, non_blocking=True)
            if warmup_steps > 0 and global_step < warmup_steps:
                scale = (global_step + 1) / warmup_steps
                for pg in opt.param_groups:
                    pg["lr"] = lr * scale
            global_step += 1
            out_dir = model(xb)
            loss_dir_k = torch.nn.functional.cross_entropy(
                out_dir["direction"].reshape(-1, 3), yb[:, None].expand(-1, int(parent.CFG.k)).reshape(-1), reduction="none",
            ).reshape(-1, int(parent.CFG.k))
            loss_dir_k = parent_script._focal_modulate(loss_dir_k, direction_focal_gamma)
            loss_qual_k = torch.nn.functional.cross_entropy(
                out_dir["quality"].reshape(-1, 3), yqb[:, None].expand(-1, int(parent.CFG.k)).reshape(-1), reduction="none",
            ).reshape(-1, int(parent.CFG.k))
            out_exit = model(xe)
            loss_exit_k = torch.nn.functional.cross_entropy(
                out_exit["exit"].reshape(-1, 2), ye[:, None].expand(-1, int(parent.CFG.k)).reshape(-1), reduction="none",
            ).reshape(-1, int(parent.CFG.k))
            loss_dir = (loss_dir_k.mean(dim=1) * wb).sum() / torch.clamp(wb.sum(), min=1.0)
            loss_qual = (loss_qual_k.mean(dim=1) * qwb).sum() / torch.clamp(qwb.sum(), min=1.0)
            loss_exit = (loss_exit_k.mean(dim=1) * we).sum() / torch.clamp(we.sum(), min=1.0)
            loss = loss_dir + float(parent.CFG.quality_loss_weight) * loss_qual + float(parent.CFG.exit_loss_weight) * loss_exit
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            opt.step()
        model.eval()
        with torch.no_grad():
            vx = torch.from_numpy(x_dir_np[val_idx]).to(device); vy = torch.from_numpy(y_dir_np[val_idx]).to(device)
            vqy = torch.from_numpy(y_qual_np[val_idx]).to(device); vw = torch.from_numpy(dir_w[val_idx]).to(device); vqw = torch.from_numpy(qual_w[val_idx]).to(device)
            ve = torch.from_numpy(x_exit_np[exit_val_idx]).to(device); vey = torch.from_numpy(y_exit_np[exit_val_idx]).to(device); vew = torch.from_numpy(ex_w[exit_val_idx]).to(device)
            vo = model(vx); veo = model(ve)
            vdir = torch.nn.functional.cross_entropy(vo["direction"].reshape(-1, 3), vy[:, None].expand(-1, int(parent.CFG.k)).reshape(-1), reduction="none").reshape(-1, int(parent.CFG.k))
            vdir = parent_script._focal_modulate(vdir, direction_focal_gamma)
            vqual = torch.nn.functional.cross_entropy(vo["quality"].reshape(-1, 3), vqy[:, None].expand(-1, int(parent.CFG.k)).reshape(-1), reduction="none").reshape(-1, int(parent.CFG.k))
            vex = torch.nn.functional.cross_entropy(veo["exit"].reshape(-1, 2), vey[:, None].expand(-1, int(parent.CFG.k)).reshape(-1), reduction="none").reshape(-1, int(parent.CFG.k))
            vloss = float((
                ((vdir.mean(dim=1) * vw).sum() / torch.clamp(vw.sum(), min=1.0))
                + float(parent.CFG.quality_loss_weight) * ((vqual.mean(dim=1) * vqw).sum() / torch.clamp(vqw.sum(), min=1.0))
                + float(parent.CFG.exit_loss_weight) * ((vex.mean(dim=1) * vew).sum() / torch.clamp(vew.sum(), min=1.0))
            ).detach().cpu())
        if variant == "aswa" and epoch >= ASWA_BURNIN_EPOCH:
            sd = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            if aswa_sum is None:
                aswa_sum = sd
            else:
                for k in aswa_sum:
                    aswa_sum[k] = aswa_sum[k] + sd[k]
            aswa_count += 1
        if vloss + 1.0e-6 < best_loss:
            best_loss = vloss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= int(parent.CFG.patience):
                break

    used_aswa = False
    if variant == "aswa" and aswa_sum is not None and aswa_count > 0:
        model.load_state_dict({k: v / aswa_count for k, v in aswa_sum.items()})
        used_aswa = True
    elif best_state is not None:
        model.load_state_dict(best_state)

    payload = {
        "model_id": parent_script.MODEL_ID,
        "expert": hard.EXPERT_NAMES[int(expert_idx)],
        "config": parent.CFG.__dict__,
        "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
        "scaler": scaler,
        "n_features": int(x_dir_np.shape[1]),
        "best_validation_loss": float(best_loss),
        "epochs_ran": int(last_epoch),
        "input_columns": list(x_dir.columns),
        "quality_target": "omega4_quality_action",
        "direction_class_weights": {str(k): float(v) for k, v in direction_class_weights.items()},
        "quality_class_weights": {str(k): float(v) for k, v in quality_class_weights.items()},
        "direction_focal_gamma": float(direction_focal_gamma),
        "hard_regime_filter": bool(hard_regime_filter),
        "variant": variant,
        "lr_used": lr,
        "weight_decay_used": weight_decay,
        "aswa_averaged_checkpoints": int(aswa_count) if variant == "aswa" else 0,
        "used_aswa_weights": used_aswa,
        "bag_train_idx_bootstrap": variant == "bag",
    }
    torch.save(payload, model_path)
    return payload
