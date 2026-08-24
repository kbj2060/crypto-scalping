#!/usr/bin/env python3
"""`eth_tabm_label_logic_5way_seed_variant_ilias_anchored_canonical_20260822.py`의 "제대로 된"
버전 -- 사용자 지시(2026-08-22): "레짐대로 데이터를 분류해서 3 expert에 데이터를 학습시켜줘.
LR 스케쥴도 추가하고 lr을 좀 더 작은 수로 시작해줘."

## 이번에 실제로 바뀌는 것

**⚠️ 사전 조사에서 발견**: 기존 러너의 `parent_script._route_probs = _uniform_route_probs`
패치는 **처음부터 아무 효과가 없었다** -- `_fit_expert_omega4`(이 파일이 실제로 override하는
함수)는 내부에서 `parent._route_probs(...)`(TabM 하위모듈 `train_eval_omega1_2_tabm_3head_
20260603`의 속성)를 부르는데, 옛 패치는 `parent_script`(omega4 모듈 자체)에 걸려서 별개
객체였다(실측 확인: `parent_script._route_probs is parent_script.parent._route_probs` ==
False). 즉 **bull expert는 계속 진짜 레짐가중치로 학습돼왔다** -- "uniform regime weight"
로그 문구 자체가 틀렸었다. 실제로 작동한 건 "bear/chop이 bull 가중치를 복사만 하고 독립학습
안 함" 쪽뿐이다. 그래서 이 스크립트가 실제로 바꾸는 건:

1. **bull-trains/bear-chop-copies shortcut 제거**: `_fit_expert_omega4`를 override하지 않고
   원본 `main()`의 3-expert 루프(`for idx, expert in enumerate(hard.EXPERT_NAMES)`)를 그대로
   돌린다 -- 이미 실제 레짐가중치가 정상 연결돼 있으므로, 이것만으로 3개 expert가 각자의
   레짐가중치로 독립 학습된다.
2. **LR 스케줄 추가**: 원본 `_fit_expert_omega4`(`train_eval_omega4_3head_parent72_loose_entry_
   quality_20260620.py:423-589`)에는 스케줄이 전혀 없다(고정 lr). 공유 파일은 무수정 원칙이라,
   그 함수 전체를 로컬로 복제한 `_fit_expert_omega4_scheduled`에 `CosineAnnealingLR`만 추가해
   `parent_script._fit_expert_omega4`에 override한다(기존 `_fit_expert_omega4_unified`와 동일한
   패턴 -- 이 함수는 omega4 모듈 자신의 전역이라 bare-name 호출이 정상적으로 패치를 본다,
   `_route_probs`와 달리 여기선 패치가 실제로 작동함을 확인함).
3. **lr을 더 작은 값으로**: `parent.CFG`는 frozen dataclass라 필드를 직접 못 바꾸므로
   `dataclasses.replace(parent.CFG, lr=2e-4)`로 새 인스턴스를 만들어 `parent.CFG` 전체를
   교체(기존 2e-3 대비 1/10). 2e-4는 이 저장소가 LOB/DC154 축에서 N≥5 시드로 이미 검증한
   값([[feedback_modern_dl_training_checklist]])을 그대로 재사용 -- 임의 선택 아님.
4. **epoch 예산 확대**: `--epochs 40`(LR-isolation cheap_gate 실험이 쓴 예산과 동일 -- 낮은
   lr+실제 patience=8이 작동할 여지를 준다).

## 그대로 유지되는 것

zigzag/h48qual/cusum 라벨 소스, TRAIN(2024-01~2026-03-31)/VAL(2026Q2)/OOS(2026-07-01~데이터상
최근일) split, 154피쳐(→ 최신 레짐분류기 states=24/sticky=0.90 오버레이 포함) 전부
`eth_directional_change_tabm_training_ilias_anchored_canonical_20260822.py`(무수정 재사용)
그대로."""
from __future__ import annotations

import argparse
import dataclasses
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from sklearn.utils.class_weight import compute_sample_weight  # noqa: E402
from torch.utils.data import DataLoader, TensorDataset  # noqa: E402

import eth_directional_change_tabm_training_ilias_anchored_canonical_20260822 as feat154  # noqa: E402
import train_omega1_regime3_expert_direction_head_volpca_20260602 as hard  # noqa: E402

parent_script = feat154.parent_script
parent = parent_script.parent   # train_eval_omega1_2_tabm_3head_20260603 -- ThreeHeadTabM/CFG/_route_probs 소유

NEW_LR = 2e-4   # 기존 2e-3의 1/10, [[feedback_modern_dl_training_checklist]] N>=5검증값 재사용
parent.CFG = dataclasses.replace(parent.CFG, lr=NEW_LR)
print(f"[regime_expert_lrschedule] parent.CFG.lr={parent.CFG.lr} (기존 2e-3 -> {NEW_LR})", flush=True)


def _fit_expert_omega4_scheduled(
    x_dir, y_dir, y_qual, route_frame, x_exit, y_exit, exit_route_frame, *,
    expert_idx, seed, epochs, device, model_path,
    direction_class_weights, quality_class_weights,
    direction_focal_gamma=0.0, hard_regime_filter=False,
):
    """`_fit_expert_omega4`(train_eval_omega4_3head_parent72_loose_entry_quality_20260620.py:423)
    전체 복제 + CosineAnnealingLR 스케줄만 추가(공유 파일 무수정, 로컬 override)."""
    torch.manual_seed(int(seed) + int(expert_idx))
    np.random.seed(int(seed) + int(expert_idx))
    model_path.parent.mkdir(parents=True, exist_ok=True)
    import pandas as pd
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
    val_idx = np.arange(split, n)
    exit_n = len(y_exit_np)
    exit_split = max(int(exit_n * 0.85), min(exit_n - 1, 256))
    exit_train_idx = np.arange(exit_split)
    exit_val_idx = np.arange(exit_split, exit_n)

    model = parent.ThreeHeadTabM(x_dir_np.shape[1], cfg=parent.CFG).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=float(parent.CFG.lr), weight_decay=float(parent.CFG.weight_decay))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=int(epochs), eta_min=float(parent.CFG.lr) * 1e-2)
    ds_dir = TensorDataset(
        torch.from_numpy(x_dir_np[train_idx]), torch.from_numpy(y_dir_np[train_idx]),
        torch.from_numpy(y_qual_np[train_idx]), torch.from_numpy(dir_w[train_idx]), torch.from_numpy(qual_w[train_idx]),
    )
    ds_exit = TensorDataset(torch.from_numpy(x_exit_np[exit_train_idx]), torch.from_numpy(y_exit_np[exit_train_idx]), torch.from_numpy(ex_w[exit_train_idx]))
    dl_dir = DataLoader(ds_dir, batch_size=int(parent.CFG.batch_size), shuffle=True, drop_last=False)
    dl_exit = DataLoader(ds_exit, batch_size=int(parent.CFG.batch_size), shuffle=True, drop_last=False)
    best_state = None
    best_loss = float("inf")
    stale = 0
    last_epoch = 0
    focal_modulate = parent_script._focal_modulate
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
            xb, yb, yqb, wb, qwb = (t.to(device, non_blocking=True) for t in (xb, yb, yqb, wb, qwb))
            xe, ye, we = (t.to(device, non_blocking=True) for t in (xe, ye, we))
            out_dir = model(xb)
            loss_dir_k = torch.nn.functional.cross_entropy(
                out_dir["direction"].reshape(-1, 3), yb[:, None].expand(-1, int(parent.CFG.k)).reshape(-1), reduction="none",
            ).reshape(-1, int(parent.CFG.k))
            loss_dir_k = focal_modulate(loss_dir_k, direction_focal_gamma)
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
        sched.step()   # ⚠️ 원본에 없던 부분 -- 이번 수정의 핵심
        model.eval()
        with torch.no_grad():
            vx = torch.from_numpy(x_dir_np[val_idx]).to(device)
            vy = torch.from_numpy(y_dir_np[val_idx]).to(device)
            vqy = torch.from_numpy(y_qual_np[val_idx]).to(device)
            vw = torch.from_numpy(dir_w[val_idx]).to(device)
            vqw = torch.from_numpy(qual_w[val_idx]).to(device)
            ve = torch.from_numpy(x_exit_np[exit_val_idx]).to(device)
            vey = torch.from_numpy(y_exit_np[exit_val_idx]).to(device)
            vew = torch.from_numpy(ex_w[exit_val_idx]).to(device)
            vo = model(vx)
            veo = model(ve)
            vdir = torch.nn.functional.cross_entropy(vo["direction"].reshape(-1, 3), vy[:, None].expand(-1, int(parent.CFG.k)).reshape(-1), reduction="none").reshape(-1, int(parent.CFG.k))
            vdir = focal_modulate(vdir, direction_focal_gamma)
            vqual = torch.nn.functional.cross_entropy(vo["quality"].reshape(-1, 3), vqy[:, None].expand(-1, int(parent.CFG.k)).reshape(-1), reduction="none").reshape(-1, int(parent.CFG.k))
            vex = torch.nn.functional.cross_entropy(veo["exit"].reshape(-1, 2), vey[:, None].expand(-1, int(parent.CFG.k)).reshape(-1), reduction="none").reshape(-1, int(parent.CFG.k))
            vloss = float(
                (((vdir.mean(dim=1) * vw).sum() / torch.clamp(vw.sum(), min=1.0))
                 + float(parent.CFG.quality_loss_weight) * ((vqual.mean(dim=1) * vqw).sum() / torch.clamp(vqw.sum(), min=1.0))
                 + float(parent.CFG.exit_loss_weight) * ((vex.mean(dim=1) * vew).sum() / torch.clamp(vew.sum(), min=1.0)))
                .detach().cpu()
            )
        print(f"    expert={hard.EXPERT_NAMES[expert_idx]} epoch={epoch:3d} "
              f"lr={sched.get_last_lr()[0]:.2e} vloss={vloss:.4f} stale={stale}", flush=True)
        if vloss + 1.0e-6 < best_loss:
            best_loss = vloss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= int(parent.CFG.patience):
                break
    if best_state is not None:
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
    }
    torch.save(payload, model_path)
    return payload


parent_script._fit_expert_omega4 = _fit_expert_omega4_scheduled
# ⚠️ _route_probs 패치, _fit_expert_omega4_unified(bull-copy shortcut)는 의도적으로 적용 안 함
# -- 위 docstring 참고, 3 expert 전부 진짜 레짐가중치로 독립학습되게 두는 게 이번 목적.

ZIGZAG_ACTION_DIR = ROOT / "tmp/ilias_labellogic_recheck_20260821/direction_labels_2024_2026q2merged/zigzag"
H48_CONSERVATIVE_QUALITY_DIR = ROOT / "tmp/ilias_labellogic_recheck_20260821/quality_labels_2024_2026q2merged/h48_conservative"
CUSUM_DENSE_LABEL_DIR = ROOT / "tmp/ilias_labellogic_recheck_20260821/direction_labels_2024_2026q2merged/cusum"

LABEL_CONFIGS = {
    "zigzag": {"direction_label_dir": str(ZIGZAG_ACTION_DIR), "quality_mode": "same_as_direction", "quality_label_dir": None},
    "h48qual": {"direction_label_dir": str(ZIGZAG_ACTION_DIR), "quality_mode": "quality_label_action", "quality_label_dir": str(H48_CONSERVATIVE_QUALITY_DIR)},
    "cusum": {"direction_label_dir": str(CUSUM_DENSE_LABEL_DIR), "quality_mode": "same_as_direction", "quality_label_dir": None},
}

COMMON_ARGS = [
    "--exit-label-mode", "independent_entry_hold_offsets",
    "--epochs", "40",   # LR-isolation cheap_gate 실험과 동일 예산(patience=8 여유)
    "--device", "auto",
]

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", required=True, choices=sorted(LABEL_CONFIGS.keys()))
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--out-suffix-tag", default="")   # 비우면 기존 baseline out_dir과 동일(하위호환) --
                                                        # 값을 주면 별도 out_dir(예: HP서치 재검증분리)
    known, extra = ap.parse_known_args()   # extra: 미인식 플래그(예: --quality-min-edge)는 그대로
                                            # parent_script.main()으로 전달 -- 하이퍼파라미터 서치용

    cfg = LABEL_CONFIGS[known.label]
    args = ["--direction-label-dir", cfg["direction_label_dir"], "--quality-mode", cfg["quality_mode"]]
    if cfg["quality_label_dir"]:
        args += ["--quality-label-dir", cfg["quality_label_dir"]]
    args += COMMON_ARGS
    args += extra

    tag = f"_{known.out_suffix_tag}" if known.out_suffix_tag else ""
    out_suffix = f"label5way_{known.label}_154feat_regime_expert_lrschedule_seed{known.seed}_20260822{tag}"
    sys.argv = [sys.argv[0], *args, "--seed", str(known.seed), "--out-suffix", out_suffix]
    print(f"stage=start label={known.label} seed={known.seed} out_suffix={out_suffix}", flush=True)
    t0 = time.time()
    result = parent_script.main()
    print(f"stage=done label={known.label} seed={known.seed} elapsed={time.time() - t0:.1f}s", flush=True)
    raise SystemExit(result)
