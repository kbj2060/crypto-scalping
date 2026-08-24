#!/usr/bin/env python3
"""zig075 pinned102 parent 재학습의 Optuna HPO -- 사용자 지시(2026-08-19)대로 lr/weight_decay/
direction_focal_gamma 3축만 탐색, patience=10/epochs cap=40은 고정.

설계 근거(연구 기반):
- **Multi-seed objective**: trial마다 단일시드가 아니라 SEEDS(6개, 이 세션 내내 쓴 고정 랜덤시드)
  전체를 순차 학습해 3-expert(bull/bear/chop) best_validation_loss 평균을 objective로 삼는다.
  이게 이 프로젝트의 과거 Optuna 시도(C2, loss weight 탐색)가 실패한 근본원인 -- "단일시드로
  탐색 후 나중에 N=5로 재검증하니 노이즈였음"(eth_odyssey4_layer_improvement_proposal_20260816)
  -- 을 설계 단계에서 원천 차단한다.
- **MedianPruner(시드를 fidelity 축으로)**: 각 seed 완료 시점마다 지금까지의 평균을 Optuna에
  report, 동시점 다른 trial들의 median보다 뚜렷이 못하면 조기중단 -- 나쁜 조합에 6시드 전부
  낭비하지 않는다. n_warmup_steps=2로 최소 3시드는 채운 뒤에만 가지치기(1-2시드 노이즈로
  성급하게 자르지 않음). HyperbandPruner 대신 MedianPruner를 쓰는 이유: fidelity 축이 세밀한
  epoch이 아니라 6단계뿐인 시드 카운트라 Hyperband의 다단 bracket 구조가 과할 수 있음(문헌
  리뷰: "MedianPruner for a simpler baseline that works well in practice").
- **Prior-informed seeding (PriorBand, arXiv:2306.12370 아이디어, `enqueue_trial`로 stock
  Optuna만으로 구현)**: 이 프로젝트가 이미 검증한 현재 기본값(lr=2e-3/wd=2e-4/focal=0)과 lr=2e-4
  단독 실험에서 나온 favorable 지점을 첫 trial들로 명시 주입 -- 맨땅 탐색이 아니라 이미 아는
  좋은 지점 주변부터 시작.
- **탐색공간에서 이미 CLOSED된 레버 제외**: GCE/AdaBelief/cosine schedule/Prechelt criterion/
  k=32/batch_size 독립축 -- 전부 feedback_modern_dl_training_checklist가 이 정확한 TabM계열에서
  N>=5시드로 이미 패배 확인. patience/epoch cap도 사용자 지시로 이번 탐색 범위 밖(고정값).

우승 trial은 이 스크립트가 이미 쓴 SEEDS 자체에 대한 objective 평균일 뿐 -- 최종 채택 전
반드시 여기 안 쓰인 새 N>=5 랜덤시드로 별도 재검증 필요(다음 단계, 이 스크립트 범위 밖)."""
from __future__ import annotations

import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import optuna  # noqa: E402
import torch  # noqa: E402
from sklearn.utils.class_weight import compute_sample_weight  # noqa: E402
from torch.utils.data import DataLoader, TensorDataset  # noqa: E402

import train_eval_omega4_3head_parent72_eth_canonicaldata_pinned102_20260818 as canon  # noqa: E402
import train_eval_omega1_2_tabm_3head_20260603 as omega1_2  # noqa: E402
import train_omega1_regime3_expert_direction_head_volpca_20260602 as hard  # noqa: E402

parent_script = canon.parent_script
_BASE_CFG = omega1_2.CFG  # original ThreeHeadConfig(lr=2e-3, weight_decay=2e-4, patience=8, ...)
_orig_fit_expert_omega4 = parent_script._fit_expert_omega4


def _fit_expert_omega4_epochlogged(
    x_dir: pd.DataFrame, y_dir, y_qual, route_frame: pd.DataFrame,
    x_exit: pd.DataFrame, y_exit, exit_route_frame: pd.DataFrame,
    *, expert_idx: int, seed: int, epochs: int, device: torch.device, model_path: Path,
    direction_class_weights: dict[int, float], quality_class_weights: dict[int, float],
    direction_focal_gamma: float = 0.0, hard_regime_filter: bool = False,
) -> dict[str, Any]:
    """train_eval_omega4_3head_parent72_loose_entry_quality_20260620.py::_fit_expert_omega4의
    로직을 byte-for-byte 그대로 복제 -- 유일한 추가는 매 epoch 끝에 val loss/accuracy를
    출력하는 것뿐(선택/학습 로직은 원본과 완전히 동일). 사용자 지시(2026-08-19)로 다른 세션의
    epoch별 로그와 동일한 가시성을 제공하기 위해 임시로 원본 함수 대신 사용."""
    torch.manual_seed(int(seed) + int(expert_idx))
    np.random.seed(int(seed) + int(expert_idx))
    model_path.parent.mkdir(parents=True, exist_ok=True)
    x_all = pd.concat([x_dir, x_exit], ignore_index=True)
    _x_np, scaler = omega1_2._standardize_fit(x_all)
    x_dir_np = omega1_2._standardize_apply(x_dir, scaler)
    x_exit_np = omega1_2._standardize_apply(x_exit, scaler)
    y_dir_np = np.asarray(y_dir, dtype=np.int64)
    y_qual_np = np.asarray(y_qual, dtype=np.int64)
    y_exit_np = np.asarray(y_exit, dtype=np.int64)
    route_probs = omega1_2._route_probs(route_frame)
    exit_route_probs = omega1_2._route_probs(exit_route_frame)
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

    model = omega1_2.ThreeHeadTabM(x_dir_np.shape[1], cfg=omega1_2.CFG).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=float(omega1_2.CFG.lr), weight_decay=float(omega1_2.CFG.weight_decay))
    ds_dir = TensorDataset(
        torch.from_numpy(x_dir_np[train_idx]), torch.from_numpy(y_dir_np[train_idx]),
        torch.from_numpy(y_qual_np[train_idx]), torch.from_numpy(dir_w[train_idx]), torch.from_numpy(qual_w[train_idx]),
    )
    ds_exit = TensorDataset(torch.from_numpy(x_exit_np[exit_train_idx]), torch.from_numpy(y_exit_np[exit_train_idx]), torch.from_numpy(ex_w[exit_train_idx]))
    dl_dir = DataLoader(ds_dir, batch_size=int(omega1_2.CFG.batch_size), shuffle=True, drop_last=False)
    dl_exit = DataLoader(ds_exit, batch_size=int(omega1_2.CFG.batch_size), shuffle=True, drop_last=False)
    best_state: dict[str, torch.Tensor] | None = None
    best_loss = float("inf")
    stale = 0
    last_epoch = 0
    tag = f"seed={seed} expert={hard.EXPERT_NAMES[expert_idx]}"
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
            loss_dir_k = torch.nn.functional.cross_entropy(out_dir["direction"].reshape(-1, 3), yb[:, None].expand(-1, int(omega1_2.CFG.k)).reshape(-1), reduction="none").reshape(-1, int(omega1_2.CFG.k))
            loss_dir_k = parent_script._focal_modulate(loss_dir_k, direction_focal_gamma)
            loss_qual_k = torch.nn.functional.cross_entropy(out_dir["quality"].reshape(-1, 3), yqb[:, None].expand(-1, int(omega1_2.CFG.k)).reshape(-1), reduction="none").reshape(-1, int(omega1_2.CFG.k))
            out_exit = model(xe)
            loss_exit_k = torch.nn.functional.cross_entropy(out_exit["exit"].reshape(-1, 2), ye[:, None].expand(-1, int(omega1_2.CFG.k)).reshape(-1), reduction="none").reshape(-1, int(omega1_2.CFG.k))
            loss_dir = (loss_dir_k.mean(dim=1) * wb).sum() / torch.clamp(wb.sum(), min=1.0)
            loss_qual = (loss_qual_k.mean(dim=1) * qwb).sum() / torch.clamp(qwb.sum(), min=1.0)
            loss_exit = (loss_exit_k.mean(dim=1) * we).sum() / torch.clamp(we.sum(), min=1.0)
            loss = loss_dir + float(omega1_2.CFG.quality_loss_weight) * loss_qual + float(omega1_2.CFG.exit_loss_weight) * loss_exit
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            opt.step()
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
            vdir = torch.nn.functional.cross_entropy(vo["direction"].reshape(-1, 3), vy[:, None].expand(-1, int(omega1_2.CFG.k)).reshape(-1), reduction="none").reshape(-1, int(omega1_2.CFG.k))
            vdir = parent_script._focal_modulate(vdir, direction_focal_gamma)
            vqual = torch.nn.functional.cross_entropy(vo["quality"].reshape(-1, 3), vqy[:, None].expand(-1, int(omega1_2.CFG.k)).reshape(-1), reduction="none").reshape(-1, int(omega1_2.CFG.k))
            vex = torch.nn.functional.cross_entropy(veo["exit"].reshape(-1, 2), vey[:, None].expand(-1, int(omega1_2.CFG.k)).reshape(-1), reduction="none").reshape(-1, int(omega1_2.CFG.k))
            vloss = float(
                (((vdir.mean(dim=1) * vw).sum() / torch.clamp(vw.sum(), min=1.0))
                 + float(omega1_2.CFG.quality_loss_weight) * ((vqual.mean(dim=1) * vqw).sum() / torch.clamp(vqw.sum(), min=1.0))
                 + float(omega1_2.CFG.exit_loss_weight) * ((vex.mean(dim=1) * vew).sum() / torch.clamp(vew.sum(), min=1.0))).detach().cpu()
            )
            # 추가(로깅 전용, 학습/선택 로직에 영향 없음): direction/quality/exit head accuracy
            dir_acc = float((vo["direction"].argmax(dim=-1) == vy[:, None]).float().mean().detach().cpu())
            qual_acc = float((vo["quality"].argmax(dim=-1) == vqy[:, None]).float().mean().detach().cpu())
            exit_acc = float((veo["exit"].argmax(dim=-1) == vey[:, None]).float().mean().detach().cpu())
        if vloss + 1.0e-6 < best_loss:
            best_loss = vloss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            stale = 0
            marker = "*BEST*"
        else:
            stale += 1
            marker = f"stale={stale}/{int(omega1_2.CFG.patience)}"
        print(f"      [{tag}] epoch={epoch + 1:02d} val_loss={vloss:.4f} dir_acc={dir_acc:.3f} "
              f"qual_acc={qual_acc:.3f} exit_acc={exit_acc:.3f} {marker}", flush=True)
        if stale >= int(omega1_2.CFG.patience):
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    payload = {
        "model_id": parent_script.MODEL_ID,
        "expert": hard.EXPERT_NAMES[int(expert_idx)],
        "config": omega1_2.CFG.__dict__,
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


parent_script._fit_expert_omega4 = _fit_expert_omega4_epochlogged

SEEDS = [260620, 121026, 337153, 390529, 640787, 794920]
FIXED_PATIENCE = 10
FIXED_EPOCHS = 40
STUDY_NAME = "zig075_pinned102_lr_wd_focalgamma_20260819"
FOCAL_GAMMA_MAX = 10.0  # widened 2026-08-19 from 5.0 -- trial6's winner (4.96) hugged the old
                         # upper bound, so this continuation round checks whether the true
                         # optimum lies beyond it before finalizing.
N_TRIALS = 6  # short, targeted continuation round only (not a full fresh 25-trial search)

_BASE_ARGS = [
    "--direction-label-dir", str(ROOT / "tmp/causal_regen_20260516/omega_current_only_all_label_candidate_parent_screen_20260629/label_contracts/zigzag_action_labels_20260531"),
    "--quality-mode", "same_as_direction",
    "--exit-label-mode", "entry_label_terminal_giveback",
    "--max-exit-samples", "30000",
    "--epochs", str(FIXED_EPOCHS),
    "--quality-thresholds", "0.55,0.60,0.65,0.70,0.75,0.80,0.85,0.90,0.95",
    "--device", "cpu",
]


def _run_one_seed(seed: int, lr: float, weight_decay: float, focal_gamma: float, trial_tag: str) -> float:
    omega1_2.CFG = replace(_BASE_CFG, lr=lr, weight_decay=weight_decay, patience=FIXED_PATIENCE)
    out_suffix = f"optuna_zig075_{trial_tag}_seed{seed}"
    sys.argv = [sys.argv[0], *_BASE_ARGS, "--seed", str(seed), "--out-suffix", out_suffix,
                "--direction-focal-gamma", str(focal_gamma)]
    t0 = time.time()
    parent_script.main()
    bundle_path = ROOT / f"tmp/causal_regen_20260516/omega4_3head_parent72_loose_entry_quality_20260620_{out_suffix}/true_3head_tabm_bundle.pt"
    bundle = torch.load(bundle_path, map_location="cpu", weights_only=False)
    losses = [payload["best_validation_loss"] for payload in bundle["models"].values()]
    epochs_ran = [payload["epochs_ran"] for payload in bundle["models"].values()]
    mean_loss = sum(losses) / len(losses)
    print(f"    seed={seed} mean_expert_val_loss={mean_loss:.4f} epochs_ran={epochs_ran} elapsed={time.time() - t0:.1f}s", flush=True)
    return mean_loss


def objective(trial: "optuna.Trial") -> float:
    lr = trial.suggest_float("lr", 1e-5, 1e-2, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-1, log=True)
    focal_gamma = trial.suggest_float("direction_focal_gamma", 0.0, FOCAL_GAMMA_MAX)
    print(f"trial={trial.number} start lr={lr:.2e} weight_decay={weight_decay:.2e} focal_gamma={focal_gamma:.3f}", flush=True)

    seed_losses: list[float] = []
    for step, seed in enumerate(SEEDS):
        loss = _run_one_seed(seed, lr, weight_decay, focal_gamma, trial_tag=f"trial{trial.number}")
        seed_losses.append(loss)
        running_mean = sum(seed_losses) / len(seed_losses)
        trial.report(running_mean, step=step)
        if trial.should_prune():
            print(f"trial={trial.number} PRUNED after {step + 1} seeds running_mean={running_mean:.4f}", flush=True)
            raise optuna.TrialPruned()
    final_mean = sum(seed_losses) / len(seed_losses)
    print(f"trial={trial.number} DONE final_mean={final_mean:.4f} n_seeds={len(seed_losses)}", flush=True)
    return final_mean


def main() -> int:
    sampler = optuna.samplers.TPESampler(seed=20260819, multivariate=True)
    pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=2)
    storage = f"sqlite:///{ROOT}/tmp/causal_regen_20260516/optuna_{STUDY_NAME}.db"
    study = optuna.create_study(direction="minimize", sampler=sampler, pruner=pruner,
                                 study_name=STUDY_NAME, storage=storage, load_if_exists=True)

    # PriorBand-style prior seeding (stock Optuna enqueue_trial, no extra dependency):
    # 1) exact current canonical default, 2) already-favored lr=2e-4 isolation-test finding,
    # 3) same + literature-standard focal gamma=2.0 (Lin et al. RetinaNet default).
    if len(study.trials) == 0:
        study.enqueue_trial({"lr": float(_BASE_CFG.lr), "weight_decay": float(_BASE_CFG.weight_decay), "direction_focal_gamma": 0.0})
        study.enqueue_trial({"lr": 2.0e-4, "weight_decay": float(_BASE_CFG.weight_decay), "direction_focal_gamma": 0.0})
        study.enqueue_trial({"lr": 2.0e-4, "weight_decay": float(_BASE_CFG.weight_decay), "direction_focal_gamma": 2.0})

    # 2026-08-19 continuation round: trial6's winner (lr=9.98e-4, wd=1.32e-4, focal_gamma=4.96)
    # hugged the old focal_gamma upper bound (5.0) -- probe past it directly (targeted, not
    # blind TPE) around trial6's otherwise-winning lr/weight_decay, before finalizing.
    _probe_done = any(
        t.params.get("direction_focal_gamma", 0) > 5.0 for t in study.trials
    )
    if not _probe_done:
        for fg in (7.0, 8.5, 10.0):
            study.enqueue_trial({"lr": 9.98e-4, "weight_decay": 1.32e-4, "direction_focal_gamma": fg})

    study.optimize(objective, n_trials=N_TRIALS)

    print(flush=True)
    print("=== STUDY DONE ===", flush=True)
    print(f"n_trials={len(study.trials)} n_pruned={len(study.get_trials(states=[optuna.trial.TrialState.PRUNED]))} "
          f"n_complete={len(study.get_trials(states=[optuna.trial.TrialState.COMPLETE]))}", flush=True)
    best = study.best_trial
    print(f"best_trial={best.number} value={best.value:.4f} params={best.params}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
