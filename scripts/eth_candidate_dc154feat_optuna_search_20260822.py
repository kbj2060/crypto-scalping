#!/usr/bin/env python3
"""DC154 tabular 트랜스포머(`eth_candidate_dc154feat_tabular_transformer_smoke_test_20260822.py`)
Optuna HP탐색 -- `load_data()`를 한 번만 호출하고 재사용, 목적함수는 VAL(2026 Q2) BCE만 본다
(OOS는 `train_and_eval(eval_oos=False)` 기본값이라 이 탐색 동안 전혀 안 건드림 -- 단일터치
보호). SQLite storage로 재실행/재개 가능.

⚠️ 기대치: 이 축은 [[eth_dc_engineered154_feature_set_20260820]]/[[eth_label_fusion_combined_
model_research_20260821]]이 이미 BCE=절편전용 이론하한임을 확인했다 -- 탐색으로 하한 아래를
찾을 걸로 기대하지 않는다. 목적은 (a) 이미 "제대로 된 기법 재검증"(사용자 지시)을 완수하는 것,
(b) 혹시 있을 수 있는 학습기법 결함(과소적합/불안정)을 놓치지 않는 것.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import optuna  # noqa: E402

import eth_candidate_dc154feat_tabular_transformer_smoke_test_20260822 as dc154  # noqa: E402

SEARCH_SEED = 20260822   # 탐색 단계 고정시드(HP효과 대 시드노이즈 혼입 최소화 목적, N시드검증은 별도 후속)
N_TRIALS = 20            # ⚠️ 2026-08-22: SDPA수정 후에도 최악조합(n_blocks=6,batch=1024)은 3에폭에
                          # 1015초 -- 탐색공간에서 그런 조합을 빼고 시행횟수도 30->20으로 낮춤
STORAGE = f"sqlite:///{ROOT}/tmp/dc154_ilias_split_20260822/optuna_study.db"
STUDY_NAME = "dc154_tabular_transformer_20260822_v2"   # v1은 죽은 첫 trial의 옛 탐색공간(n_blocks<=6
                                                          # 등)이 param distribution으로 박혀있어 재사용 불가(0 완료, 데이터손실 없음)


def objective(trial: "optuna.Trial", data: dict) -> float:
    hp = {
        "n_blocks": trial.suggest_categorical("n_blocks", [2, 3, 4]),      # 6 제거(가장 느린 축)
        "d_token": trial.suggest_categorical("d_token", [8, 16]),           # 32 제거(느린 축의 절반)
        "lr": trial.suggest_float("lr", 1e-5, 1e-3, log=True),
        "weight_decay": trial.suggest_float("weight_decay", 1e-4, 1e-1, log=True),
        "attn_dropout": trial.suggest_float("attn_dropout", 0.0, 0.3),
        "mlp_dropout": trial.suggest_float("mlp_dropout", 0.0, 0.3),
        "batch": trial.suggest_categorical("batch", [2048, 4096, 8192]),    # 1024 제거(가장 느린 축)
    }
    result = dc154.train_and_eval(data, hp, SEARCH_SEED, eval_oos=False, verbose=False)
    trial.set_user_attr("val_acc", result["val_acc"])
    trial.set_user_attr("best_epoch", result["best_epoch"])
    trial.set_user_attr("n_params", result["n_params"])
    print(f"[trial {trial.number:3d}] val_bce={result['val_bce']:.4f} "
          f"(floor={result['val_intercept_bce']:.4f}) val_acc={result['val_acc']:.3f} "
          f"n_params={result['n_params']:,} hp={hp}", flush=True)
    return result["val_bce"]


def main() -> None:
    print("데이터 로딩(1회, 전체 trial이 재사용)...", flush=True)
    data = dc154.load_data()

    Path(STORAGE.replace("sqlite:///", "")).parent.mkdir(parents=True, exist_ok=True)
    study = optuna.create_study(
        study_name=STUDY_NAME, storage=STORAGE, direction="minimize", load_if_exists=True,
    )
    n_done = len(study.trials)
    print(f"\n기존 완료 trial={n_done}, 목표 {N_TRIALS}개까지 추가 실행", flush=True)
    remaining = max(N_TRIALS - n_done, 0)
    if remaining == 0:
        print("이미 목표 trial 수 도달 -- 추가 실행 없음", flush=True)
    else:
        study.optimize(lambda t: objective(t, data), n_trials=remaining)

    print(f"\n=== Optuna 탐색 완료: {len(study.trials)} trials ===", flush=True)
    print(f"best val_bce={study.best_value:.4f} (floor={data['ytr'].mean():.3f}로부터 계산되는 이론하한 참고)", flush=True)
    print(f"best params: {study.best_params}", flush=True)
    print(f"best trial#: {study.best_trial.number}", flush=True)


if __name__ == "__main__":
    main()
