#!/usr/bin/env python3
"""zigzag+h48qual regime-expert+LR스케줄 축([[eth_tabm_label_logic_5way_regime_expert_lrschedule_
20260822]])의 direction/quality 하이퍼파라미터 Optuna 서치 + 승자 설정 N=6 시드 재검증.

## 배경
사용자 질문(2026-08-22): "direction과 quality threshold 튜닝은 해야할거 같은데" -- 지금까지
`--quality-min-edge/--quality-max-mae/--quality-min-mfe-mae/--exit-giveback-min/--direction-
focal-gamma`는 전부 파이프라인 고정 기본값이었고 한 번도 서치된 적이 없었다(quality_threshold
자체는 매 실행마다 항상 도는 고정 5칸 그리드 0.40~0.60일 뿐, 튜닝이 아님). 이후 "zigzag,
h48qual을 dev에서 튜닝해서 val oos 테스트해서 거래수, pnl 보여줘" 요청으로 두 라벨 다 포함.

이 축은 이미 BCE 기준 절편전용 이론하한과 동일함이 확인됐다([[eth_label_fusion_combined_model_
research_20260821]]) -- 그래서 이 서치로 새 신호를 발굴할 걸로 기대하진 않는다. 목적은 "고정
기본값 자체가 잘못 캘리브레이션돼서 있는 신호를 못 쓰고 있었다"는 남은 가능성을 지우는 것.

## 단일터치 OOS 보호
`train_eval_omega4_3head_parent72_loose_entry_quality_20260620.py`는 OOS를 매 실행마다 무조건
계산해서 report.json에 남긴다(끌 방법 없음, 이 서치가 새로 만드는 특성이 아니라 이 세션 내내 모든
zigzag/h48qual/cusum 실행에 이미 해당됐던 구조적 특성). "계산됨" != "선택에 사용함"이므로,
Optuna objective()는 validation_pnl만 읽고 반환한다 -- oos_pnl은 서치 도중 단 한 번도 출력/비교
하지 않는다. 승자 설정이 정해진 뒤 N=6 시드 재검증 단계에서만 OOS 숫자를 들여다본다(TabM
시드분산이 HP효과보다 큰 전례가 있어[[tabm_hp_low_signal_pattern]] 단일시드 승자만으로는
결론 못 냄).

## 방법
라벨별로 순차 실행(zigzag 완료 후 h48qual 시작, 서버 부하 관리): Optuna 20trial(서치시드 고정) ->
best_params로 N=6 established 시드 재검증(`--out-suffix-tag qdirtuned`로 baseline out_dir과
분리, baseline report.json 보존) -> 라벨별 VAL/OOS pnl+거래수 요약 출력.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np  # noqa: E402
import optuna  # noqa: E402

LABELS = ["cusum"]   # zigzag/h48qual는 이미 완료(study에 20/20 저장됨, 재실행하면 reverify만
                     # 낭비 재수행하므로 여기선 cusum만) -- 아래 objective()도 cusum 실행분부터는
                     # no-op 발견(2026-08-22) 반영해 direction-focal-gamma 1개만 서치
SEARCH_SEED = 133725056
N_TRIALS = 20
SEEDS = [133725056, 325805917, 775149439, 126593178, 286919795, 310216042]
WRAPPER = ROOT / "scripts/eth_tabm_label_logic_5way_regime_expert_lrschedule_20260822.py"
MODEL_ID = "omega4_3head_parent72_loose_entry_quality_20260620"
OUT_ROOT = ROOT / "tmp/causal_regen_20260516"
STORAGE = f"sqlite:///{ROOT}/tmp/ilias_labellogic_recheck_20260821/quality_direction_optuna_study.db"

# [[eth_tabm_label_logic_3label_split_convention_retest_20260822]]와 동일 split(VAL=2026Q2,
# OOS=2026-07-01~)에서 확인된 always-long 벤치마크. 참고용 부호비교 기준선.
BENCHMARK_VAL_PNL = -23.34
BENCHMARK_OOS_PNL = 21.51


def out_dir_for(label: str, seed: int, tag: str = "") -> Path:
    suffix = f"label5way_{label}_154feat_regime_expert_lrschedule_seed{seed}_20260822"
    if tag:
        suffix += f"_{tag}"
    return OUT_ROOT / f"{MODEL_ID}_{suffix}"


def run_wrapper(label: str, seed: int, hp: dict, tag: str = "") -> dict:
    args = [sys.executable, str(WRAPPER), "--label", label, "--seed", str(seed)]
    for flag, val in hp.items():
        args += [f"--{flag}", str(val)]
    if tag:
        args += ["--out-suffix-tag", tag]
    subprocess.run(args, cwd=str(ROOT), check=True, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
    return json.loads((out_dir_for(label, seed, tag) / "report.json").read_text())


def search_label(label: str) -> dict:
    study_name = f"{label}_regime_expert_quality_direction_20260822"

    def objective(trial: "optuna.Trial") -> float:
        # 2026-08-22 no-op 발견: quality-min-edge/max-mae/min-mfe-mae는 quality_mode=
        # risk_adjusted_barrier_meta_action 전용(cusum은 same_as_direction), exit-giveback-min은
        # exit_label_mode=entry_label_terminal_giveback 전용(이 축은 independent_entry_hold_
        # offsets로 고정) -- 전부 cusum에도 무관하므로 여기선 처음부터 뺐다. 실질 서치차원은
        # direction-focal-gamma 하나뿐(zigzag/h48qual 재검증에서 이미 확인됨).
        hp = {
            "direction-focal-gamma": trial.suggest_float("direction_focal_gamma", 0.0, 3.0),
        }
        report = run_wrapper(label, SEARCH_SEED, hp)
        top = report["ranking_by_validation_pnl"][0]   # VAL-pnl만 읽는다 -- oos_pnl 미접근
        trial.set_user_attr("variant", top["variant"])
        print(f"[{label} trial {trial.number:3d}] val_pnl={top['validation_pnl']:+7.2f}%  "
              f"variant={top['variant']}  hp={hp}", flush=True)
        return top["validation_pnl"]

    Path(STORAGE.replace("sqlite:///", "")).parent.mkdir(parents=True, exist_ok=True)
    study = optuna.create_study(study_name=study_name, storage=STORAGE, direction="maximize", load_if_exists=True)
    n_done = len(study.trials)
    print(f"\n=== [{label}] Optuna 서치: 기존 완료={n_done}, 목표={N_TRIALS} ===", flush=True)
    remaining = max(N_TRIALS - n_done, 0)
    if remaining:
        study.optimize(objective, n_trials=remaining)
    print(f"[{label}] 서치 완료 best val_pnl={study.best_value:+.2f}% params={study.best_params}", flush=True)
    return study.best_params


def reverify_label(label: str, best_params: dict) -> None:
    hp = {k.replace("_", "-"): v for k, v in best_params.items()}
    print(f"\n=== [{label}] 승자설정 N={len(SEEDS)} 시드 재검증 (tag=qdirtuned) ===", flush=True)
    rows = []
    for seed in SEEDS:
        report = run_wrapper(label, seed, hp, tag="qdirtuned")
        top = report["ranking_by_validation_pnl"][0]
        rows.append({
            "seed": seed, "variant": top["variant"],
            "val_pnl": top["validation_pnl"], "val_trades": top["validation_trades"],
            "oos_pnl": top["oos_pnl"], "oos_trades": top["oos_trades"],
        })
        print(f"  seed={seed:>10}  variant={top['variant']:<6}  "
              f"VAL={top['validation_pnl']:+7.2f}%(n={top['validation_trades']:>3})  "
              f"OOS={top['oos_pnl']:+7.2f}%(n={top['oos_trades']:>3})", flush=True)

    val = np.array([r["val_pnl"] for r in rows])
    oos = np.array([r["oos_pnl"] for r in rows])
    sign_match = int(np.sum(np.sign(val) == np.sign(oos)))
    print(f"[{label}] 튜닝후 VAL mean={val.mean():+.2f}% std={val.std(ddof=0):.2f}  "
          f"OOS mean={oos.mean():+.2f}% std={oos.std(ddof=0):.2f}  부호일치={sign_match}/{len(SEEDS)}  "
          f"(always-long 벤치마크 VAL={BENCHMARK_VAL_PNL:+.2f}% OOS={BENCHMARK_OOS_PNL:+.2f}%)", flush=True)


def main() -> None:
    for label in LABELS:
        best_params = search_label(label)
        reverify_label(label, best_params)
    print("\n=== 전체 완료 ===", flush=True)


if __name__ == "__main__":
    main()
