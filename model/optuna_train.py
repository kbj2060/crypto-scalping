"""
Optuna 기반 PPO 하이퍼파라미터 최적화
- model/config.py 파라미터를 트라이얼별로 덮어쓰고 짧은 학습 후 검증 보상으로 최적화
- 3-Action (Hold/Buy/Sell), PPOTrainer와 동일한 학습/검증 흐름 사용 (TensorBoard 비활성화)
- XLSTM: Gated Fusion, Strategy Interaction + Orthogonal Regularization, Dynamic Entropy 반영
- 설치: pip install optuna
- 실행: python -m model.optuna_train --n-trials 30 --episodes-per-trial 50
- 결과: data/optuna_best_params.json 에 최적 파라미터 저장
"""
import json
import logging
import os
import sys

import numpy as np
import optuna
from optuna.samplers import TPESampler

# 프로젝트 루트
if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model import config
from model.train_ppo import PPOTrainer

os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(message)s",
    handlers=[
        logging.FileHandler("logs/optuna_ppo.log", encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ]
)
logger = logging.getLogger(__name__)
logging.getLogger("model.feature_engineering").setLevel(logging.WARNING)
logging.getLogger("model.mtf_processor").setLevel(logging.WARNING)


# 트라이얼당 에피소드 수 (적을수록 빠르고 노이즈 많음)
N_EPISODES_PER_TRIAL = 50
# 검증 구간으로 목적 함수 계산 (True: validate_on_test_set 보상, False: 마지막 10 에피소드 평균 보상)
USE_VALIDATION_AS_OBJECTIVE = True


def apply_params_dict(params: dict) -> None:
    """파라미터 딕셔너리를 config에 반영 (best trial 저장용)."""
    if "lr" in params:
        config.PPO_LEARNING_RATE = params["lr"]
    if "gamma" in params:
        config.PPO_GAMMA = params["gamma"]
    if "lambda" in params:
        config.PPO_LAMBDA = params["lambda"]
    if "eps_clip" in params:
        config.PPO_EPS_CLIP = params["eps_clip"]
    if "k_epochs" in params:
        config.PPO_K_EPOCHS = params["k_epochs"]
    if "entropy_coef" in params:
        config.PPO_ENTROPY_COEF = params["entropy_coef"]
    if "temp_init" in params:
        config.PPO_TEMP_INIT = params["temp_init"]
    if "temp_decay" in params:
        config.PPO_TEMP_DECAY = params["temp_decay"]
    if "temp_min" in params:
        config.PPO_TEMP_MIN = params["temp_min"]
    if "lookback" in params:
        config.LOOKBACK = params["lookback"]
    if "num_layers" in params:
        config.NETWORK_NUM_LAYERS = params["num_layers"]
    if "dropout" in params:
        config.NETWORK_DROPOUT = params["dropout"]
    if "value_clip_eps" in params:
        config.PPO_VALUE_CLIP_EPS = params["value_clip_eps"]
    if "entropy_min" in params:
        config.PPO_ENTROPY_MIN = params["entropy_min"]


def apply_trial_params(trial: optuna.Trial) -> None:
    """트라이얼에서 제안된 파라미터를 config에 반영."""
    config.PPO_LEARNING_RATE = trial.suggest_float("lr", 1e-5, 1e-3, log=True)
    config.PPO_GAMMA = trial.suggest_float("gamma", 0.99, 0.999)
    config.PPO_LAMBDA = trial.suggest_float("lambda", 0.9, 0.99)
    config.PPO_EPS_CLIP = trial.suggest_float("eps_clip", 0.1, 0.25)
    config.PPO_K_EPOCHS = trial.suggest_int("k_epochs", 2, 6)
    config.PPO_ENTROPY_COEF = trial.suggest_float("entropy_coef", 0.001, 0.02, log=True)
    config.PPO_TEMP_INIT = trial.suggest_float("temp_init", 0.5, 1.0)
    config.PPO_TEMP_DECAY = trial.suggest_float("temp_decay", 0.998, 0.9999)
    config.PPO_TEMP_MIN = trial.suggest_float("temp_min", 0.2, 0.5)
    config.LOOKBACK = trial.suggest_int("lookback", 30, 120)
    config.NETWORK_NUM_LAYERS = trial.suggest_int("num_layers", 1, 2)
    config.NETWORK_DROPOUT = trial.suggest_float("dropout", 0.05, 0.2)
    config.PPO_VALUE_CLIP_EPS = trial.suggest_float("value_clip_eps", 0.2, 0.4)
    # Dynamic Entropy 하한 (Critic 오차가 클 때 탐험 증가와 함께 사용)
    config.PPO_ENTROPY_MIN = trial.suggest_float("entropy_min", 0.005, 0.05, log=True)


def objective(trial: optuna.Trial) -> float:
    """한 트라이얼: 파라미터 적용 → 짧은 학습 → 검증 보상 반환."""
    apply_trial_params(trial)

    try:
        trainer = PPOTrainer(enable_visualization=False)
        trainer.env.precompute_data()

        rewards = []
        pnls = []
        for ep in range(1, N_EPISODES_PER_TRIAL + 1):
            res = trainer.train_episode(ep)
            if res is not None:
                r, c, pnl = res
                rewards.append(r)
                pnls.append(pnl)

        if not rewards:
            return float("-inf")

        if USE_VALIDATION_AS_OBJECTIVE and hasattr(trainer, "validate_on_test_set"):
            test_reward, test_sharpe = trainer.validate_on_test_set()
            # 보상과 샤프를 조합해 목적 함수 (보상 우선, 샤프 보조)
            objective_value = test_reward + 0.1 * np.clip(test_sharpe, -2, 2)
            trial.set_user_attr("test_reward", test_reward)
            trial.set_user_attr("test_sharpe", test_sharpe)
        else:
            objective_value = float(np.mean(rewards[-10:]))
            trial.set_user_attr("mean_reward_last10", objective_value)

        trial.set_user_attr("mean_reward", float(np.mean(rewards)))
        trial.set_user_attr("mean_pnl_pct", float(np.mean(pnls)) * 100 if pnls else 0.0)

        if trainer.writer:
            trainer.writer.close()

        return objective_value

    except Exception as e:
        logger.exception(f"Trial failed: {e}")
        return float("-inf")


def run_study(
    n_trials: int = 30,
    study_name: str = "ppo_ppo",
    storage: str | None = None,
    load_if_exists: bool = True,
) -> optuna.Study:
    """Optuna 스터디 실행 및 최적 파라미터 저장."""
    sampler = TPESampler(
        n_startup_trials=5,
        n_ei_candidates=24,
        seed=42,
        multivariate=True,
    )

    if storage:
        study = optuna.create_study(
            study_name=study_name,
            storage=storage,
            load_if_exists=load_if_exists,
            direction="maximize",
            sampler=sampler,
        )
    else:
        study = optuna.create_study(
            study_name=study_name,
            direction="maximize",
            sampler=sampler,
        )

    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    # 최적 파라미터를 config에 반영하고 파일로 저장
    if study.best_trial:
        best = study.best_trial
        apply_params_dict(best.params)
        params_to_save = dict(best.params)
        out_path = os.path.join("data", "optuna_best_params.json")
        os.makedirs("data", exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(params_to_save, f, indent=2, ensure_ascii=False)
        logger.info(f"Best trial: {best.number} | value={best.value:.4f}")
        logger.info(f"Best params saved to {out_path}")

    return study


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Optuna PPO hyperparameter optimization")
    parser.add_argument("--n-trials", type=int, default=30, help="Number of Optuna trials")
    parser.add_argument("--episodes-per-trial", type=int, default=50, help="Episodes per trial")
    parser.add_argument("--study-name", type=str, default="ppo_ppo", help="Optuna study name")
    parser.add_argument("--use-validation", action="store_true", default=True, help="Use validation reward as objective")
    parser.add_argument("--no-validation", action="store_false", dest="use_validation", help="Use mean episode reward")
    args = parser.parse_args()

    global N_EPISODES_PER_TRIAL, USE_VALIDATION_AS_OBJECTIVE
    N_EPISODES_PER_TRIAL = args.episodes_per_trial
    USE_VALIDATION_AS_OBJECTIVE = args.use_validation

    run_study(n_trials=args.n_trials, study_name=args.study_name)


if __name__ == "__main__":
    main()
