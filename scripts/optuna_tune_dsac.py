import argparse
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime

import torch

try:
    import optuna
except Exception as exc:  # pragma: no cover
    optuna = None
    _OPTUNA_IMPORT_ERROR = exc
else:
    _OPTUNA_IMPORT_ERROR = None


ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAIN_SCRIPT = os.path.join(ROOT_DIR, "ensemble", "train_rl_dsac_agent.py")


def _ensure_optuna() -> None:
    if optuna is None:
        raise RuntimeError(
            "optuna is not installed in this environment. "
            f"Original import error: {_OPTUNA_IMPORT_ERROR}"
        )


def _trial_dir(base_dir: str, number: int) -> str:
    path = os.path.join(base_dir, f"trial_{number:04d}")
    os.makedirs(path, exist_ok=True)
    return path


def _build_search_space(trial: "optuna.trial.Trial") -> dict:
    entropy_min = trial.suggest_float("entropy_min", -1.10, -0.60)
    entropy_max = trial.suggest_float("entropy_max", max(entropy_min + 0.10, -0.70), -0.25)
    entropy_std_low = trial.suggest_float("entropy_std_low", 0.10, 0.26)
    entropy_std_high = trial.suggest_float("entropy_std_high", max(entropy_std_low + 0.05, 0.20), 0.48)
    pessimism_weight_min = trial.suggest_float("pessimism_weight_min", 0.35, 0.70)
    pessimism_weight_max = trial.suggest_float(
        "pessimism_weight_max",
        max(pessimism_weight_min + 0.05, 0.55),
        0.90,
    )
    soft_gate_warmup_epochs = trial.suggest_int("soft_gate_warmup_epochs", 0, 30)
    soft_gate_ramp_epochs = trial.suggest_int(
        "soft_gate_ramp_epochs",
        max(20, soft_gate_warmup_epochs + 20),
        120,
    )
    params = {
        "lr_factor": trial.suggest_float("lr_factor", 0.35, 0.80),
        "lr_patience": trial.suggest_int("lr_patience", 3, 8),
        "lr_min": trial.suggest_float("lr_min", 1e-5, 1e-4, log=True),
        "early_stop_patience": trial.suggest_int("early_stop_patience", 6, 18),
        "cvar_frac": trial.suggest_float("cvar_frac", 0.25, 0.55),
        "gamma": trial.suggest_float("gamma", 0.985, 0.9995),
        "pessimism_min_weight": trial.suggest_float("pessimism_min_weight", 0.45, 0.85),
        "adaptive_pessimism": trial.suggest_categorical("adaptive_pessimism", [False, True]),
        "pessimism_disagree_scale": trial.suggest_float("pessimism_disagree_scale", 0.05, 0.30),
        "pessimism_weight_min": pessimism_weight_min,
        "pessimism_weight_max": pessimism_weight_max,
        "dynamic_entropy": trial.suggest_categorical("dynamic_entropy", [True, False]),
        "entropy_min": entropy_min,
        "entropy_max": entropy_max,
        "entropy_std_low": entropy_std_low,
        "entropy_std_high": entropy_std_high,
        "entropy_step": trial.suggest_float("entropy_step", 0.02, 0.08),
        "critic_var_weight": trial.suggest_categorical("critic_var_weight", [False, True]),
        "critic_var_scale": trial.suggest_float("critic_var_scale", 0.5, 1.5),
        "critic_var_w_min": trial.suggest_float("critic_var_w_min", 0.10, 0.40),
        "primacy_soft_reset": trial.suggest_categorical("primacy_soft_reset", [False, True]),
        "primacy_window": trial.suggest_int("primacy_window", 40, 120),
        "primacy_imbalance_th": trial.suggest_float("primacy_imbalance_th", 0.45, 0.80),
        "primacy_entropy_low": trial.suggest_float("primacy_entropy_low", 0.25, 0.60),
        "primacy_reset_cooldown": trial.suggest_int("primacy_reset_cooldown", 60, 180),
        "direction_reg_lambda": trial.suggest_float("direction_reg_lambda", 0.0, 0.20),
        "side_balance_lambda": trial.suggest_float("side_balance_lambda", 0.0, 0.20),
        "val_side_bias_penalty": trial.suggest_float("val_side_bias_penalty", 20.0, 140.0),
        "cql_reg": trial.suggest_categorical("cql_reg", [False, True]),
        "cql_alpha": trial.suggest_float("cql_alpha", 0.005, 0.06),
        "redo_enable": trial.suggest_categorical("redo_enable", [False, True]),
        "redo_interval": trial.suggest_int("redo_interval", 250, 1000),
        "redo_tau": trial.suggest_float("redo_tau", 0.001, 0.02, log=True),
        "redo_ratio": trial.suggest_float("redo_ratio", 0.03, 0.20),
        "alpha_min": trial.suggest_float("alpha_min", 0.001, 0.02, log=True),
        "alpha_init": trial.suggest_float("alpha_init", 0.01, 0.08),
        "anti_flat_lambda": trial.suggest_float("anti_flat_lambda", 0.0, 0.16),
        "anti_flat_min_abs": trial.suggest_float("anti_flat_min_abs", 0.08, 0.30),
        "anti_flat_anneal_updates": trial.suggest_int("anti_flat_anneal_updates", 40000, 180000),
        "soft_gate_warmup_epochs": soft_gate_warmup_epochs,
        "soft_gate_ramp_epochs": soft_gate_ramp_epochs,
        "min_val_trades_for_best": trial.suggest_int("min_val_trades_for_best", 40, 120),
    }
    return params


def _build_trial_command(args: argparse.Namespace, params: dict, trial_dir: str) -> list[str]:
    trial_config = os.path.join(trial_dir, "dsac_train_config.json")
    trial_ckpt = os.path.join(trial_dir, "dsac_checkpoint.pth")
    trial_best = os.path.join(trial_dir, "best_dsac_agents.pth")

    cmd = [
        sys.executable,
        TRAIN_SCRIPT,
        "--csv-path",
        args.csv_path,
        "--train-ratio",
        str(args.train_ratio),
        "--episodes",
        str(args.episodes),
        "--fresh-start",
        "--val-interval",
        str(args.val_interval),
        "--config-json-path",
        trial_config,
        "--checkpoint-path",
        trial_ckpt,
        "--best-path",
        trial_best,
        "--hmm-cache-path",
        args.hmm_cache_path,
        "--device",
        args.device,
        "--lr-factor",
        str(params["lr_factor"]),
        "--lr-patience",
        str(params["lr_patience"]),
        "--lr-min",
        str(params["lr_min"]),
        "--early-stop-patience",
        str(params["early_stop_patience"]),
        "--cvar-frac",
        str(params["cvar_frac"]),
        "--gamma",
        str(params["gamma"]),
        "--pessimism-min-weight",
        str(params["pessimism_min_weight"]),
        "--pessimism-disagree-scale",
        str(params["pessimism_disagree_scale"]),
        "--pessimism-weight-min",
        str(params["pessimism_weight_min"]),
        "--pessimism-weight-max",
        str(params["pessimism_weight_max"]),
        "--entropy-min",
        str(params["entropy_min"]),
        "--entropy-max",
        str(params["entropy_max"]),
        "--entropy-std-low",
        str(params["entropy_std_low"]),
        "--entropy-std-high",
        str(params["entropy_std_high"]),
        "--entropy-step",
        str(params["entropy_step"]),
        "--critic-var-scale",
        str(params["critic_var_scale"]),
        "--critic-var-w-min",
        str(params["critic_var_w_min"]),
        "--primacy-window",
        str(params["primacy_window"]),
        "--primacy-imbalance-th",
        str(params["primacy_imbalance_th"]),
        "--primacy-entropy-low",
        str(params["primacy_entropy_low"]),
        "--primacy-reset-cooldown",
        str(params["primacy_reset_cooldown"]),
        "--direction-reg-lambda",
        str(params["direction_reg_lambda"]),
        "--side-balance-lambda",
        str(params["side_balance_lambda"]),
        "--val-side-bias-penalty",
        str(params["val_side_bias_penalty"]),
        "--cql-alpha",
        str(params["cql_alpha"]),
        "--redo-interval",
        str(params["redo_interval"]),
        "--redo-tau",
        str(params["redo_tau"]),
        "--redo-ratio",
        str(params["redo_ratio"]),
        "--alpha-min",
        str(params["alpha_min"]),
        "--alpha-init",
        str(params["alpha_init"]),
        "--anti-flat-lambda",
        str(params["anti_flat_lambda"]),
        "--anti-flat-min-abs",
        str(params["anti_flat_min_abs"]),
        "--anti-flat-anneal-updates",
        str(params["anti_flat_anneal_updates"]),
        "--soft-gate-warmup-epochs",
        str(params["soft_gate_warmup_epochs"]),
        "--soft-gate-ramp-epochs",
        str(params["soft_gate_ramp_epochs"]),
        "--min-val-trades-for-best",
        str(params["min_val_trades_for_best"]),
    ]

    if args.no_lr_scheduler:
        cmd.append("--no-lr-scheduler")
    if params["adaptive_pessimism"]:
        cmd.append("--adaptive-pessimism")
    else:
        cmd.append("--no-adaptive-pessimism")
    if params["dynamic_entropy"]:
        cmd.append("--dynamic-entropy")
    else:
        cmd.append("--no-dynamic-entropy")
    if params["critic_var_weight"]:
        cmd.append("--critic-var-weight")
    else:
        cmd.append("--no-critic-var-weight")
    if params["primacy_soft_reset"]:
        cmd.append("--primacy-soft-reset")
    else:
        cmd.append("--no-primacy-soft-reset")
    if params["cql_reg"]:
        cmd.append("--cql-reg")
    else:
        cmd.append("--no-cql-reg")
    if params["redo_enable"]:
        cmd.append("--redo-enable")
    else:
        cmd.append("--no-redo-enable")
    if args.hmm_force_refit:
        cmd.append("--hmm-force-refit")
    return cmd


def _load_trial_result(trial_dir: str) -> dict:
    best_path = os.path.join(trial_dir, "best_dsac_agents.pth")
    cfg_path = os.path.join(trial_dir, "dsac_train_config.json")
    result = {
        "best_val_pnl": -float("inf"),
        "best_val_score": -float("inf"),
        "best_epoch": None,
        "best_path": best_path,
        "config_json_path": cfg_path,
    }
    if os.path.exists(best_path):
        ckpt = torch.load(best_path, map_location="cpu", weights_only=False)
        result["best_val_pnl"] = float(ckpt.get("best_pnl", -float("inf")))
        result["best_val_score"] = float(ckpt.get("best_score", -float("inf")))
        result["best_epoch"] = int(ckpt.get("epoch", 0))
    return result


def _write_summary(path: str, payload: dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def _copy_best_trial(best_trial_dir: str, export_dir: str) -> dict:
    os.makedirs(export_dir, exist_ok=True)
    out = {}
    for name in ["dsac_train_config.json", "dsac_checkpoint.pth", "best_dsac_agents.pth"]:
        src = os.path.join(best_trial_dir, name)
        if os.path.exists(src):
            dst = os.path.join(export_dir, name)
            shutil.copy2(src, dst)
            out[name] = dst
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Optuna tuner for train_rl_dsac_agent.py")
    p.add_argument("--csv-path", default="data/splits/year_oos/rl_training_2025_m7.csv")
    p.add_argument("--train-ratio", type=float, default=0.8)
    p.add_argument("--episodes", type=int, default=120)
    p.add_argument("--val-interval", type=int, default=10)
    p.add_argument("--n-trials", type=int, default=20)
    p.add_argument("--timeout-sec", type=int, default=0)
    p.add_argument("--study-name", default=f"dsac_val_pnl_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    p.add_argument("--storage", default="")
    p.add_argument("--sampler-seed", type=int, default=42)
    p.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    p.add_argument("--hmm-cache-path", default="data/ensemble/ckpt/hmm_init_cache_dsac.npz")
    p.add_argument("--hmm-force-refit", action="store_true", default=False)
    p.add_argument("--no-lr-scheduler", action="store_true", default=False)
    p.add_argument("--artifact-root", default="data/ensemble/optuna")
    p.add_argument("--report-path", default="data/ensemble/reports/optuna_dsac_best_val_pnl.json")
    p.add_argument("--copy-best-to", default="data/ensemble/optuna/best_trial_export")
    return p.parse_args()


def main() -> None:
    _ensure_optuna()
    args = parse_args()

    study_dir = os.path.join(ROOT_DIR, args.artifact_root, args.study_name)
    os.makedirs(study_dir, exist_ok=True)

    sampler = optuna.samplers.TPESampler(seed=args.sampler_seed)
    study = optuna.create_study(
        study_name=args.study_name,
        storage=args.storage or None,
        load_if_exists=bool(args.storage),
        direction="maximize",
        sampler=sampler,
    )

    def objective(trial: "optuna.trial.Trial") -> float:
        params = _build_search_space(trial)
        trial_dir = _trial_dir(study_dir, trial.number)
        cmd = _build_trial_command(args, params, trial_dir)
        proc = subprocess.run(
            cmd,
            cwd=ROOT_DIR,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        stdout_path = os.path.join(trial_dir, "stdout.log")
        stderr_path = os.path.join(trial_dir, "stderr.log")
        with open(stdout_path, "w", encoding="utf-8") as f:
            f.write(proc.stdout or "")
        with open(stderr_path, "w", encoding="utf-8") as f:
            f.write(proc.stderr or "")
        if proc.returncode != 0:
            raise RuntimeError(f"trial {trial.number} failed with exit code {proc.returncode}")

        result = _load_trial_result(trial_dir)
        trial.set_user_attr("trial_dir", trial_dir)
        trial.set_user_attr("best_path", result["best_path"])
        trial.set_user_attr("config_json_path", result["config_json_path"])
        trial.set_user_attr("best_epoch", result["best_epoch"])
        for key, val in params.items():
            trial.set_user_attr(key, val)
        return float(result["best_val_pnl"])

    study.optimize(
        objective,
        n_trials=args.n_trials,
        timeout=(None if args.timeout_sec <= 0 else args.timeout_sec),
        show_progress_bar=True,
    )

    best = study.best_trial
    copied = _copy_best_trial(best.user_attrs["trial_dir"], os.path.join(ROOT_DIR, args.copy_best_to))
    summary = {
        "study_name": args.study_name,
        "created_at": datetime.now().isoformat(),
        "metric": "best_val_pnl",
        "best_value": float(best.value),
        "best_trial_number": int(best.number),
        "best_params": dict(best.params),
        "best_user_attrs": dict(best.user_attrs),
        "copied_artifacts": copied,
        "n_trials": len(study.trials),
    }
    _write_summary(os.path.join(ROOT_DIR, args.report_path), summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
