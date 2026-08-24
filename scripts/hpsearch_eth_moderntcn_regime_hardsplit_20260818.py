#!/usr/bin/env python3
"""RESEARCH ONLY -- Optuna TPE hyperparameter search for the regime-hard-split ModernTCN line
(train_eval_eth_moderntcn_direction_quality_regime_hardsplit_20260818.py), per explicit user
instruction (2026-08-18): the pilot's fixed ARCH_DEFAULT_PARAMS/ARCH_DEFAULT_TRAIN were never tuned
for this architecture/data combination, so a single-config negative result isn't a strong enough
basis to close the line (feedback_dl_needs_optimization_before_failure_verdict).

Search space and TPE approach copied from the abandoned base script's own stage_hpsearch
(train_eval_eth_direction_quality_nhits_moderntcn_20260816.py, ~line 1000) -- same architecture-
capacity + training HP ranges, same idea (cheap short-budget trials, re-score top-K longer) -- but
retargeted to _fit_one_regime (regime hard-split, extended TRAIN, the two 2026-08-18 architecture
fixes, Prechelt-UP4 + cosine LR instead of the base script's warmup/EMA/GCE/ELR/mixup machinery,
which this line deliberately does not use -- see train_eval_eth_moderntcn_direction_quality_regime_
hardsplit_20260818.py's own docstring for why).

Search space narrowed 2026-08-18 (user pushback, mid-run): the first version searched all 11 of the
base script's original dims (window/lr/weight_decay/batch_size/dropout + 5 architecture-capacity
knobs + use_revin) with only 25 trials -- the capacity knobs alone (n_stage x dim0 x large_size x
num_blocks x ffn_ratio) already span ~730 discrete combinations, hopelessly under-sampled at N=25.
Architecture capacity (n_stage/dim0/large_size/num_blocks/ffn_ratio/downsample_ratio/patch_size/
patch_stride) is now FIXED at ARCH_DEFAULT_PARAMS -- not searched at all -- because this session's
own repeated finding (TabM's R+S+B-completed architecture was consistently worse than R-only;
ModernTCN's larger input dimensionality overfit harder than GBDT on the identical exit label) is
that MORE capacity hurts under this project's weak-signal regime, so spending trials exploring
bigger models has a negative prior, not a neutral one. The search is now 6 dimensions (window, lr,
weight_decay, batch_size, dropout, use_revin) -- still thin for 25 trials but a defensible fraction
of the discrete space (18 combinations x 3 continuous dims) rather than an obviously hopeless one.

Searches on the CHOP regime only (single seed=0, fixed) as a cheap proxy for all three regimes --
chop has the most TRAIN rows (~50k) of the three, so its held-out loss is the least noisy of the
three regimes' own signals, and this repo's own convention (h48qual/zig075 don't tune per-expert
hyperparameters either -- one architecture, three independently-weighted experts) supports sharing
one HP set across bull/bear/chop rather than tripling the search cost. Objective = chop's own
selected_sel_loss (weighted CE on the embargoed TRAIN-internal held-out split, NOT VAL/OOS -- no
forward-looking data touched during search). Winning params get used for all 3 regimes' final
N-seed retrain (a separate follow-up script), matching the base script's own two-stage design
(hpsearch picks HPs cheaply, then a separate final stage spends real seeds on them).

fresh_forward_bar_by_bar=true (search only ever touches TRAIN-internal splits), trade_ledgers_used_
as_input=false, saved_parent_exit_timestamps_used=false, future_rows_used_for_entry=false.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import train_eval_eth_moderntcn_direction_quality_regime_hardsplit_20260818 as hs  # noqa: E402
import train_eval_eth_direction_quality_nhits_moderntcn_20260816 as base_nt  # noqa: E402
import train_omega1_regime3_expert_direction_head_volpca_20260602 as hard  # noqa: E402

OUT_DIR = ROOT / "tmp/causal_regen_20260516/eth_moderntcn_hardsplit_hpsearch_20260818"
N_TRIALS = 25
MAX_EPOCHS_TRIAL = 10
# 2026-08-18: moved search from server(GPU, ~9-23s/epoch) to dev(CPU-only, no GPU) after finding
# server shares its GPU with the live trading_bot.py -- CPU epoch cost measured at ~225s/epoch for
# chop (the largest regime), making the original 25x10 (~15h) budget impractical on CPU. Both
# knobs are now CLI-overridable (--n-trials, --max-epochs-trial) instead of hardcoded.
SEARCH_REGIME_IDX = hard.EXPERT_NAMES.index("chop")
TOP_K_CANDIDATES = 5


def log(msg: str) -> None:
    print(f"[hpsearch_moderntcn] {msg}", flush=True)


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    ap.add_argument("--n-trials", type=int, default=N_TRIALS)
    ap.add_argument("--max-epochs-trial", type=int, default=MAX_EPOCHS_TRIAL)
    args = ap.parse_args()

    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    device = base_nt._device(args.device)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    log("loading panel+labels+regime route...")
    data = hs.load_data_samedir_with_regime()

    def objective(trial: "optuna.Trial") -> float:
        window = trial.suggest_categorical("window", [48, 96, 192])
        lr = trial.suggest_float("lr", 1e-4, 5e-3, log=True)
        weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)
        batch_size = trial.suggest_categorical("batch_size", [256, 512, 1024])
        dropout = trial.suggest_float("dropout", 0.0, 0.3)
        # architecture capacity FIXED at ARCH_DEFAULT_PARAMS -- not searched, see module docstring
        arch_params = {**base_nt.ARCH_DEFAULT_PARAMS["moderntcn"], "dropout": dropout,
                       "use_revin": trial.suggest_categorical("use_revin", [True, False])}
        train_params = {"lr": lr, "weight_decay": weight_decay, "batch_size": batch_size, "window": window}
        log(f"  trial {trial.number} starting: arch={arch_params} train={train_params}")  # log BEFORE fitting so epoch curves below can be attributed live
        try:
            r = hs._fit_one_regime("moderntcn", arch_params, train_params, seed=0, epochs=args.max_epochs_trial,
                                    regime_idx=SEARCH_REGIME_IDX, data=data, device=device)
        except RuntimeError as exc:
            log(f"  trial pruned (RuntimeError: {exc})")
            raise optuna.TrialPruned()
        trial.set_user_attr("arch_params", arch_params)
        trial.set_user_attr("train_params", train_params)
        trial.set_user_attr("selected_bacc", r["selected_bacc"])
        log(f"  trial {trial.number} done: selected_sel_loss={r['selected_sel_loss']:.4f} selected_bacc={r['selected_bacc']:.4f}")
        return r["selected_sel_loss"]

    log(f"=== hpsearch: chop regime, n_trials={args.n_trials}, epochs<={args.max_epochs_trial}/trial ===")
    # 2026-08-18: sqlite storage + load_if_exists added after a dev WSL2 VM restart (2nd
    # infra interruption of the day, after the 2 server GPU dxg anomalies) killed the
    # in-memory-only study mid-search with zero results persisted -- rerunning this same
    # command now resumes from wherever the study.db left off instead of losing everything again.
    storage_path = OUT_DIR / "optuna_study.db"
    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=20260818),
                                 study_name="moderntcn_hardsplit_chop", storage=f"sqlite:///{storage_path}",
                                 load_if_exists=True)
    already = len(study.trials)
    remaining = max(0, args.n_trials - already)
    log(f"  resumable study at {storage_path}: {already} trial(s) already recorded, {remaining} remaining")
    t0 = time.time()
    if remaining > 0:
        study.optimize(objective, n_trials=remaining, show_progress_bar=False)
    log(f"  {len(study.trials)}/{args.n_trials} trials total ({time.time()-t0:.0f}s this run) best_sel_loss={study.best_value:.4f}")
    study.trials_dataframe().to_csv(OUT_DIR / "optuna_trials.csv", index=False)

    trials_sorted = sorted([t for t in study.trials if t.value is not None], key=lambda t: t.value)
    top = trials_sorted[:TOP_K_CANDIDATES]
    log(f"  top {len(top)} trials by chop selected_sel_loss:")
    top_rows = []
    for rank, trial in enumerate(top):
        row = {"rank": rank, "trial_number": trial.number, "sel_loss": trial.value,
               "selected_bacc": trial.user_attrs["selected_bacc"],
               "arch_params": trial.user_attrs["arch_params"], "train_params": trial.user_attrs["train_params"]}
        top_rows.append(row)
        log(f"    #{rank}: sel_loss={trial.value:.4f} bacc={trial.user_attrs['selected_bacc']:.4f} "
            f"arch={trial.user_attrs['arch_params']} train={trial.user_attrs['train_params']}")

    (OUT_DIR / "top_candidates.json").write_text(json.dumps(top_rows, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
    log(f"report={OUT_DIR / 'top_candidates.json'}")
    log("stage=done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
