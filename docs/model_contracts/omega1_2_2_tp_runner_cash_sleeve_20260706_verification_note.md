# Verification Note: omega1_2_2_tp_runner_cash_sleeve_20260615 (2026-07-06, lightweight)

Status: `structurally_plausible_pnl_unverified_high_mdd`

Scope: lighter-weight check than `omega4_3_valonly_logrisk_tail050_20260706_verification_note.md`
(time-boxed per user direction) -- code-structure review only, not a full independent bar-by-bar
re-derivation.

## Finding 1: no m7/NF dependency in this lineage

Unlike Omega4.x, the Omega1.2.x TP-runner lineage
(`train_eval_omega1_2_1_exit_only_rl_editor_20260610.py` and its dependents) has zero references
to `m7_`/`patchtst`/NeuralForecast anywhere. It still imports `train_eval_omega1_2_tabm_diffusion_
risk_20260603` for the underlying OHLC market frame (same alpha6/7-lineage `TRAIN_CSV`/`EVAL_CSV`,
Jan-Feb 2026 cap for OOS), but does not need the heavy 172-feature parent contract. This means IF
extension were pursued, it would face the same feature-vintage-drift question as Omega4.6.1 (not
the m7/NF wall) -- not checked further given the time-box.

## Finding 2: the replay engine is structurally genuine bar-by-bar

`train_eval_omega1_2_1_exit_only_rl_editor_20260610.py`'s core loop (`for i in range(0, len(frame)
- 2)`, incrementally tracking `pos.mfe`/`pos.mae`/unrealized PnL per bar) matches the same causal
pattern already verified for the Omega4 family's `_replay_with_risk`. This is a positive structural
signal (not a full re-derivation) that the reported numbers are not a post-hoc ledger edit like the
event-flat failure.

## Finding 3: the contract itself flags PnL as diagnostic-only, unverified by redteam

Direct quote from the contract doc: "Red Team PASS excludes PnL/OOS lift... These metrics are
diagnostics and are not part of the Red Team PASS/FAIL gate." This project's own redteam process
for this candidate checked only logical/data/feature-contract correctness, not PnL trustworthiness.

## Finding 4: MDD is high relative to later, stricter Omega generations

Validation MDD -27.64% (base) / -26.54% (with cash sleeve) far exceeds the ~-20% (or -8% for
Omega4.3) bounds used in later Omega4.x candidates. Even if the PnL numbers hold up under a full
replay, this MDD level would likely fail the MDD gates this project later adopted.

## Verdict

Not independently re-derived to the same depth as omega4_3 (time-boxed). Structurally plausible
(genuine bar-by-bar engine), but has two real caveats the project's own docs already surface:
PnL/OOS is explicitly unverified by redteam, and MDD is elevated versus later, stricter
generations. Lower priority than `omega4_3_valonly_logrisk_tail050` for further investment.
