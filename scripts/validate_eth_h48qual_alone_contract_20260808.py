"""Pre-registered validation of the ETH "h48qual alone" candidacy (2026-08-08).

Contract: docs/experiments/eth_h48qual_alone_candidacy_20260808.json (written first).

Design in one line: the 2026H1 window is contaminated (two prior comparisons used it), so VAL
2025Q4 -- never used for THIS comparison but the place the components were fit -- is run as a
FALSIFICATION test, and the untouched 2026-07-01..07-12 tail is reported without power to decide.
The best outcome this contract can produce is "promote to a parallel shadow", never a live swap.

Two harnesses are reused rather than reimplemented so the windows cannot drift:
  replay_omega4_6_1_greedy_router_20260706.greedy_replay / prepare_component  (the replay itself)
  replay_omega4_6_1_greedy_val_20260706.load_val_frame + VAL_PRED             (the VAL window)

Statistics note, applied from the 2026-08-08 ETH sizing-transfer lesson: a paired-bootstrap P
measures SIGN CONSISTENCY, not effect size. So every arm comparison here reports a paired
bootstrap AND a Welch t-test on per-trade returns AND the raw effect size, and the gate is written
on Calmar/MDD levels rather than on a bootstrap P.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import train_eval_omega1_2_tabm_diffusion_risk_20260603 as omega  # noqa: E402
import retest_omega4_6_1_extended_oos_20260706 as retest  # noqa: E402
import replay_omega4_6_1_greedy_val_20260706 as gval  # noqa: E402
from replay_omega4_6_1_greedy_router_20260706 import (  # noqa: E402
    DURATION_THRESHOLD, greedy_replay, prepare_component,
)

CONTRACT = ROOT / "docs/experiments/eth_h48qual_alone_candidacy_20260808.json"
OOS_PRED_DIR = ROOT / "tmp/causal_regen_20260516/omega4_6_1_extended_oos_20260706"
OUT_DIR = ROOT / "tmp/eth_h48qual_alone_contract_20260808"
G1_CALMAR_RATIO, G1_MDD_SLACK_PP = 0.70, 3.0
B_BOOT, BLOCK_DAYS, SEED = 2000, 21, 903174


def metrics(r: np.ndarray) -> dict:
    if len(r) == 0:
        return {"pnl": 0.0, "mdd": 0.0, "calmar": None, "trades": 0, "wr": None}
    eq = np.cumprod(1.0 + r)
    mdd = float((eq / np.maximum.accumulate(eq) - 1.0).min() * 100)
    pnl = float((eq[-1] - 1.0) * 100)
    return {"pnl": round(pnl, 2), "mdd": round(mdd, 2),
            "calmar": round(pnl / abs(mdd), 2) if mdd < -1e-9 else None,
            "trades": int(len(r)), "wr": round(float((r > 0).mean()), 3)}


def gated(led: pd.DataFrame, frame: pd.DataFrame) -> np.ndarray:
    m = frame[["timestamp", "ou_halflife"]].rename(columns={"timestamp": "entry_ts"})
    d = led.merge(m, on="entry_ts", how="left")
    return d.loc[d["ou_halflife"] > DURATION_THRESHOLD, "trade_return"].to_numpy(float)


def compare(a: pd.DataFrame, b: pd.DataFrame, rng) -> dict:
    """Paired calendar-block bootstrap (sign consistency) PLUS a Welch t-test and raw effect size,
    because a bootstrap P alone cannot distinguish a tiny-but-consistent gap from a real one."""
    ts = pd.concat([a["entry_ts"], b["entry_ts"]])
    edges = pd.date_range(ts.min(), ts.max() + pd.Timedelta(days=BLOCK_DAYS), freq=f"{BLOCK_DAYS}D")
    blocks = list(zip(edges[:-1], edges[1:]))
    per = {k: [d.loc[(d["entry_ts"] >= s) & (d["entry_ts"] < e), "trade_return"].to_numpy(float)
               for s, e in blocks] for k, d in (("a", a), ("b", b))}
    boot = {"a": np.empty(B_BOOT), "b": np.empty(B_BOOT)}
    for i in range(B_BOOT):
        pick = rng.integers(len(blocks), size=len(blocks))
        for k in ("a", "b"):
            r = np.concatenate([per[k][j] for j in pick]) if blocks else np.array([])
            boot[k][i] = (np.cumprod(1.0 + r)[-1] - 1.0) * 100 if len(r) else 0.0
    diff = boot["a"] - boot["b"]
    ra, rb = a["trade_return"].to_numpy(float), b["trade_return"].to_numpy(float)
    t, p = stats.ttest_ind(ra, rb, equal_var=False) if len(ra) > 1 and len(rb) > 1 else (np.nan, np.nan)
    pooled = np.sqrt((np.var(ra, ddof=1) + np.var(rb, ddof=1)) / 2) if len(ra) > 1 and len(rb) > 1 else np.nan
    return {"bootstrap_p_a_better": round(float((diff > 0).mean()), 3),
            "bootstrap_median_diff_pct": round(float(np.median(diff)), 2),
            "per_trade_welch_t": None if np.isnan(t) else round(float(t), 3),
            "per_trade_welch_p": None if np.isnan(p) else round(float(p), 4),
            "per_trade_mean_diff_pct": round(float((ra.mean() - rb.mean()) * 100), 4),
            "cohens_d": None if not np.isfinite(pooled) or pooled == 0 else round(float((ra.mean() - rb.mean()) / pooled), 3),
            "effect_size_note": "bootstrap P = sign consistency; t/d = effect size. Both required."}


def run_window(frame: pd.DataFrame, pred_paths: dict, device, tag: str) -> tuple[dict, dict]:
    """Alignment rules copied from replay_omega4_6_1_greedy_val_20260706.main(): VAL prediction
    files are OOF-format and must be renamed to the non-oof prefix, and the FRAME is shrunk to the
    timestamps common to BOTH components so every arm sees an identical bar index."""
    preds = {}
    for name in retest.COMPONENTS:
        p = pd.read_csv(pred_paths[name])
        p = p.rename(columns={c: c.replace("_expertdq_oof_", "_expertdq_") for c in p.columns})
        p["timestamp"] = pd.to_datetime(p["timestamp"])
        preds[name] = p
    common = frame["timestamp"]
    for p in preds.values():
        common = common[common.isin(p["timestamp"])]
    frame = frame.loc[frame["timestamp"].isin(common)].reset_index(drop=True)
    print(json.dumps({f"{tag}_aligned_rows": int(len(frame))}), flush=True)

    comps = {}
    for name, cfg in retest.COMPONENTS.items():
        full = preds[name]
        full = full.loc[full["timestamp"].isin(frame["timestamp"])].reset_index(drop=True)
        if len(full) != len(frame):
            raise RuntimeError(f"{tag}/{name}: aligned {len(full)} != frame {len(frame)}")
        tmp = OUT_DIR / f"_aligned_{tag}_{name}.csv"
        full.to_csv(tmp, index=False)
        comps[name] = prepare_component(frame, tmp, cfg, device)
    fee, slip = omega._load_fee_slip()
    arms = {"router": {"h48qual": comps["h48qual"], "zig075": comps["zig075"]},
            "h48qual_only": {"h48qual": comps["h48qual"]},
            "zig075_only": {"zig075": comps["zig075"]}}
    res, leds = {}, {}
    for arm, sel in arms.items():
        _, led = greedy_replay(frame, sel, fee=fee, slip=slip, cost_mult=retest.COST_MULT, device=device)
        led["entry_ts"] = pd.to_datetime(led["entry_timestamp"])
        led.to_csv(OUT_DIR / f"ledger_{tag}_{arm}.csv", index=False)
        leds[arm] = led
        res[arm] = {"no_gate": metrics(led["trade_return"].to_numpy(float)),
                    "gated": metrics(gated(led, frame))}
        print(json.dumps({f"{tag}|{arm}": res[arm]["no_gate"]}), flush=True)
    return res, leds


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    contract = json.loads(CONTRACT.read_text())
    device = retest.DEVICE
    rng = np.random.default_rng(SEED)

    print("=== VAL 2025Q4 (falsification test)", flush=True)
    val_frame = gval.load_val_frame()
    val_res, val_leds = run_window(val_frame, {k: Path(v) for k, v in gval.VAL_PRED.items()}, device, "val")

    print("=== tail 2026-07-01..07-12 (report only)", flush=True)
    tail_frame = retest.load_frame_current("2026-07-01", "2026-07-12 09:00:00")
    tail_preds = {n: OOS_PRED_DIR / n / f"oos_predictions_{c['q_tag']}.csv" for n, c in retest.COMPONENTS.items()}
    tail_res, tail_leds = run_window(tail_frame, tail_preds, device, "tail")

    cmp_val = compare(val_leds["h48qual_only"], val_leds["router"], rng)

    r_cal = val_res["router"]["no_gate"]["calmar"]
    h_cal = val_res["h48qual_only"]["no_gate"]["calmar"]
    r_mdd = val_res["router"]["no_gate"]["mdd"]
    h_mdd = val_res["h48qual_only"]["no_gate"]["mdd"]
    cal_ok = bool(h_cal is not None and r_cal is not None and h_cal >= G1_CALMAR_RATIO * r_cal)
    mdd_ok = bool(h_mdd >= r_mdd - G1_MDD_SLACK_PP)
    g1 = {"rule": f"h48qual_only Calmar >= {G1_CALMAR_RATIO} x router Calmar AND MDD >= router MDD - {G1_MDD_SLACK_PP}pp (NO-GATE)",
          "router_calmar": r_cal, "h48qual_only_calmar": h_cal, "calmar_ok": cal_ok,
          "router_mdd": r_mdd, "h48qual_only_mdd": h_mdd, "mdd_ok": mdd_ok,
          "pass": bool(cal_ok and mdd_ok)}

    out = {"contract": str(CONTRACT.relative_to(ROOT)),
           "outcome_ceiling": contract["pre_registered_gates"]["G3_outcome_ceiling"]["rule"],
           "val_2025Q4": val_res, "tail_2026_07": tail_res,
           "val_h48qual_only_vs_router": cmp_val,
           "G1_falsification": g1,
           "verdict": ("PASS -> promote to a parallel ETH SHADOW (never a live swap under this contract)"
                       if g1["pass"] else "FAIL -> close the line; the 2026-08-01 conclusion stands"),
           "contaminated_window_for_reference_only": {
               "oos_2026H1_no_gate": {"router": "+77.11/-15.48/37tr", "h48qual_only": "+58.76/-11.75/25tr"},
               "oos_2026H1_gated_20260801": {"combined": "+82.53/-15.48/31tr", "h48qual_only": "+28.22/-15.43/21tr"}},
           "fresh_forward_bar_by_bar": True, "trade_ledgers_used_as_input": False,
           "saved_parent_exit_timestamps_used": False, "future_rows_used_for_entry": False}
    (OUT_DIR / "result.json").write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(json.dumps({"G1": g1, "val_compare": cmp_val, "verdict": out["verdict"]}, indent=2, ensure_ascii=False))
    print(f"wrote {OUT_DIR / 'result.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
