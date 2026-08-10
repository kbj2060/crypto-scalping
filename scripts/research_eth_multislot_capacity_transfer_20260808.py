"""ETH multi-slot (N=3) CAPACITY transfer test -- pre-registered.

Contract: docs/experiments/eth_multislot_capacity_transfer_20260808.json (written FIRST).

Transfers ONLY the capacity lever from BTC's multislot line: N concurrent position slots, each
sized at 1/N of the sidecar margin_fraction (leverage unchanged), so the total margin budget is
identical to single-slot by construction. The 1.5x margin multiplier and the czz_trend regime
overlay are deliberately NOT transferred (see the contract for why).

N is FIXED at 3 by the contract. No slot-count search is performed, so no window is spent on
selection and the multiple-comparisons debt is zero.

Equity convention (contract: metric_basis.equity_convention). Each trade contributes a
multiplicative factor
    (1 - fee*notional) * (1 + raw_exit*notional - fee*notional)
which at N=1 is algebraically identical to the incumbent greedy_replay's
`cash/entry_equity - 1`, because nothing else touches cash between that entry and exit. The final
PnL is therefore order-independent; the curve (and hence MDD) is compounded in EXIT order, which
is the realized path. G0 enforces the N=1 identity numerically.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import train_eval_omega1_2_tabm_diffusion_risk_20260603 as omega  # noqa: E402
import train_eval_omega4_2_risk_sidecar_20260622 as sidecar  # noqa: E402
import train_omega1_regime3_expert_direction_head_volpca_20260602 as hard  # noqa: E402
import retest_omega4_6_1_extended_oos_20260706 as retest  # noqa: E402
import replay_omega4_6_1_greedy_val_20260706 as gval  # noqa: E402
from replay_omega4_6_1_greedy_router_20260706 import (  # noqa: E402
    DURATION_THRESHOLD, LEVERAGE_CAP, NOTIONAL_CAP, PRIORITY, SCALE_MAP, prepare_component,
)

CONTRACT = ROOT / "docs/experiments/eth_multislot_capacity_transfer_20260808.json"
OOS_PRED_DIR = ROOT / "tmp/causal_regen_20260516/omega4_6_1_extended_oos_20260706"
OUT_DIR = ROOT / "tmp/eth_multislot_capacity_20260808"
N_SLOTS = 3
G0_EXPECTED = {"pnl": 77.11, "mdd": -15.48, "trades": 37}
G0_TOL_PP = 0.05
G2_MDD_SLACK_PP, G3_PNL_MARGIN_PP, G3_MDD_FLOOR = 3.0, 3.0, -18.5
B_BOOT, BLOCK_DAYS, SEED, R_PERM = 2000, 21, 903174, 20000


# ----------------------------------------------------------------------------- replay
@torch.no_grad()
def multislot_replay(frame: pd.DataFrame, components: dict, *, n_slots: int, fee: float, slip: float,
                     cost_mult: float, device) -> pd.DataFrame:
    arrays = {c: pd.to_numeric(frame[c], errors="raise").to_numpy(dtype=np.float64)
              for c in ("open", "high", "low", "close")}
    n = len(frame)
    fee_eff, slip_eff = float(fee) * float(cost_mult), float(slip) * float(cost_mult)

    # precompute the per-component per-bar decision surface (pure functions of `dec`, so this is
    # semantically identical to greedy_replay's in-loop recomputation, just not O(n^2))
    pre = {}
    for name, comp in components.items():
        act = omega._active(comp["dec"])
        pre[name] = {
            "active": np.asarray(act.to_numpy() if hasattr(act, "to_numpy") else act, dtype=bool),
            "side": pd.to_numeric(comp["dec"]["side"], errors="raise").to_numpy(dtype=np.int64),
            "tp": pd.to_numeric(comp["dec"]["take_profit"], errors="raise").to_numpy(dtype=np.float64),
            "sl": pd.to_numeric(comp["dec"]["stop_loss"], errors="raise").to_numpy(dtype=np.float64),
            "margin": np.asarray(comp["margin"], dtype=np.float64),
            "leverage": np.asarray(comp["leverage"], dtype=np.float64),
        }

    slots: list[dict[str, Any] | None] = [None] * int(n_slots)
    rows: list[dict[str, Any]] = []

    for i in range(0, n - 2):
        exited_this_bar = False
        for k in range(n_slots):
            s = slots[k]
            if s is None:
                continue
            comp = components[s["comp"]]
            pos, entry_price, notional = s["pos"], s["entry_price"], s["notional"]
            move = ((arrays["close"][i] * (1 - slip_eff) - entry_price) / entry_price if pos > 0
                    else (entry_price - arrays["close"][i] * (1 + slip_eff)) / entry_price)
            s["mfe"], s["mae"] = max(s["mfe"], move), min(s["mae"], move)

            reason = ""
            if s["tp"] > 0.0 and move >= s["tp"]:
                reason = "take_profit"
            elif s["sl"] > 0.0 and move <= -abs(s["sl"]):
                reason = "stop_loss"
            if not reason:
                hold = max(i - s["entry_i"], 0)
                giveback = (s["mfe"] - move) / max(abs(s["mfe"]), 1e-8) if s["mfe"] > 0.0 else 0.0
                expert = hard.EXPERT_NAMES[int(comp["route"][i])]
                prob = sidecar._predict_exit_prob_one(
                    comp["base_np"], comp["exit_runtime"], comp["pos_idx"], row_i=int(i), expert=expert,
                    pos_values=[float(pos), float(hold), float(move), float(s["mfe"]), float(s["mae"]),
                                float(np.clip(giveback, 0.0, 10.0)), float(s["tp"] - move),
                                float(move + abs(s["sl"])), float(notional), float(s["leverage"]),
                                float(notional * s["leverage"]), float(s["tp"]), float(s["sl"])],
                    device=device)
                if prob >= comp["exit_threshold"]:
                    reason = "exit_head"
            if not reason:
                continue

            exit_px = arrays["close"][i] * (1 - slip_eff if pos > 0 else 1 + slip_eff)
            raw_exit = ((exit_px - entry_price) / entry_price if pos > 0
                        else (entry_price - exit_px) / entry_price)
            factor = (1.0 - fee_eff * notional) * (1.0 + raw_exit * notional - fee_eff * notional)
            rows.append({
                "entry_signal_i": s["entry_signal_i"], "entry_i": s["entry_i"], "exit_i": int(i),
                "entry_timestamp": str(frame["timestamp"].iloc[s["entry_signal_i"]]),
                "exit_timestamp": str(frame["timestamp"].iloc[int(i)]),
                "side": int(pos), "source_component": s["comp"], "reason": reason,
                "win": int(factor > 1.0), "trade_return": float(factor - 1.0),
                "raw_exit_price_move": float(raw_exit), "mfe_price_move": float(s["mfe"]),
                "mae_price_move": float(s["mae"]), "notional": float(notional),
                "margin_fraction": float(s["margin"]), "leverage": float(s["leverage"]), "slot": int(k),
            })
            slots[k] = None
            exited_this_bar = True

        if exited_this_bar:
            continue
        free = next((k for k in range(n_slots) if slots[k] is None), None)
        if free is None:
            continue

        for name in PRIORITY:
            if name not in components:
                continue
            p = pre[name]
            side = int(p["side"][i])
            if side == 0 or not bool(p["active"][i]):
                continue
            row_margin = float(p["margin"][i])
            if row_margin <= 0.0:
                continue
            row_margin = row_margin / float(n_slots)
            scale = SCALE_MAP.get(f"{name}_{'L' if side > 0 else 'S'}", 1.0)
            row_leverage = min(float(p["leverage"][i]) * scale, LEVERAGE_CAP)
            row_notional = min(row_margin * row_leverage, NOTIONAL_CAP)
            row_leverage = row_notional / max(row_margin, 1e-12)
            if row_notional <= 0.0:
                continue
            entry_px = arrays["open"][min(i + 1, n - 1)] * (1 + slip_eff if side > 0 else 1 - slip_eff)
            slots[free] = {
                "comp": name, "pos": side, "entry_price": float(entry_px),
                "entry_i": min(i + 1, n - 1), "entry_signal_i": i,
                "margin": row_margin, "leverage": row_leverage, "notional": row_notional,
                "tp": float(p["tp"][i]), "sl": float(p["sl"][i]), "mfe": 0.0, "mae": 0.0,
            }
            break

    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------- metrics/stats
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


def quarterly(led: pd.DataFrame) -> dict:
    out = {}
    for q, g in led.groupby(led["entry_ts"].dt.to_period("Q")):
        r = g["trade_return"].to_numpy(float)
        out[str(q)] = {"pnl": round(float((np.cumprod(1.0 + r)[-1] - 1.0) * 100), 2), "trades": int(len(r))}
    return out


def compare(a: pd.DataFrame, b: pd.DataFrame, rng) -> dict:
    """Bootstrap P (sign consistency) AND Welch t / Cohen's d (effect size). Both required --
    2026-08-08 showed a t=0.32 null difference scoring bootstrap P=0.979 on this ledger family."""
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
    bf = stats.levene(ra, rb, center="median") if len(ra) > 1 and len(rb) > 1 else None
    return {"bootstrap_p_a_better": round(float((diff > 0).mean()), 3),
            "bootstrap_median_diff_pct": round(float(np.median(diff)), 2),
            "per_trade_welch_t": None if np.isnan(t) else round(float(t), 3),
            "per_trade_welch_p": None if np.isnan(p) else round(float(p), 4),
            "per_trade_mean_diff_pct": round(float((ra.mean() - rb.mean()) * 100), 4),
            "cohens_d": None if not np.isfinite(pooled) or pooled == 0 else round(float((ra.mean() - rb.mean()) / pooled), 3),
            "variance_ratio_a_over_b": round(float(np.var(ra, ddof=1) / max(np.var(rb, ddof=1), 1e-18)), 3),
            "brown_forsythe_p": None if bf is None else round(float(bf.pvalue), 4)}


def permutation_incremental(base: np.ndarray, incr: np.ndarray, rng) -> dict:
    """How special is THIS split into 'kept by N=1' vs 'admitted only by N=3'? Reassign the same
    return multiset at random and ask how often the incremental bucket looks this good."""
    if len(incr) == 0 or len(base) == 0:
        return {"percentile": None, "note": "no incremental trades"}
    pool = np.concatenate([base, incr])
    obs = float(incr.mean())
    k = len(incr)
    null = np.array([rng.permutation(pool)[:k].mean() for _ in range(R_PERM)])
    return {"observed_incremental_mean_pct": round(obs * 100, 4),
            "null_mean_pct": round(float(null.mean()) * 100, 4),
            "percentile": round(float((null < obs).mean()), 4), "R": R_PERM}


# ----------------------------------------------------------------------------- harness
def run_window(frame: pd.DataFrame, pred_paths: dict, device, tag: str, slot_counts) -> tuple[dict, dict, pd.DataFrame]:
    """Alignment rules copied from validate_eth_h48qual_alone_contract_20260808.run_window so the
    windows cannot drift: VAL prediction files are OOF-format and must be renamed, and the frame is
    shrunk to timestamps common to BOTH components so every arm sees an identical bar index."""
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
        full = preds[name].loc[preds[name]["timestamp"].isin(frame["timestamp"])].reset_index(drop=True)
        if len(full) != len(frame):
            raise RuntimeError(f"{tag}/{name}: aligned {len(full)} != frame {len(frame)}")
        tmp = OUT_DIR / f"_aligned_{tag}_{name}.csv"
        full.to_csv(tmp, index=False)
        comps[name] = prepare_component(frame, tmp, cfg, device)

    fee, slip = omega._load_fee_slip()
    res, leds = {}, {}
    for n_slots in slot_counts:
        led = multislot_replay(frame, comps, n_slots=n_slots, fee=fee, slip=slip,
                               cost_mult=retest.COST_MULT, device=device)
        led["entry_ts"] = pd.to_datetime(led["entry_timestamp"])
        led.to_csv(OUT_DIR / f"ledger_{tag}_n{n_slots}.csv", index=False)
        leds[n_slots] = led
        res[n_slots] = {"no_gate": metrics(led["trade_return"].to_numpy(float)),
                        "gated": metrics(gated(led, frame)),
                        "by_quarter": quarterly(led),
                        "source_component": led["source_component"].value_counts().to_dict()}
        print(json.dumps({f"{tag}|N={n_slots}": res[n_slots]["no_gate"]}), flush=True)
    return res, leds, frame


def incremental_split(led1: pd.DataFrame, led3: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    kept = set(led1["entry_signal_i"].tolist())
    mask = ~led3["entry_signal_i"].isin(kept)
    return (led3.loc[~mask, "trade_return"].to_numpy(float),
            led3.loc[mask, "trade_return"].to_numpy(float),
            led3.loc[mask])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["all", "g0", "val"], default="all")
    args = ap.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    device, rng = retest.DEVICE, np.random.default_rng(SEED)
    out: dict[str, Any] = {"contract": str(CONTRACT.relative_to(ROOT)), "n_slots": N_SLOTS,
                           "outcome_ceiling": contract["pre_registered_gates"]["G5_outcome_ceiling"]["rule"]}

    # ---- G0: N=1 regression against the published incumbent -------------------------------
    print("=== G0 regression: OOS N=1 must reproduce the incumbent", flush=True)
    oos_frame = retest.load_frame_current("2026-01-01", "2026-06-30")
    oos_preds = {n: OOS_PRED_DIR / n / f"oos_predictions_{c['q_tag']}.csv" for n, c in retest.COMPONENTS.items()}
    oos_res1, oos_leds1, oos_frame_al = run_window(oos_frame, oos_preds, device, "oos", [1])
    m1 = oos_res1[1]["no_gate"]
    g0 = {"expected": G0_EXPECTED, "got": m1,
          "pass": bool(abs(m1["pnl"] - G0_EXPECTED["pnl"]) <= G0_TOL_PP
                       and abs(m1["mdd"] - G0_EXPECTED["mdd"]) <= G0_TOL_PP
                       and m1["trades"] == G0_EXPECTED["trades"])}
    out["G0_regression"] = g0
    print(json.dumps({"G0": g0}, indent=2), flush=True)
    if not g0["pass"]:
        out["verdict"] = "HALT -- N=1 does not reproduce the incumbent; no N=3 number may be reported"
        (OUT_DIR / "result.json").write_text(json.dumps(out, indent=2, ensure_ascii=False))
        print(out["verdict"])
        return 1
    if args.stage == "g0":
        (OUT_DIR / "result.json").write_text(json.dumps(out, indent=2, ensure_ascii=False))
        return 0

    # ---- G1/G2: VAL 2025Q4 premise + falsification ----------------------------------------
    print("\n=== VAL 2025Q4: premise check (G1) + falsification (G2)", flush=True)
    val_frame = gval.load_val_frame()
    val_res, val_leds, _ = run_window(val_frame, {k: Path(v) for k, v in gval.VAL_PRED.items()},
                                      device, "val", [1, N_SLOTS])
    out["val_2025Q4"] = val_res

    base_r, incr_r, incr_led = incremental_split(val_leds[1], val_leds[N_SLOTS])
    g1 = {"rule": "mean per-trade return of the trades N=3 admits and N=1 does not must be > 0 on VAL",
          "incremental_trades": int(len(incr_r)),
          "incremental_mean_pct": round(float(incr_r.mean() * 100), 4) if len(incr_r) else None,
          "incremental_total_pnl_contribution_pct": (
              round(float((np.cumprod(1.0 + incr_r)[-1] - 1.0) * 100), 2) if len(incr_r) else None),
          "kept_mean_pct": round(float(base_r.mean() * 100), 4) if len(base_r) else None,
          "permutation": permutation_incremental(base_r, incr_r, rng),
          "pass": bool(len(incr_r) > 0 and incr_r.mean() > 0)}
    out["G1_premise"] = g1
    if len(incr_led):
        incr_led.to_csv(OUT_DIR / "val_incremental_trades.csv", index=False)
        out["G1_incremental_side_mix"] = incr_led["side"].value_counts().to_dict()
        out["G1_incremental_component_mix"] = incr_led["source_component"].value_counts().to_dict()
    print(json.dumps({"G1": g1}, indent=2), flush=True)

    v1, v3 = val_res[1]["no_gate"], val_res[N_SLOTS]["no_gate"]
    g2 = {"rule": f"N=3 VAL PnL >= N=1 VAL PnL AND N=3 VAL MDD >= N=1 VAL MDD - {G2_MDD_SLACK_PP}pp",
          "n1": v1, "n3": v3,
          "pnl_ok": bool(v3["pnl"] >= v1["pnl"]), "mdd_ok": bool(v3["mdd"] >= v1["mdd"] - G2_MDD_SLACK_PP)}
    g2["pass"] = bool(g2["pnl_ok"] and g2["mdd_ok"])
    out["G2_val_falsification"] = g2
    out["val_n3_vs_n1"] = compare(val_leds[N_SLOTS], val_leds[1], rng)
    print(json.dumps({"G2": g2}, indent=2), flush=True)

    if not (g1["pass"] and g2["pass"]):
        out["verdict"] = ("CLOSE -- " + ("G1 premise failed (the admitted trades lose money on the window "
                                         "the components were fit on)" if not g1["pass"] else "G2 VAL falsification failed")
                          + "; no OOS read was spent, per the contract")
        out["oos_read_spent"] = False
        (OUT_DIR / "result.json").write_text(json.dumps(out, indent=2, ensure_ascii=False))
        print("\n" + out["verdict"])
        return 0
    if args.stage == "val":
        (OUT_DIR / "result.json").write_text(json.dumps(out, indent=2, ensure_ascii=False))
        return 0

    # ---- G3/G4: single OOS look -----------------------------------------------------------
    print("\n=== OOS 2026H1: single confirmatory read (G3/G4)", flush=True)
    oos_res3, oos_leds3, _ = run_window(oos_frame, oos_preds, device, "oos", [N_SLOTS])
    o1, o3 = oos_res1[1]["no_gate"], oos_res3[N_SLOTS]["no_gate"]
    q3 = oos_res3[N_SLOTS]["by_quarter"]
    out["oos_2026H1"] = {"n1": oos_res1[1], "n3": oos_res3[N_SLOTS]}
    out["oos_read_spent"] = True

    g3 = {"rule": f"OOS PnL >= N=1 + {G3_PNL_MARGIN_PP}pp AND MDD >= {G3_MDD_FLOOR}% AND both quarters positive",
          "n1": o1, "n3": o3,
          "pnl_ok": bool(o3["pnl"] >= o1["pnl"] + G3_PNL_MARGIN_PP),
          "mdd_ok": bool(o3["mdd"] >= G3_MDD_FLOOR),
          "quarters": q3, "quarters_ok": bool(all(v["pnl"] > 0 for v in q3.values()))}
    g3["pass"] = bool(g3["pnl_ok"] and g3["mdd_ok"] and g3["quarters_ok"])
    out["G3_oos"] = g3

    cmp_oos = compare(oos_leds3[N_SLOTS], oos_leds1[1], rng)
    ob, oi, oil = incremental_split(oos_leds1[1], oos_leds3[N_SLOTS])
    perm = permutation_incremental(ob, oi, rng)
    gate = contract["selection"]["effect_size_gate"]
    g4 = {"rule": gate, "compare": cmp_oos, "incremental_permutation": perm,
          "t_ok": bool(cmp_oos["per_trade_welch_t"] is not None
                       and abs(cmp_oos["per_trade_welch_t"]) >= gate["min_abs_t"]),
          "perm_ok": bool(perm.get("percentile") is not None
                          and perm["percentile"] >= gate["min_permutation_percentile"])}
    g4["pass"] = bool(g4["t_ok"] and g4["perm_ok"])
    out["G4_effect_size"] = g4
    if len(oil):
        oil.to_csv(OUT_DIR / "oos_incremental_trades.csv", index=False)

    passed = bool(g3["pass"] and g4["pass"])
    out["verdict"] = ("PASS -> stand up a parallel ETH multi-slot SHADOW (never a live swap under this "
                      "contract; note the one-way-mode blocker)" if passed else
                      "CLOSE -- the capacity lever does not transfer to ETH. No re-tuning of slot count, "
                      "sizing or slot policy on OOS.")
    out["fresh_forward"] = contract["fresh_forward"]
    (OUT_DIR / "result.json").write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(json.dumps({"G3": g3, "G4": g4, "verdict": out["verdict"]}, indent=2, ensure_ascii=False))
    print(f"wrote {OUT_DIR / 'result.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
