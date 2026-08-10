"""Re-measure the ETH notional multiplier UNDER THE LIVE PORTFOLIO CAP.

Contract: docs/experiments/eth_notional_multiplier_capped_remeasure_20260808.json (written FIRST).

Only 1.0x and 1.5x are decidable outcomes; the rest of the grid is reported for shape. The burden
is on 1.5x (the active intervention) to justify itself against the 1.0x default.

Importing rerun_portfolio_sweep_current_data_20260808 applies its `_eth_components` alignment
monkeypatch at import time (the oos branch of the original lacks frame/prediction alignment).
"""
from __future__ import annotations

import argparse
import json
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import rerun_portfolio_sweep_current_data_20260808 as _patch  # noqa: E402,F401  (applies alignment)
import replay_portfolio_rl_gate_2action_native_20260708 as native  # noqa: E402
import replay_portfolio_concurrent_3asset_native_20260712 as v4  # noqa: E402

CONTRACT = ROOT / "docs/experiments/eth_notional_multiplier_capped_remeasure_20260808.json"
OUT = ROOT / "tmp/eth_multiplier_capped_20260808"
Q1_CUTOFF = pd.Timestamp("2026-04-01")
SHARES = {"eth": 0.5, "btc": 0.3, "sol": 0.2}
CAP = 3.0
GRID = [1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0]
DECIDABLE = (1.0, 1.5)

# today's uncapped run (tmp/portfolio_sweep_recheck_20260808/report.json), gate off
G0_EXPECTED = {
    1.0: {"validation": (28.28, -30.94), "oos_extended": (102.84, -31.20), "oos_frozen_q1_2026": (151.72, -22.34)},
    1.5: {"validation": (22.90, -36.71), "oos_extended": (64.66, -37.82), "oos_frozen_q1_2026": (113.47, -37.10)},
}
G0_TOL = 0.05
G1_MDD_SLACK_PP = 2.0
R_PERM = 20000
SEED = 20260808


@contextmanager
def gate_off():
    orig = dict(native.DURATION_THRESHOLDS)
    native.DURATION_THRESHOLDS = {k: -999.0 for k in orig}
    try:
        yield
    finally:
        native.DURATION_THRESHOLDS = orig


def run_cfg(worlds: dict, *, cap: float | None, mult: float) -> dict[str, Any]:
    mults = {"eth": float(mult), "btc": 1.0, "sol": 1.0}
    cap_mode = "prealloc" if cap is not None else "scale"
    out: dict[str, Any] = {}
    for split, world, cutoff in (
        ("validation", worlds["val"], None),
        ("oos_extended", worlds["oos"], None),
        ("oos_frozen_q1_2026", worlds["oos"], Q1_CUTOFF),
    ):
        m, ledger, _, _ = v4._replay_concurrent(
            world, device=worlds["device"], total_notional_cap=cap, cap_mode=cap_mode,
            asset_shares=SHARES, asset_notional_multipliers=mults, entry_cutoff=cutoff,
        )
        pm = m["portfolio"]
        out[split] = {"pnl": round(float(pm["pnl"]), 2), "mdd": round(float(pm["mdd"]), 2),
                      "mtm_mdd": round(float(pm["mark_to_market_mdd"]), 2), "trades": int(pm["trades"]),
                      "ledger": ledger}
    return out


def monthly(ledger: pd.DataFrame) -> pd.Series:
    if not len(ledger):
        return pd.Series(dtype=float)
    d = ledger.copy()
    d["m"] = pd.to_datetime(d["entry_timestamp"]).dt.to_period("M")
    return d.groupby("m")["trade_return"].apply(lambda r: (np.cumprod(1.0 + r.to_numpy(float))[-1] - 1.0) * 100)


def effect_size(base: pd.Series, incumbent: pd.Series, rng) -> dict:
    """1.0x minus 1.5x, on the MONTHLY difference series (see contract G3.instrument_note:
    a per-trade t is degenerate for a deterministic sizing scale)."""
    idx = base.index.union(incumbent.index)
    diff = (base.reindex(idx).fillna(0.0) - incumbent.reindex(idx).fillna(0.0)).to_numpy(float)
    if len(diff) < 2:
        return {"n_months": int(len(diff)), "welch_t": None, "note": "too few months"}
    t, p = stats.ttest_1samp(diff, 0.0)
    null = np.array([(diff * rng.choice([-1.0, 1.0], size=len(diff))).mean() for _ in range(R_PERM)])
    obs = float(diff.mean())
    return {"n_months": int(len(diff)),
            "monthly_diff_pct": [round(float(x), 2) for x in diff],
            "mean_diff_pct": round(obs, 3),
            "welch_t": round(float(t), 3), "p": round(float(p), 4),
            "months_base_better": int((diff > 0).sum()),
            "sign_flip_permutation_percentile": round(float((null < obs).mean()), 4), "R": R_PERM}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["g0", "all"], default="all")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    rng = np.random.default_rng(SEED)
    device = native.torch.device("cpu") if not hasattr(native, "DEVICE") else native.DEVICE

    print("stage=build_world", flush=True)
    worlds = {"val": native._build_world("validation", device),
              "oos": native._build_world("oos", device), "device": device}

    res: dict[str, Any] = {"contract": str(CONTRACT.relative_to(ROOT)),
                           "config": {"total_notional_cap": CAP, "cap_mode": "prealloc",
                                      "asset_shares": SHARES, "duration_gate": "off"},
                           "declared_gaps": contract["declared_gaps_between_this_replay_and_live"],
                           "outcome_ceiling": contract["pre_registered_gates"]["G4_outcome_ceiling"]["rule"]}

    # ---------------- G0: uncapped regression ----------------
    print("\n=== G0: uncapped regression (1.0x, 1.5x) ===", flush=True)
    unc: dict[float, Any] = {}
    fails = []
    with gate_off():
        for mult in DECIDABLE:
            unc[mult] = run_cfg(worlds, cap=None, mult=mult)
            for split, (e_pnl, e_mdd) in G0_EXPECTED[mult].items():
                got = unc[mult][split]
                ok = abs(got["pnl"] - e_pnl) <= G0_TOL and abs(got["mdd"] - e_mdd) <= G0_TOL
                print(f"  {mult}x {split:20s} got={got['pnl']:8.2f}/{got['mdd']:7.2f} "
                      f"expected={e_pnl:8.2f}/{e_mdd:7.2f} {'OK' if ok else 'FAIL'}", flush=True)
                if not ok:
                    fails.append(f"{mult}x/{split}: {got['pnl']}/{got['mdd']} vs {e_pnl}/{e_mdd}")
    res["G0_regression"] = {"pass": not fails, "failures": fails,
                            "uncapped": {str(m): {s: {k: v for k, v in d.items() if k != "ledger"}
                                                  for s, d in unc[m].items()} for m in DECIDABLE}}
    if fails:
        res["verdict"] = "HALT -- uncapped regression failed; the capped run cannot be trusted"
        (OUT / "result.json").write_text(json.dumps(res, indent=2, ensure_ascii=False))
        print("\n" + res["verdict"])
        return 1
    print("  G0 PASS", flush=True)
    if args.stage == "g0":
        (OUT / "result.json").write_text(json.dumps(res, indent=2, ensure_ascii=False))
        return 0

    # ---------------- capped grid ----------------
    print(f"\n=== capped grid (cap={CAP}, prealloc, eth50/btc30/sol20, gate off) ===", flush=True)
    capped: dict[float, Any] = {}
    with gate_off():
        for mult in GRID:
            capped[mult] = run_cfg(worlds, cap=CAP, mult=mult)
            for split in ("validation", "oos_extended", "oos_frozen_q1_2026"):
                d = capped[mult][split]
                print(f"  {mult:.2f}x {split:20s} pnl={d['pnl']:8.2f}% mdd={d['mdd']:8.2f}% trades={d['trades']}", flush=True)
    res["capped_grid"] = {f"{m:.2f}": {s: {k: v for k, v in d.items() if k != "ledger"}
                                       for s, d in capped[m].items()} for m in GRID}

    # ---------------- G1: dominance ----------------
    splits = ("validation", "oos_extended", "oos_frozen_q1_2026")
    pnl_wins = sum(1 for s in splits if capped[1.5][s]["pnl"] > capped[1.0][s]["pnl"])
    mdd_ok = all(capped[1.5][s]["mdd"] >= capped[1.0][s]["mdd"] - G1_MDD_SLACK_PP for s in splits)
    g1 = {"rule": f"retain 1.5x only if it beats 1.0x on PnL in >=2/3 splits AND MDD not worse than 1.0x by >{G1_MDD_SLACK_PP}pp on any split",
          "per_split": {s: {"pnl_1.0x": capped[1.0][s]["pnl"], "pnl_1.5x": capped[1.5][s]["pnl"],
                            "mdd_1.0x": capped[1.0][s]["mdd"], "mdd_1.5x": capped[1.5][s]["mdd"]} for s in splits},
          "pnl_splits_won_by_1.5x": pnl_wins, "mdd_ok": mdd_ok,
          "retain_1.5x": bool(pnl_wins >= 2 and mdd_ok)}
    res["G1_capped_dominance"] = g1

    # ---------------- G2: convergence ----------------
    g2 = {}
    for s in splits:
        cap_gap = capped[1.5][s]["pnl"] - capped[1.0][s]["pnl"]
        unc_gap = unc[1.5][s]["pnl"] - unc[1.0][s]["pnl"]
        g2[s] = {"uncapped_gap_pp": round(unc_gap, 2), "capped_gap_pp": round(cap_gap, 2),
                 "attenuation_pct": (None if abs(unc_gap) < 1e-9
                                     else round((1 - abs(cap_gap) / abs(unc_gap)) * 100, 1))}
    res["G2_cap_attenuation"] = g2

    # ---------------- G3: effect size on monthly diffs ----------------
    g3 = {}
    for s in splits:
        g3[s] = effect_size(monthly(capped[1.0][s]["ledger"]), monthly(capped[1.5][s]["ledger"]), rng)
    gate = contract["selection"]["effect_size_gate"]
    oos = g3["oos_extended"]
    g3["gate"] = {"spec": gate,
                  "t_ok": bool(oos.get("welch_t") is not None and abs(oos["welch_t"]) >= gate["min_abs_t"]),
                  "perm_ok": bool(oos.get("sign_flip_permutation_percentile") is not None
                                  and oos["sign_flip_permutation_percentile"] >= gate["min_permutation_percentile"])}
    res["G3_effect_size"] = g3

    # ---------------- verdict ----------------
    strong = bool(g3["gate"]["t_ok"] and g3["gate"]["perm_ok"])
    small = all(abs(v["capped_gap_pp"]) < 3.0 for v in g2.values())
    if g1["retain_1.5x"]:
        verdict = "RETAIN 1.5x -- it clears its own dominance bar under the live cap."
    elif small:
        verdict = ("REVERT RECOMMENDED, but state it honestly as 'the flag barely matters under the live cap' "
                   "(capped gaps all <3pp) rather than as '1.0x is better'. Low urgency.")
    else:
        verdict = ("REVERT to 1.0x RECOMMENDED -- 1.5x fails its dominance bar under the live cap and the gap is material."
                   + ("" if strong else " NOTE: the monthly effect-size gate did NOT clear, so this is a "
                                        "'the intervention is unsupported' conclusion, not a 'reverting will earn X' claim."))
    res["verdict"] = verdict
    res["reminder"] = "This script changes NO live configuration. The .env edit is the user's decision."
    (OUT / "result.json").write_text(json.dumps(res, indent=2, ensure_ascii=False))
    for m in DECIDABLE:
        for s in splits:
            capped[m][s]["ledger"].to_csv(OUT / f"ledger_capped_{str(m).replace('.','p')}_{s}.csv", index=False)
    print("\n=== G1 ===\n" + json.dumps(g1, indent=2))
    print("\n=== G2 (cap attenuation) ===\n" + json.dumps(g2, indent=2))
    print("\n=== G3 (monthly effect size) ===\n" + json.dumps(g3, indent=2, ensure_ascii=False))
    print("\n=== VERDICT ===\n" + verdict)
    print(f"\nwrote {OUT / 'result.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
