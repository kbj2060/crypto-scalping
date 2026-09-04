import sys, json
sys.path.insert(0, "scripts")
import sweep_portfolio_prealloc_cap_shares_fresh_20260901 as sw

def main():
    new_end = sw._compute_new_end()
    orig = sw.eth_retest.load_frame_current
    sw.eth_retest.load_frame_current = lambda start, end: orig(start, new_end)
    try:
        device = sw.eth_retest.DEVICE
        sw.native.DURATION_THRESHOLDS = {k: -999.0 for k in sw.native.DURATION_THRESHOLDS}
        world_val = sw.native._build_world("validation", device)
        world_oos = sw.native._build_world("oos", device)
        points = {
            "cap1.5_shares503020(baseline shares, best cap)": (1.5, sw.SHARE_GRID["50_30_20"]),
            "cap3.0_shares333333(baseline cap, best shares)": (3.0, sw.SHARE_GRID["33_33_33"]),
            "cap1.5_shares333333(COMBINED best-of-both)": (1.5, sw.SHARE_GRID["33_33_33"]),
        }
        results = {}
        for label, (cap, shares) in points.items():
            print(f"point {label}", flush=True)
            results[label] = sw._run_point(world_val, world_oos, device, cap=cap, shares=shares)
    finally:
        sw.eth_retest.load_frame_current = orig

    for label, d in results.items():
        m = d["fresh_window"]["metrics"]["portfolio"]
        mv = d["validation"]["metrics"]["portfolio"]
        mo = d["oos_extended"]["metrics"]["portfolio"]
        print(f"{label}: fresh PnL={m['pnl']:.2f}% MDD={m['mdd']:.2f}% WR={m['wr']:.1%} n={m['trades']} | oos_ext PnL={mo['pnl']:.2f}% MDD={mo['mdd']:.2f}% | val PnL={mv['pnl']:.2f}% MDD={mv['mdd']:.2f}%")

    out = {}
    for label, d in results.items():
        out[label] = {k: v["metrics"]["portfolio"] for k, v in d.items()}
    with open("tmp/causal_regen_20260516/sweep_portfolio_prealloc_cap_shares_fresh_20260901/combined_check.json", "w") as f:
        json.dump(out, f, indent=2, default=sw._json_default)

if __name__ == "__main__":
    main()
