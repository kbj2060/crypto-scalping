#!/usr/bin/env python3
"""J — 배포 **h48qual** 부모를 zig075 와 같은 창·같은 절차로 잰다(거래량 포함).

두 부모는 **방향 라벨이 동일한 zigzag** 이고 다른 건 품질 머리의 타깃과 임계값이다:
  zig075   quality_mode=same_as_direction · q=0.75
  h48qual  quality_mode=quality_label_action · q=0.50
           품질 타깃 = `sltp_h48_conservative_padded_to_zigzag_timestamps`
           = 48봉(**4시간**) ATR 삼중배리어, TP=max(0.6%, 1.2·vol) · SL=max(0.4%, 0.8·vol)
라이브 라우터 우선순위는 **h48qual → zig075** 다(먼저 발화한 쪽이 이긴다).
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np, pandas as pd, torch

ROOT = Path.home() / "crypto-scalping"
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import train_eval_omega1_2_tabm_3head_20260603 as tabm  # noqa: E402
import train_eval_omega461_parent_zig075_longwindow_20260917 as E  # noqa: E402
import research_omega461_parent_quality_calibration_and_bias_20260917 as F  # noqa: E402
import research_omega461_parent_pnl_concentration_20260917 as G  # noqa: E402

CONTRACT = (ROOT / "tmp/causal_regen_20260516"
            / "omega4_6_2_cap220_short_boost125_time_stop120h_20260630/runtime_contract.json")


def main() -> int:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    df, base_cols = E.load()
    vm = (df.timestamp >= E.VAL[0]) & (df.timestamp <= E.VAL[1] + " 23:59:59")
    val = df[vm].reset_index(drop=True)
    yv = pd.to_numeric(val["zigzag_action"]).to_numpy(np.int64)
    xv_raw = tabm._base_input(val, base_cols)
    ev_expert = tabm._route_probs(val).argmax(1)
    vdays = val.timestamp.dt.floor("D").to_numpy()
    fwd1 = val["fwd_1h_bp"].to_numpy(np.float64)
    fwd4 = val["fwd_4h_bp"].to_numpy(np.float64)

    c = json.loads(CONTRACT.read_text())["components"]
    out, sides = [], {}
    for alias in ("h48qual", "zig075"):
        raw = c[alias]
        rep = json.loads(Path(raw["report"]).read_text())
        pdir = Path(rep["risk_model"]["precomputed_prediction_dir"])
        b = torch.load(pdir / "true_3head_tabm_bundle.pt", map_location="cpu", weights_only=False)
        qth = float(raw["quality_threshold"])
        D = np.zeros((len(val), 3)); Q = np.zeros((len(val), 3))
        for ei, ename in enumerate(("bull", "bear", "chop")):
            pay = dict(b["models"][ename])
            assert list(xv_raw.columns) == list(pay["input_columns"]), f"{alias} 입력 열 불일치"
            m = tabm.ThreeHeadTabM(int(pay["n_features"]),
                                   cfg=tabm.ThreeHeadConfig(**dict(pay["config"]))).to(device)
            m.load_state_dict(pay["state_dict"]); m.eval()
            sel = ev_expert == ei
            if sel.any():
                D[sel], Q[sel] = E.heads(m, tabm._standardize_apply(xv_raw[sel], dict(pay["scaler"])), device)
        _, side = F.gate(D, Q, qth)
        sides[alias] = side
        r = F.bias_report(alias, side, fwd1, yv, vdays)
        r.update({k: v for k, v in G.day_stats(side, fwd1, vdays).items() if not k.startswith("_")})
        ok4 = (side != 0) & np.isfinite(fwd4)
        r["gross_bp_4h"] = float((side[ok4] * fwd4[ok4]).mean())
        r.update({"alias": alias, "q_threshold": qth,
                  "pass_rate": float((side != 0).mean()),
                  "bars_per_day": float((side != 0).sum() / len(np.unique(vdays)))})
        out.append(r)

    # 라이브 라우터: h48qual 이 발화하면 그걸 쓰고, 아니면 zig075
    h, z = sides["h48qual"], sides["zig075"]
    routed = np.where(h != 0, h, z)
    r = F.bias_report("router(h48qual→zig075)", routed, fwd1, yv, vdays)
    r.update({k: v for k, v in G.day_stats(routed, fwd1, vdays).items() if not k.startswith("_")})
    r.update({"alias": "router", "q_threshold": float("nan"),
              "pass_rate": float((routed != 0).mean()),
              "bars_per_day": float((routed != 0).sum() / len(np.unique(vdays))),
              "overlap_both_fire": int(((h != 0) & (z != 0)).sum()),
              "disagree_side": int(((h != 0) & (z != 0) & (h != z)).sum()),
              "h48_only": int(((h != 0) & (z == 0)).sum()),
              "zig_only": int(((h == 0) & (z != 0)).sum())})
    out.append(r)

    print(f"\n{'모델':<24}{'q':>6}{'통과':>7}{'통과율':>8}{'일평균':>8}{'롱%':>7}"
          f"{'총bp':>8}{'중앙':>7}{'양수일':>8}{'4h총bp':>9}")
    for r in out:
        print(f"{r['name']:<24}{r['q_threshold']:>6.2f}{r['n']:>7,}{r['pass_rate']:>8.3f}"
              f"{r['bars_per_day']:>8.1f}{r['long_share']*100:>6.1f}%{r['gross_bp']:>8.2f}"
              f"{r['median_bp']:>7.2f}{r['pos_day_share']*100:>7.1f}%"
              f"{r.get('gross_bp_4h', float('nan')):>9.2f}")
    print(f"\n{'모델':<24}{'모델−롱':>10}{'CI':>22}{'모델−숏':>10}{'CI':>22}")
    for r in out:
        el, es = r["excess_vs_long"], r["excess_vs_short"]
        print(f"{r['name']:<24}{el['bp']:>+10.2f}  [{el['ci95'][0]:+7.2f},{el['ci95'][1]:+7.2f}]"
              f"{es['bp']:>+10.2f}  [{es['ci95'][0]:+7.2f},{es['ci95'][1]:+7.2f}]")
    rr = out[-1]
    print(f"\n두 부모 동시 발화 {rr['overlap_both_fire']:,}봉 · 그 중 **측면 불일치** "
          f"{rr['disagree_side']:,}봉 · h48qual 단독 {rr['h48_only']:,} · zig075 단독 {rr['zig_only']:,}")
    (E.OUT / "stageJ_h48qual_benchmark.json").write_text(json.dumps(out, indent=2, default=float))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
