#!/usr/bin/env python3
"""V자 급등락 **경제성 게이트 재실행 — HGB 서빙 확률로** (2026-09-19).

2026-09-19 라이브가 TabPFN -> 5시드 HGB 로 바뀌었다. 배포 임계값을 정당화한 시험
(`backtest_eth_v_rebound_every_bar_tabpfn_costgate_threshold_20260901.py`, f2eb2377 에서 삭제)은
**TabPFN 확률** 위에서 돌았다. 그 스크립트 자신의 문제의식이 그대로 여기 적용된다:
    "분류 AUC는 둘이 거의 같았지만, 경제성은 AUC가 아니라 **상위 확률 꼬리의 순위**가
     결정한다 -- 같은 AUC라도 어떤 봉을 호출하느냐가 다르면 bp가 달라질 수 있다."
AUC 동률(.7027 vs .7025)은 이 질문에 답이 **안 된다**. 그래서 다시 돌린다.

## 재사용 — 시뮬레이터는 한 글자도 안 고친다
`backtest_eth_v_rebound_every_bar_trailing_costgate_20260901.py` 를 복원해 `run_grid`/`pack`/
`self_check`/`simulate_trailing_*` 와 격자(SL 8 x ARM 6 x TRAIL 5 = 240)·비용 10bp·전방 200봉을
**그대로** 쓴다. 매 실행마다 scalar-vs-vec 자체점검을 돌려 0 불일치를 확인한다.

## 비교 가능성 — 임계값을 «호출률»로 맞춘다
확률값은 모델이 바뀌면 의미가 달라진다. 배포 리포트의 각 임계값 **호출률**을 재현하는 HGB
분위를 임계값으로 쓴다. 그러면 표의 각 줄이 배포판의 같은 줄과 **같은 호출 건수**를 갖는다.

## 두 가지 기준을 다 낸다
- `labeled`  : 3상태 라벨이 붙은 행만(전체의 ~53%) -- 2026-09-01 게이트와 **같은 기준**
- `all_bars` : 모든 봉 -- 라이브가 실제로 호출하는 기준. 2026-09-01 의 경제성 주장이
               철회된 이유가 정확히 이 격차다(커밋 aac1805c).

⚠️VAL/OOS 만. HOLDOUT(2026-04-01 이후) 미터치. 선정은 VAL 만, OOS 는 선정 후 1회 평가.
⚠️방향 뒤집기 대조군은 **그리드 전체**에 건다(단일 config 만 뒤집으면 오판 --
  feedback_trailing_stop_low_arm_noise_harvest_artifact).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

CODE = Path(__file__).resolve().parents[1]
for _p in (CODE, CODE / "scripts"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import research_eth_v_rebound_hgb_vs_tabpfn_20260919 as X                    # noqa: E402
import backtest_eth_v_rebound_every_bar_trailing_costgate_20260901 as BT     # noqa: E402
import live_eth_sweep_v_rebound_signal_20260829 as LIVE                      # noqa: E402

OUT = CODE / "tmp/v_rebound_hgb_costgate_20260919"
DEPLOYED = X.ROOT / ("data/research/eth_v_rebound_every_bar_tabpfn_costgate_20260901/report.json")
OOS_END = pd.Timestamp("2026-04-01", tz="UTC")     # HOLDOUT 경계 -- 여기서 자른다


def log(m):
    print(m, flush=True)


def called_rows(ts, side_dn, atr, proba, label, kl) -> pd.DataFrame:
    """임계값을 넘은 행에 전방 200봉 OHLC 를 붙인다 (09-01 `build_called` 와 같은 형식).
    🔴tz_localize(None) 이 필수다 -- tz-aware 의 to_numpy() 는 datetime64 가 아니라 Timestamp
      객체 배열이라 조회가 조용히 0건이 된다(09-01 첫 실행에서 실제로 전 임계값 0건이었다)."""
    pos = {t: i for i, t in enumerate(kl["timestamp"].to_numpy())}
    o, h, l, c = (kl[x].to_numpy() for x in ("open", "high", "low", "close"))
    n, rows = len(kl), []
    for t, dn, a, p, y in zip(ts, side_dn, atr, proba, label):
        i = pos.get(np.datetime64(t.tz_localize(None)))
        if i is None or i + BT.FORWARD_BARS + 1 >= n:
            continue
        rows.append({"side": "long" if dn else "short", "atr": float(a),
                     "entry_price": float(o[i + 1]), "model_proba": float(p), "label": float(y),
                     "fwd_open": o[i + 1:i + 1 + BT.FORWARD_BARS],
                     "fwd_high": h[i + 1:i + 1 + BT.FORWARD_BARS],
                     "fwd_low": l[i + 1:i + 1 + BT.FORWARD_BARS],
                     "fwd_close": c[i + 1:i + 1 + BT.FORWARD_BARS]})
    return pd.DataFrame(rows)


def arm_dist(grid) -> dict:
    d = {}
    for g in grid:
        if g["opt_bp"] > 0 and g["pess_bp"] > 0:
            d[str(g["arm"])] = d.get(str(g["arm"]), 0) + 1
    return d


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    dep = json.loads(DEPLOYED.read_text())
    dep_thr, dep_econ = dep["threshold_table"], dep["economics"]

    D, win, agree = X.assemble("live")
    import joblib
    art_dir = X.ROOT / "data/live/eth_v_rebound_hgb_artifact"
    models = joblib.load(art_dir / "model.joblib")
    meta = json.loads((art_dir / "meta.json").read_text())
    assert meta["features"] == LIVE.FEATURES, "아티팩트 피쳐 형상이 라이브와 다르다"
    P = np.mean([m.predict_proba(D[LIVE.FEATURES].to_numpy(float))[:, 1] for m in models], axis=0)
    log(f"[아티팩트] {meta['rule_id']} · 배포 임계값 {meta['proba_threshold']:.4f}")

    kl = X.load_klines_full()[["timestamp", "open", "high", "low", "close"]].copy()
    kl["timestamp"] = kl["timestamp"].dt.tz_localize(None)

    y3 = D.y3.to_numpy()
    lab = np.isfinite(y3)
    assert D.timestamp[win["OOS"]].max() < OOS_END, "HOLDOUT 누출"

    # ── 임계값을 배포판의 «호출률» 로 맞춘다 ──────────────────────────────────
    val_lab = win["VAL"] & lab
    thr_map = {}
    for k, row in dep_thr.items():
        rate = row["VAL"]["call_rate"]
        thr_map[k] = float(np.quantile(P[val_lab], 1.0 - rate))

    log("\n=== 임계값 스윕 (호출률 맞춤 · labeled 기준) ===")
    log(f"  {'배포thr':>7} {'HGBthr':>7} | {'창':>4} {'호출':>6} {'호출률':>7} {'정밀도':>7} "
        f"{'lift':>6} | {'배포 정밀도':>10} {'배포 lift':>9}")
    table = {}
    for k, thr in thr_map.items():
        table[k] = {"hgb_threshold": thr}
        for nm in ("VAL", "OOS"):
            m = win[nm] & lab
            called = m & (P >= thr)
            base = float(y3[m].mean())
            prec = float(y3[called].mean()) if called.sum() else float("nan")
            table[k][nm] = {"n_called": int(called.sum()),
                            "call_rate": round(float(called.sum() / m.sum()), 4),
                            "precision": round(prec, 4), "lift_vs_base": round(prec / base, 3),
                            "base": round(base, 4),
                            "deployed": dep_thr[k][nm]}
            r, dp = table[k][nm], dep_thr[k][nm]
            log(f"  {k:>7} {thr:>7.4f} | {nm:>4} {r['n_called']:>6,} {r['call_rate']*100:>6.2f}% "
                f"{r['precision']:>7.4f} {r['lift_vs_base']:>6.2f} | {dp['precision']:>10.4f} "
                f"{dp['lift_vs_base']:>9.2f}")

    # ── 경제성 ────────────────────────────────────────────────────────────────
    log("\n=== 경제성 (선정=VAL만 · OOS는 선정 후 1회) ===")
    econ = {}
    for k, thr in thr_map.items():
        by = {}
        for basis, bmask in (("labeled", lab), ("all_bars", np.ones(len(D), bool))):
            per = {}
            for nm in ("VAL", "OOS"):
                m = win[nm] & bmask & (P >= thr)
                if m.sum() < 50:
                    per[nm] = {"n": int(m.sum()), "insufficient": True}
                    continue
                df = called_rows(D.timestamp[m], D.is_downside.to_numpy()[m] == 1,
                                 D.atr.to_numpy()[m], P[m], np.nan_to_num(y3[m]), kl)
                if len(df) < 50:
                    per[nm] = {"n": int(len(df)), "insufficient": True}
                    continue
                if basis == "labeled" and nm == "VAL" and k == "0.60":
                    sc = BT.self_check(df)
                    log(f"  [자체점검] scalar-vs-vec 불일치 {sc['n_mismatches']} "
                        f"(행 {sc['n_rows']} x 설정 {sc['n_configs']})")
                    assert sc["n_mismatches"] == 0, "시뮬레이터 vec/scalar 불일치"
                per[nm] = {"n": int(len(df)), "grid": BT.run_grid(df, flip=False),
                           "flip_grid": BT.run_grid(df, flip=True)}
            by[basis] = per

        econ[k] = {"hgb_threshold": thr}
        for basis, per in by.items():
            if per.get("VAL", {}).get("insufficient", True) and "grid" not in per.get("VAL", {}):
                continue
            vg = per["VAL"]["grid"]
            ok = [g for g in vg if g["opt_bp"] > 0 and g["pess_bp"] > 0]
            best = max(ok or vg, key=lambda g: g["pess_bp"])
            key = (best["sl"], best["arm"], best["trail"])
            sel = {"selected_on": "VAL", "config": dict(zip(("sl", "arm", "trail"), key)),
                   "val_passes_both_orderings": bool(ok)}
            for nm in ("VAL", "OOS"):
                if "grid" not in per.get(nm, {}):
                    continue
                g = next(x for x in per[nm]["grid"] if (x["sl"], x["arm"], x["trail"]) == key)
                f = next(x for x in per[nm]["flip_grid"] if (x["sl"], x["arm"], x["trail"]) == key)
                sel[nm] = {"n": per[nm]["n"], "opt_bp": round(g["opt_bp"], 2),
                           "pess_bp": round(g["pess_bp"], 2), "win_rate": round(g["win_rate"], 4),
                           "flip_opt_bp": round(f["opt_bp"], 2),
                           "flip_pess_bp": round(f["pess_bp"], 2),
                           "grid_profitable": {
                               "normal": sum(1 for x in per[nm]["grid"]
                                             if x["opt_bp"] > 0 and x["pess_bp"] > 0),
                               "flipped": sum(1 for x in per[nm]["flip_grid"]
                                              if x["opt_bp"] > 0 and x["pess_bp"] > 0),
                               "total": len(per[nm]["grid"])},
                           "arm_dist_normal": arm_dist(per[nm]["grid"]),
                           "arm_dist_flipped": arm_dist(per[nm]["flip_grid"])}
            econ[k][basis] = sel

        e = econ[k]
        for basis in ("labeled", "all_bars"):
            if basis not in e:
                continue
            s = e[basis]
            cfg = s["config"]
            parts = []
            for nm in ("VAL", "OOS"):
                if nm not in s:
                    continue
                r = s[nm]
                gp = r["grid_profitable"]
                parts.append(f"{nm} n={r['n']:>5,} {r['pess_bp']:>+7.2f}bp "
                             f"(뒤집기 {r['flip_pess_bp']:>+7.2f}) 승률 {r['win_rate']:.3f} "
                             f"격자 정{gp['normal']:>3}/뒤{gp['flipped']:>3}")
            log(f"  thr={k} [{basis:>8}] SL{cfg['sl']}/ARM{cfg['arm']}/TR{cfg['trail']}  "
                + " | ".join(parts))
        if "0.60" == k:
            d = dep_econ[k]["selection"]
            log(f"  thr=0.60 [ 배포TabPFN] SL{d['config']['sl']}/ARM{d['config']['arm']}/"
                f"TR{d['config']['trail']}  VAL n={d['VAL']['n']:>5,} {d['VAL']['pess_bp']:>+7.2f}bp "
                f"(뒤집기 {d['VAL']['flip_pess_bp']:>+7.2f}) 승률 {d['VAL']['win_rate']:.3f} "
                f"격자 정{d['VAL_grid_profitable']['normal']:>3}/뒤{d['VAL_grid_profitable']['flipped']:>3}"
                f" | OOS n={d['OOS']['n']:>5,} {d['OOS']['pess_bp']:>+7.2f}bp "
                f"(뒤집기 {d['OOS']['flip_pess_bp']:>+7.2f}) 승률 {d['OOS']['win_rate']:.3f} "
                f"격자 정{d['OOS_grid_profitable']['normal']:>3}/뒤{d['OOS_grid_profitable']['flipped']:>3}")

    (OUT / "report.json").write_text(json.dumps({
        "signal": "v_rebound_hgb_costgate", "asset": "ETHUSDT",
        "fresh_forward_bar_by_bar": True, "trade_ledgers_used_as_input": False,
        "saved_parent_exit_timestamps_used": False, "future_rows_used_for_entry": False,
        "model": meta["rule_id"], "deployed_threshold": meta["proba_threshold"],
        "label_parity_vs_frozen_context": agree,
        "cost_bp": BT.STANDARD_COST_BP, "forward_bars": BT.FORWARD_BARS,
        "grid": {"sl": list(BT.SL_GRID), "arm": list(BT.ARM_GRID), "trail": list(BT.TRAIL_GRID)},
        "holdout_touched": False,
        "threshold_table": table, "economics": econ,
        "deployed_reference": str(DEPLOYED.relative_to(X.ROOT)),
    }, ensure_ascii=False, indent=1, default=float))
    log(f"\n산출물 {OUT / 'report.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
