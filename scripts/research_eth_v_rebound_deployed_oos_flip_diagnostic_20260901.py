#!/usr/bin/env python3
"""배포 중인 V자반등 설정의 OOS 방향성 진단 -- "뒤집기 우세가 한 셀뿐인가, 구간 전체인가".

## 왜

Stage 3에서 **배포 중인 설정**(K=1.5/T=0.20/FULL=12, thr 0.60)이 OOS 전체 봉에서 -0.32bp,
그런데 **방향을 뒤집으면 +4.88bp**로 나왔다. VAL에서는 뒤집기가 -2.94bp(방향이 진짜)였는데
OOS에서 무너진 것이다. 다만 그건 VAL에서 고른 **셀 하나**(SL=4.0/ARM=1.5/Trail=0.1)의 결과라,
"운 나쁜 한 셀"과 "체계적"이 아직 안 갈린다.

이 스크립트는 **선정이 아니라 진단**이다. 최적 셀을 다시 고르지 않고, ARM>=1.0 구간을 전수로
훑어 정방향/뒤집기 수익 셀 수를 센다. 이미 노출된 OOS에 대한 사후 기술통계이므로 새로운
OOS 소모가 아니다(모델/라벨/임계값 무엇도 이 결과로 재선택하지 않는다).

## 2x2 설계 -- 모집단 가설을 직접 검정한다

오늘 반복해서 나온 패턴은 "라벨 붙은 부분집합에서는 작동, 전체 봉에서는 안 작동"이었다:
  - 경제성 게이트(라벨행만, OOS n=385): +11.30bp, ARM=1.5행 정22/뒤9
  - Stage 3(전체 봉, OOS n=1066): -0.32bp, 뒤집기 +4.88bp
같은 설정·같은 구간인데 결론이 반대다. 그래서 **모집단 x 스플릿**을 교차해 확인한다:

              VAL          OOS
  전체 봉      ?            ?     <- 라이브가 실제로 마주하는 모집단
  라벨행만     ?            ?     <- 지금까지 경제성을 재던 모집단

전체 봉에서만 뒤집기가 우세하다면 원인은 시점(OOS)이 아니라 **모집단**이고, 그건
excluded-middle 봉(전체의 ~47%)에서 신호가 방향성을 못 낸다는 뜻이다.

⚠️ HOLDOUT 미터치. 라이브 코드 변경 없음. 이 결과로 파라미터를 재선택하지 않는다.

Run on the server (GPU) via handoff:
  handoff.sh launch server <job> -- python scripts/research_eth_v_rebound_deployed_oos_flip_diagnostic_20260901.py
"""
from __future__ import annotations

import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

S1 = ROOT / "scripts/research_eth_v_rebound_label_grid_screen_stage1_20260901.py"
_spec = importlib.util.spec_from_file_location("vreb_s1_diag", S1)
_s1 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_s1)

_bt = _s1._bt
FEATURE_COLUMNS = _s1.FEATURE_COLUMNS
STANDARD_COST_BP = _s1.STANDARD_COST_BP
FORWARD_BARS = _s1.FORWARD_BARS

TRAIN_END = pd.Timestamp("2025-09-01", tz="UTC")
VAL_END = pd.Timestamp("2026-01-01", tz="UTC")
OOS_END = pd.Timestamp("2026-04-01", tz="UTC")

# 배포 중인 설정 그대로 -- 이 스크립트는 이것 하나만 본다
DEPLOYED = {"atr_mult": 1.50, "t_sustain": 0.20, "full_bars": 12}
DEPLOYED_THRESHOLD = 0.60
CONTEXT_N, SEED = 18000, 20260829
ARTIFACT_FREE_MIN = 1.0     # 저ARM 노이즈수확 구간은 진단에서도 제외

SL_GRID = _bt.SL_GRID
ARM_GRID = tuple(a for a in _bt.ARM_GRID if a >= ARTIFACT_FREE_MIN)
TRAIL_GRID = _bt.TRAIL_GRID

OUT_JSON = ROOT / "data/research/eth_v_rebound_label_grid_stage1_20260901/deployed_oos_flip_diagnostic.json"


def log(msg: str) -> None:
    print(f"[diag] {msg}", flush=True)


def main() -> int:
    t0 = time.time()
    from tabpfn import TabPFNClassifier
    from sklearn.metrics import roc_auc_score
    import torch
    log(f"cuda: {torch.cuda.is_available()}")
    log("진단 전용 -- 이 결과로 파라미터/임계값을 재선택하지 않는다")

    _s1.VAL_END = OOS_END
    log("building frame...")
    sig, feat, eth = _s1.build_sig()
    kl = eth[["timestamp", "open", "high", "low", "close"]].copy()
    kl["timestamp"] = kl["timestamp"].dt.tz_localize(None)
    ts_to_pos = {t: i for i, t in enumerate(kl["timestamp"].to_numpy())}
    o, h, l, c = (kl[x].to_numpy() for x in ("open", "high", "low", "close"))

    sb = _s1.label_param(sig, True, ambig="drop", anchor="wick", **DEPLOYED)
    st = _s1.label_param(sig, False, ambig="drop", anchor="wick", **DEPLOYED)
    long = _s1.long_frame_for(sig, feat, sb, st)
    long["split"] = np.where(long["timestamp"] < TRAIN_END, "TRAIN",
                     np.where(long["timestamp"] < VAL_END, "VAL", "OOS"))
    assert long["timestamp"].max() < OOS_END, "HOLDOUT 누출"

    lab = long.loc[long["label"].notna()]
    tr = lab.loc[lab["split"] == "TRAIN"]
    rng = np.random.default_rng(SEED)
    idx = np.sort(rng.choice(len(tr), size=min(CONTEXT_N, len(tr)), replace=False))
    clf = TabPFNClassifier(device="cuda", random_state=SEED, ignore_pretraining_limits=True)
    clf.fit(tr.iloc[idx][FEATURE_COLUMNS], tr.iloc[idx]["label"].to_numpy())
    log(f"컨텍스트 {CONTEXT_N}행 | 배포 설정 K={DEPLOYED['atr_mult']} T={DEPLOYED['t_sustain']} "
        f"FULL={DEPLOYED['full_bars']} thr={DEPLOYED_THRESHOLD}")

    scored = {}
    for sp in ("VAL", "OOS"):
        s = long.loc[long["split"] == sp].copy()
        s["model_proba"] = clf.predict_proba(s[FEATURE_COLUMNS])[:, 1]
        scored[sp] = s
        sl_ = lab.loc[lab["split"] == sp]
        p = s.set_index(["timestamp", "side"]).loc[
            list(zip(sl_["timestamp"], sl_["side"])), "model_proba"].to_numpy()
        log(f"  {sp}: 전체 {len(s):,}  라벨행 {len(sl_):,}  "
            f"AUC {roc_auc_score(sl_['label'].to_numpy(), p):.4f}")

    def build(sel):
        rows = []
        for _, ev in sel.iterrows():
            i = ts_to_pos.get(np.datetime64(ev["timestamp"].tz_localize(None)))
            if i is None or i + FORWARD_BARS + 1 >= len(kl):
                continue
            rows.append({"side": "long" if ev["is_downside"] == 1 else "short",
                         "atr": float(ev["atr"]), "entry_price": float(o[i + 1]),
                         "fwd_open": o[i + 1:i + 1 + FORWARD_BARS], "fwd_high": h[i + 1:i + 1 + FORWARD_BARS],
                         "fwd_low": l[i + 1:i + 1 + FORWARD_BARS], "fwd_close": c[i + 1:i + 1 + FORWARD_BARS]})
        return pd.DataFrame(rows)

    def sweep(df):
        """ARM>=1.0 구간 전수. 정방향/뒤집기 각각의 수익 셀 수 + 최고값 + ARM행별 내역."""
        e, a, s_, H, L, C = _bt.pack(df)
        ef, af, sf, Hf, Lf, Cf = _bt.pack(df, flip=True)
        fwd, flp = [], []
        for sl in SL_GRID:
            for arm in ARM_GRID:
                for tr_ in TRAIL_GRID:
                    ov = _bt.simulate_trailing_vec(e, a, s_, H, L, C, sl, arm, tr_, False)
                    pv = _bt.simulate_trailing_vec(e, a, s_, H, L, C, sl, arm, tr_, True)
                    fo = _bt.simulate_trailing_vec(ef, af, sf, Hf, Lf, Cf, sl, arm, tr_, False)
                    fp = _bt.simulate_trailing_vec(ef, af, sf, Hf, Lf, Cf, sl, arm, tr_, True)
                    fwd.append({"sl": sl, "arm": arm, "trail": tr_,
                                "opt_bp": float(ov.mean() * 1e4 - STANDARD_COST_BP),
                                "pess_bp": float(pv.mean() * 1e4 - STANDARD_COST_BP)})
                    flp.append({"sl": sl, "arm": arm, "trail": tr_,
                                "opt_bp": float(fo.mean() * 1e4 - STANDARD_COST_BP),
                                "pess_bp": float(fp.mean() * 1e4 - STANDARD_COST_BP)})
        ok = lambda g: g["opt_bp"] > 0 and g["pess_bp"] > 0
        by_arm = {}
        for arm in ARM_GRID:
            fi = [i for i, g in enumerate(fwd) if g["arm"] == arm]
            by_arm[str(arm)] = {"cells": len(fi), "fwd": sum(1 for i in fi if ok(fwd[i])),
                                "flip": sum(1 for i in fi if ok(flp[i]))}
        return {"n_trades": int(len(df)), "n_cells": len(fwd),
                "fwd_pass": sum(map(ok, fwd)), "flip_pass": sum(map(ok, flp)),
                "fwd_best_pess": round(max(g["pess_bp"] for g in fwd), 2),
                "flip_best_pess": round(max(g["pess_bp"] for g in flp), 2),
                "fwd_median_pess": round(float(np.median([g["pess_bp"] for g in fwd])), 2),
                "flip_median_pess": round(float(np.median([g["pess_bp"] for g in flp])), 2),
                "by_arm": by_arm}

    log("")
    log(f"=== 2x2 진단 (ARM>={ARTIFACT_FREE_MIN} 구간 {len(SL_GRID)*len(ARM_GRID)*len(TRAIL_GRID)}셀 전수) ===")
    cells = {}
    for sp in ("VAL", "OOS"):
        s = scored[sp]
        for pop, sel in (("전체봉", s.loc[s["model_proba"] >= DEPLOYED_THRESHOLD]),
                         ("라벨행만", s.loc[s["label"].notna() & (s["model_proba"] >= DEPLOYED_THRESHOLD)])):
            df = build(sel)
            if len(df) < 50:
                log(f"  {sp:3s} x {pop:6s}: 호출 {len(df)}건 -- 표본 부족"); continue
            r = sweep(df)
            cells[f"{sp}|{pop}"] = r
            verdict = "정방향 우세" if r["fwd_pass"] > r["flip_pass"] else (
                      "⚠️뒤집기 우세" if r["flip_pass"] > r["fwd_pass"] else "동률")
            log(f"  {sp:3s} x {pop:6s}: n={r['n_trades']:>5d}  "
                f"정방향 {r['fwd_pass']:>3d}/{r['n_cells']} (최고 {r['fwd_best_pess']:+6.2f} 중앙 {r['fwd_median_pess']:+6.2f})  "
                f"뒤집기 {r['flip_pass']:>3d}/{r['n_cells']} (최고 {r['flip_best_pess']:+6.2f} 중앙 {r['flip_median_pess']:+6.2f})  "
                f"=> {verdict}")
            log(f"        ARM행별 정/뒤: " + "  ".join(
                f"ARM={k}: {v['fwd']}/{v['flip']}" for k, v in r["by_arm"].items()))

    log("")
    log("=== 해석 ===")
    key = lambda sp, pop: cells.get(f"{sp}|{pop}")
    both_all = [key(sp, "전체봉") for sp in ("VAL", "OOS")]
    both_lab = [key(sp, "라벨행만") for sp in ("VAL", "OOS")]
    if all(both_all) and all(both_lab):
        oos_all, oos_lab = key("OOS", "전체봉"), key("OOS", "라벨행만")
        if oos_all["flip_pass"] > oos_all["fwd_pass"] and oos_lab["fwd_pass"] >= oos_lab["flip_pass"]:
            log("  OOS에서 **전체봉만** 뒤집기 우세, 라벨행에서는 정방향 유지")
            log("  -> 원인은 시점이 아니라 **모집단**. excluded-middle 봉에서 방향성이 나오지 않는다.")
            log("  -> 화면 경제성 주장은 라벨행 기준이므로 라이브 전체에 그대로 적용하면 안 된다.")
        elif oos_all["flip_pass"] > oos_all["fwd_pass"] and oos_lab["flip_pass"] > oos_lab["fwd_pass"]:
            log("  OOS에서 **두 모집단 모두** 뒤집기 우세 -> 모집단이 아니라 시점(OOS) 문제.")
        else:
            log("  OOS 전체봉에서 정방향이 유지됨 -> Stage 3의 그 셀은 국소적 결과였다.")

    report = {
        "signal": "v_rebound_deployed_oos_flip_diagnostic", "asset": "ETHUSDT",
        "fresh_forward_bar_by_bar": True, "trade_ledgers_used_as_input": False,
        "saved_parent_exit_timestamps_used": False, "future_rows_used_for_entry": False,
        "scope": {"purpose": "진단 전용 -- Stage 3의 단일 셀 결과가 국소적인지 체계적인지",
                  "deployed_config": DEPLOYED, "threshold": DEPLOYED_THRESHOLD,
                  "arm_floor": ARTIFACT_FREE_MIN, "reselection_performed": False,
                  "holdout_touched": False, "live_code_changed": False},
        "grid": {"sl": list(SL_GRID), "arm": list(ARM_GRID), "trail": list(TRAIL_GRID)},
        "cells": cells, "runtime_sec": round(time.time() - t0, 1),
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str))
    log("")
    log(f"report saved -> {OUT_JSON}  ({report['runtime_sec']}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
