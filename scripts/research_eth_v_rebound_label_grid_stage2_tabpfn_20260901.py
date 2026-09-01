#!/usr/bin/env python3
"""V자반등 라벨 격자 -- Stage 2: 상위 후보를 **TabPFN + 240셀 전수 격자**로 VAL 재확인.

## Stage 1이 남긴 것

GBM 프록시 + 축소격자(18셀)에서 `FULL_BARS`가 지배적 축으로 나왔다. K/T를 현행 그대로 두고
12봉->24봉만 바꿔도 건당 pess가 **+3.70 -> +6.80bp**(18/18 통과, 뒤집기 0). 상위 8개 중 5개가
FULL=24였다. 반면 이번 라운드에 제안했던 두 정의 변경은 **둘 다 기각**됐다:
  - `fail_neg`(되반납 실패를 음성으로): 0/18 통과, **뒤집기가 12/18 통과** -- 신호가 망가짐
  - `anchor=body`(종가 앵커): AUC 0.697->0.601, 1/18 통과 -- 꼬리 앵커가 실제로 정보였음

## Stage 2가 답할 것

1. **GBM->TabPFN 전이되는가.** Stage 1은 프록시였다. 배포판은 TabPFN이고, 이 저장소는 둘이
   갈린 전례가 있다(V_REBOUND 자신이 TabPFN에서 GBM을 이겼다).
2. **축소격자 18/18이 240셀에서도 유지되는가.** Stage 1은 고ARM 중심 18셀만 봤다.
   저ARM까지 포함한 전수에서 방향뒤집기 대조군이 어떻게 나오는지가 진짜 관문이다
   ([[feedback_trailing_stop_low_arm_noise_harvest_artifact_20260901]]: 단일 config만 뒤집으면
   오판, fib_extension_exhaustion가 이걸로 경제성 클레임 철회됨).
3. **ARM=1.5 행(아티팩트 무영향 구간)에서 정방향이 뒤집기를 이기는가** -- 2026-09-01 임계값
   판정에서 결정적이었던 진단축.

## 컨텍스트는 배포 절차와 동일하게 만든다

변형마다 그 변형의 라벨된 TRAIN에서 **무작위 18,000행**을 뽑아 동결
(freeze_eth_v_rebound_every_bar_train_context_20260901.py와 같은 방식/시드). 재균형 안 함.

## 비교는 **동일 호출 빈도**에서

임계값을 변형 간에 고정하면 안 된다 -- 라벨이 바뀌면 확률 분포가 통째로 이동해, 고정 임계값에서는
변형이 "성능 없음"이 아니라 "측정 불가"로 탈락한다(Stage 1 첫 실행에서 6셀 중 5셀이 그렇게 죽었다).
전 변형을 TARGET_CALL_N에 맞춘다 -- "같은 매매 빈도일 때 어느 라벨이 더 나은 거래를 고르는가".

⚠️ VAL만 사용. **OOS/HOLDOUT 미터치**(Stage 3에서 최종 1개에만 OOS 1회). 라이브 코드 변경 없음.

Run on the server (GPU) via handoff:
  handoff.sh launch server <job> -- python scripts/research_eth_v_rebound_label_grid_stage2_tabpfn_20260901.py
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
_spec = importlib.util.spec_from_file_location("vreb_grid_stage1", S1)
_s1 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_s1)

_bt = _s1._bt
FEATURE_COLUMNS = _s1.FEATURE_COLUMNS
STANDARD_COST_BP = _s1.STANDARD_COST_BP
FORWARD_BARS = _s1.FORWARD_BARS
TARGET_CALL_N = _s1.TARGET_CALL_N

SL_GRID, ARM_GRID, TRAIL_GRID = _bt.SL_GRID, _bt.ARM_GRID, _bt.TRAIL_GRID  # 240셀 전수
CONTEXT_N = 18000
SEED = 20260829          # 배포 컨텍스트 동결/추론과 동일
ARTIFACT_FREE_ARM = 1.5   # 진단용: 저ARM 노이즈수확이 닿지 않는 행
ARTIFACT_FREE_MIN = 1.0   # 선정은 이 값 이상 ARM에서만 -- 저ARM은 방향 무관 수익이 나는 구간

# Stage 1 상위 + 현행. FULL=24 계열이 상위를 지배했으므로 그 축을 중심으로 고른다.
CANDIDATES = [
    {"name": "현행(baseline)",     "atr_mult": 1.50, "t_sustain": 0.20, "full_bars": 12},
    {"name": "현행+FULL24",        "atr_mult": 1.50, "t_sustain": 0.20, "full_bars": 24},
    {"name": "S1최고 K1.25/FULL24", "atr_mult": 1.25, "t_sustain": 0.20, "full_bars": 24},
    {"name": "K1.75/FULL24",       "atr_mult": 1.75, "t_sustain": 0.20, "full_bars": 24},
    {"name": "K1.25/T0.30/FULL18", "atr_mult": 1.25, "t_sustain": 0.30, "full_bars": 18},
]

OUT_JSON = ROOT / "data/research/eth_v_rebound_label_grid_stage1_20260901/stage2_tabpfn.json"


def log(msg: str) -> None:
    print(f"[grid2] {msg}", flush=True)


def main() -> int:
    t0 = time.time()
    from sklearn.metrics import roc_auc_score
    from tabpfn import TabPFNClassifier
    import torch
    log(f"cuda: {torch.cuda.is_available()}")

    log("building indicator/trigger frame...")
    sig, feat, eth = _s1.build_sig()
    kl = eth[["timestamp", "open", "high", "low", "close"]].copy()
    kl["timestamp"] = kl["timestamp"].dt.tz_localize(None)  # tz-aware면 ts_to_pos가 전부 미스
    ts_to_pos = {t: i for i, t in enumerate(kl["timestamp"].to_numpy())}
    o, h, l, c = (kl[x].to_numpy() for x in ("open", "high", "low", "close"))

    results = []
    for cand in CANDIDATES:
        kw = {k: cand[k] for k in ("atr_mult", "t_sustain", "full_bars")}
        tag = cand["name"]
        log("")
        log(f"=== {tag}  (K={kw['atr_mult']} T={kw['t_sustain']} FULL={kw['full_bars']}) ===")
        sb = _s1.label_param(sig, True, ambig="drop", anchor="wick", **kw)
        st = _s1.label_param(sig, False, ambig="drop", anchor="wick", **kw)
        long = _s1.long_frame_for(sig, feat, sb, st)

        lab = long.loc[long["label"].notna()]
        tr = lab.loc[lab["split"] == "TRAIN"]
        va_all = long.loc[long["split"] == "VAL"].copy()   # 라벨 없는 봉 포함 = 라이브 모집단
        va_lab = lab.loc[lab["split"] == "VAL"]

        rng = np.random.default_rng(SEED)
        idx = np.sort(rng.choice(len(tr), size=min(CONTEXT_N, len(tr)), replace=False))
        ctx = tr.iloc[idx]
        log(f"  컨텍스트 {len(ctx)}행 (라벨률 {ctx['label'].mean():.4f}) | "
            f"TRAIN 라벨행 {len(tr):,} | VAL 전체 {len(va_all):,}")

        clf = TabPFNClassifier(device="cuda", random_state=SEED, ignore_pretraining_limits=True)
        clf.fit(ctx[FEATURE_COLUMNS], ctx["label"].to_numpy())
        va_all["model_proba"] = clf.predict_proba(va_all[FEATURE_COLUMNS])[:, 1]

        auc = None
        if len(va_lab) and va_lab["label"].nunique() == 2:
            p_lab = va_all.set_index(["timestamp", "side"]).loc[
                list(zip(va_lab["timestamp"], va_lab["side"])), "model_proba"].to_numpy()
            auc = float(roc_auc_score(va_lab["label"].to_numpy(), p_lab))

        k = min(TARGET_CALL_N, len(va_all))
        called = va_all.nlargest(k, "model_proba")
        cutoff = float(called["model_proba"].min())

        rows = []
        for _, ev in called.iterrows():
            i = ts_to_pos.get(np.datetime64(ev["timestamp"].tz_localize(None)))
            if i is None or i + FORWARD_BARS + 1 >= len(kl):
                continue
            rows.append({"side": "long" if ev["is_downside"] == 1 else "short",
                         "atr": float(ev["atr"]), "entry_price": float(o[i + 1]),
                         "fwd_open": o[i + 1:i + 1 + FORWARD_BARS],
                         "fwd_high": h[i + 1:i + 1 + FORWARD_BARS],
                         "fwd_low": l[i + 1:i + 1 + FORWARD_BARS],
                         "fwd_close": c[i + 1:i + 1 + FORWARD_BARS]})
        df = pd.DataFrame(rows)
        r = {"tag": tag, "params": kw, "val_auc": round(auc, 4) if auc else None,
             "threshold_used": round(cutoff, 4), "n_called": int(len(df)),
             "labeled_row_pct": round(float(len(lab) / len(long) * 100), 1),
             "train_pos_rate": round(float(tr["label"].mean()), 4)}
        if len(df) < 50:
            r["skipped"] = f"호출 {len(df)}건"
            results.append(r); log(f"  SKIP {r['skipped']}"); continue

        e, a, s, H, L, C = _bt.pack(df)
        ef, af, sf, Hf, Lf, Cf = _bt.pack(df, flip=True)
        cells, flips = [], []
        for sl in SL_GRID:
            for arm in ARM_GRID:
                for trail in TRAIL_GRID:
                    opt = _bt.simulate_trailing_vec(e, a, s, H, L, C, sl, arm, trail, False)
                    pes = _bt.simulate_trailing_vec(e, a, s, H, L, C, sl, arm, trail, True)
                    fo = _bt.simulate_trailing_vec(ef, af, sf, Hf, Lf, Cf, sl, arm, trail, False)
                    fp = _bt.simulate_trailing_vec(ef, af, sf, Hf, Lf, Cf, sl, arm, trail, True)
                    cells.append({"sl": sl, "arm": arm, "trail": trail,
                                  "opt_bp": float(opt.mean() * 1e4 - STANDARD_COST_BP),
                                  "pess_bp": float(pes.mean() * 1e4 - STANDARD_COST_BP),
                                  "win_rate": float((opt * 1e4 > STANDARD_COST_BP).mean())})
                    flips.append({"arm": arm,
                                  "opt_bp": float(fo.mean() * 1e4 - STANDARD_COST_BP),
                                  "pess_bp": float(fp.mean() * 1e4 - STANDARD_COST_BP)})

        def passes(g):
            return g["opt_bp"] > 0 and g["pess_bp"] > 0
        n_pass, n_flip = sum(map(passes, cells)), sum(map(passes, flips))
        hi = [i for i, g in enumerate(cells) if g["arm"] == ARTIFACT_FREE_ARM]
        hi_pass = sum(1 for i in hi if passes(cells[i]))
        hi_flip = sum(1 for i in hi if passes(flips[i]))
        # ⚠️240셀 전체에서 pess_bp 최대를 고르면 **반드시 저ARM 아티팩트 구간이 뽑힌다** --
        # 2026-09-01 첫 실행에서 5개 후보 전부 SL=4.0/ARM=0.1/Trail=0.1이 선정됐고, 그 셀에서는
        # 방향뒤집기도 전부 수익이었다(+5.05~+6.55bp). 트레일링이 방향과 무관하게 노이즈를
        # 수확하는 것이므로 신호의 성능이 아니다
        # ([[feedback_trailing_stop_low_arm_noise_harvest_artifact_20260901]]).
        # 판정은 **ARM>=ARTIFACT_FREE_MIN 구간 안에서** 고른 셀로 한다. 전역 최고는 참고로만 남기고,
        # 그 셀에서 뒤집기도 수익이면 flag를 세워 눈에 띄게 한다.
        def sel(idxs):
            if not idxs:
                return None, None
            j = max(idxs, key=lambda i: cells[i]["pess_bp"])
            return j, cells[j]
        gi, gbest = sel(range(len(cells)))
        clean_idx = [i for i, g in enumerate(cells) if g["arm"] >= ARTIFACT_FREE_MIN]
        ci, cbest = sel(clean_idx)
        r.update({
            "n_cells": len(cells), "n_pass": n_pass, "n_flip_pass": n_flip,
            "arm15_cells": len(hi), "arm15_pass": hi_pass, "arm15_flip_pass": hi_flip,
            "best_global": {k2: round(v, 3) if isinstance(v, float) else v for k2, v in gbest.items()},
            "best_global_flip_opt_bp": round(flips[gi]["opt_bp"], 2),
            "best_global_flip_also_profits": bool(flips[gi]["opt_bp"] > 0),
            "best": {k2: round(v, 3) if isinstance(v, float) else v for k2, v in cbest.items()},
            "best_flip_opt_bp": round(flips[ci]["opt_bp"], 2),
            "best_flip_also_profits": bool(flips[ci]["opt_bp"] > 0),
            "total_bp": round(cbest["pess_bp"] * len(df), 0),
        })
        best, bi = cbest, ci
        results.append(r)
        log(f"  AUC {r['val_auc']}  thr {r['threshold_used']:.3f}  호출 {r['n_called']}")
        log(f"  240셀: 정방향 {n_pass:>3d} / 뒤집기 {n_flip:>3d}  |  "
            f"ARM=1.5행({len(hi)}셀): 정방향 {hi_pass:>2d} / 뒤집기 {hi_flip:>2d}")
        g = r["best_global"]
        log(f"  [참고] 전역최고 SL/ARM/Trail={g['sl']}/{g['arm']}/{g['trail']} "
            f"pess{g['pess_bp']:+.2f}bp (뒤집기 opt{r['best_global_flip_opt_bp']:+.2f}bp"
            f"{' ⚠️뒤집기도 수익=아티팩트' if r['best_global_flip_also_profits'] else ''})")
        log(f"  [판정] ARM>={ARTIFACT_FREE_MIN} 최고셀 SL/ARM/Trail={best['sl']}/{best['arm']}/{best['trail']}  "
            f"opt{best['opt_bp']:+.2f} pess{best['pess_bp']:+.2f}bp 승률{best['win_rate']*100:.1f}% "
            f"(뒤집기 opt{r['best_flip_opt_bp']:+.2f}bp"
            f"{' ⚠️뒤집기도 수익' if r['best_flip_also_profits'] else ''})  총{r['total_bp']:+.0f}bp")

    ok = [r for r in results if "skipped" not in r]
    ok.sort(key=lambda r: r["best"]["pess_bp"], reverse=True)
    log("")
    log(f"=== 순위 (ARM>={ARTIFACT_FREE_MIN} 구간 내 건당 pess_bp -- 저ARM 아티팩트 제외) ===")
    for r in ok:
        log(f"  {r['tag']:24s} pess{r['best']['pess_bp']:+6.2f}bp  "
            f"240셀 정{r['n_pass']:>3d}/뒤{r['n_flip_pass']:>3d}  "
            f"ARM1.5 정{r['arm15_pass']:>2d}/뒤{r['arm15_flip_pass']:>2d}  AUC {r['val_auc']}")
    base = next((r for r in ok if r["tag"] == "현행(baseline)"), None)
    if base and ok and ok[0]["tag"] != "현행(baseline)":
        d = ok[0]["best"]["pess_bp"] - base["best"]["pess_bp"]
        log("")
        log(f"최고 후보 '{ok[0]['tag']}'가 현행 대비 건당 {d:+.2f}bp")

    report = {
        "signal": "v_rebound_label_grid_stage2_tabpfn", "asset": "ETHUSDT",
        "fresh_forward_bar_by_bar": True, "trade_ledgers_used_as_input": False,
        "saved_parent_exit_timestamps_used": False, "future_rows_used_for_entry": False,
        "scope": {"stage": 2, "model": "TabPFN (배포판과 동일 계열)",
                  "context_n": CONTEXT_N, "seed": SEED,
                  "splits_used": "TRAIN+VAL only", "oos_touched": False,
                  "holdout_touched": False, "live_code_changed": False,
                  "matched_call_n": TARGET_CALL_N,
                  "economics_grid": {"sl": list(SL_GRID), "arm": list(ARM_GRID),
                                     "trail": list(TRAIL_GRID), "n_cells": 240},
                  "selection_metric": "경제성. AUC는 참고용(라벨 정의가 다르면 직접 비교 무효)"},
        "stage1_findings": {
            "dominant_axis": "FULL_BARS (12->24에서 +3.70->+6.80bp, GBM 프록시)",
            "rejected": {"fail_neg": "0/18 통과, 뒤집기 12/18 -- 신호 손상",
                         "anchor_body": "AUC 0.697->0.601, 1/18 통과"}},
        "candidates": CANDIDATES, "results": results,
        "runtime_sec": round(time.time() - t0, 1),
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str))
    log("")
    log(f"report saved -> {OUT_JSON}  ({report['runtime_sec']}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
