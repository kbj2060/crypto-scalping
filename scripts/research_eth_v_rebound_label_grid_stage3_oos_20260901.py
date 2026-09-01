#!/usr/bin/env python3
"""V자반등 라벨 격자 -- Stage 3: 최종 후보 1개를 **OOS 1회** 평가. ⚠️OOS 소모.

## 여기까지의 경위

Stage 1 (GBM 프록시, 축소격자 18셀, 54변형): `FULL_BARS`가 지배적 축. 현행 대비 최고 +3.94bp.
Stage 2 (TabPFN, 240셀 전수, 후보 5개): **결론 대부분이 깎였다.**
  - 개선폭 +3.94 -> **+0.79bp** (5분의 1)
  - Stage 1 1위(K1.25/FULL24)가 Stage 2 **최하위**로 추락 -- 972설정 VAL 훑기의 선택편향
  - GBM -> TabPFN 전이 손실도 있었다(현행 기준선 +3.70 -> +2.33bp)
  - ⚠️Stage 2 첫 실행에서 5개 후보 **전부** 전역 최고셀이 ARM=0.1이었고 그 셀에서 뒤집기도
    +5.05~+6.55bp 수익이었다. 저ARM 노이즈수확 아티팩트를 제외하지 않으면 틀린 답이 나온다
    ([[feedback_trailing_stop_low_arm_noise_harvest_artifact_20260901]]).

살아남은 것은 `FULL_BARS=24`의 **강건성**뿐이다(절대 bp가 아니라):
  ARM=1.5 통과셀 6 -> 16, 240셀 정방향:뒤집기 80:52 -> 100:35.
K값은 단계 간 순위가 뒤집혀 노이즈 대역으로 본다(bp 1.75 / 강건성 1.50 / 최하위 1.25).

## 그래서 후보는 하나뿐이다

`현행+FULL24` -- 배포판에서 **축 하나만** 움직인 변경(FULL_BARS 12->24). 과적합 여지가 가장
적고 아티팩트 무영향 구간 강건성이 최고다. K=1.75의 +0.17bp 우위는 이번 순위 변동폭에 비하면
의미 없다.

## 사전 등록한 판정 기준 (결과를 보기 전에 정한다)

**OOS에서 현행 대비 우위가 +1bp 미만이거나, 선택셀에서 뒤집기 대비 우위가 무너지면 재배포하지
않는다.** Stage1->2에서 이미 5배 깎였으므로 추가 감쇠를 예상해야 하고 OOS는 한 번뿐이다.

## 절차

  1. 라벨을 OOS 끝(2026-04-01)까지 생성, TRAIN/VAL/OOS 3분할. **HOLDOUT은 경계에서 잘라 미터치.**
  2. 변형별로 라벨된 TRAIN에서 무작위 18,000행 동결(배포 절차와 동일).
  3. TabPFN fit -> VAL/OOS 전체 봉 채점.
  4. **경제성 셀은 VAL에서만 선정**(ARM>=1.0 구간 내 pess_bp 최대). OOS에서 재최적화 금지.
  5. 그 셀 하나를 OOS에서 1회 평가 + 방향뒤집기 대조.
  6. 임계값은 두 가지로 보고: (a) VAL에서 정한 임계값을 OOS에 그대로 적용(라이브가 실제로 하는
     것), (b) OOS에서도 호출 빈도를 맞춘 통제 비교. (a)가 배포 판단의 주 기준이다.

Run on the server (GPU) via handoff:
  handoff.sh launch server <job> -- python scripts/research_eth_v_rebound_label_grid_stage3_oos_20260901.py
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
_spec = importlib.util.spec_from_file_location("vreb_grid_stage1_s3", S1)
_s1 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_s1)

_bt = _s1._bt
FEATURE_COLUMNS = _s1.FEATURE_COLUMNS
STANDARD_COST_BP = _s1.STANDARD_COST_BP
FORWARD_BARS = _s1.FORWARD_BARS

TRAIN_END = pd.Timestamp("2025-09-01", tz="UTC")
VAL_END = pd.Timestamp("2026-01-01", tz="UTC")
OOS_END = pd.Timestamp("2026-04-01", tz="UTC")   # HOLDOUT은 이 이후 -- 여기서 잘라 미터치 보장

CONTEXT_N = 18000
SEED = 20260829
ARTIFACT_FREE_MIN = 1.0
VAL_TARGET_CALL_N = _s1.TARGET_CALL_N            # 1256 -- Stage1/2와 동일
SL_GRID, ARM_GRID, TRAIL_GRID = _bt.SL_GRID, _bt.ARM_GRID, _bt.TRAIL_GRID

CANDIDATES = [
    {"name": "현행(baseline)", "atr_mult": 1.50, "t_sustain": 0.20, "full_bars": 12},
    {"name": "현행+FULL24",    "atr_mult": 1.50, "t_sustain": 0.20, "full_bars": 24},
]
DECISION_RULE = {"min_oos_edge_bp": 1.0,
                 "require_flip_negative_at_selected_cell": True,
                 "note": "결과를 보기 전에 사전 등록함"}

OUT_JSON = ROOT / "data/research/eth_v_rebound_label_grid_stage1_20260901/stage3_oos.json"


def log(msg: str) -> None:
    print(f"[grid3] {msg}", flush=True)


def called_rows(scored, kl, ts_to_pos, ohlc, k=None, thr=None):
    """상위 k건(또는 thr 이상)을 골라 전방 200봉을 붙인다."""
    if k is not None:
        sel = scored.nlargest(min(k, len(scored)), "model_proba")
    else:
        sel = scored.loc[scored["model_proba"] >= thr]
    o, h, l, c = ohlc
    rows = []
    for _, ev in sel.iterrows():
        i = ts_to_pos.get(np.datetime64(ev["timestamp"].tz_localize(None)))
        if i is None or i + FORWARD_BARS + 1 >= len(kl):
            continue
        rows.append({"side": "long" if ev["is_downside"] == 1 else "short",
                     "atr": float(ev["atr"]), "entry_price": float(o[i + 1]),
                     "fwd_open": o[i + 1:i + 1 + FORWARD_BARS], "fwd_high": h[i + 1:i + 1 + FORWARD_BARS],
                     "fwd_low": l[i + 1:i + 1 + FORWARD_BARS], "fwd_close": c[i + 1:i + 1 + FORWARD_BARS]})
    return pd.DataFrame(rows), (float(sel["model_proba"].min()) if len(sel) else None)


def cell_bp(df, sl, arm, trail):
    """한 셀의 정방향/뒤집기 성과."""
    e, a, s, H, L, C = _bt.pack(df)
    ef, af, sf, Hf, Lf, Cf = _bt.pack(df, flip=True)
    opt = _bt.simulate_trailing_vec(e, a, s, H, L, C, sl, arm, trail, False)
    pes = _bt.simulate_trailing_vec(e, a, s, H, L, C, sl, arm, trail, True)
    fo = _bt.simulate_trailing_vec(ef, af, sf, Hf, Lf, Cf, sl, arm, trail, False)
    return {"n": int(len(df)),
            "opt_bp": round(float(opt.mean() * 1e4 - STANDARD_COST_BP), 2),
            "pess_bp": round(float(pes.mean() * 1e4 - STANDARD_COST_BP), 2),
            "win_rate": round(float((opt * 1e4 > STANDARD_COST_BP).mean()), 4),
            "flip_opt_bp": round(float(fo.mean() * 1e4 - STANDARD_COST_BP), 2)}


def main() -> int:
    t0 = time.time()
    from sklearn.metrics import roc_auc_score
    from tabpfn import TabPFNClassifier
    import torch
    log(f"cuda: {torch.cuda.is_available()}")
    log(f"⚠️ 이 실행은 OOS를 1회 소모한다. 사전 등록 기준: {DECISION_RULE}")

    _s1.VAL_END = OOS_END      # 프레임을 OOS 끝까지 만든다(split은 아래에서 3분할)
    log("building indicator/trigger frame...")
    sig, feat, eth = _s1.build_sig()
    kl = eth[["timestamp", "open", "high", "low", "close"]].copy()
    kl["timestamp"] = kl["timestamp"].dt.tz_localize(None)
    ts_to_pos = {t: i for i, t in enumerate(kl["timestamp"].to_numpy())}
    ohlc = tuple(kl[x].to_numpy() for x in ("open", "high", "low", "close"))

    results = []
    for cand in CANDIDATES:
        kw = {k: cand[k] for k in ("atr_mult", "t_sustain", "full_bars")}
        tag = cand["name"]
        log("")
        log(f"=== {tag} (K={kw['atr_mult']} T={kw['t_sustain']} FULL={kw['full_bars']}) ===")
        sb = _s1.label_param(sig, True, ambig="drop", anchor="wick", **kw)
        st = _s1.label_param(sig, False, ambig="drop", anchor="wick", **kw)
        long = _s1.long_frame_for(sig, feat, sb, st)
        long["split"] = np.where(long["timestamp"] < TRAIN_END, "TRAIN",
                         np.where(long["timestamp"] < VAL_END, "VAL", "OOS"))
        assert long["timestamp"].max() < OOS_END, "HOLDOUT 누출"

        lab = long.loc[long["label"].notna()]
        tr = lab.loc[lab["split"] == "TRAIN"]
        rng = np.random.default_rng(SEED)
        idx = np.sort(rng.choice(len(tr), size=min(CONTEXT_N, len(tr)), replace=False))
        ctx = tr.iloc[idx]
        clf = TabPFNClassifier(device="cuda", random_state=SEED, ignore_pretraining_limits=True)
        clf.fit(ctx[FEATURE_COLUMNS], ctx["label"].to_numpy())
        log(f"  컨텍스트 {len(ctx)}행(라벨률 {ctx['label'].mean():.4f}) | TRAIN 라벨행 {len(tr):,}")

        scored = {}
        for sp in ("VAL", "OOS"):
            s = long.loc[long["split"] == sp].copy()
            s["model_proba"] = clf.predict_proba(s[FEATURE_COLUMNS])[:, 1]
            scored[sp] = s
            sl_ = lab.loc[lab["split"] == sp]
            auc = None
            if len(sl_) and sl_["label"].nunique() == 2:
                p = s.set_index(["timestamp", "side"]).loc[
                    list(zip(sl_["timestamp"], sl_["side"])), "model_proba"].to_numpy()
                auc = round(float(roc_auc_score(sl_["label"].to_numpy(), p)), 4)
            log(f"  {sp}: 전체 {len(s):,}행  라벨행 {len(sl_):,}  AUC {auc}")

        # --- 셀 선정: VAL만, ARM>=1.0 구간 내 ---
        val_df, val_thr = called_rows(scored["VAL"], kl, ts_to_pos, ohlc, k=VAL_TARGET_CALL_N)
        e, a, s_, H, L, C = _bt.pack(val_df)
        ef, af, sf, Hf, Lf, Cf = _bt.pack(val_df, flip=True)
        best, best_key = None, None
        for sl in SL_GRID:
            for arm in ARM_GRID:
                if arm < ARTIFACT_FREE_MIN:
                    continue          # 저ARM은 방향 무관 수익 구간 -- 선정에서 제외
                for trail in TRAIL_GRID:
                    pes = _bt.simulate_trailing_vec(e, a, s_, H, L, C, sl, arm, trail, True)
                    bp = float(pes.mean() * 1e4 - STANDARD_COST_BP)
                    if best is None or bp > best:
                        best, best_key = bp, (sl, arm, trail)
        sl, arm, trail = best_key
        val_cell = cell_bp(val_df, sl, arm, trail)
        log(f"  [VAL 선정] SL/ARM/Trail={sl}/{arm}/{trail}  n={val_cell['n']} "
            f"opt{val_cell['opt_bp']:+.2f} pess{val_cell['pess_bp']:+.2f}bp "
            f"승률{val_cell['win_rate']*100:.1f}% (뒤집기 opt{val_cell['flip_opt_bp']:+.2f}bp)  "
            f"thr={val_thr:.4f}")

        # --- OOS 1회 평가: (a) VAL 임계값 그대로, (b) 호출빈도 맞춤 ---
        oos_a, _ = called_rows(scored["OOS"], kl, ts_to_pos, ohlc, thr=val_thr)
        n_target = int(round(VAL_TARGET_CALL_N * len(scored["OOS"]) / len(scored["VAL"])))
        oos_b, oos_b_thr = called_rows(scored["OOS"], kl, ts_to_pos, ohlc, k=n_target)
        cell_a = cell_bp(oos_a, sl, arm, trail) if len(oos_a) >= 50 else {"n": int(len(oos_a)), "skipped": True}
        cell_b = cell_bp(oos_b, sl, arm, trail) if len(oos_b) >= 50 else {"n": int(len(oos_b)), "skipped": True}
        for nm, cc in (("(a) VAL임계값 그대로", cell_a), ("(b) 호출빈도 맞춤", cell_b)):
            if cc.get("skipped"):
                log(f"  [OOS {nm}] 호출 {cc['n']}건 -- 표본 부족")
            else:
                log(f"  [OOS {nm}] n={cc['n']} opt{cc['opt_bp']:+.2f} pess{cc['pess_bp']:+.2f}bp "
                    f"승률{cc['win_rate']*100:.1f}% (뒤집기 opt{cc['flip_opt_bp']:+.2f}bp)")

        results.append({"tag": tag, "params": kw, "selected_cell": {"sl": sl, "arm": arm, "trail": trail},
                        "val_threshold": round(val_thr, 4), "val": val_cell,
                        "oos_val_threshold": cell_a, "oos_matched_rate": cell_b,
                        "oos_matched_target_n": n_target})

    log("")
    log("=== 판정 ===")
    base = next(r for r in results if r["tag"] == "현행(baseline)")
    cand = next(r for r in results if r["tag"] == "현행+FULL24")
    verdict = {}
    for key, nm in (("oos_val_threshold", "(a) VAL임계값"), ("oos_matched_rate", "(b) 빈도맞춤")):
        b, c = base[key], cand[key]
        if b.get("skipped") or c.get("skipped"):
            log(f"  {nm}: 표본 부족으로 판정 불가"); continue
        edge = c["pess_bp"] - b["pess_bp"]
        flip_ok = c["flip_opt_bp"] < 0
        passed = edge >= DECISION_RULE["min_oos_edge_bp"] and flip_ok
        verdict[key] = {"edge_bp": round(edge, 2), "flip_negative": flip_ok, "passed": passed}
        log(f"  {nm}: 현행 pess{b['pess_bp']:+.2f} vs FULL24 pess{c['pess_bp']:+.2f} "
            f"-> 우위 {edge:+.2f}bp | 뒤집기 {c['flip_opt_bp']:+.2f}bp({'음수 OK' if flip_ok else '⚠️양수'}) "
            f"=> {'통과' if passed else '미달'}")
    log("")
    log(f"  사전 등록 기준: OOS 우위 >= {DECISION_RULE['min_oos_edge_bp']}bp AND 선택셀 뒤집기 음수")
    log(f"  최종: {'재배포 권고' if all(v['passed'] for v in verdict.values()) else '재배포 안 함'}")

    report = {
        "signal": "v_rebound_label_grid_stage3_oos", "asset": "ETHUSDT",
        "fresh_forward_bar_by_bar": True, "trade_ledgers_used_as_input": False,
        "saved_parent_exit_timestamps_used": False, "future_rows_used_for_entry": False,
        "scope": {"stage": 3, "model": "TabPFN", "context_n": CONTEXT_N, "seed": SEED,
                  "oos_touched": True, "oos_exposures": 1, "holdout_touched": False,
                  "live_code_changed": False,
                  "cell_selected_on": "VAL only (ARM>=1.0)", "reoptimized_on_oos": False},
        "decision_rule": DECISION_RULE, "candidates": CANDIDATES,
        "results": results, "verdict": verdict,
        "runtime_sec": round(time.time() - t0, 1),
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str))
    log("")
    log(f"report saved -> {OUT_JSON}  ({report['runtime_sec']}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
