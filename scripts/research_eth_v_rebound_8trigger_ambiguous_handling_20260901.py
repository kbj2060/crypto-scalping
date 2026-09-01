#!/usr/bin/env python3
"""8트리거 모집단에서 **ambiguous 처리 전면 재검증** -- 이전 기각을 그대로 적용하지 않는다.

## 왜 다시 하는가

`fail_neg`/`all_neg` 기각(0/18 통과, 뒤집기 12/18)은 **매 봉 모집단**에서 잰 결과다. 지금은
8트리거 게이트 모집단이고, 이건 더 정보력 있는 봉들의 부분집합이다(정직한 AUC 0.7551/0.7654).
**모집단이 다르면 기각 근거가 직접 적용되지 않는다** -- 이 저장소가 오늘 여러 번 확인한 규칙
(라벨/모집단이 다르면 성능 비교 무효)을 여기에도 적용한다.

## ambiguous의 구성 (매 봉 모집단 실측, 8트리거에서 재측정함)

  ambiguous 47.4% = 되반납 실패 29.4% + 약한이동 밴드(CHOP_MULT<=fast<ATR_MULT) 18.0%

문턱 이동은 거의 효과가 없다(T_SUSTAIN 0.20->0.30이 겨우 +2.5%p) -- 되반납 실패 대부분이
giveback 0.30보다 훨씬 큰 곳에 있기 때문. 그래서 문턱 완화가 아니라 **분류 방식**을 바꾼다.

## 축 (8개 구성)

  CHOP_MULT ∈ {1.00(현행), 1.25, 1.50}  -- **한 번도 스윕된 적 없는 축**.
      올리면 약한이동 밴드가 음성으로 편입된다. "1.2xATR 움직였다 만 것"은 V자반등보다
      "아무 일 없었다"에 가깝다는 게 논거. fail_neg가 건드린 덩어리와 **다른 쪽**이다.
  ambiguous 처리 ∈ {drop, fail_neg, all_neg}
      all_neg는 CHOP_MULT와 무관하므로(양성 아니면 전부 음성) 1개만 돈다 -> 3+3+1 = 7
  + **3class** (v_rebound / ambiguous / chop 3분류 후 P(v_rebound)를 점수로)  -- 미측정.
      버리거나 억지로 음성에 넣는 대신 "중간"을 모델이 명시적으로 배우게 한다.
  = 총 8개 구성

## ⚠️평가는 **고정 타깃**으로 한다

구성마다 라벨 정의가 달라지므로 자기 라벨 기준 AUC는 서로 비교 불가다. 전 구성을 **현행 정의
(K=1.5/T=0.20/CHOP=1.0, drop)의 라벨**에 대해 평가한다 -- 공통 잣대. 경제성은 원래 공통 단위다.
비교는 **동일 호출 빈도**에서(임계값 고정 금지 -- 오늘 3번 밟은 함정).

판정은 AUC가 아니라 **경제성 + 방향뒤집기**다.

⚠️OOS는 오늘 이미 여러 번 노출됐다. 여기서는 VAL을 1차 기준으로, OOS는 확인용으로 함께
찍되 **다중노출임을 명시**한다. HOLDOUT 미터치. 라이브 코드 변경 없음.

Run on the server (GPU) via handoff:
  handoff.sh launch server <job> -- python scripts/research_eth_v_rebound_8trigger_ambiguous_handling_20260901.py
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
_spec = importlib.util.spec_from_file_location("vreb_s1_amb", S1)
_s1 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_s1)
_feas, _bt, _vs = _s1._feas, _s1._bt, _s1._vs

FEATURE_COLUMNS = _s1.FEATURE_COLUMNS
EIGHT = [t for t in _feas.ALL9 if t != "local_extreme"]
STANDARD_COST_BP, FORWARD_BARS = _s1.STANDARD_COST_BP, _s1.FORWARD_BARS
SL_GRID, TRAIL_GRID = _bt.SL_GRID, _bt.TRAIL_GRID
ARM_GRID = tuple(a for a in _bt.ARM_GRID if a >= 1.0)   # 저ARM 아티팩트 제외

BASE = {"atr_mult": 1.50, "t_sustain": 0.20, "full_bars": 12}
CONTEXT_N, SEED = 18000, 20260829
TARGET_CALL_N_VAL = 600          # 전 구성 공통 호출 빈도(8트리거 풀 기준)
TRAIN_END = pd.Timestamp("2025-09-01", tz="UTC")
VAL_END = pd.Timestamp("2026-01-01", tz="UTC")
OOS_END = pd.Timestamp("2026-04-01", tz="UTC")

CONFIGS = ([{"name": f"drop|CHOP{c:.2f}", "ambig": "drop", "chop_mult": c} for c in (1.00, 1.25, 1.50)]
           + [{"name": f"fail_neg|CHOP{c:.2f}", "ambig": "fail_neg", "chop_mult": c} for c in (1.00, 1.25, 1.50)]
           + [{"name": "all_neg", "ambig": "all_neg", "chop_mult": 1.00},
              {"name": "3class", "ambig": "3class", "chop_mult": 1.00}])

OUT_JSON = ROOT / "data/research/eth_v_rebound_8trigger_matched_20260901/ambiguous_handling.json"


def log(msg: str) -> None:
    print(f"[amb] {msg}", flush=True)


def label_with(sig, is_down, *, chop_mult, ambig):
    """`_s1.label_param`을 CHOP_MULT까지 파라미터화. 산술은 원본 그대로."""
    close, high, low = (sig[c].to_numpy() for c in ("close", "high", "low"))
    atr = sig["atr"].to_numpy()
    extreme = low if is_down else high
    pre_atr = _vs.shifted_at(atr, -1)
    fb, K, T = BASE["full_bars"], BASE["atr_mult"], BASE["t_sustain"]
    fast_max = _vs.fwd_window(close, 1, 6, "max")
    fast_min = _vs.fwd_window(close, 1, 6, "min")
    fh = _vs.fwd_window(high, 1, fb, "max")
    fl = _vs.fwd_window(low, 1, fb, "min")
    end_price = _vs.shifted_at(close, fb)
    fast_move, peak = ((fast_max - extreme), fh) if is_down else ((extreme - fast_min), fl)
    with np.errstate(invalid="ignore", divide="ignore"):
        valid = (np.isfinite(pre_atr) & (pre_atr > 0) & np.isfinite(fh) & np.isfinite(fl)
                 & np.isfinite(end_price) & np.isfinite(extreme))
        fast_mult = fast_move / pre_atr
        denom = (peak - extreme) if is_down else (extreme - peak)
        gb = np.where(np.abs(denom) >= 1e-12,
                      (peak - end_price) / denom if is_down else (end_price - peak) / denom, np.nan)
        strong = fast_mult >= K
        is_v = strong & np.isfinite(gb) & (gb <= T)
        is_chop = fast_mult < chop_mult
        fail = strong & ~is_v
        if ambig in ("drop", "3class"):
            neg = is_chop
        elif ambig == "fail_neg":
            neg = is_chop | fail
        else:
            neg = ~is_v
    return np.where(~valid, "invalid", np.where(is_v, "v_rebound", np.where(neg, "chop", "ambiguous")))


def main() -> int:
    t0 = time.time()
    from tabpfn import TabPFNClassifier
    from sklearn.metrics import roc_auc_score
    import torch
    log(f"cuda: {torch.cuda.is_available()}  |  8트리거 모집단, ambiguous 처리 {len(CONFIGS)}개 구성")

    _s1.VAL_END = OOS_END
    log("building frame...")
    sig, feat, eth = _s1.build_sig()
    kl = eth[["timestamp", "open", "high", "low", "close"]].copy()
    kl["timestamp"] = kl["timestamp"].dt.tz_localize(None)
    ts_to_pos = {t: i for i, t in enumerate(kl["timestamp"].to_numpy())}
    o, h, l, c = (kl[x].to_numpy() for x in ("open", "high", "low", "close"))

    # 게이트 마스크
    gparts = []
    for side in ("bottom", "top"):
        g8 = np.any([sig[f"{side}_{t}"].fillna(False).to_numpy() for t in EIGHT], axis=0)
        gparts.append(pd.DataFrame({"timestamp": sig["timestamp"].to_numpy(), "side": side, "gate8": g8}))
    gates = pd.concat(gparts, ignore_index=True)

    # ⭐고정 평가 타깃: 현행 정의(CHOP=1.0, drop)의 라벨. 전 구성을 이것으로 평가한다.
    ref_b = label_with(sig, True, chop_mult=1.00, ambig="drop")
    ref_t = label_with(sig, False, chop_mult=1.00, ambig="drop")
    ref = _s1.long_frame_for(sig, feat, ref_b, ref_t)[["timestamp", "side", "label"]]
    ref = ref.rename(columns={"label": "ref_label"})

    results = []
    for cfg in CONFIGS:
        nm, ambig, cm = cfg["name"], cfg["ambig"], cfg["chop_mult"]
        sb = label_with(sig, True, chop_mult=cm, ambig=ambig)
        st = label_with(sig, False, chop_mult=cm, ambig=ambig)
        long = _s1.long_frame_for(sig, feat, sb, st)
        long = long.merge(gates, on=["timestamp", "side"], how="left").merge(ref, on=["timestamp", "side"], how="left")
        long["split"] = np.where(long["timestamp"] < TRAIN_END, "TRAIN",
                         np.where(long["timestamp"] < VAL_END, "VAL", "OOS"))
        pool = long.loc[long["gate8"].fillna(False)].copy()

        if ambig == "3class":
            cls = np.where(pool["status"] == "v_rebound", 2,
                   np.where(pool["status"] == "chop", 0, 1))
            pool["y"] = np.where(pool["status"] == "invalid", np.nan, cls)
        else:
            pool["y"] = pool["label"]
        tr = pool.loc[(pool["split"] == "TRAIN") & pool["y"].notna()]
        amb_pct = float((pool["status"] == "ambiguous").mean() * 100)
        log("")
        log(f"=== {nm}  (ambiguous {amb_pct:.1f}%, TRAIN {len(tr):,}행) ===")
        if len(tr) < 2000 or tr["y"].nunique() < 2:
            log("  표본/클래스 부족 -- 스킵"); continue

        rng = np.random.default_rng(SEED)
        idx = np.sort(rng.choice(len(tr), size=min(CONTEXT_N, len(tr)), replace=False))
        ctx = tr.iloc[idx]
        clf = TabPFNClassifier(device="cuda", random_state=SEED, ignore_pretraining_limits=True)
        clf.fit(ctx[FEATURE_COLUMNS], ctx["y"].to_numpy().astype(int))
        pos_col = list(clf.classes_).index(2 if ambig == "3class" else 1)

        rec = {"config": cfg, "ambiguous_pct": round(amb_pct, 1), "train_rows": int(len(tr)),
               "train_class_dist": {str(k): int(v) for k, v in tr["y"].value_counts().items()}}
        for sp in ("VAL", "OOS"):
            s = pool.loc[pool["split"] == sp].copy()
            s["p"] = clf.predict_proba(s[FEATURE_COLUMNS])[:, pos_col]
            # 고정 타깃 AUC (전 구성 공통 잣대)
            rl = s.loc[s["ref_label"].notna()]
            auc = float(roc_auc_score(rl["ref_label"], rl["p"])) if rl["ref_label"].nunique() == 2 else None
            # 동일 호출 빈도
            k = TARGET_CALL_N_VAL if sp == "VAL" else int(round(TARGET_CALL_N_VAL * len(s) / len(pool.loc[pool["split"] == "VAL"])))
            k = min(k, len(s))
            sel = s.nlargest(k, "p")
            rows = []
            for _, ev in sel.iterrows():
                i = ts_to_pos.get(np.datetime64(ev["timestamp"].tz_localize(None)))
                if i is None or i + FORWARD_BARS + 1 >= len(kl):
                    continue
                rows.append({"side": "long" if ev["is_downside"] == 1 else "short",
                             "atr": float(ev["atr"]), "entry_price": float(o[i + 1]),
                             "fwd_open": o[i+1:i+1+FORWARD_BARS], "fwd_high": h[i+1:i+1+FORWARD_BARS],
                             "fwd_low": l[i+1:i+1+FORWARD_BARS], "fwd_close": c[i+1:i+1+FORWARD_BARS]})
            df = pd.DataFrame(rows)
            if len(df) < 50:
                rec[sp] = {"auc_ref": round(auc, 4) if auc else None, "n": len(df), "skipped": True}
                log(f"  {sp}: 호출 {len(df)}건 -- 표본 부족"); continue
            e, a, s_, H, L, C = _bt.pack(df)
            ef, af, sf, Hf, Lf, Cf = _bt.pack(df, flip=True)
            cells = []
            for sl in SL_GRID:
                for arm in ARM_GRID:
                    for tr_ in TRAIL_GRID:
                        ov = _bt.simulate_trailing_vec(e, a, s_, H, L, C, sl, arm, tr_, False)
                        pv = _bt.simulate_trailing_vec(e, a, s_, H, L, C, sl, arm, tr_, True)
                        fo = _bt.simulate_trailing_vec(ef, af, sf, Hf, Lf, Cf, sl, arm, tr_, False)
                        fp = _bt.simulate_trailing_vec(ef, af, sf, Hf, Lf, Cf, sl, arm, tr_, True)
                        cells.append({"sl": sl, "arm": arm, "trail": tr_,
                                      "opt_bp": float(ov.mean()*1e4-STANDARD_COST_BP),
                                      "pess_bp": float(pv.mean()*1e4-STANDARD_COST_BP),
                                      "flip_opt_bp": float(fo.mean()*1e4-STANDARD_COST_BP),
                                      "flip_pess_bp": float(fp.mean()*1e4-STANDARD_COST_BP)})
            ok = lambda x: x["opt_bp"] > 0 and x["pess_bp"] > 0
            okf = lambda x: x["flip_opt_bp"] > 0 and x["flip_pess_bp"] > 0
            best = max(cells, key=lambda x: x["pess_bp"])
            rec[sp] = {"auc_ref": round(auc, 4) if auc else None, "n": len(df),
                       "n_cells": len(cells), "fwd_pass": sum(map(ok, cells)),
                       "flip_pass": sum(map(okf, cells)),
                       "best": {k2: round(v, 3) for k2, v in best.items()}}
            log(f"  {sp}: 고정타깃AUC {rec[sp]['auc_ref']}  n={len(df):>4,}  "
                f"정방향 {rec[sp]['fwd_pass']:>2d}/{len(cells)} 뒤집기 {rec[sp]['flip_pass']:>2d}/{len(cells)}  "
                f"최고 pess{best['pess_bp']:+.2f}bp (뒤집기 opt{best['flip_opt_bp']:+.2f})")
        results.append(rec)

    log("")
    log("=== 요약 (VAL 기준 정렬, 판정=경제성+방향뒤집기) ===")
    ok_res = [r for r in results if "VAL" in r and not r["VAL"].get("skipped")]
    ok_res.sort(key=lambda r: r["VAL"]["fwd_pass"] - r["VAL"]["flip_pass"], reverse=True)
    for r in ok_res:
        v, oo = r["VAL"], r.get("OOS", {})
        oos_txt = (f"OOS 정{oo['fwd_pass']:>2d}/뒤{oo['flip_pass']:>2d} pess{oo['best']['pess_bp']:+6.2f}"
                   if oo and not oo.get("skipped") else "OOS n/a")
        log(f"  {r['config']['name']:20s} amb{r['ambiguous_pct']:>5.1f}%  "
            f"VAL AUC {v['auc_ref']}  정{v['fwd_pass']:>2d}/뒤{v['flip_pass']:>2d} "
            f"pess{v['best']['pess_bp']:+6.2f}bp  |  {oos_txt}")
    log("")
    log("  ⚠️OOS는 오늘 다중 노출됨 -- 확인용이며 이것으로 재선택하지 않는다.")
    log("  기준선(현행 drop|CHOP1.00)과 비교해 정방향-뒤집기 격차가 뚜렷이 개선돼야 의미 있다.")

    report = {"signal": "v_rebound_8trigger_ambiguous_handling", "asset": "ETHUSDT",
              "scope": {"population": "8트리거 게이트 (매 봉 아님 -- 이전 기각과 다른 모집단)",
                        "eval_target": "고정: 현행 정의(CHOP=1.0, drop) 라벨",
                        "matched_call_n_val": TARGET_CALL_N_VAL,
                        "arm_floor": 1.0, "oos_multiple_exposure": True,
                        "holdout_touched": False, "live_code_changed": False},
              "configs": CONFIGS, "results": results, "runtime_sec": round(time.time()-t0, 1)}
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str))
    log("")
    log(f"report saved -> {OUT_JSON}  ({report['runtime_sec']}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
