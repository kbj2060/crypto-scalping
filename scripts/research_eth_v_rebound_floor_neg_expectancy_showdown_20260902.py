#!/usr/bin/env python3
"""F_neg 하한별 **정방향 기대값 vs 뒤집기 기대값** 정면 비교 + 귀무분포.

## 왜 이 측정이 필요한가 (선행 실험의 판정 지표 결함)

`..._entry_relative_bp_floor_neg_20260902.py`는 "80셀 중 통과 개수"로 판정했다. 개수는
**포화되고 크기를 못 담는다** -- FLOOR=40에서 정방향 최고가 +17.34bp인데 뒤집기 최고가
얼마인지는 재지 않았다. 둘이 비슷하면 방향 무관(종결)이고, 뒤집기가 훨씬 작으면 개수로만
밀렸을 뿐 **크기로는 정방향이 이긴다**.

사용자가 기대값 관점을 제기한 이후(2026-09-02, "55%만 되도 찰리 멍거는 전재산을 건다")
기대값이 옳은 지표라고 합의해놓고 정작 그 실험은 개수로 판정했다. 여기서 바로잡는다.

## 무엇을 재나 -- 전부 bp/트레이드

  A. **최고 기대값**: 정방향 best vs 뒤집기 best (80셀 max)
  B. **중앙 기대값**: 80셀의 중앙값 -- max 선택편향이 없는 강건 비교. ⭐A보다 이걸 먼저 볼 것.
  C. **승률/손익비**: 정방향 최고셀에서
  D. **귀무분포(B=200)**: 같은 모집단에서 무작위 n개를 뽑아 같은 절차를 반복.
     `best_fwd`와 **`best_fwd - best_flip`(갭)** 두 통계의 백분위를 낸다.
     갭의 백분위가 결정적이다 -- "무작위로 골라도 이 정도 갭이 나오는가".

FLOOR는 60까지 늘린다(40에서 추세가 아직 상승 중이었다).

⚠️호출 빈도는 대조군(FLOOR=0, thr 0.60)에 일치. 셀 선정은 VAL에서만. ARM>=1.0.
⚠️HOLDOUT 미터치. 라이브 코드 변경 없음.

Run on the server via handoff.
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


def _load(n, r):
    s = importlib.util.spec_from_file_location(n, ROOT / r)
    m = importlib.util.module_from_spec(s)
    s.loader.exec_module(m)
    return m


_s1 = _load("s1_show", "scripts/research_eth_v_rebound_label_grid_screen_stage1_20260901.py")
_vs, _bt = _s1._vs, _s1._bt
FEATURE_COLUMNS = _s1.FEATURE_COLUMNS
FAST_BARS, FORWARD_BARS = _s1.FAST_BARS_FIXED, _s1.FORWARD_BARS
SL_GRID, ARM_GRID, TRAIL_GRID = _bt.SL_GRID, _bt.ARM_GRID, _bt.TRAIL_GRID

DEPLOYED = {"atr_mult": 1.50, "t_sustain": 0.20, "full_bars": 12}
FLOORS = [0, 20, 40, 60]
BASE_THR, CONTEXT_N, SEED = 0.60, 18000, 20260829
COST_BP, ARTIFACT_FREE_MIN = 10.0, 1.0
NULL_B, NULL_SEED = 200, 20260902
TRAIN_END = pd.Timestamp("2025-09-01", tz="UTC")
VAL_END = pd.Timestamp("2026-01-01", tz="UTC")
OOS_END = pd.Timestamp("2026-04-01", tz="UTC")
OUT = ROOT / "data/research/eth_v_rebound_floor_neg_showdown_20260902/report.json"


def log(m): print(f"[showdown] {m}", flush=True)


def main() -> int:
    t0 = time.time()
    from tabpfn import TabPFNClassifier
    import torch
    log(f"cuda: {torch.cuda.is_available()}")

    _s1.VAL_END = OOS_END
    log("building frame ...")
    sig, feat, eth = _s1.build_sig()
    kl = eth[["timestamp", "open", "high", "low", "close"]].copy()
    kl["timestamp"] = kl["timestamp"].dt.tz_localize(None)
    pos_of = {t: i for i, t in enumerate(kl["timestamp"].to_numpy())}
    o, h, l, c = (kl[x].to_numpy() for x in ("open", "high", "low", "close"))
    nk = len(kl)

    close, high, low = (sig[x].to_numpy() for x in ("close", "high", "low"))
    op = sig["open"].to_numpy()
    fmax = _vs.fwd_window(close, 1, FAST_BARS, "max")
    fmin = _vs.fwd_window(close, 1, FAST_BARS, "min")
    ent_arr = _vs.shifted_at(op, 1)
    CAP = {True: (fmax - ent_arr) / ent_arr * 1e4, False: (ent_arr - fmin) / ent_arr * 1e4}
    base = {d: _s1.label_param(sig, d, ambig="drop", anchor="wick", **DEPLOYED) for d in (True, False)}

    def frame_for(floor):
        sts = {}
        for d in (True, False):
            st = base[d].copy()
            if floor > 0:
                st[(st == "v_rebound") & np.isfinite(CAP[d]) & (CAP[d] < floor)] = "chop"
            sts[d] = st
        lf = _s1.long_frame_for(sig, feat, sts[True], sts[False])
        lf["split"] = np.where(lf["timestamp"] < TRAIN_END, "TRAIN",
                        np.where(lf["timestamp"] < VAL_END, "VAL", "OOS"))
        assert lf["timestamp"].max() < OOS_END, "HOLDOUT 누출"
        lf["pos"] = [pos_of.get(np.datetime64(t.tz_localize(None)), -1) for t in lf["timestamp"]]
        return lf.loc[lf["pos"] >= 0].reset_index(drop=True)

    def build(s):
        rows = []
        for i_, isd, atr_ in zip(s["pos"].to_numpy(), s["is_downside"].to_numpy(), s["atr"].to_numpy()):
            i = int(i_)
            if i + FORWARD_BARS + 1 >= nk:
                continue
            rows.append({"side": "long" if isd == 1 else "short", "atr": float(atr_),
                         "entry_price": float(o[i+1]),
                         "fwd_open": o[i+1:i+1+FORWARD_BARS], "fwd_high": h[i+1:i+1+FORWARD_BARS],
                         "fwd_low": l[i+1:i+1+FORWARD_BARS], "fwd_close": c[i+1:i+1+FORWARD_BARS]})
        return pd.DataFrame(rows)

    def duel(df):
        """80셀 각각의 정방향/뒤집기 기대값(비관 기준)을 낸다."""
        if len(df) < 30:
            return None
        e, a, s_, H, L, C = _bt.pack(df)
        ef, af, sf, Hf, Lf, Cf = _bt.pack(df, flip=True)
        fw, fl_, meta = [], [], []
        for sl in SL_GRID:
            for arm in ARM_GRID:
                if arm < ARTIFACT_FREE_MIN:
                    continue
                for tr in TRAIL_GRID:
                    pv = _bt.simulate_trailing_vec(e, a, s_, H, L, C, sl, arm, tr, True)
                    fp = _bt.simulate_trailing_vec(ef, af, sf, Hf, Lf, Cf, sl, arm, tr, True)
                    net = pv * 1e4 - COST_BP
                    w = net > 0
                    fw.append(float(net.mean())); fl_.append(float(fp.mean()*1e4-COST_BP))
                    meta.append({"sl": sl, "arm": arm, "trail": tr,
                                 "win_rate": float(w.mean()),
                                 "payoff": float(net[w].mean() / -net[~w].mean())
                                 if w.any() and (~w).any() else None})
        fw, fl_ = np.array(fw), np.array(fl_)
        bi = int(fw.argmax())
        return {"n": int(len(df)), "best_fwd": float(fw.max()), "best_flip": float(fl_.max()),
                "gap_best": float(fw.max() - fl_.max()),
                "med_fwd": float(np.median(fw)), "med_flip": float(np.median(fl_)),
                "gap_med": float(np.median(fw) - np.median(fl_)),
                "best_cell": meta[bi], "flip_at_best_fwd_cell": float(fl_[bi])}

    report = {"signal": "v_rebound_floor_neg_expectancy_showdown", "asset": "ETHUSDT",
              "fresh_forward_bar_by_bar": True, "trade_ledgers_used_as_input": False,
              "saved_parent_exit_timestamps_used": False, "future_rows_used_for_entry": False,
              "scope": {"treatment": "capturable_bp < FLOOR 양성 -> 음성(0)",
                        "metric": "기대값 bp/트레이드 (비관), 정방향 vs 뒤집기 정면비교",
                        "floors": FLOORS, "cost_bp": COST_BP, "null_B": NULL_B,
                        "note": "선행 실험은 통과 '개수'로 판정 -- 크기를 못 담아 여기서 재측정",
                        "holdout_touched": False, "live_code_changed": False},
              "variants": {}}

    nrng = np.random.default_rng(NULL_SEED)
    target_n = {}
    for fl in FLOORS:
        lfr = frame_for(fl)
        tr_ = lfr.loc[(lfr["split"] == "TRAIN") & lfr["label"].notna()]
        rng = np.random.default_rng(SEED)
        ctx = tr_.iloc[np.sort(rng.choice(len(tr_), size=min(CONTEXT_N, len(tr_)), replace=False))]
        clf = TabPFNClassifier(device="cuda", random_state=SEED, ignore_pretraining_limits=True)
        clf.fit(ctx[FEATURE_COLUMNS], ctx["label"].to_numpy())
        log("")
        log(f"########## FLOOR={fl}bp  TRAIN 라벨행 {len(tr_):,} 라벨률 {tr_['label'].mean():.4f} ##########")
        ent = {"train_n": int(len(tr_)), "label_rate": round(float(tr_["label"].mean()), 5)}
        for spn in ("VAL", "OOS"):
            s = lfr.loc[lfr["split"] == spn].copy()
            CH = 20000
            s["p"] = np.concatenate([clf.predict_proba(s[FEATURE_COLUMNS].iloc[k:k+CH])[:, 1]
                                     for k in range(0, len(s), CH)])
            if fl == 0:
                target_n[spn] = int((s["p"] >= BASE_THR).sum())
            sel = s.nlargest(min(target_n[spn], len(s)), "p")
            d = duel(build(sel))
            # 귀무분포: 같은 모집단에서 무작위 n개
            gaps, bests = [], []
            for _ in range(NULL_B):
                ridx = nrng.choice(len(s), size=len(sel), replace=False)
                rd = duel(build(s.iloc[np.sort(ridx)]))
                if rd:
                    gaps.append(rd["gap_best"]); bests.append(rd["best_fwd"])
            pg = round(float((np.array(gaps) < d["gap_best"]).mean()*100), 1) if len(gaps) >= 20 else None
            pb = round(float((np.array(bests) < d["best_fwd"]).mean()*100), 1) if len(bests) >= 20 else None
            bc = d["best_cell"]
            log(f"  {spn} n={d['n']:,}")
            log(f"    ⭐최고 기대값   정방향 {d['best_fwd']:+7.2f}bp   뒤집기 {d['best_flip']:+7.2f}bp   "
                f"갭 {d['gap_best']:+7.2f}bp   (귀무 {pg}%)")
            log(f"       중앙 기대값   정방향 {d['med_fwd']:+7.2f}bp   뒤집기 {d['med_flip']:+7.2f}bp   "
                f"갭 {d['gap_med']:+7.2f}bp")
            log(f"       정방향 최고셀 SL/ARM/Tr={bc['sl']}/{bc['arm']}/{bc['trail']}  "
                f"승률 {bc['win_rate']*100:.1f}%  손익비 "
                f"{bc['payoff']:.3f}" if bc["payoff"] else "  손익비 n/a")
            log(f"       같은 셀에서 뒤집기 {d['flip_at_best_fwd_cell']:+.2f}bp   "
                f"(best_fwd 귀무 {pb}%)")
            ent[spn] = {**{k: (round(v, 3) if isinstance(v, float) else v) for k, v in d.items()},
                        "null_pctile_gap": pg, "null_pctile_best_fwd": pb}
        report["variants"][f"floor_{fl}"] = ent

    log("")
    log("=== 판정 ===")
    log("  ⭐중앙 기대값 갭이 양수여야 '크기로 정방향이 이긴다'가 성립한다(최고값은 max 선택편향).")
    log("  그리고 갭의 귀무 백분위가 95% 이상이어야 무작위 선택과 구별된다.")
    for fl in FLOORS:
        v = report["variants"][f"floor_{fl}"]
        vv, oo = v["VAL"], v["OOS"]
        ok = vv["gap_med"] > 0 and oo["gap_med"] > 0 and (vv["null_pctile_gap"] or 0) >= 95 \
             and (oo["null_pctile_gap"] or 0) >= 95
        log(f"  {'✅' if ok else '  '}FLOOR={fl:>2d}bp  VAL 중앙갭 {vv['gap_med']:+7.2f}bp(귀무 {vv['null_pctile_gap']}%)  "
            f"OOS 중앙갭 {oo['gap_med']:+7.2f}bp(귀무 {oo['null_pctile_gap']}%)")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    report["runtime_sec"] = round(time.time() - t0, 1)
    OUT.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str))
    log("")
    log(f"report saved -> {OUT}  ({report['runtime_sec']}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
