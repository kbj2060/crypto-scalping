#!/usr/bin/env python3
"""진입가 앵커 라벨의 **꼬리 구간 기대값** 측정 -- 통과셀 수가 아니라 bp/트레이드로.

## 왜 이 실험이 필요한가 (선행 실험의 구멍)

`..._entry_anchor_label_20260902.py`는 진입가 앵커 변형들의 호출 수를 **대조군에 맞춰**
(VAL 1,693 / OOS 1,367) 상위 n개만 취했다 -- **임계값 스윕을 하지 않았다.** 같은 세션에서
wick 모델에는 꼬리 조사(thr 0.60->0.90)를 해놓고 진입 앵커 모델에는 안 한 것은 일관성 결함이다.

그리고 꼬리가 정확히 문제가 되는 자리다: **전체 AUC 0.58과 "상위 5%에서 쓸 만한 엣지"는
얼마든지 공존한다.** 실제로 wick 모델에서 그 패턴이 관측됐다(전체는 밋밋한데 thr>=0.75
꼬리는 랜덤부분표집 귀무분포 100백분위 -- OOS 전이는 실패했지만 VAL에서는 진짜였다).

## 지표를 바꾼다 -- 통과셀 수가 아니라 기대값

사용자 지적(2026-09-02): "58%는 무의미한 확률이 아니야. 55%만 되도 찰리 멍거는 전재산을 건다."
맞는 지적이고, 그 논증의 정답 지표는 **승률도 AUC도 아니라 트레이드당 기대값**이다 --
왕복 10bp를 물고 대칭 배당이면 55% 승률은 지는 베팅이고, 배당이 2:1이면 40%도 이긴다.
그래서 이번엔 전부 같이 낸다:

  - **기대값(bp/트레이드)**: 비용 차감 후 평균. 이게 헤드라인.
  - **승률 + 손익비**: 평균이익/평균손실. 55%가 의미 있으려면 배당이 받쳐야 한다.
  - **십분위별 기대값**: 꼬리가 몸통보다 나은가 (단조성)
  - **임계값 스윕 0.50~0.90**: n이 줄어드는 대가로 기대값이 오르는가
  - **랜덤 부분표집 귀무분포(B=200)**: 꼬리 우세가 표본크기 부산물인지
    (2026-09-02 확립: 통과수/기대값 모두 n이 줄면 불안정해진다)

셀은 **VAL에서만** 고르고 OOS는 그 조합으로 1회 평가. 저ARM(<1.0) 제외.
⚠️HOLDOUT 미터치. 라이브 코드 변경 없음.

Run on the server (GPU) via handoff.
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


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


_s1 = _load("s1_tailexp", "scripts/research_eth_v_rebound_label_grid_screen_stage1_20260901.py")
_ea = _load("ea_tailexp", "scripts/research_eth_v_rebound_entry_anchor_label_20260902.py")
_bt = _s1._bt
FEATURE_COLUMNS = _s1.FEATURE_COLUMNS
FORWARD_BARS = _s1.FORWARD_BARS
SL_GRID, ARM_GRID, TRAIL_GRID = _bt.SL_GRID, _bt.ARM_GRID, _bt.TRAIL_GRID

ATR_MULTS = [0.75, 1.0, 1.5]
# ⚠️고정 확률 임계값은 무효다 -- 진입앵커 모델은 확률이 0.5를 넘지 않는다(최상위 십분위
# p중앙 0.226). 모델마다 확률 분포가 다르므로 **상위 분위**로 잘라야 비교가 성립한다.
# 이 세션에서 같은 함정을 이미 세 번 밟았다.
TAIL_FRACTIONS = [0.10, 0.05, 0.02, 0.01, 0.005]
CONTEXT_N, SEED = 18000, 20260829
COST_BP, ARTIFACT_FREE_MIN = 10.0, 1.0
NULL_B, NULL_SEED, NULL_MIN_N = 200, 20260902, 60
TRAIN_END = pd.Timestamp("2025-09-01", tz="UTC")
VAL_END = pd.Timestamp("2026-01-01", tz="UTC")
OOS_END = pd.Timestamp("2026-04-01", tz="UTC")
OUT_JSON = ROOT / "data/research/eth_v_rebound_entry_anchor_tail_20260902/report.json"


def log(m): print(f"[tailexp] {m}", flush=True)


def main() -> int:
    t0 = time.time()
    from tabpfn import TabPFNClassifier
    from sklearn.metrics import roc_auc_score
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

    def build(s):
        rows = []
        for i_, isd, atr_ in zip(s["pos"].to_numpy(), s["is_downside"].to_numpy(),
                                  s["atr"].to_numpy()):
            i = int(i_)
            if i + FORWARD_BARS + 1 >= nk:
                continue
            rows.append({"side": "long" if isd == 1 else "short", "atr": float(atr_),
                         "entry_price": float(o[i+1]),
                         "fwd_open": o[i+1:i+1+FORWARD_BARS], "fwd_high": h[i+1:i+1+FORWARD_BARS],
                         "fwd_low": l[i+1:i+1+FORWARD_BARS], "fwd_close": c[i+1:i+1+FORWARD_BARS]})
        return pd.DataFrame(rows)

    def cells(df):
        """전 셀의 (정방향/뒤집기) 기대값·승률·손익비를 낸다."""
        if len(df) < 30:
            return None
        e, a, s_, H, L, C = _bt.pack(df)
        ef, af, sf, Hf, Lf, Cf = _bt.pack(df, flip=True)
        out = []
        for sl in SL_GRID:
            for arm in ARM_GRID:
                if arm < ARTIFACT_FREE_MIN:
                    continue
                for tr in TRAIL_GRID:
                    ov = _bt.simulate_trailing_vec(e, a, s_, H, L, C, sl, arm, tr, False)
                    pv = _bt.simulate_trailing_vec(e, a, s_, H, L, C, sl, arm, tr, True)
                    fp = _bt.simulate_trailing_vec(ef, af, sf, Hf, Lf, Cf, sl, arm, tr, True)
                    fo = _bt.simulate_trailing_vec(ef, af, sf, Hf, Lf, Cf, sl, arm, tr, False)
                    net = pv * 1e4 - COST_BP                    # 비관 기준 트레이드별 순손익(bp)
                    win = net > 0
                    wr = float(win.mean())
                    avg_w = float(net[win].mean()) if win.any() else 0.0
                    avg_l = float(-net[~win].mean()) if (~win).any() else 0.0
                    out.append({"sl": sl, "arm": arm, "trail": tr,
                                "exp_bp": float(net.mean()),            # ⭐헤드라인: 기대값
                                "opt_bp": float(ov.mean()*1e4-COST_BP),
                                "win_rate": wr, "avg_win_bp": avg_w, "avg_loss_bp": avg_l,
                                "payoff": (avg_w/avg_l) if avg_l > 0 else float("inf"),
                                "flip_exp_bp": float(fp.mean()*1e4-COST_BP),
                                "flip_opt_bp": float(fo.mean()*1e4-COST_BP)})
        return out

    def summarize(cs):
        best = max(cs, key=lambda x: x["exp_bp"])
        fwd = sum(1 for x in cs if x["exp_bp"] > 0 and x["opt_bp"] > 0)
        flip = sum(1 for x in cs if x["flip_exp_bp"] > 0 and x["flip_opt_bp"] > 0)
        return {"best": best, "fwd_pass": fwd, "flip_pass": flip, "margin": fwd - flip,
                "n_cells": len(cs)}

    report = {"signal": "v_rebound_entry_anchor_tail_expectancy", "asset": "ETHUSDT",
              "fresh_forward_bar_by_bar": True, "trade_ledgers_used_as_input": False,
              "saved_parent_exit_timestamps_used": False, "future_rows_used_for_entry": False,
              "scope": {"anchor": "open[i+1] (진입가)", "atr_mults": ATR_MULTS,
                        "tail_fractions": TAIL_FRACTIONS, "cost_bp": COST_BP,
                        "headline_metric": "기대값 bp/트레이드 (비관 기준)",
                        "cell_selected_on": "VAL only (ARM>=1.0)", "reoptimized_on_oos": False,
                        "null_B": NULL_B, "holdout_touched": False, "live_code_changed": False},
              "variants": {}}

    nrng = np.random.default_rng(NULL_SEED)
    for am in ATR_MULTS:
        name = f"E_entry_{am:.2f}"
        sbv = _ea.label_entry_anchor(sig, True, am)
        stv = _ea.label_entry_anchor(sig, False, am)
        lf = _s1.long_frame_for(sig, feat, sbv, stv)
        lf["split"] = np.where(lf["timestamp"] < TRAIN_END, "TRAIN",
                        np.where(lf["timestamp"] < VAL_END, "VAL", "OOS"))
        assert lf["timestamp"].max() < OOS_END, "HOLDOUT 누출"
        lf["pos"] = [pos_of.get(np.datetime64(t.tz_localize(None)), -1) for t in lf["timestamp"]]
        lf = lf.loc[lf["pos"] >= 0].reset_index(drop=True)
        tr_ = lf.loc[(lf["split"] == "TRAIN") & lf["label"].notna()]
        rng = np.random.default_rng(SEED)
        ctx = tr_.iloc[np.sort(rng.choice(len(tr_), size=min(CONTEXT_N, len(tr_)), replace=False))]
        clf = TabPFNClassifier(device="cuda", random_state=SEED, ignore_pretraining_limits=True)
        clf.fit(ctx[FEATURE_COLUMNS], ctx["label"].to_numpy())

        log("")
        log(f"########## {name} (chop<{am*_ea.CHOP_RATIO:.2f})  TRAIN 라벨행 {len(tr_):,} ##########")
        ent, scored = {}, {}
        for spn in ("VAL", "OOS"):
            s = lf.loc[lf["split"] == spn].copy()
            CH = 20000
            s["p"] = np.concatenate([clf.predict_proba(s[FEATURE_COLUMNS].iloc[k:k+CH])[:, 1]
                                     for k in range(0, len(s), CH)])
            scored[spn] = s
            sl_ = s.loc[s["label"].notna()]
            ent[f"{spn}_auc"] = round(float(roc_auc_score(sl_["label"], sl_["p"])), 4) \
                if sl_["label"].nunique() == 2 else None

        # ---- 십분위 기대값 (셀 고정 없이: 전 셀 중앙값으로 요약) ----
        log("  --- 십분위별 기대값 (VAL 최적셀 기준, 비용 차감) ---")
        vcells = cells(build(scored["VAL"]))
        if vcells is None:
            log("  표본 부족 -- 건너뜀"); continue
        vb = max(vcells, key=lambda x: x["exp_bp"])
        log(f"    VAL 전체 최적셀 SL/ARM/Tr={vb['sl']}/{vb['arm']}/{vb['trail']}  "
            f"기대값 {vb['exp_bp']:+.2f}bp  승률 {vb['win_rate']*100:.1f}%  "
            f"손익비 {vb['payoff']:.2f}  (뒤집기 {vb['flip_exp_bp']:+.2f}bp)")
        dec = {}
        for spn in ("VAL", "OOS"):
            s = scored[spn]
            qd = pd.qcut(s["p"], 10, labels=False, duplicates="drop")
            rows = []
            for d in sorted(pd.unique(qd.dropna())):
                sub = s.loc[qd == d]
                df = build(sub)
                if len(df) < 30:
                    continue
                e, a, sg, H, L, C = _bt.pack(df)
                net = _bt.simulate_trailing_vec(e, a, sg, H, L, C, vb["sl"], vb["arm"],
                                                vb["trail"], True) * 1e4 - COST_BP
                w = net > 0
                rows.append({"decile": int(d), "n": int(len(df)),
                             "p_median": round(float(sub["p"].median()), 4),
                             "exp_bp": round(float(net.mean()), 2),
                             "win_rate": round(float(w.mean()), 4),
                             "payoff": round(float(net[w].mean() / -net[~w].mean()), 3)
                             if w.any() and (~w).any() else None})
            dec[spn] = rows
            log(f"    {spn}: " + " ".join(f"[{r['decile']}]{r['exp_bp']:+.1f}" for r in rows))
            if rows:
                top = rows[-1]
                log(f"      최상위 분위 n={top['n']:,} p중앙 {top['p_median']:.3f}  "
                    f"기대값 {top['exp_bp']:+.2f}bp  승률 {top['win_rate']*100:.1f}%  "
                    f"손익비 {top['payoff']}")
        ent["deciles"] = dec

        # ---- 임계값 스윕 + 귀무분포 ----
        log("  --- 임계값 스윕 (VAL) ---")
        sweep, hold = {}, {}
        for frac in TAIL_FRACTIONS:
            k = max(1, int(round(len(scored["VAL"]) * frac)))
            sel = scored["VAL"].nlargest(k, "p")
            cut = float(sel["p"].min())
            df = build(sel)
            if len(df) < 30:
                log(f"    상위{frac*100:.1f}%  n={len(df)} -- 표본 부족"); continue
            cs = cells(df); sm = summarize(cs); b = sm["best"]
            hold[frac] = (sel, cs, sm, cut)
            pct = None
            if len(df) >= NULL_MIN_N and len(scored["VAL"]) > len(sel):
                exps = []
                for _ in range(NULL_B):
                    ridx = nrng.choice(len(scored["VAL"]), size=len(sel), replace=False)
                    rc = cells(build(scored["VAL"].iloc[np.sort(ridx)]))
                    if rc:
                        exps.append(max(x["exp_bp"] for x in rc))
                if len(exps) >= 20:
                    pct = round(float((np.array(exps) < b["exp_bp"]).mean() * 100), 1)
            sweep[f"top_{frac*100:g}pct"] = {"n": int(len(df)), "p_cutoff": round(cut, 4), "best": {k: (round(v, 3)
                                   if isinstance(v, float) else v) for k, v in b.items()},
                                   "fwd_pass": sm["fwd_pass"], "flip_pass": sm["flip_pass"],
                                   "margin": sm["margin"], "null_pctile_expbp": pct}
            log(f"    상위{frac*100:>4.1f}% (p>={cut:.3f})  n={len(df):>5,}  기대값 {b['exp_bp']:+6.2f}bp  "
                f"승률 {b['win_rate']*100:4.1f}%  손익비 {b['payoff']:.2f}  "
                f"정{sm['fwd_pass']:>2d}/뒤{sm['flip_pass']:>2d}  "
                f"귀무 {(f'{pct:.1f}%' if pct is not None else 'n/a'):>7s}"
                f"{'  ✅' if (pct or 0) >= 95 and b['exp_bp'] > 0 else ''}")
        ent["val_sweep"] = sweep

        # ---- VAL 최선 -> OOS 1회 ----
        cand = [(t, v) for t, v in hold.items() if v[2]["best"]["exp_bp"] > 0]
        if cand:
            frac, (sel, cs, sm, cut) = max(cand, key=lambda kv: kv[1][2]["best"]["exp_bp"])
            b = sm["best"]
            # OOS도 **같은 분위**로 자른다(같은 확률컷을 쓰면 분포 이동 때문에 n이 달라진다)
            ok_ = max(1, int(round(len(scored["OOS"]) * frac)))
            osel = scored["OOS"].nlargest(ok_, "p")
            odf = build(osel)
            if len(odf) >= 30:
                oc = cells(odf)
                og = [x for x in oc if (x["sl"], x["arm"], x["trail"]) == (b["sl"], b["arm"], b["trail"])][0]
                osm = summarize(oc)
                log(f"  --- OOS 1회 (VAL 선정 상위{frac*100:g}%, "
                    f"SL/ARM/Tr={b['sl']}/{b['arm']}/{b['trail']}) ---")
                log(f"    n={len(odf):,}  기대값 {og['exp_bp']:+.2f}bp  승률 {og['win_rate']*100:.1f}%  "
                    f"손익비 {og['payoff']:.2f}  뒤집기 {og['flip_exp_bp']:+.2f}bp  "
                    f"격자 정{osm['fwd_pass']}/뒤{osm['flip_pass']}")
                ent["oos"] = {"tail_fraction": frac, "cell": {k: b[k] for k in ("sl", "arm", "trail")},
                              "n": int(len(odf)),
                              "result": {k: (round(v, 3) if isinstance(v, float) else v)
                                         for k, v in og.items()},
                              "grid_fwd_pass": osm["fwd_pass"], "grid_flip_pass": osm["flip_pass"]}
        else:
            log("  --- OOS: VAL에서 기대값 양수인 분위 없음 -- 넘길 후보 없음 ---")
        report["variants"][name] = ent

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    report["runtime_sec"] = round(time.time() - t0, 1)
    OUT_JSON.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str))
    log("")
    log(f"report saved -> {OUT_JSON}  ({report['runtime_sec']}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
