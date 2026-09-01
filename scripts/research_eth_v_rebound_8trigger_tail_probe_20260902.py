#!/usr/bin/env python3
"""8트리거 일치 구성의 **확률 꼬리 구간**(thr>=0.65) 정밀 조사.

## 왜 꼬리인가

2026-09-01 8트리거 실험에서 정방향이 뒤집기를 이긴 지점은 **thr=0.70 (VAL n=411,
정25/80 vs 뒤22/80) 단 하나**였다. 그보다 느슨한 모든 지점(호출 600건 기준 8개 ambiguous
구성 전부 포함)에서는 뒤집기가 이겼다. 그래서 남은 질문은 하나다:

  **꼬리의 정25 vs 뒤22는 진짜 방향성인가, 아니면 n=411짜리 격자 80셀의 노이즈인가?**

임계값을 더 올려 스윕하는 것만으로는 이 질문에 답할 수 없다. n이 같이 줄어들어 **격자
통과수 자체가 불안정**해지기 때문이다. 그래서 세 축을 같이 잰다.

## 세 축

  A. **꼬리 미세 스윕** (0.60~0.90): 각 임계값의 VAL 호출수, 정방향/뒤집기 격자 통과수,
     최고 pess. 여기에 **사건 단위 중복제거 수**(GAP=12봉)를 같이 찍는다 -- 호출 411건이
     실제로는 40개 사건의 클러스터일 수 있고, 그러면 유효 표본은 411이 아니라 40이다.

  B. ⭐**확률 십분위 단조성** -- 격자를 아예 쓰지 않는 검사. 8트리거풀 전체를 모델 확률로
     십분위 분할하고, 각 분위의 **신호 방향 기준 원시 선도수익**(6/12/24/48봉)을 잰다.
     확률에 방향성이 실려 있다면 상위 분위가 하위 분위보다 높아야 한다. 트레일링/격자/
     셀선택이 전혀 개입하지 않으므로 **아티팩트에서 자유로운 유일한 검사**다.
     여기서 단조성이 없으면 꼬리의 격자 통과수는 볼 필요가 없다.

  C. ⭐**랜덤 부분표집 귀무분포** -- 같은 풀에서 **모델 확률과 무관하게** n개를 무작위
     추출해 똑같은 격자 절차를 B회 반복한다. 모집단·표본크기·격자·비용이 전부 고정되고
     "모델이 골랐는가"만 달라진다. 관측된 (정-뒤)와 최고 pess가 이 귀무분포의 몇 백분위인지
     계산한다. 백분위가 낮으면 꼬리의 우세는 **선택이 아니라 표본크기의 부산물**이다.

## 프로토콜

임계값·셀은 **VAL에서만 선정**, OOS는 그 조합으로 1회 평가(재최적화 금지).
저ARM(<1.0)은 노이즈수확 아티팩트 구간이므로 격자에서 제외.
⚠️HOLDOUT 미터치. 라이브 코드 변경 없음.

Run on the server (GPU) via handoff:
  handoff.sh launch server <job> -- python scripts/research_eth_v_rebound_8trigger_tail_probe_20260902.py
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
_spec = importlib.util.spec_from_file_location("vreb_s1_tail", S1)
_s1 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_s1)
_feas, _bt, _vs = _s1._feas, _s1._bt, _s1._vs

FEATURE_COLUMNS = _s1.FEATURE_COLUMNS
ALL9 = _feas.ALL9
EIGHT = [t for t in ALL9 if t != "local_extreme"]
STANDARD_COST_BP, FORWARD_BARS = _s1.STANDARD_COST_BP, _s1.FORWARD_BARS
SL_GRID, ARM_GRID, TRAIL_GRID = _bt.SL_GRID, _bt.ARM_GRID, _bt.TRAIL_GRID

W = 6
DEPLOYED = {"atr_mult": 1.50, "t_sustain": 0.20, "full_bars": 12}
CONTEXT_N, SEED = 18000, 20260829
ARTIFACT_FREE_MIN = 1.0

TAIL_THRESHOLDS = [0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90]
DEDUP_GAP_BARS = 12          # 사건 단위 중복제거 간격 (기존 감사와 동일)
DECILE_HORIZONS = [6, 12, 24, 48]
NULL_B = 200                 # 랜덤 부분표집 반복수
NULL_MIN_N = 100             # 이보다 호출이 적으면 귀무분포를 돌리지 않는다
NULL_SEED = 20260902

TRAIN_END = pd.Timestamp("2025-09-01", tz="UTC")
VAL_END = pd.Timestamp("2026-01-01", tz="UTC")
OOS_END = pd.Timestamp("2026-04-01", tz="UTC")

OUT_JSON = ROOT / "data/research/eth_v_rebound_8trigger_tail_20260902/report.json"


def log(msg: str) -> None:
    print(f"[tail] {msg}", flush=True)


def main() -> int:
    t0 = time.time()
    from tabpfn import TabPFNClassifier
    from sklearn.metrics import roc_auc_score
    import torch
    log(f"cuda: {torch.cuda.is_available()}")
    log(f"8트리거: {', '.join(EIGHT)}")

    _s1.VAL_END = OOS_END
    log("building frame + labels ...")
    sig, feat, eth = _s1.build_sig()
    sb = _s1.label_param(sig, True, ambig="drop", anchor="wick", **DEPLOYED)
    st = _s1.label_param(sig, False, ambig="drop", anchor="wick", **DEPLOYED)
    long = _s1.long_frame_for(sig, feat, sb, st)
    long["split"] = np.where(long["timestamp"] < TRAIN_END, "TRAIN",
                     np.where(long["timestamp"] < VAL_END, "VAL", "OOS"))
    assert long["timestamp"].max() < OOS_END, "HOLDOUT 누출"

    # --- 8트리거 게이트 ---
    parts = []
    for side in ("bottom", "top"):
        g8 = np.any([sig[f"{side}_{t}"].fillna(False).to_numpy() for t in EIGHT], axis=0)
        parts.append(pd.DataFrame({"timestamp": sig["timestamp"].to_numpy(),
                                   "side": side, "gate8": g8}))
    long = long.merge(pd.concat(parts, ignore_index=True), on=["timestamp", "side"], how="left")
    long["gate8"] = long["gate8"].fillna(False)

    pool = long.loc[long["gate8"]].copy()
    lab = pool.loc[pool["label"].notna()]
    tr = lab.loc[lab["split"] == "TRAIN"]
    log(f"8트리거풀 {len(pool):,}행 / TRAIN 라벨행 {len(tr):,}")

    rng = np.random.default_rng(SEED)
    idx = np.sort(rng.choice(len(tr), size=min(CONTEXT_N, len(tr)), replace=False))
    ctx = tr.iloc[idx]
    clf = TabPFNClassifier(device="cuda", random_state=SEED, ignore_pretraining_limits=True)
    clf.fit(ctx[FEATURE_COLUMNS], ctx["label"].to_numpy())
    log(f"  컨텍스트 {len(ctx):,}행 (라벨률 {ctx['label'].mean():.4f})")

    scored = {}
    for sp in ("VAL", "OOS"):
        s = pool.loc[pool["split"] == sp].copy()
        s["p"] = clf.predict_proba(s[FEATURE_COLUMNS])[:, 1]
        sl_ = s.loc[s["label"].notna()]
        auc = float(roc_auc_score(sl_["label"], sl_["p"])) if sl_["label"].nunique() == 2 else None
        s.attrs["auc"] = auc
        scored[sp] = s
        log(f"  {sp}: 풀 {len(s):>6,}행  라벨행 {len(sl_):>6,}  AUC "
            + (f"{auc:.4f}" if auc else "계산불가"))

    # --- 가격 배열 + 위치 매핑 ---
    kl = eth[["timestamp", "open", "high", "low", "close"]].copy()
    kl["timestamp"] = kl["timestamp"].dt.tz_localize(None)
    ts_to_pos = {t: i for i, t in enumerate(kl["timestamp"].to_numpy())}
    o, h, l, c = (kl[x].to_numpy() for x in ("open", "high", "low", "close"))
    nk = len(kl)

    def positions(s: pd.DataFrame) -> np.ndarray:
        return np.array([ts_to_pos.get(np.datetime64(t.tz_localize(None)), -1)
                         for t in s["timestamp"]], dtype=np.int64)

    for sp in ("VAL", "OOS"):
        scored[sp]["pos"] = positions(scored[sp])
        scored[sp]["sgn"] = np.where(scored[sp]["is_downside"] == 1, 1.0, -1.0)
        bad = int((scored[sp]["pos"] < 0).sum())
        if bad:
            log(f"  ⚠️{sp}: 타임스탬프 미매칭 {bad}행 -- 제외")
        scored[sp] = scored[sp].loc[scored[sp]["pos"] >= 0].reset_index(drop=True)

    def build(s: pd.DataFrame) -> pd.DataFrame:
        """thr 통과 행을 트레일링 시뮬레이터 입력 형태로 변환."""
        rows = []
        for _, ev in s.iterrows():
            i = int(ev["pos"])
            if i + FORWARD_BARS + 1 >= nk:
                continue
            rows.append({"side": "long" if ev["is_downside"] == 1 else "short",
                         "atr": float(ev["atr"]), "entry_price": float(o[i + 1]),
                         "fwd_open": o[i + 1:i + 1 + FORWARD_BARS],
                         "fwd_high": h[i + 1:i + 1 + FORWARD_BARS],
                         "fwd_low": l[i + 1:i + 1 + FORWARD_BARS],
                         "fwd_close": c[i + 1:i + 1 + FORWARD_BARS]})
        return pd.DataFrame(rows)

    def grid(df: pd.DataFrame) -> list[dict]:
        e, a, s_, H, L, C = _bt.pack(df)
        ef, af, sf, Hf, Lf, Cf = _bt.pack(df, flip=True)
        out = []
        for sl in SL_GRID:
            for arm in ARM_GRID:
                if arm < ARTIFACT_FREE_MIN:
                    continue
                for tr_ in TRAIL_GRID:
                    ov = _bt.simulate_trailing_vec(e, a, s_, H, L, C, sl, arm, tr_, False)
                    pv = _bt.simulate_trailing_vec(e, a, s_, H, L, C, sl, arm, tr_, True)
                    fo = _bt.simulate_trailing_vec(ef, af, sf, Hf, Lf, Cf, sl, arm, tr_, False)
                    fp = _bt.simulate_trailing_vec(ef, af, sf, Hf, Lf, Cf, sl, arm, tr_, True)
                    out.append({"sl": sl, "arm": arm, "trail": tr_,
                                "opt_bp": float(ov.mean() * 1e4 - STANDARD_COST_BP),
                                "pess_bp": float(pv.mean() * 1e4 - STANDARD_COST_BP),
                                "win_rate": float((ov * 1e4 > STANDARD_COST_BP).mean()),
                                "flip_opt_bp": float(fo.mean() * 1e4 - STANDARD_COST_BP),
                                "flip_pess_bp": float(fp.mean() * 1e4 - STANDARD_COST_BP)})
        return out

    def summarize(g: list[dict]) -> dict:
        okf = sum(1 for x in g if x["opt_bp"] > 0 and x["pess_bp"] > 0)
        okr = sum(1 for x in g if x["flip_opt_bp"] > 0 and x["flip_pess_bp"] > 0)
        best = max(g, key=lambda x: x["pess_bp"])
        return {"fwd_pass": okf, "flip_pass": okr, "n_cells": len(g),
                "margin": okf - okr, "best_pess_bp": best["pess_bp"], "best": best}

    def n_events(s: pd.DataFrame) -> int:
        """GAP봉 이상 떨어진 호출만 별개 사건으로 센다(side별)."""
        tot = 0
        for side, sub in s.groupby("side"):
            ps = np.sort(sub["pos"].to_numpy())
            if not len(ps):
                continue
            tot += 1 + int((np.diff(ps) >= DEDUP_GAP_BARS).sum())
        return tot

    # =========================================================
    # A) 꼬리 미세 스윕
    # =========================================================
    log("")
    log(f"=== A) 꼬리 미세 스윕 (VAL, ARM>={ARTIFACT_FREE_MIN}, "
        f"{len(SL_GRID)*2*len(TRAIL_GRID)}셀) ===")
    log(f"  {'thr':>5s} {'호출':>6s} {'사건':>5s} {'정방향':>7s} {'뒤집기':>7s} "
        f"{'차':>4s}  {'최고pess':>9s}  {'셀(SL/ARM/Tr)':>16s}")
    sweep, val_hold = {}, {}
    for thr in TAIL_THRESHOLDS:
        sel = scored["VAL"].loc[scored["VAL"]["p"] >= thr]
        vdf = build(sel)
        if len(vdf) < 30:
            log(f"  {thr:5.2f} {len(vdf):6,d}  -- 표본 부족(<30), 건너뜀")
            sweep[f"{thr:.2f}"] = {"val_n": int(len(vdf)), "skipped": True}
            continue
        g = grid(vdf)
        sm = summarize(g)
        ev = n_events(sel)
        val_hold[thr] = (sel, vdf, g, sm)
        sweep[f"{thr:.2f}"] = {"val_n": int(len(vdf)), "val_events": ev,
                               "fwd_pass": sm["fwd_pass"], "flip_pass": sm["flip_pass"],
                               "margin": sm["margin"],
                               "best": {k: round(float(v), 3) for k, v in sm["best"].items()}}
        b = sm["best"]
        flag = "✅" if sm["margin"] > 0 else "  "
        log(f"  {thr:5.2f} {len(vdf):6,d} {ev:5d} {sm['fwd_pass']:7d} {sm['flip_pass']:7d} "
            f"{sm['margin']:+4d}{flag} {b['pess_bp']:+8.2f}bp  "
            f"{b['sl']}/{b['arm']}/{b['trail']}")

    # =========================================================
    # B) 확률 십분위 단조성 (격자 미사용)
    # =========================================================
    log("")
    log("=== B) 확률 십분위별 신호방향 원시 선도수익 (격자/트레일링 미사용, 비용 차감) ===")
    dec = {}
    for sp in ("VAL", "OOS"):
        s = scored[sp]
        q = pd.qcut(s["p"], 10, labels=False, duplicates="drop")
        pos, sgn = s["pos"].to_numpy(), s["sgn"].to_numpy()
        ent = o[np.minimum(pos + 1, nk - 1)]
        rets = {}
        for k in DECILE_HORIZONS:
            end = np.minimum(pos + 1 + k, nk - 1)
            rets[k] = sgn * (c[end] / ent - 1.0) * 1e4 - STANDARD_COST_BP
        log(f"  --- {sp} (풀 {len(s):,}행) ---")
        log(f"    {'분위':>4s} {'n':>6s} {'p중앙':>7s} " +
            " ".join(f"{f'{k}봉':>9s}" for k in DECILE_HORIZONS))
        rowsd = []
        for d in sorted(pd.unique(q.dropna())):
            m = (q == d).to_numpy()
            r = {"decile": int(d), "n": int(m.sum()),
                 "p_median": round(float(np.median(s["p"].to_numpy()[m])), 4)}
            for k in DECILE_HORIZONS:
                r[f"bp_{k}"] = round(float(np.mean(rets[k][m])), 2)
            rowsd.append(r)
            log(f"    {int(d):4d} {r['n']:6,d} {r['p_median']:7.4f} " +
                " ".join(f"{r[f'bp_{k}']:+8.2f}" for k in DECILE_HORIZONS))
        # 단조성: 확률 순위와 분위평균수익의 스피어만
        dd = pd.DataFrame(rowsd)
        mono = {f"spearman_{k}": round(float(dd["decile"].corr(dd[f"bp_{k}"], method="spearman")), 3)
                for k in DECILE_HORIZONS}
        top_minus_bot = {f"top_minus_bot_{k}": round(float(dd[f"bp_{k}"].iloc[-1] - dd[f"bp_{k}"].iloc[0]), 2)
                         for k in DECILE_HORIZONS}
        log(f"    스피어만(분위 vs 수익): " +
            "  ".join(f"{k}봉 {mono[f'spearman_{k}']:+.3f}" for k in DECILE_HORIZONS))
        log(f"    최상위-최하위 분위 차:  " +
            "  ".join(f"{k}봉 {top_minus_bot[f'top_minus_bot_{k}']:+.2f}bp" for k in DECILE_HORIZONS))
        dec[sp] = {"deciles": rowsd, **mono, **top_minus_bot}

    # =========================================================
    # C) 랜덤 부분표집 귀무분포
    # =========================================================
    log("")
    log(f"=== C) 랜덤 부분표집 귀무분포 (B={NULL_B}, 같은 풀·같은 n·같은 격자, "
        f"모델 선택만 제거) ===")
    log(f"  {'thr':>5s} {'n':>6s}  {'관측(정-뒤)':>11s} {'귀무평균':>9s} {'백분위':>7s}   "
        f"{'관측 최고pess':>13s} {'귀무평균':>9s} {'백분위':>7s}")
    nulls = {}
    vpool = scored["VAL"]
    nrng = np.random.default_rng(NULL_SEED)
    for thr in TAIL_THRESHOLDS:
        if thr not in val_hold:
            continue
        sel, vdf, g, sm = val_hold[thr]
        n = len(vdf)
        if n < NULL_MIN_N:
            log(f"  {thr:5.2f} {n:6,d}  -- n<{NULL_MIN_N}, 귀무분포 생략")
            continue
        margins, bests = [], []
        for _ in range(NULL_B):
            ridx = nrng.choice(len(vpool), size=min(n, len(vpool)), replace=False)
            rdf = build(vpool.iloc[np.sort(ridx)])
            if len(rdf) < 30:
                continue
            rs = summarize(grid(rdf))
            margins.append(rs["margin"]); bests.append(rs["best_pess_bp"])
        if len(margins) < 20:
            log(f"  {thr:5.2f} {n:6,d}  -- 유효 복제 부족({len(margins)})")
            continue
        ma, be = np.array(margins, float), np.array(bests, float)
        pct_m = float((ma < sm["margin"]).mean() * 100)
        pct_b = float((be < sm["best_pess_bp"]).mean() * 100)
        nulls[f"{thr:.2f}"] = {
            "n": int(n), "B": len(margins),
            "obs_margin": sm["margin"], "null_margin_mean": round(float(ma.mean()), 2),
            "null_margin_sd": round(float(ma.std()), 2), "margin_pctile": round(pct_m, 1),
            "obs_best_pess_bp": round(sm["best_pess_bp"], 3),
            "null_best_pess_mean": round(float(be.mean()), 3),
            "null_best_pess_sd": round(float(be.std()), 3), "best_pess_pctile": round(pct_b, 1)}
        f1 = "✅" if pct_m >= 95 else ("  " if pct_m >= 50 else "⚠️")
        f2 = "✅" if pct_b >= 95 else ("  " if pct_b >= 50 else "⚠️")
        log(f"  {thr:5.2f} {n:6,d}  {sm['margin']:+11d} {ma.mean():+9.2f} "
            f"{pct_m:6.1f}%{f1}  {sm['best_pess_bp']:+12.2f} {be.mean():+9.2f} {pct_b:6.1f}%{f2}")

    # =========================================================
    # D) VAL 선정 -> OOS 1회
    # =========================================================
    log("")
    log("=== D) VAL 선정 조합으로 OOS 1회 평가 (재최적화 금지) ===")
    oos_res = None
    cand = [(t, v) for t, v in val_hold.items() if v[3]["margin"] > 0]
    if not cand:
        log("  ⚠️VAL에서 정방향이 뒤집기를 이긴 임계값이 없다 -- OOS로 넘길 후보 없음")
    else:
        thr, (sel, vdf, g, sm) = max(cand, key=lambda kv: (kv[1][3]["margin"],
                                                           kv[1][3]["best_pess_bp"]))
        cell = sm["best"]
        log(f"  [VAL 선정] thr={thr:.2f}  SL/ARM/Trail={cell['sl']}/{cell['arm']}/{cell['trail']}  "
            f"(margin {sm['margin']:+d}, pess {cell['pess_bp']:+.2f}bp)")
        osel = scored["OOS"].loc[scored["OOS"]["p"] >= thr]
        odf = build(osel)
        if len(odf) < 30:
            log(f"  [OOS] 호출 {len(odf)}건 -- 표본 부족")
        else:
            og_all = grid(odf)
            og = [x for x in og_all
                  if (x["sl"], x["arm"], x["trail"]) == (cell["sl"], cell["arm"], cell["trail"])][0]
            osm = summarize(og_all)
            log(f"  [OOS 1회]  n={len(odf):,} (사건 {n_events(osel)})  "
                f"opt{og['opt_bp']:+.2f} pess{og['pess_bp']:+.2f}bp  승률{og['win_rate']*100:.1f}%  "
                f"뒤집기 opt{og['flip_opt_bp']:+.2f}bp"
                f"{'  ⚠️뒤집기도 수익' if og['flip_opt_bp'] > 0 else '  ✅뒤집기 음수'}")
            log(f"  [OOS 격자] 정방향 {osm['fwd_pass']}/{osm['n_cells']}  "
                f"뒤집기 {osm['flip_pass']}/{osm['n_cells']}  차 {osm['margin']:+d}"
                f"{'  ✅' if osm['margin'] > 0 else '  ⚠️뒤집기 우세'}")
            oos_res = {"threshold": thr, "cell": {k: cell[k] for k in ("sl", "arm", "trail")},
                       "n": int(len(odf)), "events": n_events(osel),
                       "result": {k: round(float(v), 3) for k, v in og.items()},
                       "grid_fwd_pass": osm["fwd_pass"], "grid_flip_pass": osm["flip_pass"],
                       "grid_margin": osm["margin"], "n_cells": osm["n_cells"]}

    report = {"signal": "v_rebound_8trigger_tail_probe", "asset": "ETHUSDT",
              "fresh_forward_bar_by_bar": True, "trade_ledgers_used_as_input": False,
              "saved_parent_exit_timestamps_used": False, "future_rows_used_for_entry": False,
              "scope": {"triggers": EIGHT, "excluded": "local_extreme (held_up 얽힘)",
                        "train_pop": "8트리거풀", "serve_pop": "8트리거풀 (일치)",
                        "cell_selected_on": f"VAL only (ARM>={ARTIFACT_FREE_MIN})",
                        "reoptimized_on_oos": False, "holdout_touched": False,
                        "live_code_changed": False, "cost_bp": STANDARD_COST_BP,
                        "dedup_gap_bars": DEDUP_GAP_BARS, "null_B": NULL_B},
              "auc": {sp: scored[sp].attrs.get("auc") for sp in scored},
              "A_tail_sweep": sweep, "B_decile_monotonicity": dec,
              "C_random_subsample_null": nulls, "D_oos": oos_res,
              "runtime_sec": round(time.time() - t0, 1)}
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str))
    log("")
    log(f"report saved -> {OUT_JSON}  ({report['runtime_sec']}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
