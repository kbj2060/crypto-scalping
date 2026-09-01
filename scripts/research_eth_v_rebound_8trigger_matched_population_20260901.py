#!/usr/bin/env python3
"""8트리거 **일치 구성**: local_extreme을 학습셋 구성에서도 빼고, 학습·서빙 모집단을 맞춘다.

## 왜 이 구성인가

2026-09-01 실측 두 가지:
  9트리거 풀 학습 -> 매 봉 서빙 : AUC **0.5287** (붕괴)
  매 봉 학습      -> 매 봉 서빙 : AUC 0.6992 / OOS 0.7051

붕괴 원인은 **학습/서빙 모집단 불일치**다. 후보풀은 `local_extreme` 때문에 `held_up`
(= low[i+1:i+7].min() >= low[i], 라벨 fast_move의 선행조건)이 이미 보장된 봉만 담고 있어,
모델은 held_up을 **예측하는 법을 배울 기회가 없었다**. 그 모델을 임의의 봉에 적용하면 held_up이
보장되지 않는데 평가할 능력이 없다. 순위 자체가 무너지므로 확률 보정으로도 못 고친다.

그러면 **얽힌 트리거(local_extreme) 하나만 빼고 나머지 8개로 학습·서빙을 모두 게이트**하면?
  - 얽힘 제거: 나머지 8개는 증거신호라 held_up을 보장하지 않는다 (이 스크립트가 **직접 검증**)
  - 모집단 일치: 0.5287의 원인이 사라진다
  - 라이브는 5분마다 8트리거 발동을 확인하고, 발동하면 채점 -> "5분마다 진행"이 이 형태

**아직 한 번도 측정한 적 없는 조합이다.**

## 이 스크립트가 답할 것

  1. **얽힘 검증**: held_up 비율과 라벨률을 전체봉 / 8트리거풀 / 9트리거풀에서 비교.
     8트리거풀이 전체봉과 비슷하면 깨끗한 것이고, 9트리거풀처럼 부풀면 이 구성도 무효다.
  2. 게이트된 모집단에서의 AUC. ⚠️전체봉 0.6992와 **직접 비교 불가**(모집단이 다르면 문제
     난이도가 다르다) -- 참고용으로만 본다.
  3. ⭐**게이트된 모집단에서의 경제성 + 방향뒤집기**. 오늘 경제성이 무너진 이유가 모집단
     (excluded-middle 봉에서 방향성 소실)이었다면, 게이트를 되살렸을 때 방향성이 돌아올 수 있다.
     이게 이 실험의 핵심 질문이다.

## 프로토콜

임계값·경제성 셀은 **VAL에서만 선정**, OOS는 그 조합으로 1회 평가(재최적화 금지).
저ARM은 노이즈수확 아티팩트 구간이므로 선정에서 제외(ARM>=1.0).
⚠️HOLDOUT 미터치. 라이브 코드 변경 없음.

Run on the server (GPU) via handoff:
  handoff.sh launch server <job> -- python scripts/research_eth_v_rebound_8trigger_matched_population_20260901.py
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
_spec = importlib.util.spec_from_file_location("vreb_s1_8t", S1)
_s1 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_s1)
_feas, _bt, _vs = _s1._feas, _s1._bt, _s1._vs

FEATURE_COLUMNS = _s1.FEATURE_COLUMNS
ALL9 = _feas.ALL9
EIGHT = [t for t in ALL9 if t != "local_extreme"]
STANDARD_COST_BP, FORWARD_BARS = _s1.STANDARD_COST_BP, _s1.FORWARD_BARS

W = 6
DEPLOYED = {"atr_mult": 1.50, "t_sustain": 0.20, "full_bars": 12}
CONTEXT_N, SEED = 18000, 20260829
ARTIFACT_FREE_MIN = 1.0
THRESHOLDS = [0.40, 0.50, 0.60, 0.70]
SL_GRID, ARM_GRID, TRAIL_GRID = _bt.SL_GRID, _bt.ARM_GRID, _bt.TRAIL_GRID

TRAIN_END = pd.Timestamp("2025-09-01", tz="UTC")
VAL_END = pd.Timestamp("2026-01-01", tz="UTC")
OOS_END = pd.Timestamp("2026-04-01", tz="UTC")

OUT_JSON = ROOT / "data/research/eth_v_rebound_8trigger_matched_20260901/report.json"


def log(msg: str) -> None:
    print(f"[8t] {msg}", flush=True)


def main() -> int:
    t0 = time.time()
    from tabpfn import TabPFNClassifier
    from sklearn.metrics import roc_auc_score
    import torch
    log(f"cuda: {torch.cuda.is_available()}")
    log(f"8트리거: {', '.join(EIGHT)}")

    _s1.VAL_END = OOS_END
    log("building frame + labels...")
    sig, feat, eth = _s1.build_sig()
    sb = _s1.label_param(sig, True, ambig="drop", anchor="wick", **DEPLOYED)
    st = _s1.label_param(sig, False, ambig="drop", anchor="wick", **DEPLOYED)
    long = _s1.long_frame_for(sig, feat, sb, st)
    long["split"] = np.where(long["timestamp"] < TRAIN_END, "TRAIN",
                     np.where(long["timestamp"] < VAL_END, "VAL", "OOS"))
    assert long["timestamp"].max() < OOS_END, "HOLDOUT 누출"

    # --- 게이트 마스크 + held_up (얽힘 검증용) ---
    n = len(sig)
    lowv, highv = sig["low"].to_numpy(), sig["high"].to_numpy()
    fwd_low_min = _vs.fwd_window(lowv, 1, W, "min")
    fwd_high_max = _vs.fwd_window(highv, 1, W, "max")
    lo_f = np.zeros(n, bool); hi_f = np.zeros(n, bool)
    for i in range(W, n - W):
        if lowv[i] == lowv[i - W:i + W + 1].min():
            lo_f[i] = True
        if highv[i] == highv[i - W:i + W + 1].max():
            hi_f[i] = True

    parts = []
    for side, is_down in (("bottom", True), ("top", False)):
        g8 = np.any([sig[f"{side}_{t}"].fillna(False).to_numpy() for t in EIGHT], axis=0)
        le = lo_f if is_down else hi_f
        held = (fwd_low_min >= lowv) if is_down else (fwd_high_max <= highv)
        parts.append(pd.DataFrame({"timestamp": sig["timestamp"].to_numpy(), "side": side,
                                   "gate8": g8, "gate9": g8 | le, "held_up": held}))
    gates = pd.concat(parts, ignore_index=True)
    long = long.merge(gates, on=["timestamp", "side"], how="left")
    long[["gate8", "gate9"]] = long[["gate8", "gate9"]].fillna(False)

    # --- 1) 얽힘 검증 ---
    log("")
    log("=== 1) 얽힘 검증 (held_up / 라벨률) ===")
    ent = {}
    for nm, mask in (("전체봉", pd.Series(True, index=long.index)),
                     ("8트리거풀", long["gate8"]), ("9트리거풀", long["gate9"])):
        sub = long.loc[mask]
        lb = sub.loc[sub["label"].notna()]
        ent[nm] = {"rows": int(len(sub)), "pct_of_all": round(len(sub) / len(long) * 100, 2),
                   "held_up_rate": round(float(sub["held_up"].mean()), 4),
                   "label_rate": round(float(lb["label"].mean()), 4) if len(lb) else None,
                   "labeled_pct": round(len(lb) / len(sub) * 100, 1) if len(sub) else None}
        e = ent[nm]
        log(f"  {nm:8s} 행 {e['rows']:>7,} (전체의 {e['pct_of_all']:>5.2f}%)  "
            f"**held_up {e['held_up_rate']:.4f}**  라벨률 {e['label_rate']}  라벨행 {e['labeled_pct']}%")
    base_h = ent["전체봉"]["held_up_rate"]
    r8 = ent["8트리거풀"]["held_up_rate"] / base_h
    r9 = ent["9트리거풀"]["held_up_rate"] / base_h
    log(f"  held_up 배수: 8트리거풀 {r8:.2f}x / 9트리거풀 {r9:.2f}x  (1.0에 가까우면 깨끗)")
    if r8 > 1.5:
        log("  ⚠️8트리거풀도 held_up이 부풀어 있다 -- 이 구성 역시 얽힘에서 자유롭지 않다")
    else:
        log("  ✅8트리거풀은 held_up이 전체봉과 비슷 -- 얽힘 제거 확인")

    # --- 2) 일치 구성 학습/평가 ---
    pool = long.loc[long["gate8"]].copy()
    lab = pool.loc[pool["label"].notna()]
    tr = lab.loc[lab["split"] == "TRAIN"]
    log("")
    log(f"=== 2) 8트리거 일치 구성 학습 (TRAIN 라벨행 {len(tr):,}) ===")
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
        scored[sp] = s
        sl_ = s.loc[s["label"].notna()]
        auc = float(roc_auc_score(sl_["label"], sl_["p"])) if sl_["label"].nunique() == 2 else None
        log(f"  {sp}: 풀 {len(s):>6,}행  라벨행 {len(sl_):>6,}  AUC {auc:.4f}"
            if auc else f"  {sp}: AUC 계산불가")
        scored[sp].attrs["auc"] = auc

    # --- 3) 경제성 + 방향뒤집기 (VAL 선정 -> OOS 1회) ---
    kl = eth[["timestamp", "open", "high", "low", "close"]].copy()
    kl["timestamp"] = kl["timestamp"].dt.tz_localize(None)
    ts_to_pos = {t: i for i, t in enumerate(kl["timestamp"].to_numpy())}
    o, h, l, c = (kl[x].to_numpy() for x in ("open", "high", "low", "close"))

    def build(s, thr):
        rows = []
        for _, ev in s.loc[s["p"] >= thr].iterrows():
            i = ts_to_pos.get(np.datetime64(ev["timestamp"].tz_localize(None)))
            if i is None or i + FORWARD_BARS + 1 >= len(kl):
                continue
            rows.append({"side": "long" if ev["is_downside"] == 1 else "short",
                         "atr": float(ev["atr"]), "entry_price": float(o[i + 1]),
                         "fwd_open": o[i + 1:i + 1 + FORWARD_BARS], "fwd_high": h[i + 1:i + 1 + FORWARD_BARS],
                         "fwd_low": l[i + 1:i + 1 + FORWARD_BARS], "fwd_close": c[i + 1:i + 1 + FORWARD_BARS]})
        return pd.DataFrame(rows)

    def grid(df):
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

    log("")
    log(f"=== 3) 경제성 (ARM>={ARTIFACT_FREE_MIN} {len(SL_GRID)*2*len(TRAIL_GRID)}셀, VAL 선정 -> OOS 1회) ===")
    econ, best_overall = {}, None
    for thr in THRESHOLDS:
        vdf = build(scored["VAL"], thr)
        if len(vdf) < 50:
            log(f"  thr={thr:.2f} VAL 호출 {len(vdf)}건 -- 표본 부족"); continue
        g = grid(vdf)
        ok = lambda x: x["opt_bp"] > 0 and x["pess_bp"] > 0
        npass, nflip = sum(map(ok, g)), sum(1 for x in g if x["flip_opt_bp"] > 0 and x["flip_pess_bp"] > 0)
        best = max(g, key=lambda x: x["pess_bp"])
        econ[f"{thr:.2f}"] = {"val_n": len(vdf), "val_pass": npass, "val_flip_pass": nflip,
                              "n_cells": len(g), "val_best": {k: round(v, 3) for k, v in best.items()}}
        log(f"  thr={thr:.2f}  VAL n={len(vdf):>5,}  정방향 {npass:>2d}/{len(g)} 뒤집기 {nflip:>2d}/{len(g)}  "
            f"최고 SL/ARM/Tr={best['sl']}/{best['arm']}/{best['trail']} "
            f"pess{best['pess_bp']:+.2f}bp (뒤집기 opt{best['flip_opt_bp']:+.2f}bp)")
        if best_overall is None or best["pess_bp"] > best_overall[1]["pess_bp"]:
            best_overall = (thr, best)

    if best_overall:
        thr, cell = best_overall
        odf = build(scored["OOS"], thr)
        log("")
        log(f"  [VAL 선정] thr={thr:.2f} SL/ARM/Trail={cell['sl']}/{cell['arm']}/{cell['trail']}")
        if len(odf) >= 50:
            og = [x for x in grid(odf)
                  if (x["sl"], x["arm"], x["trail"]) == (cell["sl"], cell["arm"], cell["trail"])][0]
            ofull = grid(odf)
            ok = lambda x: x["opt_bp"] > 0 and x["pess_bp"] > 0
            log(f"  [OOS 1회]  n={len(odf):,}  opt{og['opt_bp']:+.2f} pess{og['pess_bp']:+.2f}bp "
                f"승률{og['win_rate']*100:.1f}%  뒤집기 opt{og['flip_opt_bp']:+.2f}bp"
                f"{'  ⚠️뒤집기도 수익' if og['flip_opt_bp'] > 0 else '  ✅뒤집기 음수'}")
            log(f"  [OOS 격자] 정방향 {sum(map(ok, ofull)):>2d}/{len(ofull)}  "
                f"뒤집기 {sum(1 for x in ofull if x['flip_opt_bp'] > 0 and x['flip_pess_bp'] > 0):>2d}/{len(ofull)}")
            econ["oos"] = {"threshold": thr, "cell": {k: cell[k] for k in ("sl", "arm", "trail")},
                           "n": len(odf), "result": {k: round(v, 3) for k, v in og.items()},
                           "grid_fwd_pass": sum(map(ok, ofull)),
                           "grid_flip_pass": sum(1 for x in ofull
                                                 if x["flip_opt_bp"] > 0 and x["flip_pess_bp"] > 0),
                           "n_cells": len(ofull)}
        else:
            log(f"  [OOS] 호출 {len(odf)}건 -- 표본 부족")

    log("")
    log("=== 비교 기준 ===")
    log(f"  매 봉 구성(현행 배포):  VAL AUC 0.6992 / OOS 0.7051, OOS 전체봉 경제성 정21/뒤31 ⚠️")
    log(f"  9트리거풀 학습->매봉:   AUC 0.5287 (모집단 불일치로 붕괴)")
    log(f"  9트리거 게이트 시절:    0.8292/0.8127/0.8465 (held_up 얽힘으로 과대평가)")

    report = {"signal": "v_rebound_8trigger_matched_population", "asset": "ETHUSDT",
              "fresh_forward_bar_by_bar": True, "trade_ledgers_used_as_input": False,
              "saved_parent_exit_timestamps_used": False, "future_rows_used_for_entry": False,
              "scope": {"triggers": EIGHT, "excluded": "local_extreme (held_up 얽힘)",
                        "train_pop": "8트리거풀", "serve_pop": "8트리거풀 (일치)",
                        "cell_selected_on": "VAL only (ARM>=1.0)", "reoptimized_on_oos": False,
                        "holdout_touched": False, "live_code_changed": False},
              "entanglement_check": ent,
              "auc": {sp: scored[sp].attrs.get("auc") for sp in scored},
              "economics": econ, "runtime_sec": round(time.time() - t0, 1)}
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str))
    log("")
    log(f"report saved -> {OUT_JSON}  ({report['runtime_sec']}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
