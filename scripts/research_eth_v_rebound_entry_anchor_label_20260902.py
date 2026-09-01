#!/usr/bin/env python3
"""라벨 기준점을 **실제 진입가(open[i+1])**로 옮겨 재정의하고 경제성까지 측정.

## 왜

2026-09-02 진단(`..._atr_scale_wick_gap_diagnostic_20260902.py`)에서 확정된 사실:

  현행 라벨의 기준점은 그 봉의 **저가**(`extreme = low[i]`)인데 실제 진입가는 `open[i+1]`이다.
  진입 시점에 목표(1.5*ATR)가 **이미 중앙값 110~115% 소진**돼 있고, 호출의 **60~63%는 남은
  몫이 0 이하**다. 목표 크기 자체는 문제가 없다(1.5*ATR 중앙 37~42bp vs 왕복비용 10bp).

즉 모델은 진짜 패턴을 보지만 **그 패턴이 가리키는 움직임은 진입 시점에 이미 끝나 있다**.
라벨은 `저가→종가`를 재고 거래는 `시가→청산`을 먹는다. 모델·필터로는 못 고친다.

## 이 실험

**기준점 하나만** 바꾼다(한 번에 하나만 바꿔야 원인이 특정된다):

    현행: extreme = low[i]        (bottom) / high[i]        (top)
    신규: extreme = open[i+1]     (양쪽 동일 -- 실제 진입가)

나머지 산술(fast 창 6봉, giveback 공식, full 창 12봉, end_price, ambiguous 제외)은 **그대로**.
giveback 분모도 앵커를 따라 `peak - entry`가 된다("진입~정점 중 얼마를 반납했나").

⚠️**문턱은 같이 스윕해야 한다** -- 1.5*ATR은 wick 앵커에 맞춰 튜닝된 값이고, 진입 앵커에서는
같은 사건이 훨씬 작게 측정된다. `atr_mult in {0.5, 0.75, 1.0, 1.5}`를 돌리고 chop 문턱은
v7b의 비율(1.0/1.5)을 보존해 `chop_mult = atr_mult * 2/3`으로 둔다.

## 판정

⚠️**AUC로 현행과 비교 금지** -- 라벨 정의가 다르면 문제 난이도가 다르다(이 저장소 규칙).
판정은 **방향뒤집기 통제 경제성**(ARM>=1.0 80셀)이고, 현행 배포판과는 **호출 빈도를 맞춰**
같은 n에서 비교한다.

부수 검증: 신규 라벨의 양성은 **구성상 진입 후 남은 목표가 문턱 이상**이어야 한다 --
진단의 Q3(남은 목표 중앙 -3~-5bp)가 양수로 뒤집히는지 확인한다.

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


_s1 = _load("s1_entry", "scripts/research_eth_v_rebound_label_grid_screen_stage1_20260901.py")
_vs, _bt = _s1._vs, _s1._bt
FEATURE_COLUMNS = _s1.FEATURE_COLUMNS
FAST_BARS, FORWARD_BARS = _s1.FAST_BARS_FIXED, _s1.FORWARD_BARS
SL_GRID, ARM_GRID, TRAIL_GRID = _bt.SL_GRID, _bt.ARM_GRID, _bt.TRAIL_GRID

FULL_BARS, T_SUSTAIN = 12, 0.20
ATR_MULT_GRID = [0.5, 0.75, 1.0, 1.5]
CHOP_RATIO = 1.0 / 1.5              # v7b의 chop/success 비율 보존
CONTEXT_N, SEED = 18000, 20260829
COST_BP, ARTIFACT_FREE_MIN = 10.0, 1.0
TRAIN_END = pd.Timestamp("2025-09-01", tz="UTC")
VAL_END = pd.Timestamp("2026-01-01", tz="UTC")
OOS_END = pd.Timestamp("2026-04-01", tz="UTC")
OUT_JSON = ROOT / "data/research/eth_v_rebound_entry_anchor_20260902/report.json"


def log(m): print(f"[entry] {m}", flush=True)


def label_entry_anchor(sig: pd.DataFrame, is_down: bool, atr_mult: float) -> np.ndarray:
    """`_s1.label_param`을 그대로 옮기되 **앵커만** open[i+1]로 교체."""
    close, high, low = (sig[c].to_numpy() for c in ("close", "high", "low"))
    op = sig["open"].to_numpy()
    atr = sig["atr"].to_numpy()

    extreme = _vs.shifted_at(op, 1)                 # ⭐진입가 = 다음 봉 시가 (양쪽 동일)
    pre_atr = _vs.shifted_at(atr, -1)
    fast_close_max = _vs.fwd_window(close, 1, FAST_BARS, "max")
    fast_close_min = _vs.fwd_window(close, 1, FAST_BARS, "min")
    full_high_max = _vs.fwd_window(high, 1, FULL_BARS, "max")
    full_low_min = _vs.fwd_window(low, 1, FULL_BARS, "min")
    end_price = _vs.shifted_at(close, FULL_BARS)

    if is_down:
        fast_move, peak = fast_close_max - extreme, full_high_max
    else:
        fast_move, peak = extreme - fast_close_min, full_low_min

    with np.errstate(invalid="ignore", divide="ignore"):
        valid = (np.isfinite(pre_atr) & (pre_atr > 0) & np.isfinite(full_high_max)
                 & np.isfinite(full_low_min) & np.isfinite(end_price) & np.isfinite(extreme))
        fast_mult = fast_move / pre_atr
        denom = (peak - extreme) if is_down else (extreme - peak)
        giveback = np.where(np.abs(denom) >= 1e-12,
                            (peak - end_price) / denom if is_down else (end_price - peak) / denom,
                            np.nan)
        strong = fast_mult >= atr_mult
        is_v = strong & np.isfinite(giveback) & (giveback <= T_SUSTAIN)
        is_chop = fast_mult < (atr_mult * CHOP_RATIO)

    out = np.full(len(sig), "ambiguous", dtype=object)
    out[valid & is_v] = "v_rebound"
    out[valid & is_chop & ~is_v] = "chop"
    out[~valid] = "invalid"
    return out


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

    # 현행(wick) 라벨 -- 대조군
    wb = _s1.label_param(sig, True, ambig="drop", anchor="wick",
                         atr_mult=1.50, t_sustain=T_SUSTAIN, full_bars=FULL_BARS)
    wt = _s1.label_param(sig, False, ambig="drop", anchor="wick",
                         atr_mult=1.50, t_sustain=T_SUSTAIN, full_bars=FULL_BARS)

    def grid(df, cost=COST_BP):
        if len(df) < 30:
            return None
        e, a, s_, H, L, C = _bt.pack(df)
        ef, af, sf, Hf, Lf, Cf = _bt.pack(df, flip=True)
        fwd = flip = 0; best = None
        for sl in SL_GRID:
            for arm in ARM_GRID:
                if arm < ARTIFACT_FREE_MIN:
                    continue
                for tr in TRAIL_GRID:
                    ov = _bt.simulate_trailing_vec(e, a, s_, H, L, C, sl, arm, tr, False)
                    pv = _bt.simulate_trailing_vec(e, a, s_, H, L, C, sl, arm, tr, True)
                    fo = _bt.simulate_trailing_vec(ef, af, sf, Hf, Lf, Cf, sl, arm, tr, False)
                    fp = _bt.simulate_trailing_vec(ef, af, sf, Hf, Lf, Cf, sl, arm, tr, True)
                    ob, pb = float(ov.mean()*1e4-cost), float(pv.mean()*1e4-cost)
                    fwd += int(ob > 0 and pb > 0)
                    flip += int(float(fo.mean()*1e4-cost) > 0 and float(fp.mean()*1e4-cost) > 0)
                    if best is None or pb > best["pess_bp"]:
                        best = {"sl": sl, "arm": arm, "trail": tr,
                                "opt_bp": round(ob, 2), "pess_bp": round(pb, 2)}
        return {"n": int(len(df)), "fwd_pass": fwd, "flip_pass": flip,
                "margin": fwd - flip, "best": best}

    def build(s):
        rows = []
        for i_, isd, atr_ in zip(s["pos"].to_numpy(), s["is_downside"].to_numpy(),
                                  s["atr"].to_numpy()):
            i = int(i_)
            if i + FORWARD_BARS + 1 >= nk:
                continue
            rows.append({"side": "long" if isd == 1 else "short", "atr": float(atr_),
                         "entry_price": float(o[i + 1]),
                         "fwd_open": o[i+1:i+1+FORWARD_BARS], "fwd_high": h[i+1:i+1+FORWARD_BARS],
                         "fwd_low": l[i+1:i+1+FORWARD_BARS], "fwd_close": c[i+1:i+1+FORWARD_BARS]})
        return pd.DataFrame(rows)

    report = {"signal": "v_rebound_entry_anchor_label", "asset": "ETHUSDT",
              "fresh_forward_bar_by_bar": True, "trade_ledgers_used_as_input": False,
              "saved_parent_exit_timestamps_used": False, "future_rows_used_for_entry": False,
              "scope": {"change": "라벨 앵커만 low[i]/high[i] -> open[i+1]로 교체(나머지 산술 불변)",
                        "atr_mult_grid": ATR_MULT_GRID, "chop_ratio": CHOP_RATIO,
                        "t_sustain": T_SUSTAIN, "full_bars": FULL_BARS, "fast_bars": FAST_BARS,
                        "context_n": CONTEXT_N, "seed": SEED, "cost_bp": COST_BP,
                        "auc_cross_label_comparison": "금지(라벨 난이도 다름) -- 판정은 경제성",
                        "holdout_touched": False, "live_code_changed": False},
              "variants": {}}

    # ---- 대조군: 현행 wick 앵커 ----
    log("")
    log("=== 대조군: 현행 wick 앵커 (atr_mult=1.5) ===")
    base_long = _s1.long_frame_for(sig, feat, wb, wt)
    base_long["split"] = np.where(base_long["timestamp"] < TRAIN_END, "TRAIN",
                           np.where(base_long["timestamp"] < VAL_END, "VAL", "OOS"))
    assert base_long["timestamp"].max() < OOS_END, "HOLDOUT 누출"
    base_long["pos"] = [pos_of.get(np.datetime64(t.tz_localize(None)), -1)
                        for t in base_long["timestamp"]]
    base_long = base_long.loc[base_long["pos"] >= 0].reset_index(drop=True)
    log(f"  라벨률(TRAIN) {base_long.loc[base_long['split']=='TRAIN','label'].mean():.4f}  "
        f"라벨행 비율 {base_long.loc[base_long['split']=='TRAIN','label'].notna().mean()*100:.1f}%")

    variants = {"W_wick_1.50": (wb, wt, 1.50)}
    for am in ATR_MULT_GRID:
        variants[f"E_entry_{am:.2f}"] = (label_entry_anchor(sig, True, am),
                                         label_entry_anchor(sig, False, am), am)

    call_targets = {}
    for name, (sbv, stv, am) in variants.items():
        lf = _s1.long_frame_for(sig, feat, sbv, stv)
        lf["split"] = np.where(lf["timestamp"] < TRAIN_END, "TRAIN",
                        np.where(lf["timestamp"] < VAL_END, "VAL", "OOS"))
        lf["pos"] = [pos_of.get(np.datetime64(t.tz_localize(None)), -1) for t in lf["timestamp"]]
        lf = lf.loc[lf["pos"] >= 0].reset_index(drop=True)
        lab = lf.loc[lf["label"].notna()]
        tr = lab.loc[lab["split"] == "TRAIN"]
        log("")
        log(f"=== {name}  (atr_mult={am}, chop<{am*CHOP_RATIO:.2f}) ===")
        log(f"  TRAIN 라벨행 {len(tr):,} ({len(tr)/len(lf.loc[lf['split']=='TRAIN'])*100:.1f}% of bars)"
            f"  라벨률 {tr['label'].mean():.4f}")
        if len(tr) < 2000 or tr["label"].nunique() < 2 or tr["label"].mean() < 0.005:
            log("  ⚠️학습 불가(표본/양성 부족) -- 건너뜀")
            report["variants"][name] = {"train_n": int(len(tr)),
                                        "label_rate": round(float(tr['label'].mean()), 5),
                                        "skipped": "표본/양성 부족"}
            continue

        rng = np.random.default_rng(SEED)
        idx = np.sort(rng.choice(len(tr), size=min(CONTEXT_N, len(tr)), replace=False))
        ctx = tr.iloc[idx]
        clf = TabPFNClassifier(device="cuda", random_state=SEED, ignore_pretraining_limits=True)
        clf.fit(ctx[FEATURE_COLUMNS], ctx["label"].to_numpy())

        ent = {"train_n": int(len(tr)), "context_n": int(len(ctx)),
               "label_rate": round(float(tr["label"].mean()), 5),
               "labeled_pct": round(float(len(tr) / len(lf.loc[lf['split'] == 'TRAIN']) * 100), 1)}
        for spn in ("VAL", "OOS"):
            s = lf.loc[lf["split"] == spn].copy()
            CH = 20000
            s["p"] = np.concatenate([clf.predict_proba(s[FEATURE_COLUMNS].iloc[k:k+CH])[:, 1]
                                     for k in range(0, len(s), CH)])
            sl_ = s.loc[s["label"].notna()]
            auc = (float(roc_auc_score(sl_["label"], sl_["p"]))
                   if sl_["label"].nunique() == 2 else None)
            # 호출 빈도: 대조군이 세운 목표에 맞춘다(첫 변형이 기준을 세움)
            if name == "W_wick_1.50":
                sel = s.loc[s["p"] >= 0.60]
                call_targets[spn] = len(sel)
            else:
                sel = s.nlargest(min(call_targets[spn], len(s)), "p")
            g = grid(build(sel))
            # 부수 검증: 진입 후 남은 목표(bp)
            i_ = sel["pos"].to_numpy().astype(int)
            dnm = sel["is_downside"].to_numpy() == 1
            sgn = np.where(dnm, 1.0, -1.0)
            anch = o[i_ + 1] if name != "W_wick_1.50" else np.where(dnm, l[i_], h[i_])
            remain = sgn * ((anch + sgn * am * sel["atr"].to_numpy()) - o[i_ + 1]) / o[i_ + 1] * 1e4
            ent[spn] = {"auc_own_label": round(auc, 4) if auc else None,
                        "n_calls": int(len(sel)), "grid": g,
                        "remaining_target_bp_median": round(float(np.median(remain)), 2)}
            if g:
                log(f"  {spn}: 호출 {len(sel):,}  자기라벨AUC {auc:.4f}  "
                    f"정{g['fwd_pass']}/뒤{g['flip_pass']} (차 {g['margin']:+d})  "
                    f"최고pess {g['best']['pess_bp']:+.2f}bp  "
                    f"진입후남은목표 중앙 {ent[spn]['remaining_target_bp_median']:+.2f}bp")
            else:
                log(f"  {spn}: 호출 {len(sel):,} -- 격자 표본 부족")
        report["variants"][name] = ent

    log("")
    log("=== 판정 (⚠️AUC 교차비교 금지 -- 경제성만) ===")
    for name, e in report["variants"].items():
        if "VAL" not in e:
            continue
        v, oo = e["VAL"].get("grid"), e["OOS"].get("grid")
        if not (v and oo):
            continue
        ok = v["margin"] > 0 and oo["margin"] > 0
        log(f"  {'✅' if ok else '  '}{name:16s} VAL 차{v['margin']:+4d}  OOS 차{oo['margin']:+4d}  "
            f"(OOS 최고pess {oo['best']['pess_bp']:+.2f}bp)")

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    report["runtime_sec"] = round(time.time() - t0, 1)
    OUT_JSON.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str))
    log("")
    log(f"report saved -> {OUT_JSON}  ({report['runtime_sec']}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
