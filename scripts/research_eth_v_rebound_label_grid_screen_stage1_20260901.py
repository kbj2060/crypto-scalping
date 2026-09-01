#!/usr/bin/env python3
"""V자반등 라벨 파라미터 그리드 스크리닝 -- Stage 1 (GBM 프록시, VAL만).

## 왜 하는가

이 신호의 라벨 상수 4개(FAST_BARS/FULL_BARS/ATR_MULT/T_SUSTAIN)는 **한 번도 격자로 훑린 적이
없다**. v1~v7b 진화 과정에서 하나씩 손으로 정해진 값이고, BTC 스크리닝 스크립트도 이를 명시한다
("FAST_BARS/FULL_BARS/ATR_MULT/T_SUSTAIN are fixed, reused as-is ... not a re-optimization").
반면 호메로스 증거신호 8종은 horizon x gap x K 그리드 스크리닝이 표준 관문이다 -- V자반등만
그 관문을 건너뛰었다(계보가 증거신호가 아니라 별도 특화 감지기라서).

## 두 갈래를 **곱하지 않고 따로** 돌린다

곱하면 288변형이라 Stage 1에 과하고, 추가 축이 의미 있는지부터 확인하는 게 순서다.

  (A) 파라미터 격자 48셀: ATR_MULT x T_SUSTAIN x FULL_BARS
      -- 현행 정의(ambiguous=drop, anchor=wick) 고정
  (B) 정의 ablation 6셀: ambiguous 처리 3종 x anchor 2종
      -- 현행 파라미터(1.5/0.20/12) 고정

### (B)의 ambiguous 처리 3종 -- 이번 스크리닝의 핵심 아이디어

현행은 `is_v = fast_mult>=K AND giveback<=T`, `is_chop = fast_mult<CHOP_MULT`, 나머지는 전부
`ambiguous`로 **학습에서 제외**한다. 그런데 그 나머지는 정반대 두 종류가 섞여 있다:

  (a) 1.0 <= fast_mult < K            -- 애매하게 약한 움직임. 버리는 게 타당
  (b) fast_mult >= K 인데 giveback > T -- **K*ATR 튀었다가 되반납**한 명백한 실패

(b)는 애매한 게 아니라 이 신호가 가장 피해야 할 거짓양성이다. "튀고 유지"와 "튀고 반납"을
가르는 게 과제의 본질인데 후자를 통째로 안 가르치고 있다. 게다가 매 봉 스코어링이라 **추론
시엔 그 봉들도 반드시 채점**하고, 2026-09-01 실측에서 **라벨 없는 봉의 발동률이 라벨 있는 봉보다
높았다** -- 모델이 한 번도 배운 적 없는 구간에서 가장 자신 있게 신호를 낸다는 뜻이다.

  drop     : 현행 그대로(ambiguous 전부 제외)
  fail_neg : (b)만 음성으로 편입, (a)는 여전히 제외   <- 제안하는 안
  all_neg  : (a)(b) 모두 음성으로 편입(= 이진 분류, 제외 없음)

### (B)의 anchor 2종

`fast_move = max(close[i+1..i+6]) - low[i]`는 **저가(꼬리)에서 재고 종가로 끝난다**. 아랫꼬리가
길면 실제로 없던 이동폭이 부풀려지는데, 라이브는 봉 i **종가**에 신호가 뜨고 다음 봉에 들어가므로
그 꼬리를 못 먹는다. 2026-09-01에 고친 진입시점 비현실성과 같은 계열의 두 번째 문제.
`wick`(현행) vs `body`(종가 앵커)를 비교한다.

## 선정 기준: **경제성**이지 AUC가 아니다

라벨 정의가 다르면 문제 난이도가 달라져 AUC 직접 비교는 무효다
([[feedback_cross_model_auc_comparison_requires_matched_label_difficulty_20260901]]).
AUC는 참고로만 찍고, 순위는 트레일링 경제성으로 매긴다 -- bp는 같은 가격계열 위의 공통 단위다.

Stage 1은 **축소 격자**(ARM>=1.0 구간 중심)만 쓴다. 2026-09-01 실측에서 진짜 엣지는 고ARM에
있었고 저ARM은 노이즈수확 아티팩트였다. 통과 후보만 Stage 2에서 240셀 전수 + 방향뒤집기.

⚠️**다중검정**: 54변형 x 18셀 = 972 설정을 VAL에서 본다. 이 저장소는 DSR/PBO를 통과한 적이
없다([[eth_live_stack_never_passed_dsr_pbo]]). 그래서 최고값만이 아니라 **몇 개가 통과했는지와
방향뒤집기 대비 우위**를 함께 보고한다. OOS는 Stage 3에서 최종 1개에만 쓴다.

⚠️ VAL만 사용. **OOS/HOLDOUT 미터치**. 라이브 코드 변경 없음.

Run on the server via handoff:
  handoff.sh launch server <job> -- python scripts/research_eth_v_rebound_label_grid_screen_stage1_20260901.py
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

FEAS = ROOT / "scripts/research_eth_v_rebound_every_bar_scoring_feasibility_20260901.py"
_spec = importlib.util.spec_from_file_location("everybar_feas_grid", FEAS)
_feas = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_feas)
_vs = _feas._vs

BT = ROOT / "scripts/backtest_eth_v_rebound_every_bar_trailing_costgate_20260901.py"
_bspec = importlib.util.spec_from_file_location("everybar_costgate_grid", BT)
_bt = importlib.util.module_from_spec(_bspec)
_bspec.loader.exec_module(_bt)

FEATURE_COLUMNS = _feas.FEATURE_COLUMNS
STANDARD_COST_BP = _bt.STANDARD_COST_BP
FORWARD_BARS = _bt.FORWARD_BARS

TRAIN_END = pd.Timestamp("2025-09-01", tz="UTC")
VAL_END = pd.Timestamp("2026-01-01", tz="UTC")   # Stage 1은 여기서 끊는다 -- OOS 미터치
START = pd.Timestamp("2024-01-01", tz="UTC")

SEEDS = [20260829, 141592]
PROBA_THRESHOLD = 0.60          # 현행 배포판 운영점 -- 아래 TARGET_CALL_N 산출에만 쓴다
# 모든 변형을 **같은 호출 빈도**에서 비교한다(임계값은 변형마다 그 빈도에 맞춰 자동 결정).
# 값은 현행 배포 설정(라벨 1.5/0.20/12, thr 0.60)이 VAL 전체봉에서 내는 호출수 -- 즉 "지금과
# 같은 매매 빈도일 때 어느 라벨이 더 나은 거래를 고르는가"를 묻는 것.
TARGET_CALL_N = 1256
FAST_BARS_FIXED = 6             # 30분 -- 이번 라운드에서는 고정(격자를 더 키우지 않기 위해)
CHOP_MULT = 1.0

# (A) 파라미터 격자 -- 현행 정의 고정
GRID_ATR_MULT = [1.25, 1.50, 1.75, 2.00]
GRID_T_SUSTAIN = [0.15, 0.20, 0.25, 0.30]
GRID_FULL_BARS = [12, 18, 24]
# (B) 정의 ablation -- 현행 파라미터 고정
ABLATION_AMBIG = ["drop", "fail_neg", "all_neg"]
ABLATION_ANCHOR = ["wick", "body"]
BASE = {"atr_mult": 1.50, "t_sustain": 0.20, "full_bars": 12, "ambig": "drop", "anchor": "wick"}

# Stage 1 축소 경제성 격자 -- 고ARM 중심(저ARM은 노이즈수확 아티팩트 구간)
SL_GRID = (2.5, 4.0, 5.0)
ARM_GRID = (1.0, 1.5)
TRAIL_GRID = (0.10, 0.15, 0.20)

OUT_JSON = ROOT / "data/research/eth_v_rebound_label_grid_stage1_20260901/report.json"


def log(msg: str) -> None:
    print(f"[grid1] {msg}", flush=True)


def label_param(sig: pd.DataFrame, is_down: bool, *, atr_mult: float, t_sustain: float,
                full_bars: int, ambig: str, anchor: str) -> np.ndarray:
    """`label_variant()`의 파라미터화 버전. shift=0 고정.

    ⚠️산술은 원본에서 **그대로** 옮겼다(재유도 금지 -- 원본과 어긋나면 격자 전체가 무효).
    아래 self-check가 BASE 설정에서 원본과 문자 그대로 일치하는지 매 실행 확인한다."""
    close, high, low = (sig[c].to_numpy() for c in ("close", "high", "low"))
    atr = sig["atr"].to_numpy()

    extreme = (low if is_down else high) if anchor == "wick" else close
    pre_atr = _vs.shifted_at(atr, -1)
    fast_close_max = _vs.fwd_window(close, 1, FAST_BARS_FIXED, "max")
    fast_close_min = _vs.fwd_window(close, 1, FAST_BARS_FIXED, "min")
    full_high_max = _vs.fwd_window(high, 1, full_bars, "max")
    full_low_min = _vs.fwd_window(low, 1, full_bars, "min")
    end_price = _vs.shifted_at(close, full_bars)

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
        is_v = strong & np.isfinite(giveback) & (giveback <= t_sustain)
        is_chop = fast_mult < CHOP_MULT
        # ⭐ambiguous 처리: (b)=강하게 튀었으나 되반납 -- 애매한 게 아니라 명백한 실패
        fail = strong & ~is_v
        if ambig == "drop":
            neg = is_chop
        elif ambig == "fail_neg":
            neg = is_chop | fail
        elif ambig == "all_neg":
            neg = ~is_v
        else:
            raise ValueError(ambig)

    return np.where(~valid, "invalid", np.where(is_v, "v_rebound", np.where(neg, "chop", "ambiguous")))


def build_sig():
    """지표/트리거 프레임 -- feasibility의 것을 그대로 재사용."""
    feat = _feas.build_all_bar_frame()
    eth = _vs.load_klines(_feas.ETH_CSV)
    impl = _vs.load_impl()
    causal = impl.add_causal_columns(eth[["timestamp", "open", "high", "low", "close"]].copy())
    from live_evidence_signal_dashboard_20260823 import compute_signals
    btc = _vs.load_klines(_feas.BTC_CSV)
    sig = compute_signals(eth, btc_df=btc, funding_df=None)
    sig["atr"] = causal["atr"].to_numpy()
    return sig, feat, eth


def long_frame_for(sig, feat, status_b, status_t) -> pd.DataFrame:
    """(bar, side) long frame + 주어진 status. 피쳐 계산은 feasibility와 동일 공식.

    ⚠️`sig`(280,471행)와 `feat`(280,363행)은 길이가 다르다 -- 위치로 붙이면 안 되고 **timestamp로
    merge**해야 한다(원본 build_long_frame()도 `how="inner"` merge를 쓴다). 위치 정렬로 붙였다가
    길이 불일치로 터진 적 있음(2026-09-01 첫 실행)."""
    st_frame = pd.DataFrame({"timestamp": sig["timestamp"].to_numpy(),
                             "status_b": status_b, "status_t": status_t})
    merged = st_frame.merge(feat, on="timestamp", how="inner", suffixes=("", "_f"))
    rows = []
    for side, is_down, col in (("bottom", True, "status_b"), ("top", False, "status_t")):
        d = merged
        st = d[col].to_numpy()
        sub = pd.DataFrame({"timestamp": d["timestamp"], "side": side})
        sub["is_downside"] = np.int8(1 if is_down else 0)
        level = d["sweep_level_low"].to_numpy() if is_down else d["sweep_level_high"].to_numpy()
        atr = d["atr"].to_numpy(dtype=float)
        pen = (level - d["low"].to_numpy()) if is_down else (d["high"].to_numpy() - level)
        sub["sweep_penetration_atr"] = pen / atr
        sub["atr"] = atr
        for col in ["atr_percentile_864", "range_width_pct", "hour_utc", "weekday", "p_fast",
                    "p_slow", "vwap_dev_z", "cvd_roll_roc_48", "vol_z", "lower_wick_ratio",
                    "upper_wick_ratio", "bb_pctb", "adx14", "pdi", "ndi", "bb_width_pctile",
                    "ret3_z", "rsi"]:
            sub[col] = d[col].to_numpy()
        dz = d["delta_z"].to_numpy(dtype=float)
        sub["delta_z"] = dz
        sub["flow_aligned_delta_z"] = dz if is_down else -dz
        sub["status"] = st
        rows.append(sub)
    long = pd.concat(rows, ignore_index=True)
    long = long.loc[(long["timestamp"] >= START) & (long["timestamp"] < VAL_END)]
    long = long.dropna(subset=FEATURE_COLUMNS).reset_index(drop=True)
    long["label"] = np.where(long["status"] == "v_rebound", 1.0,
                     np.where(long["status"] == "chop", 0.0, np.nan))
    long["split"] = np.where(long["timestamp"] < TRAIN_END, "TRAIN", "VAL")
    return long


def evaluate(long: pd.DataFrame, kl: pd.DataFrame, tag: str, target_call_n: int) -> dict:
    """GBM 학습 -> VAL 전체봉 채점 -> 축소 경제성 격자 + 방향뒤집기."""
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.metrics import roc_auc_score

    lab = long.loc[long["label"].notna()]
    tr = lab.loc[lab["split"] == "TRAIN"]
    if len(tr) < 2000 or tr["label"].nunique() < 2:
        return {"tag": tag, "skipped": "TRAIN 표본 부족"}
    va_all = long.loc[long["split"] == "VAL"].copy()      # 라벨 없는 봉 포함(라이브 모집단)
    va_lab = lab.loc[lab["split"] == "VAL"]

    probs = []
    for sd in SEEDS:
        m = HistGradientBoostingClassifier(random_state=sd, max_iter=300, early_stopping=True,
                                           validation_fraction=0.15)
        m.fit(tr[FEATURE_COLUMNS], tr["label"].to_numpy())
        probs.append(m.predict_proba(va_all[FEATURE_COLUMNS])[:, 1])
    va_all["model_proba"] = np.mean(probs, axis=0)

    auc = None
    if len(va_lab):
        p_lab = va_all.set_index(["timestamp", "side"]).loc[
            list(zip(va_lab["timestamp"], va_lab["side"])), "model_proba"].to_numpy()
        if len(np.unique(va_lab["label"])) == 2:
            auc = float(roc_auc_score(va_lab["label"].to_numpy(), p_lab))

    # ⚠️임계값을 변형 간에 고정하면 안 된다. 음성을 늘리거나(fail_neg/all_neg) 라벨을 어렵게
    # 하면(anchor=body) 확률 분포 전체가 내려가, 고정 0.60에서는 호출이 0건이 되어 변형이
    # "성능 없음"이 아니라 "측정 불가"로 탈락한다(2026-09-01 첫 실행에서 B 6셀 중 5셀이 그렇게
    # 스킵됐다). 임계값도 라벨과 함께 달라지는 양이므로, **동일 호출 빈도**에서 비교한다 --
    # "같은 매매 빈도일 때 어느 라벨이 더 나은 거래를 고르는가"가 답해야 할 질문이다.
    # AUC를 라벨 간에 비교하면 안 되는 것과 정확히 같은 이유.
    k = min(target_call_n, len(va_all))
    cutoff = float(np.partition(va_all["model_proba"].to_numpy(), -k)[-k]) if k > 0 else 1.0
    called = va_all.nlargest(k, "model_proba")
    ts_to_pos = {t: i for i, t in enumerate(kl["timestamp"].to_numpy())}
    o, h, l, c = (kl[x].to_numpy() for x in ("open", "high", "low", "close"))
    rows = []
    for _, ev in called.iterrows():
        i = ts_to_pos.get(np.datetime64(ev["timestamp"].tz_localize(None)))
        if i is None or i + FORWARD_BARS + 1 >= len(kl):
            continue
        rows.append({"side": "long" if ev["is_downside"] == 1 else "short",
                     "atr": float(ev["atr"]), "entry_price": float(o[i + 1]),
                     "fwd_open": o[i + 1:i + 1 + FORWARD_BARS], "fwd_high": h[i + 1:i + 1 + FORWARD_BARS],
                     "fwd_low": l[i + 1:i + 1 + FORWARD_BARS], "fwd_close": c[i + 1:i + 1 + FORWARD_BARS]})
    df = pd.DataFrame(rows)
    out = {"tag": tag, "train_n": int(len(tr)), "train_pos_rate": round(float(tr["label"].mean()), 4),
           "val_labeled_n": int(len(va_lab)), "val_auc": round(auc, 4) if auc else None,
           "labeled_row_pct": round(float(len(lab) / len(long) * 100), 1),
           "n_called": int(len(df)), "call_rate": round(float(len(called) / len(va_all)), 4),
           "matched_call_n": int(k), "threshold_used": round(cutoff, 4)}
    if len(df) < 50:
        out["skipped"] = f"호출 {len(df)}건 -- 표본 부족"
        return out

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
                flips.append({"opt_bp": float(fo.mean() * 1e4 - STANDARD_COST_BP),
                              "pess_bp": float(fp.mean() * 1e4 - STANDARD_COST_BP)})
    n_pass = sum(1 for g in cells if g["opt_bp"] > 0 and g["pess_bp"] > 0)
    n_flip_pass = sum(1 for g in flips if g["opt_bp"] > 0 and g["pess_bp"] > 0)
    best = max(cells, key=lambda g: g["pess_bp"])
    bi = cells.index(best)
    out.update({
        "n_cells": len(cells), "n_pass": n_pass, "n_flip_pass": n_flip_pass,
        "best": {k: round(v, 3) if isinstance(v, float) else v for k, v in best.items()},
        "best_flip_opt_bp": round(flips[bi]["opt_bp"], 2),
        # 총 기대이익 = 건당 bp x 호출수. 건당만 보면 드물게 잘 맞는 라벨이 과대평가된다.
        "total_bp": round(best["pess_bp"] * len(df), 0),
    })
    return out


def main() -> int:
    t0 = time.time()
    log("building indicator/trigger frame...")
    sig, feat, eth = build_sig()
    kl = eth[["timestamp", "open", "high", "low", "close"]].copy()
    kl["timestamp"] = kl["timestamp"].dt.tz_localize(None)  # tz-aware면 ts_to_pos가 전부 미스

    # self-check: BASE 설정이 원본 label_variant와 문자 그대로 일치해야 한다
    for is_down in (True, False):
        mine = label_param(sig, is_down, atr_mult=BASE["atr_mult"], t_sustain=BASE["t_sustain"],
                           full_bars=BASE["full_bars"], ambig="drop", anchor="wick")
        ref = _vs.label_variant(sig, is_down=is_down, anchor_mode="wick", shift=0)
        n_mis = int((mine != ref).sum())
        log(f"self-check {'bottom' if is_down else 'top'}: 원본과 불일치 {n_mis}건 (0이어야 정상)")
        if n_mis:
            log("  ⛔ 파라미터화가 원본과 어긋남 -- 중단")
            return 1

    results = []

    def run(tag, **kw):
        sb = label_param(sig, True, **kw)
        st = label_param(sig, False, **kw)
        long = long_frame_for(sig, feat, sb, st)
        r = evaluate(long, kl, tag, target_call_n=TARGET_CALL_N)
        r["params"] = kw
        results.append(r)
        if "skipped" in r:
            log(f"  {tag:44s} SKIP({r['skipped']})")
        else:
            log(f"  {tag:44s} 라벨행 {r['labeled_row_pct']:4.1f}% 양성 {r['train_pos_rate']:.3f} "
                f"AUC {r['val_auc']}  thr {r['threshold_used']:.3f}  호출 {r['n_called']:>4d}  "
                f"통과 {r['n_pass']:>2d}/{r['n_cells']}(뒤집기 {r['n_flip_pass']:>2d})  "
                f"최고 pess{r['best']['pess_bp']:+7.2f}bp 총{r['total_bp']:+9.0f}bp")

    log("")
    log("=== (B) 정의 ablation 6셀 (파라미터는 현행 고정) ===")
    for ambig in ABLATION_AMBIG:
        for anchor in ABLATION_ANCHOR:
            run(f"B|ambig={ambig:8s}|anchor={anchor}",
                atr_mult=BASE["atr_mult"], t_sustain=BASE["t_sustain"],
                full_bars=BASE["full_bars"], ambig=ambig, anchor=anchor)

    log("")
    log("=== (A) 파라미터 격자 48셀 (정의는 현행 고정: ambig=drop, anchor=wick) ===")
    for k in GRID_ATR_MULT:
        for t in GRID_T_SUSTAIN:
            for fb in GRID_FULL_BARS:
                run(f"A|K={k:.2f}|T={t:.2f}|FULL={fb:2d}",
                    atr_mult=k, t_sustain=t, full_bars=fb, ambig="drop", anchor="wick")

    scored = [r for r in results if "skipped" not in r]
    scored.sort(key=lambda r: r["best"]["pess_bp"], reverse=True)
    log("")
    log("=== 건당 bp 상위 8 ===")
    for r in scored[:8]:
        log(f"  {r['tag']:44s} pess{r['best']['pess_bp']:+7.2f}bp  통과{r['n_pass']:>2d}/"
            f"{r['n_cells']}(뒤집기{r['n_flip_pass']:>2d})  호출{r['n_called']:>4d}  총{r['total_bp']:+9.0f}bp")
    by_total = sorted(scored, key=lambda r: r["total_bp"], reverse=True)
    log("")
    log("=== 총 bp 상위 8 (건당만 보면 드물게 맞는 라벨이 과대평가된다) ===")
    for r in by_total[:8]:
        log(f"  {r['tag']:44s} 총{r['total_bp']:+9.0f}bp  건당 pess{r['best']['pess_bp']:+7.2f}bp  호출{r['n_called']:>4d}")
    base_row = next((r for r in scored if r["tag"].startswith("A|K=1.50|T=0.20|FULL=12")), None)
    if base_row:
        log("")
        log(f"현행 기준선: pess{base_row['best']['pess_bp']:+.2f}bp  총{base_row['total_bp']:+.0f}bp  "
            f"호출{base_row['n_called']}  통과{base_row['n_pass']}/{base_row['n_cells']}")

    report = {
        "signal": "v_rebound_label_grid_screen_stage1", "asset": "ETHUSDT",
        "fresh_forward_bar_by_bar": True, "trade_ledgers_used_as_input": False,
        "saved_parent_exit_timestamps_used": False, "future_rows_used_for_entry": False,
        "scope": {"stage": 1, "model": "GBM proxy (TabPFN과 -0.0011로 검증됨)",
                  "splits_used": "TRAIN+VAL only", "oos_touched": False, "holdout_touched": False,
                  "live_code_changed": False, "proba_threshold": PROBA_THRESHOLD,
                  "economics_grid": {"sl": list(SL_GRID), "arm": list(ARM_GRID),
                                     "trail": list(TRAIL_GRID),
                                     "note": "축소격자(고ARM 중심) -- Stage 2에서 240셀 전수"},
                  "selection_metric": "경제성(건당 pess_bp + 총 bp). AUC는 참고용 -- 라벨 정의가 "
                                      "다르면 난이도가 달라 직접 비교 무효",
                  "multiple_testing": "54변형 x 18셀 = 972설정. 최고값과 함께 통과 개수/뒤집기 "
                                      "대비를 보고. OOS는 Stage 3에서 최종 1개에만."},
        "fixed": {"FAST_BARS": FAST_BARS_FIXED, "CHOP_MULT": CHOP_MULT, "seeds": SEEDS},
        "grids": {"A": {"atr_mult": GRID_ATR_MULT, "t_sustain": GRID_T_SUSTAIN,
                        "full_bars": GRID_FULL_BARS},
                  "B": {"ambig": ABLATION_AMBIG, "anchor": ABLATION_ANCHOR}},
        "base": BASE, "results": results,
        "top_by_per_trade": [r["tag"] for r in scored[:8]],
        "top_by_total": [r["tag"] for r in by_total[:8]],
        "runtime_sec": round(time.time() - t0, 1),
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str))
    log("")
    log(f"report saved -> {OUT_JSON}  ({report['runtime_sec']}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
