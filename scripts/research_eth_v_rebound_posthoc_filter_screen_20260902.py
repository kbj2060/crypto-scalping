#!/usr/bin/env python3
"""배포판 V자반등 모델에 **사후 필터**를 걸어 방향성이 살아나는지 검정.

## 왜 이 실험인가

2026-09-02까지 모델 축은 전부 소진됐다: 라벨 격자/타이트화 · 8트리거 피쳐화 · 154피쳐 ·
ambiguous 처리 8구성 · 확률 꼬리 · GBM프록시 6건 TabPFN 재확인 -- 어느 것도 OOS 방향성을
만들지 못했다. 남은 축은 **모델을 바꾸는 게 아니라 모델의 호출 중 어느 것을 취할지**다.

## ⚠️ 이 방향의 사전 확률은 낮다 -- 두 가지가 이미 막혀 있다

1. **베이스 임계값 조이기는 실패했다**(같은 날 꼬리 조사). thr 0.60->0.90에서 VAL은 진짜였으나
   (랜덤부분표집 귀무분포 98.5~100백분위) **OOS 전이 실패**(격자 정17/뒤36).
2. **복합 AND-필터가 기각됐다**(같은 날, 증거신호 맥락). 풀링에서 7/60셀 생존 + 계열 일관성까지
   보였으나 **VAL/OOS 창 분리에서 0/7**. 거기서 나온 결론이 "발동을 줄이려면 직교 필터 말고
   자기 임계값을 조일 것"인데, 1이 그것도 안 된다는 뜻이라 양쪽이 막혀 있다.

그래도 안 해본 것은 맞다 -- 1은 **모델 자기 확률**로 잘랐고 이건 **모델 밖 조건**으로 자른다.
대신 관문이 하나 늘었다: 필터는 **이미 OOS에서 실패하는 베이스에 없던 방향성을 만들어야** 한다.
"정밀도가 올랐다"로는 부족하다(AND-필터는 부분집합+n축소로 정밀도가 기계적으로 오른다).

## 베이스 = 실제 배포판 (재학습 아님)

`live_eth_sweep_v_rebound_signal_20260829.py`가 서빙에 쓰는 것과 **동일**:
동결 컨텍스트 CSV(18,000행) + TabPFN(seed 20260829, ignore_pretraining_limits) + thr 0.60.
자체검증으로 VAL AUC가 배포 실측 0.6942에 근접하는지 확인하고, 어긋나면 즉시 중단한다.

## 필터 -- 전부 인과적, 3쌍(각 쌍의 짝이 서로의 플라시보)

  R1a 레짐 chop        / R1b 레짐 trend      (배포된 GBM2 trend/chop 분류기)
  R2a 증거신호 합류>=2 / R2b 합류==0         (8종, **후방 [i-3,i]창만** -- 전방은 룩어헤드)
  R3a basis 정렬극단   / R3b basis 중간대     (현물-선물, 인과적 864봉 롤링 백분위)

⚠️ATR백분위·시간대·요일은 **의도적으로 제외** -- 이미 Tier0 피쳐라 모델이 보고도 안 쓴 정보다.
⚠️청산맵 자석은 수집이 2026-08 시작이라 VAL(2025-09~12)을 못 덮어 제외.

## 관문 셋 (2026-09-02 확립, README §5.15)

  1. **랜덤 부분표집 귀무분포**(B=200) -- 베이스 호출에서 같은 n을 무작위로 뽑아 비교.
     "필터가 정보인가, 아니면 그냥 호출을 줄인 것인가"를 가른다.
  2. **임계값-매칭 베이스** -- 같은 n이 되도록 베이스 thr을 올린 것(확률 상위 n)과 비교.
     못 이기면 정보가 아니라 희소성 손잡이다.
  3. **VAL/OOS 창 분리** -- 풀링 금지. 1·2를 통과하고도 여기서 전멸한 전례가 있다.

판정 지표는 **방향뒤집기 통제 경제성**(ARM>=1.0 80셀 격자의 정방향-뒤집기 통과수 차)이다.
lift도 AUC도 아니다 -- V자반등이 무너지는 지점이 정확히 방향성이라서.

⚠️HOLDOUT(2026-04-01~) 미터치. 라이브 코드 변경 없음.

Run on the server (GPU) via handoff:
  handoff.sh launch server <job> -- python scripts/research_eth_v_rebound_posthoc_filter_screen_20260902.py
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


def _load(name: str, rel: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_s1 = _load("vreb_s1_filter", "scripts/research_eth_v_rebound_label_grid_screen_stage1_20260901.py")
_feas, _bt = _s1._feas, _s1._bt

EIGHT = [t for t in _feas.ALL9 if t != "local_extreme"]
STANDARD_COST_BP, FORWARD_BARS = _s1.STANDARD_COST_BP, _s1.FORWARD_BARS
SL_GRID, ARM_GRID, TRAIL_GRID = _bt.SL_GRID, _bt.ARM_GRID, _bt.TRAIL_GRID

CTX_CSV = ROOT / "data/labels/eth_5m_v_rebound_every_bar_20260901/tabpfn_train_context_frozen_every_bar_20260901.csv"
REGIME_MODEL = ROOT / "tmp/eth_regime_gbm2_trend_chop_20260827/model.joblib"
CANON = [ROOT / "data/splits/year_oos/training_features_2025.csv",
         ROOT / "data/splits/year_oos/training_features_2026_rebuilt.csv"]
SPOT_CSV = ROOT / "binance_data/klines_spot/ETHUSDT/ETHUSDT-5m-spot.csv"

DEPLOYED_LABEL = {"atr_mult": 1.50, "t_sustain": 0.20, "full_bars": 12}
PROBA_THRESHOLD = 0.60
LIVE_SEED = 20260829
DEPLOYED_VAL_AUC = 0.6942          # 배포 실측 (context_report.json)
AUC_TOLERANCE = 0.02               # 이보다 벌어지면 베이스 재현 실패로 간주하고 중단

ARTIFACT_FREE_MIN = 1.0
CONFLUENCE_LOOKBACK = 3            # 후방 창만 (전방은 룩어헤드)
BASIS_WINDOW = 864                 # 3일, atr_percentile_864와 같은 관례
NULL_B = 200
NULL_SEED = 20260902
NULL_MIN_N = 60

TRAIN_END = pd.Timestamp("2025-09-01", tz="UTC")
VAL_END = pd.Timestamp("2026-01-01", tz="UTC")
OOS_END = pd.Timestamp("2026-04-01", tz="UTC")

OUT_JSON = ROOT / "data/research/eth_v_rebound_posthoc_filter_20260902/report.json"


def log(msg: str) -> None:
    print(f"[filter] {msg}", flush=True)


def regime_chop_prob() -> pd.DataFrame:
    """배포된 GBM2 trend/chop 분류기를 캐노니컬 피쳐에 적용해 과거 구간 chop_prob를 만든다."""
    import joblib
    from retrain_clean_regime_hmm_raw_state12_20260517 import _with_raw_state12
    p = joblib.load(REGIME_MODEL)
    cols, med = p["feature_cols"], pd.Series(p["feature_medians"])
    ci = list(p["classes"]).index("chop")
    out = []
    for f in CANON:
        d = pd.read_csv(f)
        d = _with_raw_state12(d)
        miss = [c for c in cols if c not in d.columns]
        if miss:
            raise RuntimeError(f"{f.name}: 레짐 피쳐 결측 {len(miss)} -- {miss[:5]}")
        x = (d[cols].apply(pd.to_numeric, errors="coerce")
             .replace([np.inf, -np.inf], np.nan).fillna(med).fillna(0.0))
        out.append(pd.DataFrame({"timestamp": pd.to_datetime(d["timestamp"], utc=True),
                                 "chop_prob": p["model"].predict_proba(x)[:, ci]}))
    return pd.concat(out, ignore_index=True).drop_duplicates("timestamp")


def basis_pctile(sig: pd.DataFrame) -> pd.DataFrame:
    """현물-선물 베이시스의 **인과적** 롤링 백분위 (현재봉 포함 과거 BASIS_WINDOW봉 내 순위)."""
    sp = pd.read_csv(SPOT_CSV, usecols=["timestamp", "close"])
    sp["timestamp"] = pd.to_datetime(sp["timestamp"], utc=True)
    sp = sp.rename(columns={"close": "spot_close"})
    m = pd.DataFrame({"timestamp": sig["timestamp"], "perp_close": sig["close"].to_numpy()})
    m = m.merge(sp, on="timestamp", how="left")
    basis = (m["perp_close"] - m["spot_close"]) / m["spot_close"]
    pct = basis.rolling(BASIS_WINDOW, min_periods=BASIS_WINDOW // 2).rank(pct=True)
    return pd.DataFrame({"timestamp": m["timestamp"], "basis_pct": pct.to_numpy()})


def confluence_count(sig: pd.DataFrame) -> pd.DataFrame:
    """side별, 후방 [i-LOOKBACK, i] 창에서 발동한 **서로 다른** 증거신호 개수."""
    rows = []
    for side in ("bottom", "top"):
        fires = np.vstack([sig[f"{side}_{t}"].fillna(False).to_numpy().astype(np.int8)
                           for t in EIGHT])                       # (8, n)
        # 각 신호별로 후방창 내 1회 이상 발동 여부 -> 합
        rolled = np.vstack([pd.Series(fires[k]).rolling(CONFLUENCE_LOOKBACK + 1, min_periods=1)
                            .max().to_numpy() for k in range(len(EIGHT))])
        rows.append(pd.DataFrame({"timestamp": sig["timestamp"].to_numpy(), "side": side,
                                  "confluence": rolled.sum(axis=0)}))
    return pd.concat(rows, ignore_index=True)


def main() -> int:
    t0 = time.time()
    from tabpfn import TabPFNClassifier
    from sklearn.metrics import roc_auc_score
    import torch
    log(f"cuda: {torch.cuda.is_available()}")

    # ---------- 1) 배포 베이스 재현 ----------
    ctx = pd.read_csv(CTX_CSV)
    FEATURES = [c for c in ctx.columns if c not in ("timestamp", "label")]
    log(f"동결 컨텍스트 {len(ctx):,}행 / 피쳐 {len(FEATURES)}개 (라벨률 {ctx['label'].mean():.4f})")
    clf = TabPFNClassifier(device="cuda", random_state=LIVE_SEED, ignore_pretraining_limits=True)
    clf.fit(ctx[FEATURES], ctx["label"].to_numpy())

    _s1.VAL_END = OOS_END
    log("building every-bar frame + labels (VAL+OOS) ...")
    sig, feat, eth = _s1.build_sig()
    sb = _s1.label_param(sig, True, ambig="drop", anchor="wick", **DEPLOYED_LABEL)
    st = _s1.label_param(sig, False, ambig="drop", anchor="wick", **DEPLOYED_LABEL)
    long = _s1.long_frame_for(sig, feat, sb, st)
    long["split"] = np.where(long["timestamp"] < TRAIN_END, "TRAIN",
                     np.where(long["timestamp"] < VAL_END, "VAL", "OOS"))
    assert long["timestamp"].max() < OOS_END, "HOLDOUT 누출"
    long = long.loc[long["split"] != "TRAIN"].dropna(subset=FEATURES).reset_index(drop=True)
    # 12만행을 한 번에 던지면 메모리/속도가 불리하다 -- 청크로 나눠 예측한다.
    CH = 20000
    long["p"] = np.concatenate([clf.predict_proba(long[FEATURES].iloc[i:i + CH])[:, 1]
                                for i in range(0, len(long), CH)])

    selfchk = {}
    for spname in ("VAL", "OOS"):
        s = long.loc[(long["split"] == spname) & long["label"].notna()]
        selfchk[spname] = round(float(roc_auc_score(s["label"], s["p"])), 4)
    log(f"자체검증 AUC  VAL {selfchk['VAL']} (배포 실측 {DEPLOYED_VAL_AUC})  OOS {selfchk['OOS']}")
    if abs(selfchk["VAL"] - DEPLOYED_VAL_AUC) > AUC_TOLERANCE:
        log("  ⚠️배포 베이스 재현 실패 -- 중단")
        return 1
    log("  ✅배포 베이스 재현 확인")

    # ---------- 2) 필터 컬럼 부착 ----------
    log("attaching filters (regime / confluence / basis) ...")
    long = long.merge(regime_chop_prob(), on="timestamp", how="left")
    long = long.merge(confluence_count(sig), on=["timestamp", "side"], how="left")
    long = long.merge(basis_pctile(sig), on="timestamp", how="left")
    cov = {c: round(float(long[c].notna().mean()), 4) for c in ("chop_prob", "confluence", "basis_pct")}
    log(f"  필터 커버리지: {cov}")
    for c, v in cov.items():
        if v < 0.90:
            log(f"  ⚠️{c} 커버리지 {v:.1%} -- 결측 행은 해당 필터에서 자동 탈락(편향 주의)")

    # ---------- 3) 경제성 도구 ----------
    kl = eth[["timestamp", "open", "high", "low", "close"]].copy()
    kl["timestamp"] = kl["timestamp"].dt.tz_localize(None)     # ⚠️tz-aware .to_numpy() 함정
    ts_to_pos = {t: i for i, t in enumerate(kl["timestamp"].to_numpy())}
    o, h, l, c = (kl[x].to_numpy() for x in ("open", "high", "low", "close"))
    nk = len(kl)
    long["pos"] = [ts_to_pos.get(np.datetime64(t.tz_localize(None)), -1) for t in long["timestamp"]]
    bad = int((long["pos"] < 0).sum())
    if bad:
        log(f"  ⚠️타임스탬프 미매칭 {bad}행 제외")
    long = long.loc[long["pos"] >= 0].reset_index(drop=True)

    # ⚠️필터는 **컬럼**으로 만든다 -- 넘파이 배열로 들고 있으면 위의 pos 절단/reset_index와
    # 조용히 어긋난다(행이 빠졌는데 마스크는 옛 길이 그대로).
    dn = long["is_downside"].to_numpy() == 1
    bp = long["basis_pct"].to_numpy(dtype=float)
    long["R1a_regime_chop"] = long["chop_prob"].to_numpy() >= 0.5
    long["R1b_regime_trend"] = long["chop_prob"].to_numpy() < 0.5
    long["R2a_confluence_ge2"] = long["confluence"].to_numpy() >= 2
    long["R2b_confluence_eq0"] = long["confluence"].to_numpy() == 0
    long["R3a_basis_aligned_ext"] = np.where(dn, bp <= 1 / 3, bp >= 2 / 3) & np.isfinite(bp)
    long["R3b_basis_mid"] = (bp > 1 / 3) & (bp < 2 / 3)
    FILTERS = ["R1a_regime_chop", "R1b_regime_trend", "R2a_confluence_ge2",
               "R2b_confluence_eq0", "R3a_basis_aligned_ext", "R3b_basis_mid"]
    for k in FILTERS:
        log(f"  {k:24s} 발동률 {long[k].mean()*100:5.1f}%")

    def build(s: pd.DataFrame) -> pd.DataFrame:
        rows = []
        for pos_i, isd, atr in zip(s["pos"].to_numpy(), s["is_downside"].to_numpy(),
                                    s["atr"].to_numpy()):
            i = int(pos_i)
            if i + FORWARD_BARS + 1 >= nk:
                continue
            rows.append({"side": "long" if isd == 1 else "short", "atr": float(atr),
                         "entry_price": float(o[i + 1]),
                         "fwd_open": o[i + 1:i + 1 + FORWARD_BARS],
                         "fwd_high": h[i + 1:i + 1 + FORWARD_BARS],
                         "fwd_low": l[i + 1:i + 1 + FORWARD_BARS],
                         "fwd_close": c[i + 1:i + 1 + FORWARD_BARS]})
        return pd.DataFrame(rows)

    def grid_summary(df: pd.DataFrame) -> dict | None:
        if len(df) < 30:
            return None
        e, a, s_, H, L, C = _bt.pack(df)
        ef, af, sf, Hf, Lf, Cf = _bt.pack(df, flip=True)
        fwd = flip = 0
        best = None
        for sl in SL_GRID:
            for arm in ARM_GRID:
                if arm < ARTIFACT_FREE_MIN:
                    continue
                for tr_ in TRAIL_GRID:
                    ov = _bt.simulate_trailing_vec(e, a, s_, H, L, C, sl, arm, tr_, False)
                    pv = _bt.simulate_trailing_vec(e, a, s_, H, L, C, sl, arm, tr_, True)
                    fo = _bt.simulate_trailing_vec(ef, af, sf, Hf, Lf, Cf, sl, arm, tr_, False)
                    fp = _bt.simulate_trailing_vec(ef, af, sf, Hf, Lf, Cf, sl, arm, tr_, True)
                    ob = float(ov.mean() * 1e4 - STANDARD_COST_BP)
                    pb = float(pv.mean() * 1e4 - STANDARD_COST_BP)
                    fwd += int(ob > 0 and pb > 0)
                    flip += int(float(fo.mean() * 1e4 - STANDARD_COST_BP) > 0
                                and float(fp.mean() * 1e4 - STANDARD_COST_BP) > 0)
                    if best is None or pb > best["pess_bp"]:
                        best = {"sl": sl, "arm": arm, "trail": tr_, "opt_bp": round(ob, 3),
                                "pess_bp": round(pb, 3)}
        return {"n": int(len(df)), "fwd_pass": fwd, "flip_pass": flip, "margin": fwd - flip,
                "n_cells": 80, "best": best}

    # ---------- 4) 필터별 3관문 ----------
    log("")
    log(f"=== 사후 필터 검정 (베이스 thr={PROBA_THRESHOLD}, ARM>={ARTIFACT_FREE_MIN} 80셀) ===")
    results = {}
    nrng = np.random.default_rng(NULL_SEED)
    for spname in ("VAL", "OOS"):
        s = long.loc[long["split"] == spname]
        base_sel = s.loc[s["p"] >= PROBA_THRESHOLD]
        base_g = grid_summary(build(base_sel))
        log("")
        log(f"--- {spname}  베이스 호출 {len(base_sel):,}건  "
            f"정{base_g['fwd_pass']}/뒤{base_g['flip_pass']} (차 {base_g['margin']:+d})  "
            f"최고pess {base_g['best']['pess_bp']:+.2f}bp ---")
        log(f"    {'필터':24s} {'n':>6s} {'정':>4s} {'뒤':>4s} {'차':>5s} {'최고pess':>9s}   "
            f"{'매칭베이스 차':>12s}  {'귀무백분위':>10s}")
        sp_res = {"base": base_g}
        for fname in FILTERS:
            fsel = base_sel.loc[base_sel[fname].to_numpy()]
            fg = grid_summary(build(fsel))
            if fg is None:
                log(f"    {fname:24s} {len(fsel):6,d}  -- 표본 부족(<30)")
                sp_res[fname] = {"n": int(len(fsel)), "skipped": True}
                continue
            n_f = fg["n"]
            # 관문 2: 임계값-매칭 베이스 (확률 상위 n_f)
            matched = base_sel.nlargest(len(fsel), "p")
            mg = grid_summary(build(matched))
            # 관문 1: 랜덤 부분표집 귀무분포 (베이스 호출에서 같은 n)
            pct = None
            if n_f >= NULL_MIN_N and len(base_sel) > len(fsel):
                margins = []
                for _ in range(NULL_B):
                    ridx = nrng.choice(len(base_sel), size=len(fsel), replace=False)
                    rg = grid_summary(build(base_sel.iloc[np.sort(ridx)]))
                    if rg:
                        margins.append(rg["margin"])
                if len(margins) >= 20:
                    pct = round(float((np.array(margins) < fg["margin"]).mean() * 100), 1)
            md = fg["margin"] - (mg["margin"] if mg else 0)
            flag = "✅" if (pct is not None and pct >= 95 and md > 0 and fg["margin"] > 0) else "  "
            log(f"    {fname:24s} {n_f:6,d} {fg['fwd_pass']:4d} {fg['flip_pass']:4d} "
                f"{fg['margin']:+5d} {fg['best']['pess_bp']:+8.2f}bp {md:+12d}  "
                f"{(f'{pct:.1f}%' if pct is not None else 'n/a'):>10s}{flag}")
            sp_res[fname] = {**fg, "matched_base": mg, "margin_vs_matched": md,
                             "null_pctile": pct}
        results[spname] = sp_res

    # ---------- 5) 판정 ----------
    log("")
    log("=== 판정: 세 관문 동시 통과 + VAL/OOS 양쪽 ===")
    winners = []
    for fname in FILTERS:
        v, o_ = results["VAL"].get(fname, {}), results["OOS"].get(fname, {})
        ok = all(x.get("margin", -1) > 0 and (x.get("null_pctile") or 0) >= 95
                 and x.get("margin_vs_matched", -1) > 0 for x in (v, o_))
        if ok:
            winners.append(fname)
            log(f"  ✅{fname}: VAL 차{v['margin']:+d}(null {v['null_pctile']}%) / "
                f"OOS 차{o_['margin']:+d}(null {o_['null_pctile']}%)")
    if not winners:
        log("  ❌통과 필터 없음 -- 사후 필터링으로도 OOS 방향성은 만들어지지 않는다")

    report = {"signal": "v_rebound_posthoc_filter_screen", "asset": "ETHUSDT",
              "fresh_forward_bar_by_bar": True, "trade_ledgers_used_as_input": False,
              "saved_parent_exit_timestamps_used": False, "future_rows_used_for_entry": False,
              "scope": {"base": "배포판 그대로(동결 컨텍스트+TabPFN seed 20260829+thr 0.60)",
                        "base_reproduction_check": selfchk,
                        "deployed_val_auc_reference": DEPLOYED_VAL_AUC,
                        "filters": FILTERS, "filter_coverage": cov,
                        "gates": ["랜덤 부분표집 귀무분포 B=200(베이스 호출 내)",
                                  "임계값-매칭 베이스(확률 상위 n)",
                                  "VAL/OOS 창 분리(풀링 금지)"],
                        "excluded_filters": {"atr/hour/weekday": "이미 Tier0 피쳐",
                                             "liquidation_magnet": "수집 2026-08 시작, VAL 미커버"},
                        "confluence_window": f"후방 [i-{CONFLUENCE_LOOKBACK}, i] (전방 배제)",
                        "holdout_touched": False, "live_code_changed": False},
              "results": results, "winners": winners,
              "runtime_sec": round(time.time() - t0, 1)}
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str))
    log("")
    log(f"report saved -> {OUT_JSON}  ({report['runtime_sec']}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
