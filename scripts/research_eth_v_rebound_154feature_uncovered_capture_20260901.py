#!/usr/bin/env python3
"""154피쳐 확장이 **미포착 사건을 보이게 하는가** -- 피쳐 축의 결정적 시험.

## 판정 기준은 AUC가 아니다

2026-09-01 확인: V자반등 사건의 38%가 9트리거 미포착이고, 라벨 품질은 포착 사건과 같은데
**모델이 거의 못 잡는다**(포착률 27.9% vs **8.5%**, OOS 27.4% vs 12.6%). 결과는 진짜인데
사전 피쳐로 안 보이는 사건이다. 라벨 타이트화는 지렛대가 아니었고(조일수록 미포착 비중 38%->58%),
8트리거 피쳐화는 구조적으로 불가능하다(미포착 사건에서 트리거는 정의상 전부 0).

그래서 이 실험의 판정 기준은 **미포착 사건 포착률**이다. AUC가 올라도 그게 포착 사건 쪽에서만
오르면 이 문제는 안 풀린 것이다. AUC는 참고로만 찍는다.

## 피쳐셋

  F0  Tier0 23            현행 배포판 (기준선)
  F1  Tier0 + 미시구조     문헌상 유일하게 강증거인 축(OFI/유동성/스프레드 근사)만 선별.
                          과적합 위험이 F2보다 훨씬 낮다.
  F2  Tier0 + 통과 150     감사 통과 전부. 상한을 보는 용도.

154셋 감사(audit_eth_154feature_lookahead_20260901.py): 150 pass / 3 모델출력 제외
(`regime3_*`, 순환성) / 1 경계 아티팩트. 정보량 상위 피쳐가 전부 다음 봉보다 직전 봉과 강하게
관계 -- 인과적 서명. ⚠️단 그 감사는 총체적 누출 탐지기이지 인과성 증명이 아니다.

## 모델

GBM(HistGradientBoosting) -- 154피쳐를 그대로 먹고, TabPFN 프록시로 2026-09-01 검증됨
(전체봉 AUC 0.6953 vs TabPFN 0.6942). 다시드로 노이즈와 구분한다.

⚠️ VAL/OOS만. HOLDOUT 미터치. 라이브 코드 변경 없음.

Run on the server via handoff:
  handoff.sh launch server <job> -- python scripts/research_eth_v_rebound_154feature_uncovered_capture_20260901.py
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
_spec = importlib.util.spec_from_file_location("vreb_s1_154", S1)
_s1 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_s1)
_feas = _s1._feas

AUDIT = ROOT / "scripts/research_eth_v_rebound_every_bar_label_event_audit_20260901.py"
_aspec = importlib.util.spec_from_file_location("vreb_audit_154", AUDIT)
_audit = importlib.util.module_from_spec(_aspec)
_aspec.loader.exec_module(_audit)

TIER0 = _s1.FEATURE_COLUMNS
ALL9 = _feas.ALL9
F154 = ROOT / "tmp/ilias_eth_154feature_dataset_20260821/ilias_eth_154feature_2024_2026H1_combined.csv"
AUDIT_JSON = ROOT / "data/research/eth_154feature_audit_20260901/report.json"

W, GAP, THRESHOLD = 6, 12, 0.60
DEPLOYED = {"atr_mult": 1.50, "t_sustain": 0.20, "full_bars": 12}
SEEDS = [20260829, 141592, 271828]
TRAIN_END = pd.Timestamp("2025-09-01", tz="UTC")
VAL_END = pd.Timestamp("2026-01-01", tz="UTC")
OOS_END = pd.Timestamp("2026-04-01", tz="UTC")

# 문헌상 강증거 축(OFI/유동성/스프레드/장기기억) -- F1용 선별
MICRO = ["kyle_lambda_48", "vpin_approx_48", "amihud_illiquidity_z", "corwin_schultz_spread",
         "roll_implied_spread_48", "ofi_acceleration", "ofti", "trade_intensity",
         "big_trade_ratio", "net_taker_ratio", "smart_money_flow", "execution_quality",
         "hurst_48", "hurst_288", "ffd_close_d03", "ffd_close_d05",
         "variance_ratio_q4_96", "variance_ratio_q12_96", "entropy_return_sign_48",
         "realized_semivar_ratio_96", "parkinson_vol", "garch_vol_z"]

OUT_JSON = ROOT / "data/research/eth_154feature_audit_20260901/uncovered_capture.json"


def log(msg: str) -> None:
    print(f"[f154] {msg}", flush=True)


def main() -> int:
    t0 = time.time()
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.metrics import roc_auc_score

    passed = [f["feature"] for f in json.loads(AUDIT_JSON.read_text())["features"]
              if f["verdict"] == "pass"]
    log(f"감사 통과 피쳐 {len(passed)}개")

    _s1.VAL_END = OOS_END
    log("building frame + labels...")
    sig, feat, eth = _s1.build_sig()
    sb = _s1.label_param(sig, True, ambig="drop", anchor="wick", **DEPLOYED)
    st = _s1.label_param(sig, False, ambig="drop", anchor="wick", **DEPLOYED)
    long = _s1.long_frame_for(sig, feat, sb, st)

    ex = pd.read_csv(F154)
    ex["timestamp"] = pd.to_datetime(ex["timestamp"]).dt.tz_localize("UTC")
    ex_cols = [c for c in passed if c in ex.columns]
    long = long.merge(ex[["timestamp"] + ex_cols], on="timestamp", how="left")
    log(f"154셋 병합: {len(ex_cols)}컬럼, 결측행 {int(long[ex_cols].isna().any(axis=1).sum()):,}/{len(long):,}")

    long["split"] = np.where(long["timestamp"] < TRAIN_END, "TRAIN",
                     np.where(long["timestamp"] < VAL_END, "VAL", "OOS"))
    assert long["timestamp"].max() < OOS_END, "HOLDOUT 누출"

    micro = [c for c in MICRO if c in ex_cols]
    SETS = {"F0 Tier0 23": TIER0,
            f"F1 +미시구조 {len(micro)}": TIER0 + micro,
            f"F2 +통과 {len(ex_cols)}": TIER0 + ex_cols}
    log(f"F1 미시구조 실제 사용 {len(micro)}개: {', '.join(micro[:8])}...")

    # 사건 구성(포착/미포착) -- 피쳐셋과 무관하므로 한 번만
    n = len(sig)
    lowv, highv = sig["low"].to_numpy(), sig["high"].to_numpy()
    ts_arr = sig["timestamp"].to_numpy()
    lo_f = np.zeros(n, bool); hi_f = np.zeros(n, bool)
    for i in range(W, n - W):
        if lowv[i] == lowv[i - W:i + W + 1].min():
            lo_f[i] = True
        if highv[i] == highv[i - W:i + W + 1].max():
            hi_f[i] = True
    events = {}
    for side, is_down in (("bottom", True), ("top", False)):
        trig = np.any([sig[f"{side}_{nm}"].fillna(False).to_numpy()
                       for nm in ALL9 if nm != "local_extreme"], axis=0)
        trig = trig | (lo_f if is_down else hi_f)
        status = sb if is_down else st
        for ev in _audit.cluster_events(np.flatnonzero(status == "v_rebound"), GAP):
            ts0 = pd.Timestamp(ts_arr[int(ev[0])])
            # ⚠️sig는 현재까지 전 구간이라 잘라주지 않으면 HOLDOUT(>=2026-04-01) 사건까지 "OOS"로
            # 세어 로그가 오도된다(첫 실행에서 OOS 1,006이어야 할 것이 2,736으로 찍혔다).
            # 확률이 없어 계산에선 자동 제외됐지만 출력은 틀렸으므로 여기서 명시적으로 자른다.
            if ts0 < TRAIN_END or ts0 >= OOS_END:
                continue
            sp = "VAL" if ts0 < VAL_END else "OOS"
            events.setdefault(sp, []).append(
                {"side": side, "bars": [pd.Timestamp(ts_arr[i]) for i in ev],
                 "covered": bool(trig[ev].any())})
    for sp, evs in events.items():
        u = sum(1 for e in evs if not e["covered"])
        log(f"  {sp}: 사건 {len(evs):,} (미포착 {u/len(evs)*100:.1f}%)")

    results = {}
    target_calls = {}   # F0가 세운 호출 빈도 -- 전 피쳐셋 공통
    for name, cols in SETS.items():
        cols = [c for c in cols if c in long.columns]
        sub = long.dropna(subset=cols).copy()
        lab = sub.loc[sub["label"].notna()]
        tr = lab.loc[lab["split"] == "TRAIN"]
        log("")
        log(f"=== {name} ({len(cols)}피쳐)  TRAIN 라벨행 {len(tr):,} ===")

        per_split = {}
        for sp in ("VAL", "OOS"):
            s = sub.loc[sub["split"] == sp].copy()
            probs = []
            for sd in SEEDS:
                m = HistGradientBoostingClassifier(random_state=sd, max_iter=300,
                                                   early_stopping=True, validation_fraction=0.15)
                m.fit(tr[cols], tr["label"].to_numpy())
                probs.append(m.predict_proba(s[cols])[:, 1])
            s["p"] = np.mean(probs, axis=0)
            # ⚠️피쳐셋마다 확률 분포가 이동하므로 임계값을 고정하면 안 된다 -- 고정하면 피쳐를
            # 늘린 쪽이 그냥 더 많이 발동해 포착률이 오르고, 그건 "더 잘 본다"가 아니라 "더
            # 자주 부른다"이다(첫 실행에서 F2가 호출 1.53배, 미포착 포착률도 정확히 1.53배로
            # 증가 = 순증분 0). 라벨 격자에서 이미 두 번 밟은 함정. F0의 호출수에 맞춘다.
            sl_ = lab.loc[lab["split"] == sp]
            auc = None
            if len(sl_) and sl_["label"].nunique() == 2:
                pk = s.set_index(["timestamp", "side"])
                key = [k for k in zip(sl_["timestamp"], sl_["side"]) if k in pk.index]
                if key:
                    auc = float(roc_auc_score(
                        sl_.set_index(["timestamp", "side"]).loc[key, "label"].to_numpy(),
                        pk.loc[key, "p"].to_numpy()))
            k_target = target_calls.get(sp)
            if k_target is None:
                k_target = int((s["p"] >= THRESHOLD).sum())
                target_calls[sp] = k_target      # F0가 기준을 세운다
            k_use = min(k_target, len(s))
            cutoff = float(np.partition(s["p"].to_numpy(), -k_use)[-k_use]) if k_use > 0 else 1.0
            called_mask = s["p"] >= cutoff
            pmap = {(t, sd_): (pv if cm else -1.0)
                    for t, sd_, pv, cm in zip(s["timestamp"], s["side"], s["p"], called_mask)}
            cov_c, unc_c = [], []
            for e in events.get(sp, []):
                ps = [pmap.get((b, e["side"])) for b in e["bars"]]
                ps = [x for x in ps if x is not None]
                if not ps:
                    continue
                (cov_c if e["covered"] else unc_c).append(max(ps) >= cutoff)
            cr = float(np.mean(cov_c)) if cov_c else float("nan")
            ur = float(np.mean(unc_c)) if unc_c else float("nan")
            per_split[sp] = {"auc": round(auc, 4) if auc else None,
                             "n_called_rows": int(called_mask.sum()),
                             "cutoff": round(cutoff, 4), "matched_target": int(k_use),
                             "capture_covered": round(cr, 4), "capture_uncovered": round(ur, 4),
                             "n_cov": len(cov_c), "n_unc": len(unc_c)}
            log(f"  {sp}: AUC {per_split[sp]['auc']}  호출행 {per_split[sp]['n_called_rows']:>5,}"
                f"(컷오프 {cutoff:.3f})  "
                f"포착률 포착 {cr*100:>5.1f}% / **미포착 {ur*100:>5.1f}%**")
        results[name] = {"n_features": len(cols), "splits": per_split}

    log("")
    log("=== 판정: 미포착 사건 포착률이 오르는가 (AUC 아님) ===")
    base = results["F0 Tier0 23"]["splits"]
    for name, r in results.items():
        if name.startswith("F0"):
            continue
        for sp in ("VAL", "OOS"):
            b, c = base[sp]["capture_uncovered"], r["splits"][sp]["capture_uncovered"]
            d = (c - b) * 100
            log(f"  {name:22s} {sp}: 미포착 {b*100:.1f}% -> {c*100:.1f}%  ({d:+.1f}%p)")
    log("")
    log("  기준: 미포착 포착률이 유의하게 오르면 피쳐 축이 지렛대. 제자리면 klines 파생 공간에")
    log("  그 정보가 없다는 뜻이고, 남는 길은 진짜 대체 데이터(L2/청산)를 1년 이상 모으는 것뿐.")

    report = {"signal": "v_rebound_154feature_uncovered_capture", "asset": "ETHUSDT",
              "scope": {"decision_metric": "미포착 사건 포착률 (AUC 아님)",
                        "threshold": THRESHOLD, "seeds": SEEDS, "model": "GBM",
                        "holdout_touched": False, "live_code_changed": False,
                        "audit": str(AUDIT_JSON.relative_to(ROOT))},
              "feature_sets": {k: v["n_features"] for k, v in results.items()},
              "micro_features": micro, "results": results,
              "runtime_sec": round(time.time() - t0, 1)}
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str))
    log("")
    log(f"report saved -> {OUT_JSON}  ({report['runtime_sec']}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
