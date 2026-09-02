#!/usr/bin/env python3
"""XRP 증거신호 5종 **TabPFN 메타라벨** — 그리드스크린이 고른 셀로 VAL/OOS/HOLDOUT 평가.

## 왜 신호마다 스크립트를 포팅하지 않고 통합했나

XRP의 확정 HIT_TYPE 중 **2종이 BTC와 계열이 다르다**:

    taker_delta_climax        BTC close_at_h  ->  XRP touch_giveback_sustained
    fib_extension_exhaustion  BTC close_at_h  ->  XRP touch_mfe

BTC 메타라벨 스크립트를 포팅하면 그 안의 hit 계산 코드를 **손으로 다른 계열로 고쳐야 한다**.
그게 정확히 "재구현" 함정이고, BTC에서 라이브 hit률이 2.6배 과대평가된 원인이었다.

대신 **그리드스크린이 실제로 쓴 hit 함수를 그대로 import**한다
(`research_btc_short_term_return_z_gridscreen_hittype_20260901.py`의 4개 구현 —
`hit_touch_mfe` / `hit_close_at_h` / `hit_touch_mae_capped` / `hit_touch_giveback_sustained`).
선정에 쓴 함수와 학습에 쓴 함수가 **같은 객체**이므로 어긋날 수가 없다.

피쳐 목록도 각 BTC 메타라벨 모듈의 `FEATURE_COLUMNS`를 그대로 import한다(신호별로 다르다).

## 확정 셀 (2026-09-03 그리드스크린 + 표본 두께 감사)

    신호                       HIT_TYPE                  H   K     GAP  해상봉
    taker_delta_climax        touch_giveback_sustained   9   1.5    3    18
    liquidity_sweep           touch_giveback_sustained  15   2.0    6    30
    short_term_return_z       touch_mae_capped          12   1.5   12    12
    orthogonal_combo          touch_mfe                  8   2.0    6     8
    fib_extension_exhaustion  touch_mfe                 10   1.5    6    10

⭐giveback 계열 2종은 **해상에 2xH 봉이 필요**하다. 서빙 코드가 H로 착각하면 절반 시점에
잘못 확정한다(BTC `liquidity_sweep`에서 실제 발생).

⚠️`orthogonal_combo`/`fib_extension_exhaustion`은 그리드스크린의 기계적 argmax
(`touch_giveback_sustained`, hit 79/34·39/33)를 **표본이 얇아 기각**하고 두꺼운 family를 택했다.
근거: `docs/experiments/xrp_evidence_signal_and_regime_20260903.md` 3단계.

⚠️**HOLDOUT은 이 실행에서 1회 노출된다.** 셀은 위 표로 이미 확정됐고 재조정하지 않는다.
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


def _mod(name: str, rel: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


# ⭐선정에 쓴 hit 함수를 그대로 쓴다(재구현 금지)
_HIT = _mod("xrp_hitfns", "scripts/research_btc_short_term_return_z_gridscreen_hittype_20260901.py")

CSV = ROOT / "data/labels/xrp_5m_evidence_signal_candidates_20260903/xrp_5m_evidence_signal_candidates_tier0.csv"
OUT_DIR = ROOT / "data/labels/xrp_5m_evidence_signal_candidates_20260903"

TRAIN_END = pd.Timestamp("2025-09-01")
VAL_END = pd.Timestamp("2026-01-01")
OOS_END = pd.Timestamp("2026-04-01")          # 이후는 HOLDOUT
SEED = 20260903

SPEC = {
    "taker_delta_z_climax":     {"sig": "taker_delta_z_climax",     "hit": "touch_giveback_sustained", "h": 9,  "k": 1.5, "gap": 3},
    "liquidity_sweep":          {"sig": "liquidity_sweep",          "hit": "touch_giveback_sustained", "h": 15, "k": 2.0, "gap": 6},
    "short_term_return_z":      {"sig": "short_term_return_z",      "hit": "touch_mae_capped",         "h": 12, "k": 1.5, "gap": 12},
    "orthogonal_combo":         {"sig": "orthogonal_combo",         "hit": "touch_mfe",                "h": 8,  "k": 2.0, "gap": 6},
    "fib_extension_exhaustion": {"sig": "fib_extension_exhaustion", "hit": "touch_mfe",                "h": 10, "k": 1.5, "gap": 6},
}
# ⚠️피쳐 파생 함수 이름이 모듈마다 다르다(BTC 동결 컨텍스트 빌더에서 이미 겪은 지점).
# 후보 CSV에는 원재료만 있고 nyse_open_flag/er_24/realized_vol_ratio/rsi/dem 등은
# 각 모듈의 파생 함수가 만든다 -- 재구현하지 않고 그 함수를 그대로 태운다.
PREP_FN = {
    "taker_delta_z_climax": "add_missing_features",
    "liquidity_sweep": "add_missing_features",
    "short_term_return_z": "add_derived_features",
    "orthogonal_combo": "add_missing_features",
    "fib_extension_exhaustion": "augment_features",
}
BTC_MODULE = {                                  # FEATURE_COLUMNS를 가져올 모듈(신호별로 다르다)
    "taker_delta_z_climax": "scripts/research_btc_taker_delta_climax_metalabel_tabpfn_20260901.py",
    "liquidity_sweep": "scripts/research_btc_liquidity_sweep_metalabel_tabpfn_20260901.py",
    "short_term_return_z": "scripts/research_btc_short_term_return_z_metalabel_tabpfn_20260901.py",
    "orthogonal_combo": "scripts/research_btc_orthogonal_combo_metalabel_tabpfn_20260901.py",
    "fib_extension_exhaustion": "scripts/research_btc_fib_extension_exhaustion_metalabel_tabpfn_20260901.py",
}


def log(m): print(f"[xrp-meta] {m}", flush=True)


def resolve_bars(spec: dict) -> int:
    """해상에 필요한 봉 수. giveback은 FULL_WINDOW(=2xH)를 본다."""
    return spec["h"] * _HIT.GIVEBACK_FULL_MULT if spec["hit"] == "touch_giveback_sustained" else spec["h"]


def build(df: pd.DataFrame, name: str, spec: dict, feats: list[str]) -> pd.DataFrame:
    """발동 -> 클러스터 디둡 -> hit 라벨 -> 피쳐 행. 방향-상대 피쳐는 여기서 유도한다."""
    n = len(df)
    high, low, close = (df[c].to_numpy(dtype=float) for c in ("high", "low", "close"))
    atr = df["atr"].to_numpy(dtype=float)
    ret3z = df["ret3_z"].to_numpy(dtype=float)
    dz = df["delta_z"].to_numpy(dtype=float)
    lvl_lo = df["sweep_level_low"].to_numpy(dtype=float)
    lvl_hi = df["sweep_level_high"].to_numpy(dtype=float)
    need = resolve_bars(spec)
    rows = []
    for side, col, most_neg in (("bottom", f"bottom_{spec['sig']}", True),
                                 ("top", f"top_{spec['sig']}", False)):
        raw = np.flatnonzero(df[col].fillna(False).to_numpy())
        raw = raw[(raw < n - need) & np.isfinite(atr[raw]) & np.isfinite(ret3z[raw])]
        idx = _HIT.cluster_dedup_gap(raw, ret3z[raw], most_negative=most_neg, gap=spec["gap"])
        if len(idx) == 0:
            continue
        hit = _HIT.compute_hit(spec["hit"], high, low, close, atr, idx, spec["h"], spec["k"], side)
        sub = df.iloc[idx].copy().reset_index(drop=True)
        sub["hit"] = hit.astype(float)
        sub["side"] = side
        sub["pos"] = idx
        # 방향-상대 피쳐 (후보 CSV는 원재료만 담는다 -- 빌더 docstring 참조)
        sub["is_bottom"] = 1 if side == "bottom" else 0
        pen = (lvl_lo[idx] - low[idx]) if side == "bottom" else (high[idx] - lvl_hi[idx])
        with np.errstate(invalid="ignore", divide="ignore"):
            sub["sweep_penetration_atr"] = np.where(np.isfinite(atr[idx]) & (atr[idx] > 0),
                                                    pen / atr[idx], np.nan)
        sub["flow_aligned_delta_z"] = dz[idx] if side == "bottom" else -dz[idx]
        sub["atr_pct"] = atr[idx] / np.where(close[idx] != 0, close[idx], np.nan)
        rows.append(sub)
    if not rows:
        return pd.DataFrame()
    out = pd.concat(rows, ignore_index=True)
    out["timestamp"] = pd.to_datetime(out["timestamp"])
    if out["timestamp"].dt.tz is not None:
        out["timestamp"] = out["timestamp"].dt.tz_localize(None)
    miss = [c for c in feats if c not in out.columns]
    if miss:
        raise KeyError(f"{name}: 피쳐 누락 {miss}")
    return out.dropna(subset=feats).sort_values("pos").reset_index(drop=True)


def main() -> int:
    t0 = time.time()
    from sklearn.metrics import roc_auc_score
    from tabpfn import TabPFNClassifier

    df = pd.read_csv(CSV)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    if df["timestamp"].dt.tz is not None:
        df["timestamp"] = df["timestamp"].dt.tz_localize(None)
    log(f"후보 CSV {len(df):,}행  {df.timestamp.min()} ~ {df.timestamp.max()}")
    log("⚠️HOLDOUT 1회 노출 -- 셀은 그리드스크린+두께감사로 이미 확정됨, 재조정 없음")

    rep = {"asset": "XRPUSDT", "seed": SEED, "holdout_start": str(OOS_END),
           "splits": {"train_end": str(TRAIN_END), "val_end": str(VAL_END), "oos_end": str(OOS_END)},
           "signals": {}}
    log("")
    log(f"{'신호':<26}{'HIT':<26}{'H':>3}{'K':>6}{'해상':>5}{'n':>7}{'hit률':>8}"
        f"{'VAL':>8}{'OOS':>8}{'HOLD':>8}")
    for name, spec in SPEC.items():
        mod = _mod(f"btc_{name}", BTC_MODULE[name])
        feats = list(mod.FEATURE_COLUMNS)
        try:
            fn = getattr(mod, PREP_FN[name], None)
            frame = fn(df.copy()) if fn is not None else df.copy()
            fires = build(frame, name, spec, feats)
        except Exception as e:                                    # noqa: BLE001
            log(f"{name:<26} ⚠️{type(e).__name__}: {str(e)[:60]}")
            rep["signals"][name] = {"error": f"{type(e).__name__}: {e}"}
            continue
        if fires.empty:
            log(f"{name:<26} ⚠️발동 0건"); continue
        sp = np.where(fires["timestamp"] < TRAIN_END, "TRAIN",
              np.where(fires["timestamp"] < VAL_END, "VAL",
               np.where(fires["timestamp"] < OOS_END, "OOS", "HOLDOUT")))
        fires["split"] = sp
        tr = fires[fires.split == "TRAIN"]
        aucs = {}
        if len(tr) >= 50 and tr["hit"].nunique() > 1:
            clf = TabPFNClassifier(device="cuda", random_state=SEED, ignore_pretraining_limits=True)
            clf.fit(tr[feats], tr["hit"].to_numpy().astype(int))
            for s_ in ("VAL", "OOS", "HOLDOUT"):
                g = fires[fires.split == s_]
                if len(g) >= 30 and g["hit"].nunique() > 1:
                    p = np.concatenate([clf.predict_proba(g[feats].iloc[k:k+20000])[:, 1]
                                        for k in range(0, len(g), 20000)])
                    aucs[s_] = round(float(roc_auc_score(g["hit"].astype(int), p)), 4)
                else:
                    aucs[s_] = None
        f_ = lambda v: f"{v:.4f}" if isinstance(v, float) else "-"
        log(f"{name:<26}{spec['hit']:<26}{spec['h']:>3}{spec['k']:>6}{resolve_bars(spec):>5}"
            f"{len(tr):>7}{tr['hit'].mean():>8.4f}"
            f"{f_(aucs.get('VAL')):>8}{f_(aucs.get('OOS')):>8}{f_(aucs.get('HOLDOUT')):>8}")
        rep["signals"][name] = {
            "hit_type": spec["hit"], "horizon": spec["h"], "k": spec["k"], "gap": spec["gap"],
            "resolve_bars": resolve_bars(spec), "n_features": len(feats), "features": feats,
            "n_by_split": {s_: int((fires.split == s_).sum()) for s_ in ("TRAIN", "VAL", "OOS", "HOLDOUT")},
            "train_hit_rate": round(float(tr["hit"].mean()), 4), "auc": aucs}
        fires.to_csv(OUT_DIR / f"{name}_xrp_fires_labeled.csv", index=False)
    rep["runtime_sec"] = round(time.time() - t0, 1)
    (OUT_DIR / "xrp_metalabel_report.json").write_text(json.dumps(rep, ensure_ascii=False, indent=2))
    log("")
    log(f"report -> {OUT_DIR/'xrp_metalabel_report.json'}  ({rep['runtime_sec']}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
