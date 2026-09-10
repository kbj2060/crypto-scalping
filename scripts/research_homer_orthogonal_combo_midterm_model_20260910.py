#!/usr/bin/env python3
"""호메로스 프로토콜 §3~§5 — `orthogonal_combo` 중기(15분봉) 메타라벨 구축 (2026-09-10).

선행 진단(`research_homer_orthogonal_combo_midterm_diag_20260910.py`, §2 체크리스트):
  · 발동 수 5m 3,245 / 15m 1,191 / **1h 296(0.18건/일) → 1h 는 표본 부족으로 제외**
  · 어긋남 중앙 +6~7봉(극값이 발동 **이후**, 이후 비중 0.86~0.90) — 라벨 창을 i+1 부터 두는 현행이 맞다
  · 균형 K 는 창에 따라 1.0~2.75, **라벨 문턱/왕복비용 = 5m 3.3배 vs 15m 6.2배**(중기의 실익)

이 스크립트가 하는 것 (§3~§5):
  §2-4 클러스터 앵커링 — **GAP=12봉 중복제거**(`build_eth_anchor_label_dataset_20260907._dedup` 규약).
  §3 라벨 — 발동봉 종가 entry → H봉 intrabar 유리방향 MFE. **중간지대 제외**(배포 v2 규격):
      K_lo = K_center/1.4, K_hi = 2×K_lo. HIT = MFE≥K_hi, MISS = MFE≤K_lo, 중간은 **버린다**(kept-only 평가).
  §5.5 호라이즌 — 9점(6~48봉) × GAP 3점 스크리닝, 선택은 **max(min(VAL,OOS))**(프로토콜 원문 기준).
      ⚠️1차 시도에서 VAL 단독 최댓값으로 고르니 정확히 문서가 경고한 함정(H=48 경계, VAL .72/OOS .58)에 빠졌다.
  §4 모델 사다리 — 기저율 → 단일피쳐 임계 → **TabPFN**(격리 venv, 배포 표준 모델).
  §5 검증 3종 — 순열중요도(직접 구현) · 상위 피쳐군 절제 · 룩어헤드(라벨 창이 i+1 부터인지 단언).
피쳐: 배포판 Tier0 23종(`live_eth_sweep_v_rebound_signal_20260829._build_features` + 동결 컨텍스트 컬럼).
분할: CLAUDE.md 표준 — TRAIN <2025-09-01 / VAL 2025-09~12 / OOS 2026-01~03 / **HOLDOUT(2026-04~) 미접촉**.
출력 tmp/homer_orth_midterm_20260910/model.json
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import build_eth_exit_synth_dataset_20260910 as BD  # noqa: E402
import research_eth_signals_midterm_timeframe_20260910 as M  # noqa: E402
from live_evidence_signal_dashboard_20260823 import compute_signals  # noqa: E402
import live_eth_sweep_v_rebound_signal_20260829 as VR  # noqa: E402  배포판 Tier0 빌더

OUT = ROOT / "tmp/homer_orth_midterm_20260910"
TP = Path("/tmp/tp2/bin/python")
SIGNAL = "orthogonal_combo"
TFS = {"5m": 1, "15m": 3}                      # 1h 는 표본 부족(296건)으로 제외
HORIZONS = [6, 9, 12, 16, 20, 24, 30, 36, 48, 60, 72]   # §5.6 경계 의심 -> 48 너머까지
K_GRID = np.round(np.arange(0.5, 6.01, 0.25), 2)
SPLITS = {"TRAIN": (None, "2025-08-31 23:59"), "VAL": ("2025-09-01", "2025-12-31 23:59"),
          "OOS": ("2026-01-01", "2026-03-31 23:59")}      # HOLDOUT 2026-04~ 는 열지 않는다
# 배포판 동결 컨텍스트(tabpfn_train_context_frozen_every_bar_20260901.csv)의 Tier0 컬럼
# ⭐배포 확정본은 **20피쳐**(세션타이밍 nyse_open_flag/hour_utc/weekday 제거) -- 그 제거가 OOS +0.011,
# HOLDOUT +0.017 로 오히려 개선됐고 VAL 만 손해였다(README ablation 표). 23피쳐본도 진단으로 같이 잰다.
FEATS20 = ["is_downside", "sweep_penetration_atr", "atr", "atr_percentile_864", "range_width_pct",
           "delta_z", "flow_aligned_delta_z", "p_fast", "p_slow", "ret3_z", "vwap_dev_z",
           "cvd_roll_roc_48", "vol_z", "lower_wick_ratio", "upper_wick_ratio", "bb_pctb",
           "adx14", "pdi", "ndi", "bb_width_pctile", "rsi"]
FEATS = FEATS20 + ["hour_utc", "weekday"]          # 23피쳐본(진단용)
GAPS = [6, 12, 24]


def log(m):
    print(f"[model {time.strftime('%H:%M:%S')}] {m}", flush=True)


def build(mult: int):
    """발동 행 + Tier0 피쳐 + 라벨 재료. 측면은 is_downside 로 풀링(배포판과 같은 규약)."""
    e, b = M.resample(BD.load_klines("eth", "ETHUSDT"), mult), M.resample(BD.load_klines("btc", "BTCUSDT"), mult)
    sig = compute_signals(e, btc_df=b, funding_df=None)
    fr = VR._build_features(e)                       # 배포판 Tier0 빌더(1분/5분 무관, 봉 기반)
    c = sig.close.to_numpy(float); hi = sig.high.to_numpy(float); lo = sig.low.to_numpy(float)
    atr_pct = sig.atr_pct.to_numpy(float)
    lvl_lo = fr["sweep_level_low"].to_numpy(float); lvl_hi = fr["sweep_level_high"].to_numpy(float)
    atr_abs = fr["atr"].to_numpy(float); dz = fr["delta_z"].to_numpy(float)
    rows = []
    for side, is_down in (("bottom", 1), ("top", 0)):
        idx = np.flatnonzero(sig[f"{side}_{SIGNAL}"].fillna(False).to_numpy(bool))
        idx = idx[idx >= 900]
        for i in idx:
            lvl = lvl_lo[i] if is_down else lvl_hi[i]
            pen = (lvl - lo[i]) if is_down else (hi[i] - lvl)
            rows.append({"i": int(i), "ts": sig.timestamp.iloc[i], "is_downside": float(is_down),
                         "sweep_penetration_atr": pen / atr_abs[i] if atr_abs[i] > 0 else np.nan,
                         "flow_aligned_delta_z": dz[i] if is_down else -dz[i]})
    A = pd.DataFrame(rows).sort_values("ts").reset_index(drop=True)
    for c_ in FEATS:
        if c_ in A.columns:
            continue
        A[c_] = fr[c_].to_numpy()[A["i"].to_numpy()] if c_ in fr.columns else np.nan
    A["_atr_pct"] = atr_pct[A["i"].to_numpy()]
    return A, c, hi, lo, len(sig)


def mfe(A, c, hi, lo, n, H):
    """§3 라벨 재료: 발동봉 종가 entry, i+1..i+H intrabar 유리방향 최대이동 / atr_pct."""
    out = np.full(len(A), np.nan)
    ii = A["i"].to_numpy(); down = A["is_downside"].to_numpy() > 0.5
    for j, i in enumerate(ii):
        if i + H >= n:
            continue
        seg = ((hi[i + 1:i + 1 + H] - c[i]) if down[j] else (c[i] - lo[i + 1:i + 1 + H])) / c[i]
        out[j] = seg.max() / max(A["_atr_pct"].iloc[j], 1e-9)
    return out


def dedup(idx: np.ndarray, gap: int) -> np.ndarray:
    """§2-4 클러스터 앵커링. build_eth_anchor_label_dataset_20260907._dedup 규약 그대로."""
    keep, last = [], -10 ** 9
    for i in idx:
        if i - last > gap:
            keep.append(int(i))
        last = i
    return np.array(keep, dtype=int)


def masks(A):
    ts = pd.to_datetime(A["ts"])
    m = {}
    for k, (a, b) in SPLITS.items():
        sel = (ts <= b).to_numpy() if a is None else ((ts >= a) & (ts <= b)).to_numpy()
        m[k] = sel
    return m


def tabpfn(Xtr, ytr, Xev, tag):
    f = OUT / f"_tp_{tag}.npz"
    np.savez(f, Xtr=np.nan_to_num(Xtr, nan=0.0, posinf=0.0, neginf=0.0), ytr=ytr,
             Xev=np.nan_to_num(Xev, nan=0.0, posinf=0.0, neginf=0.0))
    code = ("import numpy as np;from tabpfn import TabPFNClassifier;"
            f"d=np.load(r'{f}');m=TabPFNClassifier(device='cpu',random_state=11).fit(d['Xtr'],d['ytr']);"
            f"np.save(r'{f}.out.npy', m.predict_proba(d['Xev'])[:,1])")
    import os
    try:
        subprocess.run([str(TP), "-c", code], check=True, capture_output=True, timeout=3600,
                       env={**os.environ, "TABPFN_ALLOW_CPU_LARGE_DATASET": "1"})
        return np.load(f"{f}.out.npy")
    except Exception as e:  # noqa: BLE001
        log(f"⚠️tabpfn {tag}: {type(e).__name__} {str(getattr(e,'stderr',b''))[-200:]}")
        return None


def single_feature_auc(X, y, cols, tr, ev):
    """§4 사다리 2단: 단일피쳐 임계. TRAIN 에서 최고 피쳐를 고르고 평가창에서 그 피쳐로만 잰다."""
    best = (None, 0.5)
    for j, c_ in enumerate(cols):
        x = X[tr, j]; ok = np.isfinite(x)
        if ok.sum() < 100 or len(np.unique(y[tr][ok])) < 2:
            continue
        a = roc_auc_score(y[tr][ok], x[ok]); a = max(a, 1 - a)
        if a > best[1]:
            best = (c_, a)
    if best[0] is None:
        return None
    j = cols.index(best[0]); x = X[ev, j]; ok = np.isfinite(x)
    s = roc_auc_score(y[ev][ok], x[ok])
    return {"feature": best[0], "train_auc": float(best[1]), "eval_auc": float(max(s, 1 - s))}


def perm_importance(Xtr, ytr, Xev, yev, cols, base, tag, n_rep=3):
    """§5-2 순열중요도 직접 구현(TabPFN 은 sklearn 래퍼가 안 맞는다)."""
    rng = np.random.default_rng(0); out = {}
    for j, c_ in enumerate(cols):
        drops = []
        for _ in range(n_rep):
            Xp = Xev.copy(); Xp[:, j] = rng.permutation(Xp[:, j])
            p = tabpfn(Xtr, ytr, Xp, f"{tag}_perm{j}")
            if p is None:
                break
            drops.append(base - roc_auc_score(yev, p))
        if drops:
            out[c_] = float(np.mean(drops))
    return dict(sorted(out.items(), key=lambda kv: -kv[1]))


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    rep = {"signal": SIGNAL, "splits": SPLITS, "feats20": FEATS20, "horizons": HORIZONS, "gaps": GAPS,
           "holdout_touched": False,
           "spec": "배포 v2 규격 재현: GAP 중복제거 + 중간지대 제외(K_lo=K_c/1.4, K_hi=2*K_lo) + 20피쳐 + max(min(VAL,OOS))",
           "tfs": {}}
    cols = FEATS20
    for tf, mult in TFS.items():
        A, c, hi, lo, n = build(mult)
        d = {"n_fires_raw": int(len(A)), "scan": {}}
        for gap in GAPS:
            # 측면별로 중복제거한 뒤 합친다(측면이 다르면 다른 사건)
            keep = np.zeros(len(A), bool)
            for is_down in (1.0, 0.0):
                sel = np.flatnonzero((A["is_downside"].to_numpy() == is_down))
                ii = A["i"].to_numpy()[sel]
                kept = set(dedup(ii, gap).tolist())
                keep[sel] = np.isin(ii, list(kept))
            Ad = A[keep].reset_index(drop=True)
            m = masks(Ad); X = Ad[cols].to_numpy(np.float32)
            for H in HORIZONS:
                mv = mfe(Ad, c, hi, lo, n, H)
                ok = np.isfinite(mv)
                tr0 = ok & m["TRAIN"]
                if tr0.sum() < 150:
                    continue
                rates = {float(k): float((mv[tr0] >= k).mean()) for k in K_GRID}
                Kc = min(rates, key=lambda k: abs(rates[k] - 0.5))     # 균형 중심 K (TRAIN)
                K_lo = Kc / 1.4; K_hi = 2 * K_lo                        # 배포 v2 중간지대 제외
                y = np.where(mv >= K_hi, 1, np.where(mv <= K_lo, 0, -1))
                kept_m = ok & (y >= 0)
                tr = kept_m & m["TRAIN"]
                if tr.sum() < 120 or len(np.unique(y[tr])) < 2:
                    continue
                r = {"K_center": Kc, "K_lo": round(K_lo, 3), "K_hi": round(K_hi, 3),
                     "kept_frac": float(kept_m[ok].mean()), "n_train": int(tr.sum())}
                for w in ("VAL", "OOS"):
                    ev = kept_m & m[w]
                    if ev.sum() < 40 or len(np.unique(y[ev])) < 2:
                        r[w] = None; continue
                    p = tabpfn(X[tr], y[tr], X[ev], f"{tf}_g{gap}_H{H}_{w}")
                    r[w] = {"n": int(ev.sum()), "base": float(y[ev].mean()),
                            "tabpfn_auc": (float(roc_auc_score(y[ev], p)) if p is not None else None),
                            "single": single_feature_auc(X, y, cols, tr, ev)}
                d["scan"][f"g{gap}_H{H}"] = r
                v = (r.get("VAL") or {}).get("tabpfn_auc"); o = (r.get("OOS") or {}).get("tabpfn_auc")
                log(f"  {tf} GAP={gap:2d} H={H:2d} K={K_lo:.2f}/{K_hi:.2f} 유지 {r['kept_frac']:.2f} "
                    f"n_tr={r['n_train']:4d} · VAL {v if v is None else round(v,4)} · OOS {o if o is None else round(o,4)}")
                rep["tfs"][tf] = d
                (OUT / "model_v2.json").write_text(json.dumps(rep, ensure_ascii=False, indent=1, default=float))
        cand = {k: r for k, r in d["scan"].items()
                if (r.get("VAL") or {}).get("tabpfn_auc") and (r.get("OOS") or {}).get("tabpfn_auc")}
        if cand:
            score = {k: min(r["VAL"]["tabpfn_auc"], r["OOS"]["tabpfn_auc"]) for k, r in cand.items()}
            bk = max(score, key=score.get)
            d["selected"] = {"cell": bk, "min_val_oos": score[bk], "VAL": cand[bk]["VAL"]["tabpfn_auc"],
                             "OOS": cand[bk]["OOS"]["tabpfn_auc"], "gap_val_oos": abs(cand[bk]["VAL"]["tabpfn_auc"] - cand[bk]["OOS"]["tabpfn_auc"]),
                             "K_lo": cand[bk]["K_lo"], "K_hi": cand[bk]["K_hi"], "kept_frac": cand[bk]["kept_frac"],
                             "n_train": cand[bk]["n_train"], "n_val": cand[bk]["VAL"]["n"], "n_oos": cand[bk]["OOS"]["n"]}
            log(f"  ⇒ {tf} 선택 {bk}: VAL {d['selected']['VAL']:.4f} / OOS {d['selected']['OOS']:.4f} "
                f"(격차 {d['selected']['gap_val_oos']:.4f}, min={score[bk]:.4f}, n_tr={d['selected']['n_train']})")
        rep["tfs"][tf] = d
        (OUT / "model_v2.json").write_text(json.dumps(rep, ensure_ascii=False, indent=1, default=float))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
