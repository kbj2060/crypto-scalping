#!/usr/bin/env python3
"""증거신호 8종 -- ATR 정규화 라벨의 저변동성 결함 진단 (docs/homer/README.md 5.9)절 1번 항목).

## 배경

2026-09-01 V_REBOUND에서 발견된 결함: 이 저장소의 라벨은 거의 전부 ATR 정규화(K x ATR 터치)를
쓰는데, **저변동성 구간에서는 문턱 자체가 거래비용 아래로 내려가** 라벨이 "성공"이라 말하는
움직임이 실제로는 왕복 수수료도 못 건진다. V_REBOUND 실측: 죽은시장(ATR<10bp) 사건의 반등폭
중앙값 18~19bp = 왕복비용 10bp의 2배 미만, 배포모델 학습데이터에도 4.0% 포함.

**8개 증거신호도 전부 K x ATR 터치 라벨이라 구조적으로 같은 결함을 가질 수 있다** -- 사용자
지시로 이 진단을 실행한다.

## 진단 지표

라벨이 hit(1)이 되려면 가격이 최소 `K x ATR`만큼 움직여야 하므로, **최소 자격 움직임(bp)
= K x atr_pct x 10000**이 이 신호의 "그 시점 라벨이 요구하는 최소 수익폭"이다. 이게 왕복비용
(10bp)보다 작으면 그 양성은 경제적으로 무의미하다.

  - hit_threshold_bp 중앙값 (전체 / ATR 하위 10% 구간)
  - 왕복비용 대비 배수 -- **5.9)절 판정선: 저변동성 구간에서 2배 미만이면 결함 보유**
  - 비용 미만(<10bp) / 2배 미만(<20bp) 양성의 비율

## 데이터 출처 (중요)

6개 신호는 **실제 배포된 frozen train context CSV**를 그대로 읽는다 -- 배포 모델이 문자 그대로
학습한 데이터이므로 진단 대상으로 가장 정확하다. `demarker_extreme`/`kalman_deviation_meanrev`
2개는 서버에서 빌드돼 로컬에 없어, compute_signals()+causal ATR로 **로컬 재구성**하고 리포트에
`source="local_reconstruction"`으로 명시한다(서버 원본과 미세한 차이 가능).

⚠️ 진단 전용: 라벨/라이브 코드 변경 없음, 모델 재학습 없음, OOS/HOLDOUT 무관(배포 학습데이터
자체를 보는 것). 하한 도입 여부는 이 진단이 아니라 **경제성 게이트**로 판단해야 한다(5.9)절 2번 --
V_REBOUND에서 분류 +0.021이 경제성으로 전이되지 않은 전례).

Run with the quant_ai conda env:
  ~/anaconda3/envs/quant_ai/bin/python3 scripts/research_eth_evidence_signal_atr_label_bp_floor_diagnostic_20260901.py
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
for _p in (ROOT, ROOT / "scripts"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from live_evidence_signal_metalabel_20260829 import METALABEL_SIGNALS  # noqa: E402
from live_evidence_signal_dashboard_20260823 import compute_signals  # noqa: E402

VARIANT_SCRIPT = ROOT / "scripts/research_eth_v_rebound_label_redesign_variant_screen_20260901.py"
_vspec = importlib.util.spec_from_file_location("label_variants_evdiag_20260901", VARIANT_SCRIPT)
_vs = importlib.util.module_from_spec(_vspec)
_vspec.loader.exec_module(_vs)

OUT_JSON = ROOT / "data/labels/evidence_signal_atr_bp_floor_diagnostic_20260901.json"
COST_BP = 10.0
LOW_ATR_PCTILE = 0.10
DEFECT_RATIO = 2.0  # 5.9)절 판정선: 저변동성 구간 배수가 이 미만이면 결함 보유

# local reconstruction thresholds for the two signals whose train contexts live on the server
RECON = {
    "demarker_extreme": ("dem", 0.10, "le"),           # bottom: dem<=0.10, top: dem>=0.90
    "kalman_deviation_meanrev": ("kalman_dev_z", 2.0, "z"),  # bottom: z<=-2.0, top: z>=+2.0
}


def log(msg: str) -> None:
    print(f"[evsig_bp_diag] {msg}", flush=True)


def summarise(atr_pct: np.ndarray, atr_rank: np.ndarray, k: float, name: str, source: str,
              n_rows: int, hit_rate: float | None) -> dict:
    thr_bp = k * atr_pct * 1e4
    ok = np.isfinite(thr_bp)
    thr_bp = thr_bp[ok]
    atr_rank = atr_rank[ok]
    low = np.isfinite(atr_rank) & (atr_rank <= LOW_ATR_PCTILE)

    med_all = float(np.median(thr_bp))
    med_low = float(np.median(thr_bp[low])) if low.sum() else None
    out = {
        "source": source, "K": k, "n_rows": int(n_rows), "hit_rate": hit_rate,
        "n_scored": int(len(thr_bp)),
        "hit_threshold_bp": {
            "median_all": round(med_all, 1),
            "p10_all": round(float(np.percentile(thr_bp, 10)), 1),
            "median_low_atr_decile": round(med_low, 1) if med_low is not None else None,
            "n_low_atr_decile": int(low.sum()),
        },
        "vs_cost": {
            "median_all_x_cost": round(med_all / COST_BP, 2),
            "median_low_atr_x_cost": round(med_low / COST_BP, 2) if med_low is not None else None,
            "pct_below_cost_10bp": round(float((thr_bp < COST_BP).mean()) * 100, 2),
            "pct_below_2x_cost_20bp": round(float((thr_bp < 2 * COST_BP).mean()) * 100, 2),
        },
    }
    ratio = out["vs_cost"]["median_low_atr_x_cost"]
    out["verdict_defect_present"] = bool(ratio is not None and ratio < DEFECT_RATIO)
    log(f"  {name:28s} K={k:<6.3f} [{source[:12]:12s}] n={len(thr_bp):5d}  "
        f"문턱bp 중앙값 전체={med_all:6.1f} 저ATR10%={med_low if med_low is None else round(med_low,1)}  "
        f"| 비용대비 전체={out['vs_cost']['median_all_x_cost']:.2f}x 저ATR={ratio}x  "
        f"| <10bp {out['vs_cost']['pct_below_cost_10bp']:.1f}% <20bp {out['vs_cost']['pct_below_2x_cost_20bp']:.1f}%  "
        f"=> {'⚠️결함' if out['verdict_defect_present'] else 'OK'}")
    return out


def reconstruct(sig: pd.DataFrame, name: str) -> tuple[np.ndarray, np.ndarray, int]:
    col, th, mode = RECON[name]
    v = sig[col].to_numpy(dtype=float)
    if mode == "le":
        fire = (v <= th) | (v >= 1.0 - th)
    else:
        fire = (v <= -th) | (v >= th)
    fire &= np.isfinite(v)
    atr_pct = (sig["atr"].to_numpy() / sig["close"].to_numpy())
    rank = pd.Series(sig["atr"].to_numpy() / sig["close"].to_numpy()).rolling(864, min_periods=864).rank(pct=True).to_numpy()
    return atr_pct[fire], rank[fire], int(fire.sum())


def main() -> int:
    t0 = time.time()
    results = {}
    need_recon = []

    log("=== 배포 frozen train context 기반 진단 (6개 신호) ===")
    for name, cfg in METALABEL_SIGNALS.items():
        p = Path(cfg["train_context"])
        k = float(cfg["k"])
        if not p.exists():
            need_recon.append(name)
            continue
        df = pd.read_csv(p)
        if "atr_pct" not in df.columns:
            log(f"  {name}: atr_pct 컬럼 없음 -- 스킵")
            continue
        hit_rate = float(df["hit"].mean()) if "hit" in df.columns else None
        rank = df["atr_percentile_864"].to_numpy() if "atr_percentile_864" in df.columns else np.full(len(df), np.nan)
        results[name] = summarise(df["atr_pct"].to_numpy(dtype=float), rank, k, name,
                                   "deployed_train_context", len(df), hit_rate)
        # realized move where available (only some signals carry move_atr_mult)
        if "move_atr_mult" in df.columns and "hit" in df.columns:
            hits = df.loc[df["hit"] == 1]
            if len(hits):
                mv = (hits["move_atr_mult"].to_numpy(dtype=float) * hits["atr_pct"].to_numpy(dtype=float) * 1e4)
                mv = mv[np.isfinite(mv)]
                results[name]["realized_move_bp_of_hits"] = {
                    "median": round(float(np.median(mv)), 1),
                    "p10": round(float(np.percentile(mv, 10)), 1),
                    "pct_below_2x_cost": round(float((mv < 2 * COST_BP).mean()) * 100, 2),
                }

    if need_recon:
        log(f"=== 로컬 재구성 기반 진단 ({len(need_recon)}개: {need_recon}) ===")
        eth = _vs.load_klines(ROOT / "binance_data/klines/ETHUSDT/ETHUSDT-5m-api.csv")
        btc = _vs.load_klines(ROOT / "binance_data/klines/BTCUSDT/BTCUSDT-5m-api.csv")
        impl = _vs.load_impl()
        causal = impl.add_causal_columns(eth[["timestamp", "open", "high", "low", "close"]].copy())
        sig = compute_signals(eth, btc_df=btc, funding_df=None)
        sig["atr"] = causal["atr"].to_numpy()
        for name in need_recon:
            if name not in RECON:
                log(f"  {name}: 재구성 규칙 미정의 -- 스킵")
                continue
            atr_pct, rank, n = reconstruct(sig, name)
            results[name] = summarise(atr_pct, rank, float(METALABEL_SIGNALS[name]["k"]), name,
                                       "local_reconstruction", n, None)

    defects = [n for n, r in results.items() if r.get("verdict_defect_present")]
    log("")
    log(f"=== 판정 요약 (저ATR 10% 구간에서 최소자격움직임이 왕복비용의 {DEFECT_RATIO}배 미만) ===")
    log(f"  결함 보유: {len(defects)}/{len(results)}개 -> {defects}")
    log(f"  정상      : {[n for n in results if n not in defects]}")

    report = {
        "diagnostic": "evidence_signal_atr_normalized_label_low_volatility_bp_floor",
        "asset": "ETHUSDT", "cost_bp": COST_BP, "low_atr_pctile": LOW_ATR_PCTILE,
        "defect_ratio_threshold": DEFECT_RATIO,
        "methodology_doc": "docs/homer/README.md 5.9)절",
        "scope": {
            "diagnostic_only": True, "label_changed": False, "model_retrained": False,
            "live_code_changed": False,
            "note": ("hit_threshold_bp = K * atr_pct * 1e4 = the minimum move the label demands at "
                     "that bar. Adoption of any floor must be decided by the ECONOMICS gate, not "
                     "this diagnostic nor classification AUC (5.9 item 2)."),
        },
        "results": results,
        "defect_signals": defects,
        "runtime_sec": round(time.time() - t0, 1),
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str))
    log(f"report saved -> {OUT_JSON}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
