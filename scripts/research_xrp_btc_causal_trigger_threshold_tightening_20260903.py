#!/usr/bin/env python3
"""XRP·BTC **인과적 첫발동 + 임계값 조이기** -- 기준을 낮추지 않고 신호 쪽을 개선한다.

## 왜 이 방향인가

사용자 지시: "경제성 테스트 기준은 낮추지 말고 기존 기준에서 신호를 개선해보자."

오늘까지 확인된 것:
  · 기존 경제성게이트 수치는 **앵커(`cluster_dedup`) 미래참조**로 10/10 무효.
  · 비용을 낮추는 건 답이 아니다(7bp에서 10종 중 1종, 그것도 VAL +0.74bp).
  · 지정가 진입은 XRP 2종만 플라시보 귀무를 통과했고 BTC는 전멸
    (귀무 평균이 48셀 중 1.7~5.8셀 -- "4셀 통과"는 우연 수준이었다).

⇒ 남은 축은 **발동 규칙 자체**다. 이 저장소에 기록된 유일한 미시도 방향:

> ⭐부수 결론: 생존 7셀을 뺀 대부분에서 **임계값-매칭 베이스가 모든 필터를 이긴다** --
> "더 적게 더 좋은 발동"이 목표면 답은 직교필터가 아니라 **베이스 임계값 조이기**.
> (`eth_composite_and_filter_rejected_pooling_artifact_20260902`)

XRP·BTC엔 시도된 적이 없다.

## 설계 -- 기준은 하나도 안 건드린다

**발동 규칙만 바꾼다:**

  1. **인과적 첫발동**: raw 임계 돌파 중 **직전 GAP봉 안에 같은 쪽 발동이 없는 봉**만 채택.
     그 봉에서 과거만 보면 판정된다 ⇒ **앵커 선택이 없다 = 미래참조가 없다.**
     (`cluster_dedup`은 "클러스터가 끝나야 최극단을 안다"라 미래참조였다.)
  2. **임계값 조이기**: 각 신호의 트리거 임계를 단계적으로 조인다. 발동이 줄고 질이 오르는지 본다.

**평가 기준은 기존 그대로:**
  · 왕복 **10bp**(수수료 우대 가정 없음), margin 0.30 x leverage 3.0
  · 96셀 그리드 SL[1.5~4.0] x ARM[0.5~2.0] x Trail[0.1~0.5]
  · **방향뒤집기 대조군 96셀 전량**
  · **ARM >= 1.0**(ARM<1.0은 노이즈 수확 아티팩트)
  · 판정: VAL>0 AND OOS>0 AND 정방향>뒤집기(양 구간)
  · `purged_decision_mask` + `simulate_single_position` (게이트와 동일 회계)

⚠️**표본 두께**를 같이 낸다. 임계를 조이면 발동이 줄어드는데, 오늘 반복 확인했듯 얇은 셀의
높은 수치는 믿을 수 없다. 후보 수 < 100이면 판정에서 제외한다.

⚠️HOLDOUT 미터치. VAL+OOS만.
"""
from __future__ import annotations

import json
import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT, ROOT / "scripts"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import numpy as np   # noqa: E402
import pandas as pd  # noqa: E402

from core.causal_futures_backtest import purged_decision_mask, simulate_single_position  # noqa: E402
from live_evidence_signal_dashboard_20260823 import compute_signals   # noqa: E402

OUT = ROOT / "data/research/xrp_btc_causal_threshold_tightening_20260903.json"

MARGIN_FRACTION, LEVERAGE, ROUNDTRIP_COST_RATE = 0.30, 3.0, 0.001   # ⭐10bp 유지
VAL_START = pd.Timestamp("2025-09-01")
OOS_START = pd.Timestamp("2026-01-01")
HOLDOUT_START = pd.Timestamp("2026-04-01")
SL_GRID = [1.5, 2.0, 2.5, 3.0, 3.5, 4.0]
ARM_GRID = [0.5, 1.0, 1.5, 2.0]
TRAIL_GRID = [0.1, 0.2, 0.3, 0.5]
MIN_CANDIDATES = 100          # 두께 하한 (VAL·OOS 각각)

ASSETS = {
    "XRP": {"klines": ROOT / "binance_data/klines/XRPUSDT/XRPUSDT-5m-api.csv",
            "partner": ROOT / "binance_data/klines/BTCUSDT/BTCUSDT-5m-api.csv",
            "funding": "research_xrp_regime_label_conditional_lift_20260903.py:load_xrp_funding_z"},
    "BTC": {"klines": ROOT / "binance_data/klines/BTCUSDT/BTCUSDT-5m-api.csv",
            "partner": ROOT / "binance_data/klines/ETHUSDT/ETHUSDT-5m-api.csv",
            "funding": "research_btc_regime_label_conditional_lift_20260902.py:load_btc_funding_z"},
}

# 신호별: 원자료 컬럼, 임계 스윕(느슨 -> 조임), GAP(첫발동 판정 창), H(보유한도)
# ⭐첫 값이 현행 배포 임계다 -- 그게 기준선(baseline)이 된다.
SIGNALS = {
    "demarker_extreme": {"kind": "bounded", "col": "dem",
                         "thr": [0.10, 0.07, 0.05, 0.03, 0.02], "gap": 6, "H": {"XRP": 2, "BTC": 8}},
    "short_term_return_z": {"kind": "z", "col": "ret3_z",
                            "thr": [2.5, 3.0, 3.5, 4.0, 4.5], "gap": 12, "H": {"XRP": 12, "BTC": 6}},
    "taker_delta_z_climax": {"kind": "z", "col": "delta_z",
                             "thr": [2.0, 2.5, 3.0, 3.5, 4.0], "gap": 3, "H": {"XRP": 9, "BTC": 6}},
    "kalman_deviation_meanrev": {"kind": "z", "col": "kalman_dev_z",
                                 "thr": [2.0, 2.5, 3.0, 3.5, 4.0], "gap": 6, "H": {"XRP": 5, "BTC": 10}},
}


def log(m): print(f"[tighten] {m}", flush=True)


def load_funding(spec: str):
    import importlib.util
    rel, fn = spec.split(":")
    sp = importlib.util.spec_from_file_location(f"f_{Path(rel).stem}", ROOT / "scripts" / rel)
    m = importlib.util.module_from_spec(sp); sp.loader.exec_module(m)
    out = getattr(m, fn)()
    for c in out.columns:
        if str(out[c].dtype).startswith("datetime64"):
            out[c] = out[c].astype("datetime64[ns]")
    return out


def first_of_cluster(fire: np.ndarray, gap: int) -> np.ndarray:
    """⭐**인과적**: 직전 gap봉 안에 같은 쪽 발동이 없으면 첫발동. 그 봉에서 과거만 보면 안다.
    (`cluster_dedup`은 클러스터가 끝나야 최극단을 알아 미래참조였다.)"""
    out = np.zeros(len(fire), dtype=bool)
    last = -10**9
    for i in np.flatnonzero(fire):
        if i - last > gap:
            out[i] = True
        last = i
    return out


def run_grid(kl, dec, is_long, atr, H):
    ts = kl["timestamp"]
    o, h, l, c = (kl[x].to_numpy() for x in ("open", "high", "low", "close"))
    masks = {}
    for wn, (s, e) in {"val": (VAL_START, OOS_START), "oos": (OOS_START, HOLDOUT_START)}.items():
        el = set(np.flatnonzero(purged_decision_mask(ts, start=s, end=e, horizon_bars=H)).tolist())
        masks[wn] = np.array([d in el for d in dec])
    tp = np.full(len(dec), 999.0)
    cells = []
    for sl in SL_GRID:
        for arm in ARM_GRID:
            for tr in TRAIL_GRID:
                row = {"sl": sl, "arm": arm, "trail": tr}
                for wn, m in masks.items():
                    for tag, sgn in (("fwd", 1.0), ("flip", -1.0)):
                        sc = np.where(is_long, 1.0, -1.0) * sgn
                        r = simulate_single_position(
                            timestamps=ts, open_px=o, high=h, low=l, close=c,
                            decision_indices=dec[m], scores=sc[m], tp_moves=tp[m],
                            sl_moves=(sl * atr)[m], upper_threshold=1.0, lower_threshold=-1.0,
                            horizon_bars=H, margin_fraction=MARGIN_FRACTION, leverage=LEVERAGE,
                            roundtrip_cost_rate=ROUNDTRIP_COST_RATE,
                            arm_moves=(arm * atr)[m], trail_moves=(tr * atr)[m])
                        led = r.ledger
                        row[f"{wn}_{tag}_bp"] = (float(led["trade_return"].mean() * 1e4)
                                                 if len(led) else float("nan"))
                        if tag == "fwd":
                            row[f"{wn}_n"] = int(len(led))
                cells.append(row)
    return cells, {k: int(v.sum()) for k, v in masks.items()}


def main() -> int:
    t0 = time.time()
    rep = {"cost_bp": 10.0, "grid": {"sl": SL_GRID, "arm": ARM_GRID, "trail": TRAIL_GRID},
           "min_candidates": MIN_CANDIDATES, "holdout_touched": False,
           "trigger_rule": "인과적 첫발동(직전 GAP봉 내 같은쪽 발동 없음) -- dedup/앵커 없음",
           "criteria_unchanged": True, "assets": {}}

    for asset, cfg in ASSETS.items():
        log(""); log("#" * 78); log(asset); log("#" * 78)
        raw = pd.read_csv(cfg["klines"], parse_dates=["timestamp"])
        raw = raw.sort_values("timestamp").drop_duplicates("timestamp", keep="last").reset_index(drop=True)
        partner = pd.read_csv(cfg["partner"], usecols=["timestamp", "high", "low"], parse_dates=["timestamp"])
        funding = load_funding(cfg["funding"])
        for d in (raw, partner):
            d["timestamp"] = d["timestamp"].astype("datetime64[ns]")
        frame = compute_signals(raw, btc_df=partner, funding_df=funding)
        frame["timestamp"] = frame["timestamp"].astype("datetime64[ns]")
        kl = frame[["timestamp", "open", "high", "low", "close"]].reset_index(drop=True)
        atr_all = frame["atr_pct"].to_numpy(float)
        log(f"프레임 {len(frame):,}봉  {frame['timestamp'].min()} ~ {frame['timestamp'].max()}")

        res = {}
        for sname, spec in SIGNALS.items():
            col = frame[spec["col"]].to_numpy(float)
            H = spec["H"][asset]
            log("")
            log(f"=== {sname}  (원자료 `{spec['col']}`, GAP={spec['gap']}, H={H}) ===")
            log(f"{'임계':>7}{'발동(b/t)':>14}{'후보 V/O':>13}{'진짜ARM>=1':>11}"
                f"{'최선 VAL':>10}{'최선 OOS':>10}  판정")
            rows = []
            for thr in spec["thr"]:
                if spec["kind"] == "bounded":
                    fb, ft = (col <= thr), (col >= 1.0 - thr)
                else:
                    fb, ft = (col <= -thr), (col >= thr)
                fb = np.nan_to_num(fb, nan=False); ft = np.nan_to_num(ft, nan=False)
                cb = first_of_cluster(fb, spec["gap"])
                ct = first_of_cluster(ft, spec["gap"])
                idx = np.flatnonzero(cb | ct)
                idx = idx[np.isfinite(atr_all[idx]) & (atr_all[idx] > 0) & (idx < len(kl) - 1)]
                if len(idx) < 50:
                    log(f"{thr:>7} 발동 부족({len(idx)}) -- 건너뜀"); continue
                is_long = cb[idx]
                cells, ns = run_grid(kl, idx, is_long, atr_all[idx], H)
                thin = ns["val"] < MIN_CANDIDATES or ns["oos"] < MIN_CANDIDATES
                gen = [c for c in cells
                       if c["val_fwd_bp"] > 0 and c["oos_fwd_bp"] > 0
                       and c["val_fwd_bp"] > c["val_flip_bp"] and c["oos_fwd_bp"] > c["oos_flip_bp"]
                       and c["arm"] >= 1.0]
                best = max(gen, key=lambda c: min(c["val_fwd_bp"], c["oos_fwd_bp"])) if gen else None
                mark = ("⚠️얇음" if thin else ("✅" if gen else "❌"))
                log(f"{thr:>7}{str(int(cb.sum())) + '/' + str(int(ct.sum())):>14}"
                    f"{str(ns['val']) + '/' + str(ns['oos']):>13}{len(gen):>11}"
                    f"{(best['val_fwd_bp'] if best else float('nan')):>+10.2f}"
                    f"{(best['oos_fwd_bp'] if best else float('nan')):>+10.2f}  {mark}")
                rows.append({"thr": thr, "n_fire_bottom": int(cb.sum()), "n_fire_top": int(ct.sum()),
                             "n_candidates": ns, "thin": bool(thin),
                             "n_genuine_arm1": len(gen), "best": best})
            res[sname] = {"gap": spec["gap"], "H": H, "col": spec["col"], "by_threshold": rows}
            base = rows[0] if rows else None
            good = [r for r in rows if r["n_genuine_arm1"] > 0 and not r["thin"]]
            if base is not None:
                log(f"  기준선(현행 임계 {base['thr']}): 진짜 {base['n_genuine_arm1']}셀"
                    f"{' (얇음)' if base['thin'] else ''}")
            log(f"  ⇒ 조여서 두께 유지하며 통과: {[r['thr'] for r in good] if good else '없음'}")
        rep["assets"][asset] = res

    log(""); log("=" * 82)
    log("종합 -- 임계값을 조이면 기존 기준에서 통과하는가 (10bp, 96셀, 뒤집기, ARM>=1.0)")
    log("=" * 82)
    log(f"{'자산':<5}{'신호':<26}{'현행':>7}{'최선임계':>9}{'통과셀':>8}{'VAL':>9}{'OOS':>9}  판정")
    tot = 0
    for asset, res in rep["assets"].items():
        for sname, v in res.items():
            good = [r for r in v["by_threshold"] if r["n_genuine_arm1"] > 0 and not r["thin"]]
            if not good:
                log(f"{asset:<5}{sname:<26}{v['by_threshold'][0]['thr'] if v['by_threshold'] else '-':>7}"
                    f"{'-':>9}{0:>8}{'-':>9}{'-':>9}  ❌")
                continue
            b = max(good, key=lambda r: min(r["best"]["val_fwd_bp"], r["best"]["oos_fwd_bp"]))
            tot += 1
            log(f"{asset:<5}{sname:<26}{v['by_threshold'][0]['thr']:>7}{b['thr']:>9}"
                f"{b['n_genuine_arm1']:>8}{b['best']['val_fwd_bp']:>+9.2f}"
                f"{b['best']['oos_fwd_bp']:>+9.2f}  ✅")
    log("")
    log(f"⇒ 기준 유지 + 임계 조이기로 통과: **{tot}종**")
    log("  ⚠️통과해도 다음 관문이 남는다: 무작위진입 귀무 / 순환이동 플라시보 / HOLDOUT 단일노출")
    rep["n_passed"] = tot
    rep["runtime_sec"] = round(time.time() - t0, 1)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(rep, ensure_ascii=False, indent=2, default=str))
    log(f"report -> {OUT}  ({rep['runtime_sec']}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
