#!/usr/bin/env python3
"""BTC 증거신호 7종 **1시간봉** 트레일링스톱 경제성게이트.

## 왜 이걸 하는가 (5분봉 게이트가 0/672로 끝난 뒤)

5분봉 실패의 원인은 신호 사멸이 아니라 **비용이 엣지의 2배**였다
(총이익 +2.2~4.9bp vs 비용 9.0bp). 그리고 진단에서 나온 두 값이 검증 가능한 예측을 만든다:

    비용 = 9bp로 **고정**
    총이익 = 포착효율 x ATR x sqrt(H)      (BTC 실측 포착효율 5.36%)

1시간봉 ATR은 66.0bp(실측)이므로 **보유 8시간이면 총이익 10.0bp > 비용 9bp**로 넘어선다.
손익분기는 6.5시간. 이 스크립트는 그 예측을 직접 검정한다.

⭐**투영이 맞다면 H에 대해 단조 증가하며 H~6~7에서 0을 통과해야 한다.**
단일 셀이 우연히 통과하는 것과 이 **패턴**은 전혀 다르다 -- 패턴이 근거고 단일 셀은 노이즈다.

## 5분봉 게이트와 무엇이 같고 무엇이 다른가

같음: 96셀 SL/ARM/Trail, 표준비용 10bp, notional 0.90, 방향뒤집기 전량 적용,
      ARM<1.0 별도집계, `purged_decision_mask`+`simulate_single_position`, HOLDOUT 미터치.
다름: (a) 봉이 1시간, (b) **보유 H를 스윕한다**(4/6/8/12/20시간).

## ⚠️1시간봉 트리거 재유도 -- 판단이 들어간 지점

`build_btc_5m_evidence_signal_candidates_tier0_20260901.py::main()`의 파이프라인을 1시간봉에
그대로 적용한다. 트리거는 z-score/percentile/오실레이터 임계 규칙이라 스케일에 상대적이므로
이식된다. **롤링 윈도는 봉 개수를 유지한다**(ret3_z 288, atr_percentile 864) --
시간 길이가 아니라 표본 수를 보존해야 z-score 분포 성질이 유지되고, 이미 커밋된
`research_btc_v_rebound_econ_1h_20260902.py::build_1h_frame()`도 같은 선택을 했다.
⚠️이건 판단이지 유일한 정답이 아니다. 시간 길이를 보존하는 대안도 있다.

⭐**라벨(K)은 재스크리닝하지 않는다** -- 경제성게이트는 hit 라벨을 쓰지 않고 발동 봉을 전부
트리거 방향으로 매매한다. 필요한 건 (timestamp, bar_idx, side, atr_pct)뿐이다.
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

_S = importlib.util.spec_from_file_location(
    "btcgate", ROOT / "scripts/gate_btc_evidence_signals_trailing_economics_20260902.py")
_g = importlib.util.module_from_spec(_S)
_S.loader.exec_module(_g)

OUT = ROOT / "data/research/btc_evidence_signals_costgate_1h_20260902/report.json"
BTC_5M = ROOT / "binance_data/klines/BTCUSDT/BTCUSDT-5m-api.csv"
HORIZONS = [4, 6, 8, 12, 20]                  # 시간 (1시간봉 개수)
CAPTURE_EFF = 0.0536                          # 5분봉 실측 포착효율 -- 투영 대조용


def log(m): print(f"[btc-1h-gate] {m}", flush=True)


def build_1h_tier0() -> tuple[pd.DataFrame, pd.DataFrame]:
    """5분봉 후보 빌더의 파이프라인을 1시간봉에 그대로 적용. (frame, klines_1h) 반환."""
    bspec = importlib.util.spec_from_file_location(
        "bcand", ROOT / "scripts/build_btc_5m_evidence_signal_candidates_tier0_20260901.py")
    b = importlib.util.module_from_spec(bspec); bspec.loader.exec_module(b)
    from analyze_eth_broad_evidence_signal_sweep_20260814 import add_broad_indicators
    from analyze_eth_creative_reversal_evidence_signals_20260814 import add_creative_indicators
    from backtest_eth_slowk_williamsr_persistence_confluence_20260814 import compute_indicators

    raw = pd.read_csv(BTC_5M, usecols=["timestamp", "open", "high", "low", "close",
                                       "volume", "taker_buy_base"])
    raw["timestamp"] = pd.to_datetime(raw["timestamp"], utc=True)
    raw = (raw.dropna().sort_values("timestamp").drop_duplicates("timestamp", keep="last")
           .loc[lambda d: d["timestamp"] >= b.START].reset_index(drop=True))
    agg = {"open": "first", "high": "max", "low": "min", "close": "last",
           "volume": "sum", "taker_buy_base": "sum"}
    h = (raw.set_index("timestamp").resample("1h").agg(agg)
         .dropna(subset=["open", "high", "low", "close"]).reset_index())
    # resample이 datetime64[ns,UTC]를 내는데 funding CSV는 [us,UTC]라 merge가 터진다 -- 원본 dtype로 되돌린다
    h["timestamp"] = h["timestamp"].astype(raw["timestamp"].dtype)
    log(f"1시간봉 {len(h):,}행 ({h.timestamp.min()} ~ {h.timestamp.max()})")

    frame = add_broad_indicators(add_creative_indicators(compute_indicators(h)))
    ret3 = frame["close"] / frame["close"].shift(3) - 1.0
    frame["ret3_z"] = ((ret3 - ret3.rolling(288, min_periods=288).mean())
                       / ret3.rolling(288, min_periods=288).std().replace(0.0, np.nan))
    causal = b.load_sweep_impl().add_causal_columns(
        h[["timestamp", "open", "high", "low", "close"]].copy())
    for c in ("sweep_level_low", "sweep_level_high", "atr"):
        frame[c] = causal[c]
    frame["atr_percentile_864"] = frame["atr"].rolling(864, min_periods=864).rank(pct=True)
    frame["range_width_pct"] = (frame["sweep_level_high"] - frame["sweep_level_low"]) / frame["close"]
    frame["hour_utc"] = frame["timestamp"].dt.hour
    frame["weekday"] = frame["timestamp"].dt.weekday
    frame["rsi"] = b.rsi_wilder(frame["close"])

    # ⚠️funding CSV는 `.dt.as_unit("us")`인데 resample 결과는 [ns] -- merge_asof가 dtype 일치를 요구한다
    fund = b.load_btc_funding()
    fund["calc_time"] = fund["calc_time"].astype(h["timestamp"].dtype)
    log(f"  dtype 정렬: bars {h['timestamp'].dtype} / funding {fund['calc_time'].dtype}")
    sig = b.compute_signals(h, btc_df=None, funding_df=fund)
    for name in b.NAMED_TRIGGERS:
        frame[f"bottom_{name}"] = sig[f"bottom_{name}"].fillna(False).to_numpy()
        frame[f"top_{name}"] = sig[f"top_{name}"].fillna(False).to_numpy()
    ll, lh = b.local_extreme_flags(frame["low"].to_numpy(), frame["high"].to_numpy(),
                                  b.LOCAL_EXTREME_W)
    frame["bottom_local_extreme"], frame["top_local_extreme"] = ll, lh
    dn = up = np.zeros(len(frame), dtype=bool)
    for name in b.NAMED_TRIGGERS + ["local_extreme"]:
        dn = dn | frame[f"bottom_{name}"].to_numpy()
        up = up | frame[f"top_{name}"].to_numpy()
    frame["any_bottom_trigger"], frame["any_top_trigger"] = dn, up
    frame["timestamp"] = frame["timestamp"].dt.tz_localize(None)
    nfire = {n: int(frame[f"bottom_{n}"].sum() + frame[f"top_{n}"].sum())
             for n in b.NAMED_TRIGGERS + ["local_extreme"]}
    log(f"  트리거 발동(양측 합): {nfire}")
    h["timestamp"] = h["timestamp"].dt.tz_localize(None)
    return frame, h


def fires_1h(frame, rel, builder, prep, kind):
    """신호별 빌더를 1h 프레임에 적용. prep[0](로더)는 건너뛴다 -- 5분 간격을 assert하므로."""
    mod = _g.load_mod(rel)
    f = frame.copy()
    for pname in prep[1:]:
        fnp = getattr(mod, pname, None)
        if fnp is not None:
            f = fnp(f)
    fn = getattr(mod, builder)
    if kind == "demarker":
        g = _g.GRID_CHOSEN["demarker"]
        out = fn(f, g["horizon"], g["k"], mod.CLUSTER_GAP)
    elif kind == "kalman":
        g = _g.GRID_CHOSEN["kalman"]
        f["kalman_dev_z"] = mod.compute_kalman_dev_z(f["close"].to_numpy())
        bt = (f["kalman_dev_z"] <= -2.0).fillna(False).to_numpy()
        tt = (f["kalman_dev_z"] >= 2.0).fillna(False).to_numpy()
        out = fn(f, bt, tt, g["horizon"], g["k"], mod.CLUSTER_GAP)
    else:
        out = fn(f)
    return out[0] if isinstance(out, tuple) else out


def main() -> int:
    t0 = time.time()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    frame, kl = build_1h_tier0()
    log(f"그리드 96셀 x 보유 {HORIZONS} x 정방향/뒤집기 x VAL/OOS")
    log(f"투영: 총이익 = {CAPTURE_EFF*100:.2f}% x ATR x sqrt(H), 비용 9bp -> 손익분기 H~6.5")
    log("⚠️HOLDOUT 미터치")

    rep = {"asset": "BTCUSDT", "bar": "1h", "cost_bp": 10.0, "horizons": HORIZONS,
           "capture_eff_5m": CAPTURE_EFF, "holdout_touched": False,
           "rolling_window_convention": "bar-count preserved (288/864), not time-span",
           "signals": {}}
    any_pass = []
    for name, rel, builder, prep, kind in _g.SIGNALS:
        log("")
        log(f"=== {name} ===")
        try:
            fr = fires_1h(frame, rel, builder, prep, kind)
        except Exception as e:                                  # noqa: BLE001
            log(f"  ⚠️fires 실패: {type(e).__name__}: {e}")
            rep["signals"][name] = {"error": f"{type(e).__name__}: {e}"}
            continue
        fr["timestamp"] = pd.to_datetime(fr["timestamp"])
        if fr["timestamp"].dt.tz is not None:
            fr["timestamp"] = fr["timestamp"].dt.tz_localize(None)
        fr = fr.loc[fr["timestamp"] < _g.HOLDOUT_START].reset_index(drop=True)
        atr_bp = float(np.median(fr.loc[(fr["timestamp"] >= _g.VAL_START), "atr_pct"]) * 1e4)
        log(f"  fires {len(fr):,}건  진입시점 ATR 중앙 {atr_bp:.1f}bp")

        per_h = {}
        for H in HORIZONS:
            _g.ROUNDTRIP_COST_RATE = 0.001
            cells, ns = _g.run_grid(kl, fr, H)
            _g.ROUNDTRIP_COST_RATE = 0.0
            gcells, _ = _g.run_grid(kl, fr, H)
            _g.ROUNDTRIP_COST_RATE = 0.001

            passing = [c for c in cells if c["val_fwd_bp"] > 0 and c["oos_fwd_bp"] > 0]
            genuine = [c for c in passing if c["val_fwd_bp"] > c["val_flip_bp"]
                       and c["oos_fwd_bp"] > c["oos_flip_bp"]]
            g1 = [c for c in genuine if c["arm"] >= 1.0]
            gb = max(gcells, key=lambda c: min(c["val_fwd_bp"], c["oos_fwd_bp"]))
            gross = (gb["val_fwd_bp"] + gb["oos_fwd_bp"]) / 2
            proj = CAPTURE_EFF * atr_bp * np.sqrt(H)
            nb = max(cells, key=lambda c: min(c["val_fwd_bp"], c["oos_fwd_bp"]))
            log(f"  H={H:>2}시간  후보 {ns['val']}/{ns['oos']}  "
                f"총이익 {gross:>6.2f}bp (투영 {proj:>5.2f})  "
                f"순최선 VAL {nb['val_fwd_bp']:>7.2f} OOS {nb['oos_fwd_bp']:>7.2f}  "
                f"통과 {len(passing):>2}/96 진짜 {len(genuine):>2} ARM>=1 **{len(g1)}**")
            per_h[H] = {"n_val": ns["val"], "n_oos": ns["oos"],
                        "gross_bp": round(gross, 2), "projected_gross_bp": round(float(proj), 2),
                        "net_best_val": nb["val_fwd_bp"], "net_best_oos": nb["oos_fwd_bp"],
                        "n_passing": len(passing), "n_genuine": len(genuine),
                        "n_genuine_arm_ge_1": len(g1), "genuine_arm_ge_1": g1[:12]}
            if g1:
                any_pass.append((name, H, len(g1)))
        rep["signals"][name] = {"atr_bp": round(atr_bp, 1), "n_fires": int(len(fr)),
                                "per_horizon": per_h}

    log("")
    log("=== 종합 ===")
    for k, v in rep["signals"].items():
        if "error" in v:
            log(f"  {k:<26} ⚠️{v['error'][:44]}"); continue
        s = "  ".join(f"H{H}:{v['per_horizon'][H]['n_genuine_arm_ge_1']}" for H in HORIZONS)
        log(f"  {k:<26} ARM>=1.0 진짜  {s}")
    log("")
    log(f"  ⇒ 통과 (신호, H, 셀수): {any_pass if any_pass else '없음'}")
    rep["passed"] = [{"signal": a, "H": b, "n_cells": c} for a, b, c in any_pass]
    rep["runtime_sec"] = round(time.time() - t0, 1)
    OUT.write_text(json.dumps(rep, ensure_ascii=False, indent=2, default=str))
    log(f"report -> {OUT}  ({rep['runtime_sec']}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
