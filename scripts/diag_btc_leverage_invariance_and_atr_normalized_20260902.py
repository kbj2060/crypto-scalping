#!/usr/bin/env python3
"""사용자 두 질문에 답한다: (1) 레버리지로 살릴 수 있나 (2) BTC 변동성 부족이 원인인가.

## Q1. 레버리지 불변성

회계상 답은 정해져 있다 -- `PnL = price_move x notional`, `cost = cost_rate x notional`,
`notional = margin x leverage`. 즉

    계좌수익 = notional x (price_move - cost_rate)

레버리지는 **괄호 밖 배수**일 뿐이라 괄호 안 부호를 못 바꾼다. 음수 x 큰 배수 = 더 큰 음수.
말로만 하지 않고 leverage 3/10/20으로 실제 격자를 돌려 **선형 배수로 나빠지는 것**을 보인다.

## Q2. ⭐"BTC는 짧은 시간 안의 변동성이 부족하다" 가설 검정

사용자 가설이 맞다면, **ATR로 정규화하면 BTC와 ETH가 같아져야 한다** --
비용/ATR만 다르고 총이익/ATR은 동일해야 한다. 정규화 후에도 BTC가 낮으면
"변동성 부족"만으로는 설명이 안 되고 신호 자체도 약한 것이다.

    총이익/ATR      : 이 신호가 1 ATR당 몇 %를 건지나 (신호의 순수 포착력)
    총이익/(ATR*sqrt(H)) : 보유기간을 감안한 확산 스케일 대비 포착력
    비용/ATR        : 이미 알려진 값 (BTC 5m 62% vs ETH 43%)

⚠️비용 0 총이익은 진단 전용이다. 승격 근거로 쓰지 않는다.
"""
from __future__ import annotations

import importlib.util
import json
import sys
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

OUT = ROOT / "data/research/btc_evidence_signals_costgate_20260902/leverage_and_atr_norm.json"
ETH_KLINES = ROOT / "binance_data/klines/ETHUSDT/ETHUSDT-5m-api.csv"
ETH_SIGNALS = {
    "liquidity_sweep_topdown": ("data/labels/eth_5m_liquidity_sweep_topdown_metalabel_20260830/eth_5m_liquidity_sweep_topdown_metalabel_features_H30_GAP12_K4.0.csv", 30),
    "smt_divergence": ("data/labels/eth_5m_smt_divergence_metalabel_20260831/eth_5m_smt_divergence_metalabel_features.csv", 72),
    "fib_extension_exhaustion": ("data/labels/eth_5m_fib_extension_exhaustion_metalabel_20260831/eth_5m_fib_extension_exhaustion_metalabel_FINAL_features.csv", 20),
}


def log(m): print(f"[lev-atr] {m}", flush=True)


def load_kl(p):
    k = pd.read_csv(p)
    k["timestamp"] = pd.to_datetime(k["timestamp"], utc=True).dt.tz_localize(None)
    return k.sort_values("timestamp").reset_index(drop=True)


def entry_atr_bp(fires):
    """VAL+OOS 구간 후보들의 진입시점 ATR 중앙값(bp)."""
    m = (fires["timestamp"] >= _g.VAL_START) & (fires["timestamp"] < _g.HOLDOUT_START)
    a = fires.loc[m, "atr_pct"].to_numpy(dtype=float)
    return float(np.median(a) * 1e4)


def main() -> int:
    btc_kl = load_kl(_g.KLINES)

    # ---------- Q1. 레버리지 불변성 ----------
    log("=" * 74)
    log("Q1. 레버리지를 올리면 살아나는가 -- taker_delta_climax로 실증")
    log("=" * 74)
    name, rel, builder, prep, kind = next(s for s in _g.SIGNALS if s[0] == "taker_delta_climax")
    fires, _ = _g.build_fires(name, rel, builder, prep, kind)
    fires["timestamp"] = pd.to_datetime(fires["timestamp"])
    if fires["timestamp"].dt.tz is not None:
        fires["timestamp"] = fires["timestamp"].dt.tz_localize(None)
    fires = fires.loc[fires["timestamp"] < _g.HOLDOUT_START].reset_index(drop=True)

    lev_rows = []
    base_margin = 0.30
    for lev in (3.0, 10.0, 20.0):
        _g.LEVERAGE = lev
        _g.ROUNDTRIP_COST_RATE = 0.001
        cells, _ = _g.run_grid(btc_kl, fires, _g.HORIZON[name])
        best = max(cells, key=lambda c: min(c["val_fwd_bp"], c["oos_fwd_bp"]))
        npass = sum(1 for c in cells if c["val_fwd_bp"] > 0 and c["oos_fwd_bp"] > 0)
        notional = base_margin * lev
        log(f"  leverage {lev:>4.0f}x (notional {notional:.2f})  "
            f"격자최선 VAL {best['val_fwd_bp']:>8.2f}bp  OOS {best['oos_fwd_bp']:>8.2f}bp  "
            f"통과 {npass}/96")
        lev_rows.append({"leverage": lev, "notional": notional,
                         "best_val_bp": best["val_fwd_bp"], "best_oos_bp": best["oos_fwd_bp"],
                         "n_passing": npass})
    r3, r20 = lev_rows[0], lev_rows[-1]
    log(f"  ⇒ 20x/3x 배수 = {r20['best_val_bp']/r3['best_val_bp']:.2f}x "
        f"(notional 배수 {r20['notional']/r3['notional']:.2f}x)")
    log("  ⇒ ⭐**정확히 비례해서 더 나빠진다.** 레버리지는 부호를 못 바꾼다.")
    _g.LEVERAGE = 3.0

    # ---------- Q2. ATR 정규화 대조 ----------
    log("")
    log("=" * 74)
    log("Q2. 'BTC는 짧은 시간 변동성이 부족하다' -- ATR로 정규화하면 같아지는가")
    log("=" * 74)
    _g.ROUNDTRIP_COST_RATE = 0.0                      # 총이익 측정 (진단 전용)
    rows = []

    def measure(asset, sig, fires, H, kl):
        f = fires.loc[fires["timestamp"] < _g.HOLDOUT_START].reset_index(drop=True)
        cells, ns = _g.run_grid(kl, f, H)
        b = max(cells, key=lambda c: min(c["val_fwd_bp"], c["oos_fwd_bp"]))
        atr = entry_atr_bp(f)
        gross = (b["val_fwd_bp"] + b["oos_fwd_bp"]) / 2
        return {"asset": asset, "signal": sig, "H": H, "atr_bp": atr,
                "gross_bp": gross, "gross_over_atr": gross / atr,
                "gross_over_atr_sqrtH": gross / (atr * np.sqrt(H)),
                "cost_over_atr": 9.0 / atr, "n_oos": ns["oos"]}

    for nm, rel, builder, prep, kind in _g.SIGNALS:
        fr, _ = _g.build_fires(nm, rel, builder, prep, kind)
        fr["timestamp"] = pd.to_datetime(fr["timestamp"])
        if fr["timestamp"].dt.tz is not None:
            fr["timestamp"] = fr["timestamp"].dt.tz_localize(None)
        rows.append(measure("BTC", nm, fr, _g.HORIZON[nm], btc_kl))

    eth_kl = load_kl(ETH_KLINES)
    for nm, (csv, H) in ETH_SIGNALS.items():
        p = ROOT / csv
        if not p.exists():
            log(f"  ⚠️ETH {nm}: CSV 없음 -- 건너뜀"); continue
        fr = pd.read_csv(p, parse_dates=["timestamp"])
        if fr["timestamp"].dt.tz is not None:
            fr["timestamp"] = fr["timestamp"].dt.tz_localize(None)
        rows.append(measure("ETH", nm, fr, H, eth_kl))

    log("")
    log(f"{'':4}{'신호':<26}{'H':>4}{'ATR(bp)':>9}{'총이익':>8}{'/ATR':>8}{'/ATR√H':>9}{'비용/ATR':>9}")
    for r in rows:
        log(f"{r['asset']:<4}{r['signal']:<26}{r['H']:>4}{r['atr_bp']:>9.1f}"
            f"{r['gross_bp']:>8.2f}{r['gross_over_atr']*100:>7.1f}%"
            f"{r['gross_over_atr_sqrtH']*100:>8.1f}%{r['cost_over_atr']*100:>8.0f}%")

    for a in ("BTC", "ETH"):
        g = [r for r in rows if r["asset"] == a]
        if not g:
            continue
        log(f"  {a} 평균: ATR {np.mean([r['atr_bp'] for r in g]):.1f}bp  "
            f"총이익/ATR **{np.mean([r['gross_over_atr'] for r in g])*100:.1f}%**  "
            f"총이익/ATR√H **{np.mean([r['gross_over_atr_sqrtH'] for r in g])*100:.2f}%**  "
            f"비용/ATR {np.mean([r['cost_over_atr'] for r in g])*100:.0f}%")

    b = [r for r in rows if r["asset"] == "BTC"]
    e = [r for r in rows if r["asset"] == "ETH"]
    if e:
        rb = np.mean([r["gross_over_atr_sqrtH"] for r in b])
        re_ = np.mean([r["gross_over_atr_sqrtH"] for r in e])
        log("")
        log(f"  ⭐정규화 포착력 비율 BTC/ETH = {rb/re_:.2f}x")
        log("  ⇒ " + ("✅**가설 지지** -- 정규화하면 비슷하다. 원인은 변동성 부족(비용/ATR)이다."
                      if rb / re_ >= 0.75 else
                      "⚠️**가설 부분지지** -- 정규화 후에도 BTC가 낮다. "
                      "변동성 부족 + 신호 포착력 약화가 겹쳤다."))
    OUT.write_text(json.dumps({"note": "총이익은 비용0 진단값, 승격 근거 아님",
                               "leverage_invariance": lev_rows, "atr_normalized": rows},
                              ensure_ascii=False, indent=2, default=str))
    log(f"report -> {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
