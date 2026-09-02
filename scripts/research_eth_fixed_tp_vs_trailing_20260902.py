#!/usr/bin/env python3
"""STEP A -- 익절선을 넓히면 더 버는가? 기존 발동 그대로, 청산만 고정 TP로 바꿔 검정.
2026-09-02 사용자 제안("익절라인이 진입가랑 너무 붙어있어서 늘려서 테스트하고 싶다").

전제는 이미 데이터로 확인됐다 -- 발동들의 MFE 중앙값이 1.80~4.23 ATR인데 현재 트레일링 청산은
ARM 1.0~3.0에서 발동해 0.05 ATR 되돌림에 나간다(short_term_return_z는 중앙 이동의 44%를 남김).
다만 MFE는 완전예지로만 잡히는 최대치이므로 "여지가 있다"가 "먹을 수 있다"를 뜻하지 않는다.

이 스크립트는 재학습 없이 **청산만** 바꿔 그 격차를 얼마나 회수할 수 있는지 잰다. 여기서 고정 TP가
트레일링에 크게 지면, 라벨 K를 올려 재학습하는 Step B는 "선택만으로 그 격차를 뒤집어야" 하는
훨씬 강한 주장이 된다 -- 그 사전 판단을 저렴하게 얻는 게 목적이다.

⚠️선례: 2026-08-30 liquidity_sweep에서 "고정 TP/SL + 시간청산"이 트레일링보다 나빴고, 그리드를
넓히면 TP가 사실상 도달불가능해지고 시간청산이 대신 일하는 퇴화 패턴으로 기각된 바 있다. 그건
V자반등 라벨 기준이었고, 여기서는 배포된 5개 신호에 MFE 증거를 쥐고 재검정한다.

⚠️봉내 순서: core의 브래킷은 한 봉 안에서 SL을 먼저 검사한다(비관적 = 보수적). 이 저장소는
낙관/비관 이중검증을 요구하므로, 코어를 고치는 대신 **모호 발생률**(청산 봉에서 TP와 SL 레벨이
둘 다 닿은 비율)을 함께 측정해 관례가 결론을 바꿀 수 있는지 수치로 남긴다.
"""
from __future__ import annotations

import json, sys, warnings
warnings.filterwarnings("ignore")
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import numpy as np
import pandas as pd
from core.causal_futures_backtest import purged_decision_mask, simulate_single_position  # noqa: E402
from research_eth_regime_gated_costgate_ensemble_20260902 import (  # noqa: E402
    HOLDOUT_START, KLINES_PATH, LEVERAGE, MARGIN_FRACTION, MIN_WINDOW_N, OOS_START,
    ROUNDTRIP_COST_RATE, SIGNALS, VAL_START,
)

TP_GRID = [1.0, 1.5, 2.0, 2.5, 3.0, 4.0]
SL_GRID = [1.5, 2.0, 3.0, 4.0]
TRAIL_BASE = {  # 각 신호의 현행 트레일링 최적(VAL 선택본) -- 동일 실행 내 like-for-like 대조
    "taker_delta_z_climax": (2.0, 1.0, 0.05), "short_term_return_z": (3.0, 1.0, 0.05),
    "liquidity_sweep": (5.0, 3.0, 0.05), "orthogonal_combo": (4.0, 1.5, 0.05),
    "smt_divergence": (3.5, 2.0, 0.05)}
OUT_DIR = ROOT / "tmp/eth_fixed_tp_vs_trailing_20260902"


def log(m): print(f"[fixedTP] {m}", flush=True)


def run(ts, o, h, l, c, dec, sc, atr, H, *, tp=None, sl=None, arm=None, trail=None):
    n = len(dec)
    res = simulate_single_position(
        timestamps=ts, open_px=o, high=h, low=l, close=c, decision_indices=dec, scores=sc,
        tp_moves=(tp * atr) if tp is not None else np.full(n, 999.0), sl_moves=sl * atr,
        upper_threshold=1.0, lower_threshold=-1.0, horizon_bars=H,
        margin_fraction=MARGIN_FRACTION, leverage=LEVERAGE, roundtrip_cost_rate=ROUNDTRIP_COST_RATE,
        **({"arm_moves": arm * atr, "trail_moves": trail * atr} if arm is not None else {}))
    L = res.ledger
    if L.empty: return None
    r = L["trade_return"].to_numpy(); w, ls = r[r > 0].sum(), -r[r < 0].sum()
    return {"n": int(len(r)), "mean_bp": float(r.mean() * 1e4), "total_bp": float(r.sum() * 1e4),
            "pf": float(w / ls) if ls > 0 else float("inf"),
            "profit_wr": float((r > 0).mean()), "ledger": L}


def ambiguity_rate(kl, L, atr_map, tp, sl):
    """청산 봉에서 TP·SL 레벨이 둘 다 닿았던 비율 -- 봉내 순서 관례가 좌우하는 거래의 비중."""
    idx = pd.DatetimeIndex(kl["timestamp"]); amb = 0
    for r in L.itertuples():
        try:
            ei, xi = idx.get_loc(r.entry_timestamp), idx.get_loc(r.exit_timestamp)
        except KeyError:
            continue
        a = atr_map.get(r.decision_timestamp)
        if a is None: continue
        entry = kl["open"].iloc[ei]
        if r.side > 0: tl, sll = entry * (1 + tp * a), entry * (1 - sl * a)
        else:          tl, sll = entry * (1 - tp * a), entry * (1 + sl * a)
        hi, lo = kl["high"].iloc[xi], kl["low"].iloc[xi]
        if (hi >= max(tl, sll) if r.side > 0 else lo <= min(tl, sll)):
            if (lo <= sll and hi >= tl) if r.side > 0 else (hi >= sll and lo <= tl): amb += 1
    return amb / max(1, len(L))


def main() -> int:
    kl = pd.read_csv(KLINES_PATH, parse_dates=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    ts = kl["timestamp"]; o, h, l, c = (kl[k].to_numpy() for k in ("open", "high", "low", "close"))
    out = {}
    for name, cfg in SIGNALS.items():
        H = cfg["horizon"]
        f = pd.read_csv(ROOT / cfg["fires"], parse_dates=["timestamp"])
        f = f.loc[f["timestamp"] < HOLDOUT_START].sort_values("pos").reset_index(drop=True)
        dec = f["pos"].to_numpy(np.int64); sc = np.where(f["side"].to_numpy() == "bottom", 1.0, -1.0)
        atr = f["atr_pct"].to_numpy(float)
        atr_map = dict(zip(f["timestamp"], atr))
        ev = purged_decision_mask(ts, start=VAL_START, end=OOS_START, horizon_bars=H)
        eo = purged_decision_mask(ts, start=OOS_START, end=HOLDOUT_START, horizon_bars=H)
        vs, os_ = set(np.flatnonzero(ev).tolist()), set(np.flatnonzero(eo).tolist())
        vm = np.array([d in vs for d in dec]); om = np.array([d in os_ for d in dec])
        if vm.sum() < MIN_WINDOW_N or om.sum() < MIN_WINDOW_N: continue

        bs, ba, bt = TRAIL_BASE[name]
        tv = run(ts, o, h, l, c, dec[vm], sc[vm], atr[vm], H, sl=bs, arm=ba, trail=bt)
        to = run(ts, o, h, l, c, dec[om], sc[om], atr[om], H, sl=bs, arm=ba, trail=bt)
        log(f"\n=== {name} (H={H}) ===")
        log(f"  트레일링 baseline (SL{bs}/ARM{ba}/Tr{bt}): "
            f"VAL {tv['mean_bp']:+6.2f}bp -> OOS {to['mean_bp']:+6.2f}bp PF {to['pf']:.2f} 총 {to['total_bp']:+8.1f}")

        cands = []
        for tp in TP_GRID:
            for sl in SL_GRID:
                v = run(ts, o, h, l, c, dec[vm], sc[vm], atr[vm], H, tp=tp, sl=sl)
                if v is None: continue
                fv = run(ts, o, h, l, c, dec[vm], -sc[vm], atr[vm], H, tp=tp, sl=sl)
                if not (v["mean_bp"] > 0 and fv and v["mean_bp"] > fv["mean_bp"] and fv["mean_bp"] < 0):
                    continue
                oo = run(ts, o, h, l, c, dec[om], sc[om], atr[om], H, tp=tp, sl=sl)
                cands.append({"tp": tp, "sl": sl, "val": v["mean_bp"], "oos": oo["mean_bp"],
                              "oos_pf": oo["pf"], "oos_total": oo["total_bp"], "oos_n": oo["n"],
                              "_L": oo["ledger"]})
        if not cands:
            log("  고정TP: VAL 통과 조합 없음"); out[name] = {"fixed_tp": None}; continue
        b = max(cands, key=lambda x: x["val"])
        amb = ambiguity_rate(kl, b["_L"], atr_map, b["tp"], b["sl"])
        log(f"  고정TP VAL선택 TP{b['tp']}/SL{b['sl']}: VAL {b['val']:+6.2f}bp -> OOS {b['oos']:+6.2f}bp "
            f"PF {b['oos_pf']:.2f} 총 {b['oos_total']:+8.1f} n={b['oos_n']}")
        log(f"  → OOS 차이 (고정TP − 트레일링): mean {b['oos']-to['mean_bp']:+6.2f}bp, "
            f"총 {b['oos_total']-to['total_bp']:+8.1f} | 봉내순서 모호율 {amb:.1%}")
        log(f"  통과 조합 {len(cands)}/{len(TP_GRID)*len(SL_GRID)}, TP별 OOS mean: " +
            ", ".join(f"TP{t}:{max([x['oos'] for x in cands if x['tp']==t], default=float('nan')):+.1f}"
                      for t in TP_GRID))
        out[name] = {"trailing": {"val": tv["mean_bp"], "oos": to["mean_bp"], "oos_total": to["total_bp"], "oos_pf": to["pf"]},
                     "fixed_tp": {k: v for k, v in b.items() if k != "_L"},
                     "ambiguity_rate": amb, "n_pass": len(cands)}
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "summary.json").write_text(json.dumps(out, indent=2, ensure_ascii=False, default=float))
    log(f"\nWrote {OUT_DIR}/summary.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
