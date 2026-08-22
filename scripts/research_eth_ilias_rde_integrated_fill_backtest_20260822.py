#!/usr/bin/env python3
"""RDE(레짐 직결 노출) × maker 체결 시뮬 — 레이어 통합 백테스트 (2026-08-22).

배경: 지금까지 두 축이 따로 존재했다 —
  (1) `research_eth_ilias_regime_direct_exposure_backtest_20260822.py`: HMM 레짐확률로 방향
      결정, 비용은 "평균 3.1bp/leg" 같은 고정 가정을 일괄 차감.
  (2) `research_eth_maker_fill_simulation_{l2,trades_only}_20260822.py`: RDE와 무관하게
      5분 간격 가상주문으로 체결비용 "분포"만 계측.
이 스크립트는 둘을 실제로 연결한다 — RDE 정책이 VAL 구간에서 실제로 포지션을 전환하는
**정확한 timestamp**마다, 그 순간 진짜 체결 시뮬(v2, aggTrades 재구성)을 돌려 **개별 전환의
실제 체결가**로 PnL을 재계산한다. "평균 비용을 가정"이 아니라 "각 전환이 실제로 얼마에
체결됐을지"를 하나하나 시뮬레이션한 것 — 지금까지 중 가장 현실에 가까운 수치다.

범위: VAL(2026-04-01~06-30)만 사용(이미 튜닝 티어로 반복 조회 중인 창, 신규 OOS 접촉 없음).
정책: span=36(EWMA)/hi=0.6/lo=0.3(문서에서 "중심셀"로 반복 인용된 설정), peg, T=120s.
"""
from __future__ import annotations

import importlib.util
import json
import sys
import types
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

stub = types.ModuleType("mamba_ssm")
stub.Mamba = object
sys.modules["mamba_ssm"] = stub

from experiment_regime3_current_hmm_wide24_20260529 import _transform  # noqa: E402

spec = importlib.util.spec_from_file_location(
    "fillsim_v2", ROOT / "scripts/research_eth_maker_fill_simulation_trades_only_20260822.py")
fillsim = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fillsim)

HMM_MODEL = ROOT / "tmp/eth_hmm_wide24_resweep_train2026h1_20260821/states24_sticky0.90/models/regime3_current_sensitive_v2_hmm_wide24_2024.joblib"
CANON_2026 = ROOT / "data/splits/year_oos/training_features_2026_rebuilt.csv"
OUT_DIR = ROOT / "tmp/eth_maker_fill_simulation_20260822"

VAL_START, VAL_END = "2026-04-01", "2026-07-01"
SPAN, HI, LO = 36, 0.6, 0.3
POLICY, TIMEOUT_S = "peg", 120
MAKER_FEE_BP = 2.0
TAKER_FEE_BP = 5.0


def rde_positions() -> pd.DataFrame:
    frame = pd.read_csv(CANON_2026)
    frame["timestamp"] = pd.to_datetime(frame["timestamp"])
    frame = frame.sort_values("timestamp").reset_index(drop=True)
    payload = joblib.load(HMM_MODEL)
    ov, _ = _transform(payload, frame)
    prefix = f"{payload['prefix_stem']}_{payload['feature_set']}_"
    df = pd.DataFrame({"ts": ov["timestamp"], "bull": ov[prefix + "bull_prob"].to_numpy(),
                       "bear": ov[prefix + "bear_prob"].to_numpy(), "close": frame["close"].to_numpy()})
    w = df[(df.ts >= VAL_START) & (df.ts < VAL_END)].reset_index(drop=True)
    b = w.bull.ewm(span=SPAN, adjust=False).mean().to_numpy()
    s = w.bear.ewm(span=SPAN, adjust=False).mean().to_numpy()
    pos = np.zeros(len(b), dtype=int)
    cur = 0
    for t in range(len(b)):
        if cur == 1 and b[t] < LO:
            cur = 0
        elif cur == -1 and s[t] < LO:
            cur = 0
        if cur == 0:
            if b[t] > HI:
                cur = 1
            elif s[t] > HI:
                cur = -1
        pos[t] = cur
    w["pos"] = pos
    return w


def extract_transitions(w: pd.DataFrame) -> pd.DataFrame:
    pos = w.pos.to_numpy()
    idx = np.flatnonzero(np.diff(pos)) + 1
    rows = []
    for i in idx:
        old, new = int(pos[i - 1]), int(pos[i])
        rows.append({"ts": w.ts.iloc[i], "side": "buy" if new > old else "sell",
                    "mag": abs(new - old), "old": old, "new": new})
    return pd.DataFrame(rows)


def simulate_transitions(trans: pd.DataFrame) -> pd.DataFrame:
    dates = sorted(trans.ts.dt.date.astype(str).unique())
    print(f"downloading/loading aggTrades for {len(dates)} unique days...", flush=True)
    books: dict[str, "fillsim.Book"] = {}
    for i, d in enumerate(dates):
        tr = fillsim.load_days([d])
        books[d] = fillsim.Book(tr)
        if (i + 1) % 10 == 0:
            print(f"  {i+1}/{len(dates)} days loaded", flush=True)

    results = []
    for _, row in trans.iterrows():
        d = row.ts.strftime("%Y-%m-%d")
        book = books[d]
        t0 = int(row.ts.timestamp() * 1000)
        # 자정 근접(다음날 데이터 필요) 방어: 타임아웃 창이 그날 마지막 체결보다 늦으면 다음날 이어붙임
        if t0 + TIMEOUT_S * 1000 > book.ts[-1]:
            nd = (row.ts + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
            if nd not in books:
                books[nd] = fillsim.Book(fillsim.load_days([nd]))
            merged_tr = pd.concat([
                pd.DataFrame({"price": book.px, "quantity": book.qty, "transact_time": book.ts, "is_buyer_maker": book.bm}),
                pd.DataFrame({"price": books[nd].px, "quantity": books[nd].qty, "transact_time": books[nd].ts, "is_buyer_maker": books[nd].bm}),
            ], ignore_index=True)
            book = fillsim.Book(merged_tr)
        r = fillsim.simulate_leg(book, t0, row.side, TIMEOUT_S, POLICY)
        if r is None:
            r = {"filled": False, "cost_bp": None, "mode": "no_quote_at_arrival"}
        results.append({**row.to_dict(), **r})
    return pd.DataFrame(results)


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    w = rde_positions()
    trans = extract_transitions(w)
    print(f"transitions: {len(trans)} (buy={sum(trans.side=='buy')}, sell={sum(trans.side=='sell')})", flush=True)

    ret = np.zeros(len(w))
    ret[1:] = w.close.to_numpy()[1:] / w.close.to_numpy()[:-1] - 1.0
    sig = np.roll(w.pos.to_numpy(), 1)
    sig[0] = 0
    gross = float((sig[1:] * ret[1:]).sum())
    print(f"gross (no cost): {gross*100:.2f}%", flush=True)

    filled = simulate_transitions(trans)
    n_missing = filled.cost_bp.isna().sum()
    print(f"legs with no cost (no_quote_at_arrival): {n_missing}/{len(filled)}", flush=True)

    total_cost_frac = (filled.cost_bp.fillna(0.0) / 1e4).sum()
    net_integrated = gross - total_cost_frac
    avg_cost_bp = filled.cost_bp.mean()
    filled_rate = filled.filled.mean()

    flat_31 = gross - len(trans) * 3.1e-4
    flat_40 = gross - len(trans) * 4.0e-4
    flat_75 = gross - len(trans) * 7.5e-4

    report = {
        "experiment": "eth_ilias_rde_integrated_fill_backtest_20260822",
        "policy": {"span": SPAN, "hi": HI, "lo": LO, "fill_policy": POLICY, "timeout_s": TIMEOUT_S},
        "window": [VAL_START, VAL_END],
        "n_transitions": int(len(trans)),
        "gross_pnl_pct_no_cost": gross * 100,
        "avg_realized_cost_bp": float(avg_cost_bp),
        "fill_rate": float(filled_rate),
        "n_legs_no_quote_at_arrival": int(n_missing),
        "net_pnl_pct_integrated_per_transition_sim": net_integrated * 100,
        "net_pnl_pct_flat_assumption_3.1bp": flat_31 * 100,
        "net_pnl_pct_flat_assumption_4.0bp": flat_40 * 100,
        "net_pnl_pct_flat_assumption_7.5bp_tail": flat_75 * 100,
        "fill_mode_counts": filled["mode"].value_counts().to_dict() if "mode" in filled.columns else {},
        "honesty_caveats": [
            "VAL window (in-sample-ish, inside HMM fit period) -- research/dev score only.",
            "Single seed (7529) for HMM regime probs -- not re-run across N=5 seeds for this integration.",
            "Self-order market impact still ignored (small-size assumption), same as v1/v2 sims.",
        ],
    }
    (OUT_DIR / "report_rde_integrated_val.json").write_text(json.dumps(report, indent=2))
    filled.to_csv(OUT_DIR / "rde_integrated_val_transitions_filled.csv", index=False)

    print(json.dumps({k: v for k, v in report.items() if not isinstance(v, (list, dict))}, indent=2))
    print(f"\nreport -> {OUT_DIR / 'report_rde_integrated_val.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
