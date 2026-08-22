#!/usr/bin/env python3
"""RDE × maker 체결 시뮬 — 레이어 통합 백테스트 N=5 시드 확장 (2026-08-22).

`research_eth_ilias_rde_integrated_fill_backtest_20260822.py`(단일 시드 7529)의 N=5 확장 —
CLAUDE.md Seed-Diversity Ensemble Promotion Gate(N≥5 진짜 무작위 시드) 정신을, 이번엔
"방향 신호"뿐 아니라 **레이어 통합(실제 전환 timestamp별 체결 시뮬)까지 포함한 최종 net PnL**
에 직접 적용한다. 시드는 레짐 분류기 시드검증(N=5 CONFIRMED,
[[eth_regime_classifier_wide24_vs_jm_sjm_investigation_20260821]])과 동일한 5개
(7529 baseline + 534964/116595/666940/505456, 진짜무작위 `random.sample` 추출 — 고정간격
증가 아님, 게이트 요건 충족).

VAL(2026-04-01~06-30)만 사용(신규 OOS 접촉 없음). 정책: span=36(EWMA)/hi=0.6/lo=0.3, peg,
T=120s — 단일시드판과 동일. 사전 확인: 5개 시드의 VAL 포지션 시계열은 사실상 동일
(match rate ≈1.0000, 레짐분류기 자체의 시드안정성과 정합) → 필요한 aggTrades 날짜 집합도
거의 동일(64일 그대로, 추가 다운로드 최소) → Book 캐시를 시드 간 공유해 재다운로드 방지.
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

RESWEEP = ROOT / "tmp/eth_hmm_wide24_resweep_train2026h1_20260821"
SEED_MODELS = {
    7529: RESWEEP / "states24_sticky0.90/models/regime3_current_sensitive_v2_hmm_wide24_2024.joblib",
    534964: RESWEEP / "seedcheck_states24_sticky0.90_seed534964/models/regime3_current_sensitive_v2_hmm_wide24_2024.joblib",
    116595: RESWEEP / "seedcheck_states24_sticky0.90_seed116595/models/regime3_current_sensitive_v2_hmm_wide24_2024.joblib",
    666940: RESWEEP / "seedcheck_states24_sticky0.90_seed666940/models/regime3_current_sensitive_v2_hmm_wide24_2024.joblib",
    505456: RESWEEP / "seedcheck_states24_sticky0.90_seed505456/models/regime3_current_sensitive_v2_hmm_wide24_2024.joblib",
}
CANON_2026 = ROOT / "data/splits/year_oos/training_features_2026_rebuilt.csv"
OUT_DIR = ROOT / "tmp/eth_maker_fill_simulation_20260822"

VAL_START, VAL_END = "2026-04-01", "2026-07-01"
SPAN, HI, LO = 36, 0.6, 0.3
POLICY, TIMEOUT_S = "peg", 120


def positions_for_seed(frame: pd.DataFrame, model_path: Path) -> pd.DataFrame:
    payload = joblib.load(model_path)
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


def gross_pnl(w: pd.DataFrame) -> float:
    ret = np.zeros(len(w))
    ret[1:] = w.close.to_numpy()[1:] / w.close.to_numpy()[:-1] - 1.0
    sig = np.roll(w.pos.to_numpy(), 1)
    sig[0] = 0
    return float((sig[1:] * ret[1:]).sum())


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(CANON_2026)
    frame["timestamp"] = pd.to_datetime(frame["timestamp"])
    frame = frame.sort_values("timestamp").reset_index(drop=True)

    per_seed = {}
    all_dates: set[str] = set()
    for seed, path in SEED_MODELS.items():
        w = positions_for_seed(frame, path)
        trans = extract_transitions(w)
        per_seed[seed] = {"w": w, "trans": trans, "gross": gross_pnl(w)}
        all_dates.update(trans.ts.dt.date.astype(str).unique())
        print(f"seed={seed} n_transitions={len(trans)} gross={per_seed[seed]['gross']*100:.2f}%", flush=True)

    dates = sorted(all_dates)
    print(f"\nunion of unique dates across 5 seeds: {len(dates)}", flush=True)
    books: dict[str, "fillsim.Book"] = {}
    for i, d in enumerate(dates):
        tr = fillsim.load_days([d])
        books[d] = fillsim.Book(tr)
        if (i + 1) % 20 == 0:
            print(f"  {i+1}/{len(dates)} days loaded", flush=True)

    def book_for(ts: pd.Timestamp) -> "fillsim.Book":
        d = ts.strftime("%Y-%m-%d")
        book = books[d]
        t0 = int(ts.timestamp() * 1000)
        if t0 + TIMEOUT_S * 1000 > book.ts[-1] and (ts + pd.Timedelta(days=1)).strftime("%Y-%m-%d") in books:
            nd = (ts + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
            merged = pd.concat([
                pd.DataFrame({"price": book.px, "quantity": book.qty, "transact_time": book.ts, "is_buyer_maker": book.bm}),
                pd.DataFrame({"price": books[nd].px, "quantity": books[nd].qty, "transact_time": books[nd].ts, "is_buyer_maker": books[nd].bm}),
            ], ignore_index=True)
            return fillsim.Book(merged)
        return book

    results = {}
    for seed, data in per_seed.items():
        trans = data["trans"]
        rows = []
        for _, row in trans.iterrows():
            book = book_for(row.ts)
            t0 = int(row.ts.timestamp() * 1000)
            r = fillsim.simulate_leg(book, t0, row.side, TIMEOUT_S, POLICY)
            if r is None:
                r = {"filled": False, "cost_bp": None, "mode": "no_quote_at_arrival"}
            rows.append({**row.to_dict(), **r})
        filled = pd.DataFrame(rows)
        total_cost_frac = (filled.cost_bp.fillna(0.0) / 1e4).sum()
        net = data["gross"] - total_cost_frac
        results[str(seed)] = {
            "n_transitions": int(len(trans)),
            "gross_pnl_pct": data["gross"] * 100,
            "avg_realized_cost_bp": float(filled.cost_bp.mean()),
            "fill_rate": float(filled.filled.mean()),
            "n_no_quote": int(filled.cost_bp.isna().sum()),
            "net_pnl_pct": net * 100,
        }
        print(f"seed={seed} net={net*100:.2f}% avg_cost={filled.cost_bp.mean():.2f}bp "
              f"fill_rate={filled.filled.mean():.3f}", flush=True)
        filled.to_csv(OUT_DIR / f"rde_integrated_val_transitions_filled_seed{seed}.csv", index=False)

    nets = [v["net_pnl_pct"] for v in results.values()]
    report = {
        "experiment": "eth_ilias_rde_integrated_fill_backtest_5seed_20260822",
        "policy": {"span": SPAN, "hi": HI, "lo": LO, "fill_policy": POLICY, "timeout_s": TIMEOUT_S},
        "window": [VAL_START, VAL_END],
        "seeds": list(map(str, SEED_MODELS)),
        "per_seed": results,
        "net_pnl_mean_pct": float(np.mean(nets)),
        "net_pnl_std_pct": float(np.std(nets)),
        "net_pnl_min_max_pct": [float(min(nets)), float(max(nets))],
        "sign_consistency": f"{sum(1 for n in nets if n > 0)}/{len(nets)} positive",
        "honesty_caveats": [
            "VAL window (in-sample-ish, inside HMM fit period) -- research/dev score only.",
            "Position series near-identical across seeds (match rate ~1.0 vs seed 7529) -- this is",
            "primarily a check that fill-cost integration doesn't introduce seed sensitivity, not an",
            "independent re-derivation of the direction signal (which was already known stable).",
            "Self-order market impact still ignored (small-size assumption).",
        ],
    }
    (OUT_DIR / "report_rde_integrated_val_5seed.json").write_text(json.dumps(report, indent=2))
    print("\n" + json.dumps({k: v for k, v in report.items() if k not in ("per_seed", "honesty_caveats")}, indent=2))
    print(f"\nreport -> {OUT_DIR / 'report_rde_integrated_val_5seed.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
