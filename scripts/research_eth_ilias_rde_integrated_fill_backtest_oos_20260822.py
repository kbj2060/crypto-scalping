#!/usr/bin/env python3
"""RDE × maker 체결 시뮬 — 레이어 통합 백테스트, OOS 조기실행 (2026-08-22).

⚠️ Fresh-Forward OOS "09-30까지 대기" 규칙에 대한 **명시적 사용자 override**
(`.claude/CLAUDE.md` 규칙, 일리아스 계약 "Dataset Split" 절 예외 기록 절차 적용) — 사용자가
"현재까지 데이터로 OOS 진행해줘, 09-30까지 기다리지 말고"라고 직접 지시(2026-08-22).
같은 종류의 선례 2건(레짐 분류기 N=5 시드검증, DC154 트랜스포머 스모크테스트)과 동일 패턴,
동일 OOS 창(2026-07-01~08-19, 즉시가용 데이터 — 09-30까지 기다리지 않음).

이 실행은 이 OOS 창을 **이 정책(RDE 레짐직결노출, span36/hi0.6/lo0.3, peg, T120,
seed=[7529,534964,116595,666940,505456]) 전용으로 단일터치 소진**한다. 결과가 어떻게
나오든 이 창에서 다른 정책/파라미터로 재시도하지 않는다 — VAL에서 이미 정한 "중심셀"
정책을 그대로, 딱 한 번만 평가한다(`research_eth_ilias_rde_integrated_fill_backtest_5seed_20260822.py`
의 OOS 버전, 로직 동일·창만 교체).
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

OOS_START, OOS_END = "2026-07-01", "2026-08-20"   # 데이터상 실질 종점 2026-08-19 23:55
SPAN, HI, LO = 36, 0.6, 0.3          # VAL에서 확정한 정책, 변경 없음
POLICY, TIMEOUT_S = "peg", 120


def positions_for_seed(frame: pd.DataFrame, model_path: Path) -> pd.DataFrame:
    payload = joblib.load(model_path)
    ov, _ = _transform(payload, frame)
    prefix = f"{payload['prefix_stem']}_{payload['feature_set']}_"
    df = pd.DataFrame({"ts": ov["timestamp"], "bull": ov[prefix + "bull_prob"].to_numpy(),
                       "bear": ov[prefix + "bear_prob"].to_numpy(), "close": frame["close"].to_numpy()})
    w = df[(df.ts >= OOS_START) & (df.ts < OOS_END)].reset_index(drop=True)
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

    always_long = (frame[(frame.timestamp >= OOS_START) & (frame.timestamp < OOS_END)].close.iloc[-1]
                  / frame[(frame.timestamp >= OOS_START) & (frame.timestamp < OOS_END)].close.iloc[0] - 1) * 100

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
        if (i + 1) % 10 == 0:
            print(f"  {i+1}/{len(dates)} days loaded", flush=True)

    def book_for(ts: pd.Timestamp) -> "fillsim.Book":
        d = ts.strftime("%Y-%m-%d")
        book = books[d]
        t0 = int(ts.timestamp() * 1000)
        if t0 + TIMEOUT_S * 1000 > book.ts[-1]:
            nd = (ts + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
            if nd not in books:
                try:
                    books[nd] = fillsim.Book(fillsim.load_days([nd]))
                except Exception:
                    return book
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
            "avg_realized_cost_bp": float(filled.cost_bp.mean()) if filled.cost_bp.notna().any() else None,
            "fill_rate": float(filled.filled.mean()),
            "n_no_quote": int(filled.cost_bp.isna().sum()),
            "net_pnl_pct": net * 100,
        }
        print(f"seed={seed} net={net*100:.2f}% avg_cost={results[str(seed)]['avg_realized_cost_bp']}bp "
              f"fill_rate={filled.filled.mean():.3f}", flush=True)
        filled.to_csv(OUT_DIR / f"rde_integrated_oos_transitions_filled_seed{seed}.csv", index=False)

    nets = [v["net_pnl_pct"] for v in results.values()]
    report = {
        "experiment": "eth_ilias_rde_integrated_fill_backtest_oos_20260822",
        "⚠️_oos_early_access_override": (
            "Fresh-Forward '09-30까지 대기' 규칙의 명시적 사용자 override — 사용자 직접 지시"
            "(2026-08-22, '현재까지 데이터로 OOS 진행해줘, 09-30까지 기다리지 말고'). "
            "레짐분류기/DC154 트랜스포머와 동일 패턴의 3번째 사례."
        ),
        "policy": {"span": SPAN, "hi": HI, "lo": LO, "fill_policy": POLICY, "timeout_s": TIMEOUT_S,
                  "note": "VAL에서 확정한 정책 그대로 재사용 -- OOS를 보고 나서 고른 것 아님"},
        "window": [OOS_START, OOS_END],
        "always_long_benchmark_pct": always_long,
        "seeds": list(map(str, SEED_MODELS)),
        "per_seed": results,
        "net_pnl_mean_pct": float(np.mean(nets)),
        "net_pnl_std_pct": float(np.std(nets)),
        "net_pnl_min_max_pct": [float(min(nets)), float(max(nets))],
        "sign_consistency": f"{sum(1 for n in nets if n > 0)}/{len(nets)} positive",
        "single_touch_status": "CONSUMED -- 이 정책(span36/hi0.6/lo0.3)에 대해 이 OOS 창은 이제 재조회 금지",
        "honesty_caveats": [
            "이 창(2026-07-01~08-19)은 이미 다른 축(maker 체결시뮬 pilot window 07-19~21 raw L2,"
            " 최초 RDE 단일정책 grid-selected hi0.7/lo0.45 OOS평가)에서 조회된 적 있음 -- 완전히"
            " 처음 보는 창은 아니나, 이 특정 정책(span36/hi0.6/lo0.3)에 대해서는 최초 조회.",
            "단일 정책만 평가 -- 결과가 부정적이어도 VAL 재탐색으로 새 정책을 찾아 재시도하지 않음.",
            "self-order market impact 여전히 미반영.",
        ],
    }
    (OUT_DIR / "report_rde_integrated_oos.json").write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print("\n" + json.dumps({k: v for k, v in report.items() if k not in ("per_seed", "honesty_caveats", "⚠️_oos_early_access_override")}, indent=2, ensure_ascii=False))
    print(f"\nreport -> {OUT_DIR / 'report_rde_integrated_oos.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
