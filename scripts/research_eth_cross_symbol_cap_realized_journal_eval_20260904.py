#!/usr/bin/env python3
"""A4 크로스심볼 캡 -- 실현 저널 기반 재사이징 평가 (2026-09-04).

`data/live/trade_journal.jsonl`(ETH 실경로 + BTC/SOL 인프로세스 슬롯, 2026-07-07~)에 이미 기록된
트레이드에 prealloc 캡 구성을 사후 적용해 포트폴리오 실현 PnL/MDD를 비교한다.

왜 이게 정확한가: prealloc은 진입 시점·방향을 바꾸지 않고 크기만 min(요청, 예산)으로 줄인다
(예산이 min_notional 이상이면 기각 없음) -> 트레이드 집합은 구성과 무관하게 동일하고, 트레이드
손익은 notional에 선형이다(pnl_frac = (gross - roundtrip_fee) * notional -- 행마다 검증). 따라서
기록된 크기보다 *좁은* 예산 구성의 반사실은 정확하다. 기록된 notional은 이미 서버 캡(3.0 /
50-30-20, >=2026-08-20 활성)으로 잘린 값이라 "무캡" 반사실은 재구성할 수 없다(요청값이 저널에
없음 -- 2026-09-04 `portfolio_cap` 텔레메트리 이후 행부터 가능. 그 행부터는 requested/approved
비율로 최종 notional을 되돌려 무캡 반사실도 계산한다).

이건 백테스트/OOS 조회가 아니라 이미 일어난 라이브(페이퍼) 결정의 재집계다.
실행: bash scripts/ops/handoff.sh pull server data/live/trade_journal.jsonl 후
      python scripts/research_eth_cross_symbol_cap_realized_journal_eval_20260904.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
JOURNAL = ROOT / "data/live/trade_journal.jsonl"
OUT_DIR = ROOT / "tmp/eth_cross_symbol_cap_realized_journal_eval_20260904"
MIN_NOTIONAL = 0.05  # PortfolioRiskConfig default == replay MIN_NOTIONAL

# name -> (total_notional_cap, shares) ; shares normalized like PortfolioRiskManager
CONFIGS: dict[str, tuple[float | None, dict[str, float]]] = {
    "recorded": (None, {}),
    "cap3.0_50-30-20 (server since >=08-20)": (3.0, {"eth": 0.5, "btc": 0.3, "sol": 0.2}),
    "cap1.5_equal (A4 fresh optimum, original grid)": (1.5, {"eth": 1.0, "btc": 1.0, "sol": 1.0}),
    "cap1.0_equal (A4 extended-grid optimum)": (1.0, {"eth": 1.0, "btc": 1.0, "sol": 1.0}),
}


def budget(cfg: tuple[float | None, dict[str, float]], asset: str) -> float | None:
    cap, shares = cfg
    if cap is None:
        return None
    s = sum(shares.values())
    return cap * shares.get(asset, 0.0) / s


def load_trades(path: Path = JOURNAL) -> tuple[pd.DataFrame, list[dict]]:
    rows = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    for r in rows:
        r["asset"] = (r.get("asset") or "eth").lower()
    # single slot per asset: pair each CLOSE with that asset's currently-open OPEN row (journal order),
    # trade_id only as a cross-check -- a restart-restored position can carry a re-minted trade_id.
    open_by_asset: dict[str, dict] = {}
    trades, open_now, id_mismatch = [], [], 0
    for r in rows:
        if r["kind"] == "OPEN":
            open_by_asset[r["asset"]] = r
            continue
        if r["kind"] != "CLOSE":
            continue
        o = open_by_asset.pop(r["asset"], None)
        if o is not None and o.get("trade_id") != r.get("trade_id"):
            id_mismatch += 1
        notional = float(r["notional_exposure"])
        pnl = float(r["pnl_frac"])
        gross = float(r.get("gross_return_frac", 0.0))
        fee = float(r.get("roundtrip_fee_rate", 0.0))
        cap_tr = dict(r.get("portfolio_cap") or (o or {}).get("portfolio_cap") or {})
        req, appr = cap_tr.get("requested_notional"), cap_tr.get("approved_notional")
        # final notional after later modifiers = approved * downstream_mult ; uncapped final = requested * same mult
        uncapped = notional * float(req) / float(appr) if (req and appr and float(appr) > 0) else None
        trades.append({
            "trade_id": r["trade_id"], "asset": r["asset"], "side": r["side"],
            "open_ts": pd.Timestamp((o or r)["ts"]), "close_ts": pd.Timestamp(r["ts"]),
            "notional": notional, "pnl_frac": pnl, "r_unit": pnl / notional if notional else 0.0,
            "linear_check_err": abs(pnl - (gross - fee) * notional),
            "uncapped_notional": uncapped, "has_cap_trace": bool(cap_tr), "open_row_missing": o is None,
        })
    for a, o in open_by_asset.items():
        open_now.append({"trade_id": o.get("trade_id", ""), "asset": a, "side": o["side"], "open_ts": o["ts"],
                         "notional": float(o["notional_exposure"]), "portfolio_cap": o.get("portfolio_cap") or {}})
    if id_mismatch:
        print(f"note: {id_mismatch} CLOSE row(s) paired by asset order with a different trade_id than their OPEN (restart-restored)")
    return pd.DataFrame(trades).sort_values("close_ts").reset_index(drop=True), open_now


def size_under(cfg, t: pd.Series) -> float | None:
    """Returns None when the config cannot be evaluated for this trade (needs the unrecorded request)."""
    cap, _ = cfg
    base = t["notional"]
    if cap is None:
        return base
    b = budget(cfg, t["asset"])
    if b is None:
        return base
    # widening beyond the recorded (already-capped) size is only knowable with the requested value
    if b > base + 1e-9 and t["uncapped_notional"] is not None:
        return min(t["uncapped_notional"], b)
    return min(base, b)


def simulate(trades: pd.DataFrame, cfg) -> dict:
    """Shared-cash compounding like the replay: cash_at_open * notional * r_unit credited at close;
    closes and opens interleaved by timestamp (closes first on ties)."""
    events = []
    for i, t in trades.iterrows():
        events.append((t["open_ts"], 1, i))
        events.append((t["close_ts"], 0, i))
    events.sort()
    cash, peak, mdd = 1.0, 1.0, 0.0
    cash_at_open: dict[int, float] = {}
    per_asset = {a: 0.0 for a in trades["asset"].unique()}
    scaled, ratios, skipped = 0, [], 0
    for ts, kind, i in events:
        t = trades.loc[i]
        if kind == 1:
            cash_at_open[i] = cash
            continue
        n = size_under(cfg, t)
        if n is None:
            n = t["notional"]
        if n < MIN_NOTIONAL - 1e-9:
            skipped += 1
            continue
        if n < t["notional"] - 1e-9:
            scaled += 1
            ratios.append(n / t["notional"])
        # a CLOSE whose OPEN predates the journal (or was never tagged) is sized off current cash
        gain = cash_at_open.get(i, cash) * n * t["r_unit"]
        per_asset[t["asset"]] += gain
        cash += gain
        peak = max(peak, cash)
        mdd = min(mdd, cash / peak - 1.0)
    wins = int((trades["r_unit"] > 0).sum())
    return {
        "pnl_pct": (cash - 1.0) * 100.0, "realized_mdd_pct": mdd * 100.0, "trades": int(len(trades)),
        "wr": wins / len(trades) if len(trades) else 0.0, "scaled_trades": scaled,
        "mean_scale_ratio": sum(ratios) / len(ratios) if ratios else 1.0, "skipped_below_floor": skipped,
        "per_asset_gain_pct": {a: v * 100.0 for a, v in per_asset.items()},
    }


def main() -> None:
    trades, open_now = load_trades()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"journal: {JOURNAL}  closed trades={len(trades)}  window {trades['open_ts'].min()} ~ {trades['close_ts'].max()}")
    print(f"linearity check |pnl - (gross-fee)*notional| max = {trades['linear_check_err'].max():.2e}"
          f"  (rows > 1e-6: {int((trades['linear_check_err'] > 1e-6).sum())})")
    print(f"rows with 09-04 portfolio_cap telemetry: {int(trades['has_cap_trace'].sum())}")
    print(f"CLOSE rows without a journal OPEN (position predates journal tagging): {int(trades['open_row_missing'].sum())}\n")

    print("=== 자산별 기록 크기 ===")
    for a, g in trades.groupby("asset"):
        print(f"  {a}: n={len(g)} notional min/median/max = {g['notional'].min():.3f}/{g['notional'].median():.3f}/{g['notional'].max():.3f}"
              f"  sum pnl_frac={g['pnl_frac'].sum()*100:+.2f}%")

    results = {}
    print("\n=== 구성별 포트폴리오(공유현금 복리, 실현 기준) ===")
    print(f"{'config':52} {'PnL%':>8} {'MDD%':>8} {'WR':>6} {'scaled':>6} {'ratio':>6}  per-asset gain%")
    for name, cfg in CONFIGS.items():
        res = simulate(trades, cfg)
        res["budgets"] = {a: budget(cfg, a) for a in ("eth", "btc", "sol")}
        results[name] = res
        pa = " ".join(f"{a}={v:+.2f}" for a, v in sorted(res["per_asset_gain_pct"].items()))
        print(f"{name:52} {res['pnl_pct']:8.2f} {res['realized_mdd_pct']:8.2f} {res['wr']:6.1%} {res['scaled_trades']:6d} {res['mean_scale_ratio']:6.3f}  {pa}")

    print("\n=== 현재 오픈 포지션(구성별 크기) ===")
    for o in open_now:
        sizes = {name: (min(o["notional"], budget(cfg, o["asset"])) if budget(cfg, o["asset"]) is not None else o["notional"]) for name, cfg in CONFIGS.items()}
        print(f"  {o['asset']} {o['side']} since {o['open_ts']} recorded={o['notional']:.3f} -> " + ", ".join(f"{k.split(' ')[0]}={v:.3f}" for k, v in sizes.items()))

    # 동시노출 통계(08-31 스크립트 재사용)
    sys.path.insert(0, str(ROOT / "scripts"))
    try:
        import research_eth_cross_symbol_exposure_concurrency_check_20260831 as cc  # noqa: E402
        iv, _, _ = cc.load_intervals(JOURNAL)
        overlap = cc.pairwise_overlap(iv)
        print("\n=== 자산쌍 겹침(동방향) ===")
        print(overlap.to_string(index=False))
        overlap_json = overlap.to_dict(orient="records")
    except Exception as e:  # pragma: no cover
        print("overlap stats unavailable:", e)
        overlap_json = None

    report = {
        "journal": str(JOURNAL), "n_closed_trades": int(len(trades)),
        "window": [str(trades["open_ts"].min()), str(trades["close_ts"].max())],
        "linearity_max_abs_err": float(trades["linear_check_err"].max()),
        "rows_with_cap_telemetry": int(trades["has_cap_trace"].sum()),
        "configs": results, "open_positions": open_now, "overlap": overlap_json,
        "note": "realized-only (no mark-to-market path); 'recorded' sizes already carry the server cap 3.0/50-30-20 where it bound, so an uncapped counterfactual needs the portfolio_cap telemetry rows",
    }
    (OUT_DIR / "report.json").write_text(json.dumps(report, indent=2, default=str, ensure_ascii=False))
    trades.to_csv(OUT_DIR / "trades.csv", index=False)
    print(f"\nwrote {OUT_DIR / 'report.json'}")


if __name__ == "__main__":
    main()
