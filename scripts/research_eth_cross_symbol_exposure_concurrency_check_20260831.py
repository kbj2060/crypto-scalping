#!/usr/bin/env python3
"""ETH/BTC/SOL 라이브+섀도우 동시노출(크로스심볼 스태킹) 실측 — A4 설계용.

trading_bot.py의 공유 `data/live/trade_journal.jsonl`(ETH 암묵 + BTC/SOL 명시 태깅,
단일슬롯 GovernorPositionRouter 3개)에서 이미 실현된 진입/청산 이벤트만 읽어 자산쌍별
시간 겹침·동방향 여부를 집계한다. 이건 백테스트/OOS 조회가 아니라 **이미 일어난 라이브/
섀도우 결정을 세는 것**이라 09-30 fresh-forward 데이터 세대교체 규율과 무관하다
(2026-08-24에 한 번 수기로 이 카운팅을 했으나 스크립트로 남기지 않아 이번에 재사용
가능하도록 저장 — 다음에는 이 스크립트를 그대로 재실행하면 됨).

BTC의 독립 3-슬롯 실험(`run_btc_multislot_shadow_loop_20260807.py`, 별도 state/ledger
파일)은 여기 포함하지 않는다 — 이 스크립트는 trading_bot.py 인프로세스 단일슬롯 BTC/SOL
섀도우만 다룬다.

실행: `bash scripts/ops/handoff.sh pull server data/live/trade_journal.jsonl` 로 먼저
서버의 최신 원장을 받은 뒤 이 스크립트를 돌릴 것 — 로컬 dev 체크아웃엔 라이브 프로세스가
안 돌아 파일이 갱신되지 않는다.
"""
from __future__ import annotations

import json
from itertools import combinations
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
JOURNAL = ROOT / "data/live/trade_journal.jsonl"


def load_intervals(journal_path: Path = JOURNAL) -> pd.DataFrame:
    rows = [json.loads(line) for line in journal_path.read_text().splitlines() if line.strip()]
    for r in rows:
        r["asset"] = (r.get("asset") or r.get("symbol") or "eth").lower()
    df = pd.DataFrame(rows)[["ts", "kind", "side", "asset"]]
    df["ts"] = pd.to_datetime(df["ts"])
    df = df.sort_values("ts")

    now = df["ts"].max()
    intervals: list[dict] = []
    open_by_asset: dict[str, pd.Series] = {}
    for _, r in df.iterrows():
        a = r["asset"]
        if r["kind"] == "OPEN":
            open_by_asset[a] = r
        elif r["kind"] == "CLOSE":
            o = open_by_asset.pop(a, None)
            if o is not None:
                intervals.append({"asset": a, "side": o["side"], "open_ts": o["ts"], "close_ts": r["ts"], "still_open": False})
    for a, o in open_by_asset.items():
        intervals.append({"asset": a, "side": o["side"], "open_ts": o["ts"], "close_ts": now, "still_open": True})

    return pd.DataFrame(intervals), df["ts"].min(), df["ts"].max()


def pairwise_overlap(iv: pd.DataFrame) -> pd.DataFrame:
    assets = sorted(iv["asset"].unique())
    rows = []
    for a, b in combinations(assets, 2):
        ivs_a, ivs_b = iv[iv["asset"] == a], iv[iv["asset"] == b]
        n_overlap = n_same_dir = 0
        overlap_dur = pd.Timedelta(0)
        for _, ra in ivs_a.iterrows():
            for _, rb in ivs_b.iterrows():
                lo, hi = max(ra["open_ts"], rb["open_ts"]), min(ra["close_ts"], rb["close_ts"])
                if lo < hi:
                    n_overlap += 1
                    overlap_dur += hi - lo
                    if ra["side"] == rb["side"]:
                        n_same_dir += 1
        rows.append({"pair": f"{a}-{b}", "overlap_events": n_overlap, "same_direction": n_same_dir,
                      "same_direction_pct": n_same_dir / n_overlap if n_overlap else float("nan"),
                      "overlap_duration": str(overlap_dur)})
    return pd.DataFrame(rows)


def main() -> None:
    iv, ts_min, ts_max = load_intervals()
    print(f"journal window: {ts_min} ~ {ts_max}\n")
    print("=== 재구성된 포지션 구간 ===")
    print(iv.to_string(index=False))

    print("\n=== 자산쌍별 겹침(pairwise overlap) ===")
    pw = pairwise_overlap(iv)
    print(pw.to_string(index=False))
    print(f"\nTOTAL: overlap_events={pw['overlap_events'].sum()}, same_direction={pw['same_direction'].sum()}")

    still_open = iv[iv["still_open"]]
    print("\n=== 현재 오픈 포지션(원장 마지막 시점 기준) ===")
    print(still_open[["asset", "side", "open_ts"]].to_string(index=False) if not still_open.empty else "없음")


if __name__ == "__main__":
    main()
