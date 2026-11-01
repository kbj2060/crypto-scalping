#!/usr/bin/env python3
"""횡보/추세 분별 전진 기록 크론 (2026-11-02). 매 시 02분: 4코인 5분봉(닫힌 봉만)을 서버에서 받아 예측 기록 + 4시간 지난 행 채점 + 요약 JSON.
거래 로직과 무관(읽기 전용 시장 데이터 → data/live/regime_band_forward*.jsonl). 실패 시 아무것도 쓰지 않고 종료코드 1.
⚠️ 이 파일은 서버에서만 실행한다(로컬 REST 호출은 서버와 IP 한도를 공유해 밴 위험). 로컬 테스트는 fetch 주입(test/test_live_regime_band_cron_20261102.py).
  python scripts/live_regime_band_cron_20261102.py            (크론)     python scripts/live_regime_band_cron_20261102.py --dry   (기록 없이 예측만 출력)
"""
from __future__ import annotations

import json
import sys
import time
import urllib.request
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import live_regime_band_recorder_20261102 as R  # noqa: E402

S = R.S
SUMMARY = S.ROOT / "data" / "live" / "regime_band_forward_summary.json"
NEED = 2700                                       # 서빙 요구 ≥ 2500봉 + 여유
BASE = "https://fapi.binance.com/fapi/v1/klines"


def rest_klines(symbol: str, end_ms: int | None = None, limit: int = 1500) -> list:
    url = f"{BASE}?symbol={symbol}&interval=5m&limit={limit}" + (f"&endTime={end_ms}" if end_ms else "")
    with urllib.request.urlopen(url, timeout=20) as r:
        return json.loads(r.read())


def panel(symbol: str, now_ms: int, fetch=rest_klines) -> pd.DataFrame:
    rows = fetch(symbol)
    rows = fetch(symbol, rows[0][0] - 1) + rows
    df = pd.DataFrame(rows, columns=["open_time", "open", "high", "low", "close", "volume", "close_time", "quote_volume", "trades", "taker_buy_base", "taker_buy_quote", "ignore"])
    df = df.drop_duplicates("open_time").sort_values("open_time")
    df = df[df.close_time < now_ms]                                   # 형성 중인 봉 제외(마감 봉만)
    df["timestamp"] = pd.to_datetime(df.open_time, unit="ms")
    return S.from_klines(df.tail(NEED).reset_index(drop=True))


def main(dry: bool = False, now_ms: int | None = None, fetch=rest_klines, log=R.LOG, scored=R.SCORED, summary=SUMMARY) -> dict:
    now_ms = now_ms or int(time.time() * 1000)
    panels = {c: panel(f"{c}USDT", now_ms, fetch) for c in R.COINS}
    last = {c: p.index[-1] for c, p in panels.items()}
    assert len({t.minute for t in last.values()}) == 1 and all(t.minute == 55 for t in last.values()), f"마지막 봉이 정시 마감 봉이 아님: {last}"
    if dry:
        return {c: {k: v for k, v in S.predict_latest(panels, c).items() if k != "features"} for c in R.COINS}
    new = R.record(panels, log)
    n_scored = R.score_new(panels, log, scored)
    s = R.summarize(scored)
    summary.parent.mkdir(parents=True, exist_ok=True)
    summary.write_text(json.dumps(s, ensure_ascii=False, indent=1))
    return dict(recorded=len(new), newly_scored=n_scored, n_scored_total=s.get("n_scored", 0), decision_ts=str(last["XRP"] + pd.Timedelta(minutes=5)))


if __name__ == "__main__":
    try:
        print(json.dumps(main(dry="--dry" in sys.argv), ensure_ascii=False, default=str))
    except Exception as e:                      # 실패는 조용히 쓰지 않고 종료코드로 알린다
        print("FAIL", type(e).__name__, e, file=sys.stderr)
        sys.exit(1)
