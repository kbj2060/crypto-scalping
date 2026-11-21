"""QX4I 옵션 ATM IV 받기(scripts/live_vol4h_qx4_20261025.py `fetch_option_trades`·`iv_now`) 정답지 (2026-11-22 · tests/qx4.rs qx4i_iv_now).
서버 qx4_payload 가 ETH·BTC 에 부르는 길: iv_now(코인별 시간 칸 기억, 지금 ms, deribit_get) → (iv, 정시) → 있으면 qx4i 칸 덧붙이기(add_iv 는 이미 대조됨).
가짜 deribit_get(합성 체결 · 쪽 나눔 has_more · 같은 trade_id 겹침 · 블록·청산·다른 종목·iv 없음 · 델타 밖 · 빈 시간 · 받기 실패 한 번) -- 네트워크 없음.
    (tools/server_env.sh 의 환경) python rtmodel/tools/gen_qx4i_fixture.py      # → tests/fixtures/qx4i_fixture.json
"""
import json
import sys
import urllib.parse
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
import live_vol4h_qx4_20261025 as Q  # noqa: E402

H1 = 3_600_000


def main():
    rng = np.random.default_rng(4)
    t0 = 1_795_100_000_000 // H1 * H1
    empty = {t0 - 2 * H1}                                                           # 이 시간엔 쓸 체결 없음 → 앞 시간으로
    trades = {}
    for coin, px in (("ETH", 3000.0), ("BTC", 90000.0)):
        rows, tid = [], 0
        for h in range(t0 - 12 * H1, t0 + 6 * H1, H1):
            for j in range(int(rng.integers(5, 40))):
                ts = h + int(rng.integers(0, H1))
                exp = datetime.fromtimestamp((h + int(rng.integers(1, 12)) * 86_400_000) / 1000, timezone.utc)
                k = round(px * float(np.exp(rng.normal(0, 0.05))) / (25 if coin == "ETH" else 500)) * (25 if coin == "ETH" else 500)
                name = f"{coin}-{exp.day}{exp.strftime('%b').upper()}{exp.strftime('%y')}-{k}-{'C' if rng.random() < .5 else 'P'}"
                if j % 17 == 3:
                    name = f"{coin}-PERPETUAL"
                tid += 1
                t = {"trade_id": f"{coin}-{tid}", "timestamp": ts, "instrument_name": name, "iv": None if j % 13 == 5 else round(float(rng.uniform(30, 90)), 2),
                     "index_price": px * float(np.exp(rng.normal(0, 0.002)))}
                if j % 11 == 7:
                    t["block_trade_id"] = "BLOCK-1"
                if j % 19 == 9:
                    t["liquidation"] = "M"
                if h in empty and coin == "ETH":
                    t["iv"] = None
                rows.append(t)
        rows.sort(key=lambda r: r["timestamp"])
        trades[coin] = rows
    calls, fail_at = [], {3}                                                         # 넷째 요청은 실패(None)

    def get(params):
        url = f"{Q.DERIBIT_TRADES}?{urllib.parse.urlencode(params)}"
        if len(calls) in fail_at:
            calls.append([url, None]); return None
        a, b = params["start_timestamp"], params["end_timestamp"]
        sel = [r for r in trades[params["currency"]] if a <= r["timestamp"] <= b][:7]   # 쪽 = 7개(원본은 1000 -- 이어 받기 고리를 시험)
        more = len([r for r in trades[params["currency"]] if a <= r["timestamp"] <= b]) > 7
        res = {"trades": sel + (sel[-1:] if more else []), "has_more": more}          # 마지막 체결을 한 번 더 넣음 = 같은 trade_id 겹침
        calls.append([url, res]); return res
    runs, state = [], {"ETH": {}, "BTC": {}}
    for now in (t0 + 30_000, t0 + 90_000, t0 + 90_000, t0 + 300_000, t0 + H1 + 120_000, t0 + 3 * H1 + 61_000):
        for coin in ("ETH", "BTC"):
            n0 = len(calls)
            iv, hour = Q.iv_now(state[coin], now, get, coin)
            runs.append({"now": now, "coin": coin, "calls": calls[n0:], "iv": iv, "hour": hour, "keys": sorted(state[coin])})
            print(now, coin, len(calls) - n0, iv, hour)
    (ROOT / "rtmodel/tests/fixtures/qx4i_fixture.json").write_text(json.dumps({"runs": runs}))


if __name__ == "__main__":
    main()
