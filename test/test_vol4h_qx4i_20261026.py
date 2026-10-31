"""QX4I 서빙(live_vol4h_qx4_20261025) = 연구(research_vol4h_round12_option_iv_20261025) 패리티. 연구 체결 파일이 없으면 그 시험만 건너뜀.
2026-10-31 BTC(12b 계수 · docs/experiments/btc_sol_vol4h_qx4_20261031.md) 추가."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import live_vol4h_qx4_20261025 as Q  # noqa: E402

TRADES = {"ETH": Path("/home/kbj20/crypto-scalping/tmp/deribit_optflow_20261025/trades"),
          "BTC": Path("/home/kbj20/crypto-scalping/tmp/vol4h_r12b_btc_20261026/trades")}
R12B = Path("/home/kbj20/crypto-scalping/tmp/vol4h_r12b_btc_20261026/report.json")
DAYS = ["2026-01-14", "2026-01-15", "2026-03-03", "2026-07-20", "2026-09-01", "2026-10-22"]   # 전진 창(10-24~) 전만


def test_qx4i_formula_matches_research():
    R12 = pytest.importorskip("research_vol4h_round12_option_iv_20261025")
    assert Q.BETA_I["ETH"] == R12.BETA_I
    if R12B.exists():
        assert Q.BETA_I["BTC"] == tuple(json.loads(R12B.read_text())["beta_I"])
    assert Q.qx4i_log(-7.0, 50.0, "BTC") != Q.qx4i_log(-7.0, 50.0)          # 코인별 계수가 실제로 쓰인다
    ts = pd.date_range("2026-03-01 00:55", periods=3, freq="1h")            # 5분봉 시작 = 정시 − 5분
    d = (ts + pd.Timedelta("5min")).astype("int64") // 10**6
    yh, iv = np.array([-7.1, -6.4, -8.0]), pd.Series([48.0, np.nan, 61.5], index=d)
    want = R12.qx4i(ts, yh, iv)
    got = [Q.qx4i_log(p, None if np.isnan(v) else v) for p, v in zip(yh, iv)]
    assert np.allclose(got, want, atol=1e-12, rtol=0)


def test_fetch_pages_and_dedup():
    pages = [dict(trades=[dict(trade_id=1, timestamp=10), dict(trade_id=2, timestamp=20)], has_more=True),
             dict(trades=[dict(trade_id=2, timestamp=20), dict(trade_id=3, timestamp=30)], has_more=False)]
    calls = []
    got = Q.fetch_option_trades(0, 100, lambda p: (calls.append(p["start_timestamp"]), pages[len(calls) - 1])[1])
    assert [x["trade_id"] for x in got] == [1, 2, 3] and calls == [0, 20]
    assert Q.fetch_option_trades(0, 100, lambda p: None) is None
    seen = []
    Q.fetch_option_trades(0, 100, lambda p: (seen.append(p["currency"]), pages[1])[1], "BTC")
    assert seen == ["BTC"]


@pytest.mark.parametrize("coin", ["ETH", "BTC"])
def test_atm_iv_and_ffill_match_research(coin):
    if not TRADES[coin].exists():
        pytest.skip("연구 체결 파일 없음")
    R12 = pytest.importorskip("research_vol4h_round12_option_iv_20261025")
    O = R12.O
    raw = pd.concat([pd.read_parquet(TRADES[coin] / f"{d}.parquet") for d in DAYS], ignore_index=True)
    tr = raw[raw.instrument_name.str.match(rf"^{coin}-\d+[A-Z]{{3}}\d{{2}}-\d+-[CP]$") & ~raw.block & ~raw.liq].reset_index(drop=True)   # 연구 iv_hourly 와 같은 준비
    exp = pd.to_datetime(tr.instrument_name.str.split("-").str[1], format="%d%b%y").astype("int64") // 10**6 + 8 * Q.H1
    tr["T"] = (exp - tr.timestamp) / 86_400_000; tr["d"] = O.deltas(tr); tr = tr[np.isfinite(tr.d)]
    tr["dnb"] = np.where(tr.direction == "buy", 1, -1) * tr.d * tr.amount
    grid = np.unique(np.concatenate([np.arange(pd.Timestamp(d).value // 10**6, pd.Timestamp(d).value // 10**6 + 86_400_000, Q.H1) + Q.H1 for d in DAYS]))
    want = R12.hourly(tr, grid).iv
    rows = [dict(timestamp=int(r.timestamp), instrument_name=r.instrument_name, iv=None if pd.isna(r.iv) else float(r.iv), index_price=float(r.index_price),
                 block_trade_id="b" if r.block else None, liquidation="M" if r.liq else None) for r in raw.itertuples()]
    by_hour = {}
    for r in rows:
        by_hour.setdefault((r["timestamp"] // Q.H1 + 1) * Q.H1, []).append(r)
    cache = {h: Q.atm_iv(by_hour.get(h, []), h, coin) for h in grid}
    n = 0
    for h in grid:
        got, _ = Q.iv_now({k: v for k, v in cache.items() if k <= h}, h + Q.SETTLE_MS, get=lambda p: pytest.fail("미리 채운 칸만 써야 함"), coin=coin)
        if h - 6 * Q.H1 < grid[0] or any(k not in cache for k in range(h - 6 * Q.H1, h + 1, Q.H1)):
            continue                                   # 날짜 경계(이어지지 않은 격자)는 ffill 비교 밖
        w = want.loc[h]
        assert (got is None and np.isnan(w)) or got == pytest.approx(w, abs=1e-12), (h, got, w)
        n += 1
    assert n > 100
