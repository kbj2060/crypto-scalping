"""옵션 요약(scripts/collect_deribit_option_gex_20260815.py::options_summary) 계산 점검 -- 네트워크 없이 합성 체인으로.
max pain · 25Δ 리스크 리버설 부호 · 콜/풋 미결제 · 감마 플립 없음(콜만 있는 체인) · 보험 OTM 목록."""
import sys, time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import collect_deribit_option_gex_20260815 as gex  # noqa: E402


def _chain():
    now = datetime.now(timezone.utc)
    exp = (now + timedelta(days=2)).replace(microsecond=0)
    rows = []
    for k in (1800, 1900, 2000, 2100, 2200):
        for t in ("call", "put"):
            iv = 50.0 + (8.0 if t == "put" and k < 2000 else 0.0)        # 풋 쪽 스큐 -> RR 음수
            rows.append({"recorded_at_utc": now, "currency": "ETH", "instrument_name": f"x{k}{t}", "option_type": t,
                         "strike": float(k), "expiration_ts": pd.Timestamp(exp), "days_to_expiry": 2.0,
                         "open_interest": 100.0 if k == 2000 else 10.0, "mark_iv": iv, "underlying_price": 2000.0,
                         "mark_price": 0.01, "volume": 0.0,
                         "gamma_bs": gex._bs_gamma(2000.0, k, iv, 2 / 365)})
    return pd.DataFrame(rows)


def test_options_summary_synthetic(monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("offline")
    monkeypatch.setattr(gex, "_pub", boom)                 # index/dvol/rv7 조회 실패 -> None, 계산은 계속
    o = gex.options_summary(_chain(), "ETH")
    assert o["index"] is None and o["dvol"] is None and o["rv7"] is None
    e = o["expiries"][0]
    assert e["pain"] == 2000.0, "미결제가 몰린 행사가가 max pain"
    assert e["rr25"] < 0, "풋 IV 가 높으면 리스크 리버설은 음수"
    assert abs(e["call_oi_usd"] - 140 * 2000) < 1e-6 and abs(e["pc"] - 1.0) < 1e-9
    assert "gamma" not in o and "gamma_by" not in o, "2026-10-02 딜러 감마 제거"
    assert all(h["type"] in ("P", "C") and h["usd"] > 0 for h in o["hedge"])
    assert {h["type"] for h in o["hedge"]} == {"P", "C"}, "지수 아래 풋 · 위 콜"
    fr = {r[0]: r for r in o["strikes"]["front"]}
    assert set(fr) == {1900.0, 2000.0, 2100.0}, "사다리 = 지수 ±8% 행사가만(2000 기준 1840~2160)"
    assert fr[2000.0][1] == fr[2000.0][2] == 100 * 2000 and all(len(r) == 3 for r in fr.values()), "행 = [행사가, 콜$, 풋$]"


def test_unrecoverable_gap_written_and_read(tmp_path, monkeypatch):
    """09-30 검증: 통합 수집기 write() 가 공백을 여러 건 기록한다(10-01 부터 커버 판정은 trade_seq 완결)."""
    import duckdb
    import live_deribit_block_trade_collector_20260928 as col
    monkeypatch.setattr(col, "DB", tmp_path / "o.duckdb")
    monkeypatch.setattr(col, "STATE_PATH", tmp_path / "st.json")
    row = lambda ts, n, d, a: (ts, f"t{ts}", 1, n, d, a, 0.01, 0.01, 2000.0, 50.0, 0, None, False, None, None, None, None, None)
    col.write([row(1000, "ETH-1OCT26-2700-C", "buy", 5.0), row(3000, "ETH-3OCT26-2700-P", "sell", 4.0)],
              [(500, 900, "ConnectionClosedError", 3), (1500, 2600, "unrecoverable ETH", 0)])
    con = duckdb.connect(str(tmp_path / "o.duckdb"))
    assert con.execute("SELECT count(*) FROM gaps").fetchone()[0] == 2


def test_flush_buf_keeps_rows_arriving_during_write_and_on_failure(monkeypatch):
    """09-30 검증: 체인 폴링 직전 flush. 쓰는 동안 들어온 체결은 남고, 쓰기 실패면 버퍼 그대로."""
    import asyncio
    import live_deribit_block_trade_collector_20260928 as col
    r = lambda k: (k,) * 12 + (False,)           # 13번째 칸 = is_block
    col._BUF[:] = [r("a"), r("b")]
    def w(rows, gaps=None):
        col._BUF.append(r("late"))               # 쓰는 사이 WS 가 붙인 행
        return len(rows)
    monkeypatch.setattr(col, "write", w)
    n, added, _ = asyncio.run(col.flush_buf())
    assert n == 2 and col._BUF == [r("late")]
    monkeypatch.setattr(col, "write", lambda rows, gaps=None: (_ for _ in ()).throw(OSError("locked")))
    try:
        asyncio.run(col.flush_buf())
    except OSError:
        pass
    assert col._BUF == [r("late")], "실패하면 지우지 않는다"
    col._BUF.clear()


def test_constant_maturity_skew(monkeypatch):
    """고정만기 RR/BF: 델타 보간(외가격만) + 만기 사이 시간 보간 · 양쪽 만기가 없으면 None(외삽 금지)."""
    monkeypatch.setattr(gex, "_pub", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("offline")))
    now = datetime.now(timezone.utc)
    def exp_rows(days, skew):
        exp = (now + timedelta(days=days)).replace(microsecond=0); rows = []
        for k in range(1500, 2525, 25):
            for t in ("call", "put"):
                iv = 50.0 + (skew if t == "put" and k < 2000 else 0.0) * (2000 - k) / 500   # 풋 날개로 갈수록 비싸다
                rows.append({"recorded_at_utc": now, "currency": "ETH", "instrument_name": f"{days}-{k}{t}", "option_type": t,
                             "strike": float(k), "expiration_ts": pd.Timestamp(exp), "days_to_expiry": float(days),
                             "open_interest": 10.0, "mark_iv": iv, "underlying_price": 2000.0, "mark_price": 0.01, "volume": 0.0,
                             "gamma_bs": gex._bs_gamma(2000.0, k, iv, days / 365)})
        return rows
    ch = pd.DataFrame(exp_rows(3, 4.0) + exp_rows(10, 8.0) + exp_rows(40, 8.0))
    o = gex.options_summary(ch, "ETH")
    e3, e10 = o["expiries"][0], o["expiries"][1]
    assert e3["rr25i"] < 0 and e10["rr25i"] < e3["rr25i"], "풋 스큐가 크면 리버설이 더 음수"
    assert abs(e3["atm_i"] - 50.0) < 1e-9 and e3["bf25i"] > 0
    cm7 = o["cm"]["7"]
    w = (7 * 24 - e3["hours"]) / (e10["hours"] - e3["hours"])
    assert abs(cm7["rr"] - ((1 - w) * e3["rr25i"] + w * e10["rr25i"])) < 1e-9, "7일 = 3일·10일 만기 시간 보간"
    assert abs(cm7["atm"] - 50.0) < 1e-6
    assert o["cm"]["30"] is not None and o["cm"]["60"] is None, "60일보다 먼 만기(40일이 끝)가 없으면 None"
    assert abs(sum(e["oi_share"] for e in o["expiries"]) - 1.0) < 1e-9
    only_short = gex.options_summary(pd.DataFrame(exp_rows(3, 4.0)), "ETH")
    assert only_short["cm"]["7"] is None, "7일보다 먼 만기가 없으면 외삽하지 않는다"


def test_surface_stats_flat_smile():
    """평평한 스마일(IV 60%, 7일)이면 모델프리 IV ≈ 60, 꼬리확률 ≈ 로그정규 값, 위험중립 첨도 ≈ 3 · 호가 폭 IV pt 환산."""
    import math
    yrs, F = 7 / 365, 2000.0
    ks = [float(k) for k in range(1200, 2825, 25)]
    g = pd.DataFrame([{"option_type": t, "strike": k, "mark_iv": 60.0, "bid_price": 0.0, "ask_price": 0.0} for k in ks for t in ("call", "put")])
    st = gex._surface_stats(g, F, yrs)
    assert abs(st["mfiv"] - 60.0) < 0.6, st["mfiv"]
    s = 0.6 * math.sqrt(yrs)
    N = lambda x: 0.5 * (1 + math.erf(x / math.sqrt(2)))   # noqa: E731
    p5 = N((math.log(0.95) + 0.5 * s * s) / s) + 1 - N((math.log(1.05) + 0.5 * s * s) / s)
    assert abs(st["tail"]["5"] - p5) < 0.01, (st["tail"], p5)
    assert st["tail"]["2"] > st["tail"]["3"] > st["tail"]["5"] > 0
    assert abs(st["rn_kurt"] - 3.0) < 0.3 and abs(st["rn_skew"]) < 0.3, st
    # 풋 날개가 비싸면 왜도가 음수
    sk = g.assign(mark_iv=[60.0 + (20.0 * (2000 - k) / 800 if k < 2000 else 0.0) for k in g["strike"]])
    assert gex._surface_stats(sk, F, yrs)["rn_skew"] < st["rn_skew"]
    # 호가 폭: ATM 콜 하나, 매도−매수 0.001 ETH(역옵션) ÷ 베가 → IV pt
    one = pd.DataFrame([{"option_type": "call", "strike": 2000.0, "mark_iv": 60.0, "bid_price": 0.020, "ask_price": 0.021}])
    vega = F * math.exp(-0.5 * (0.5 * s) ** 2) / math.sqrt(2 * math.pi) * math.sqrt(yrs) / 100
    assert abs(gex._atm_spread_iv(one, F, yrs, False) - 0.001 * F / vega) < 1e-9


def test_write_state_opt_hist(tmp_path, monkeypatch):
    """2026-10-01 VoV(시간별 DVOL 로그 변화 SD, %) · 만기 ATM 미결제 분위(지난 30일 07:00~07:10 UTC 값 대비)."""
    import json, math
    import duckdb
    con = duckdb.connect(":memory:")
    gex.ensure_tables(con)
    monkeypatch.setattr(gex, "STATE_PATH", tmp_path / "st.json")
    now = datetime.now(timezone.utc)
    for h in range(24, -1, -1):                                   # DVOL 50·51 번갈아 → 로그 변화 ±0.0198
        con.execute("INSERT INTO option_summary VALUES (?, 'ETH', 2000, ?, NULL, NULL, NULL, NULL, NULL, '{}')",
                    [now - timedelta(hours=h, minutes=1), 50.0 if h % 2 else 51.0])
    for d in range(1, 11):                                        # 지난 10일 07:05 UTC 의 ATM 미결제 = 1..10
        t = (now - timedelta(days=d)).replace(hour=7, minute=5, second=0, microsecond=0)
        con.execute("INSERT INTO option_summary VALUES (?, 'ETH', 2000, 50, NULL, NULL, NULL, NULL, NULL, ?)",
                    [t, json.dumps({"front_atm_oi_usd": float(d)})])
    con.execute("INSERT INTO option_summary VALUES (?, 'ETH', 2000, 50, NULL, NULL, NULL, NULL, NULL, ?)",
                [now, json.dumps({"front_atm_oi_usd": 7.5})])
    gex.write_state(con)
    st = json.loads((tmp_path / "st.json").read_text())["currencies"]["ETH"]
    oh = st["opt_hist"]
    assert st["spot_price"] == 2000 and "dex_1h_ago" not in st and "total_gex_usd" not in st
    assert oh["atm_oi_n"] == 10 and abs(oh["atm_oi_pct"] - 0.7) < 1e-9, oh
    assert oh["vov24"] is not None and abs(oh["vov24"] - math.log(51 / 50) * 100) < 0.2, oh
