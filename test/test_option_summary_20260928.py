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
    assert o["gamma"]["now_usd"] != 0 and len(o["gamma"]["profile"]) == 25
    assert all(h["type"] in ("P", "C") and h["usd"] > 0 for h in o["hedge"])
    assert {h["type"] for h in o["hedge"]} == {"P", "C"}, "지수 아래 풋 · 위 콜"
    fr = {r[0]: r for r in o["strikes"]["front"]}
    assert set(fr) == {1900.0, 2000.0, 2100.0}, "사다리 = 지수 ±8% 행사가만(2000 기준 1840~2160)"
    assert fr[2000.0][1] == fr[2000.0][2] == 100 * 2000 and fr[2000.0][3] < fr[2100.0][3] + 1e9


def test_gamma_uses_nearest_expiry_only(monkeypatch):
    """2026-09-29: 딜러 감마 = 가장 가까운 만기 하나. 먼 만기에 콜을 잔뜩 쌓아도 now_usd 부호가 안 바뀐다."""
    monkeypatch.setattr(gex, "_pub", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("offline")))
    near = _chain()
    near.loc[near.option_type == "call", "open_interest"] = 1.0          # 가까운 만기는 풋 우세 -> 음감마
    far = _chain().assign(expiration_ts=near.expiration_ts.iloc[0] + pd.Timedelta(days=60), days_to_expiry=62.0)
    far = far[far.option_type == "call"].assign(open_interest=1e5)       # 먼 만기 콜 산더미(+)
    o = gex.options_summary(pd.concat([near, far], ignore_index=True), "ETH")
    only = gex.options_summary(near, "ETH")
    assert o["gamma"]["now_usd"] < 0 and abs(o["gamma"]["now_usd"] - only["gamma"]["now_usd"]) < 1e-6
    assert o["gamma"]["exp_ms"] == int(near.expiration_ts.iloc[0].timestamp() * 1000)


def test_gamma_by_follows_ladder_scopes(monkeypatch):
    """2026-09-29 사용자 «사다리 칩 따라가게»: gamma_by 는 사다리 칩과 같은 범위. 먼 만기 콜 산더미는 all 만 +로 뒤집는다."""
    monkeypatch.setattr(gex, "_pub", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("offline")))
    near = _chain()
    near.loc[near.option_type == "call", "open_interest"] = 1.0
    far = _chain().assign(expiration_ts=near.expiration_ts.iloc[0] + pd.Timedelta(days=60), days_to_expiry=62.0)
    far = far[far.option_type == "call"].assign(open_interest=1e5)
    o = gex.options_summary(pd.concat([near, far], ignore_index=True), "ETH")
    gb = o["gamma_by"]
    assert gb["front"]["now_usd"] == o["gamma"]["now_usd"] < 0, "front = 이전 gamma 키"
    assert abs(gb["week"]["now_usd"] - gb["front"]["now_usd"]) < 1e-6, "7일 안 = 가까운 만기뿐(먼 만기 62일 제외)"
    assert gb["all"]["now_usd"] > 0, "전 만기는 먼 콜로 양수"
    assert len(gb["all"]["profile"]) == 25


def test_dex_holder_sign_and_scale(monkeypatch):
    """2026-09-29 미결제 기반 DEX(보유자 기준): 콜만 → +, 풋만 → −, ATM 콜 100개 ≈ 0.5 × 100 × 2000."""
    monkeypatch.setattr(gex, "_pub", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("offline")))
    ch = _chain()
    calls = gex.options_summary(ch.assign(open_interest=ch.open_interest.where(ch.option_type == "call", 0.0)), "ETH")
    puts = gex.options_summary(ch.assign(open_interest=ch.open_interest.where(ch.option_type == "put", 0.0)), "ETH")
    assert calls["gamma"]["dex_usd"] > 0 > puts["gamma"]["dex_usd"]
    atm = ch[(ch.strike == 2000) & (ch.option_type == "call")]
    d = gex.options_summary(atm, "ETH")["gamma"]["dex_usd"]
    assert 0.45 * 100 * 2000 < d < 0.56 * 100 * 2000, d
    prof = calls["gamma_by"]["all"]["profile"]
    assert len(prof[0]) == 6 and prof[0][2] < prof[-1][2], "가격이 오르면 콜 델타(DEX)가 커진다 (행 = 가격·감마·보유자·딜러가정·딜러체결 DEX·딜러체결 감마)"
    # 2026-09-29 딜러 가정(콜 매수·풋 매도): 풋만 있는 체인에서 보유자 DEX 는 −, 딜러 가정 DEX 는 + (풋을 판 딜러는 롱 델타)
    assert puts["gamma"]["dex_asm_usd"] > 0 > puts["gamma"]["dex_usd"]
    assert abs(calls["gamma"]["dex_asm_usd"] - calls["gamma"]["dex_usd"]) < 1e-6, "콜만이면 두 기준이 같다"


def test_write_state_dex_1h_ago(tmp_path, monkeypatch):
    """2026-09-29 ΔDEX: write_state 가 50~75분 전 스냅샷의 칩별 DEX 를 dex_1h_ago 로 싣는다(5분 전 것은 안 쓴다)."""
    import json
    import duckdb
    con = duckdb.connect(":memory:")
    gex.ensure_tables(con)
    monkeypatch.setattr(gex, "STATE_PATH", tmp_path / "st.json")
    now = datetime.now(timezone.utc)
    con.execute("INSERT INTO gex_summary VALUES (?, 'ETH', 2000, 1e6, 1e5, 10, 5)", [now])
    for mins, dex in ((60, 111.0), (5, 999.0), (0, 500.0)):
        pay = {"gamma_by": {"week": {"now_usd": 1.0, "dex_usd": dex, "dealer_gex_usd": -dex, "dealer_cov": dex / 1000},
                            "front": {"now_usd": 1.0, "dex_usd": dex, "exp_ms": 7}}}
        con.execute("INSERT INTO option_summary VALUES (?, 'ETH', 2000, NULL, NULL, NULL, NULL, NULL, NULL, ?)",
                    [now - timedelta(minutes=mins), json.dumps(pay)])
    gex.write_state(con)
    st = json.loads((tmp_path / "st.json").read_text())["currencies"]["ETH"]
    assert st["dex_1h_ago"]["week"]["dex_usd"] == 111.0 and st["dex_1h_ago"]["front"]["exp_ms"] == 7
    assert st["dex_1h_ago"]["week"]["dealer_gex_usd"] == -111.0 and st["dex_1h_ago"]["week"]["dealer_cov"] == 0.111
    assert st["options"]["gamma_by"]["week"]["dex_usd"] == 500.0


def test_dealer_dex_from_taker_flow(monkeypatch):
    """2026-09-29 체결 기반 딜러 DEX: 수집 뒤 상장(covered) 종목만. 테이커가 ATM 콜 50개 순매수 → 딜러 −50콜 → 딜러 DEX ≈ −0.5×50×2000.
    커버 = 그 종목 미결제 비중 · |테이커 순|/미결제 = 50/100. flow 가 없으면 딜러 키 자체가 없다(옛 단독 수집기)."""
    monkeypatch.setattr(gex, "_pub", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("offline")))
    ch = _chain()
    atm_c = "x2000call"
    flow = {"net": {atm_c: 50.0, "x2000put": 999.0}, "covered": {atm_c}}     # 풋은 커버 밖 -- 순수량이 있어도 안 센다
    g = gex.options_summary(ch, "ETH", flow)["gamma"]
    assert -0.56 * 50 * 2000 < g["dealer_dex_usd"] < -0.45 * 50 * 2000, g["dealer_dex_usd"]
    assert abs(g["dealer_cov"] - 100 / ch.open_interest.sum()) < 1e-9
    assert abs(g["dealer_net_oi"] - 0.5) < 1e-9
    assert "dealer_dex_usd" not in gex.options_summary(ch, "ETH")["gamma"]
    none_cov = gex.options_summary(ch, "ETH", {"net": {atm_c: 50.0}, "covered": set()})["gamma"]
    assert none_cov["dealer_dex_usd"] is None and none_cov["dealer_cov"] == 0


def test_taker_flow_sql_and_covered(monkeypatch):
    """option_trades 에서 종목별 테이커 순수량 · trade_seq 가 1 부터 끊김 없으면 covered(2026-10-01). 접두어 밖(BTC) 종목은 안 섞인다."""
    import duckdb
    con = duckdb.connect()
    con.execute("CREATE TABLE option_trades (ts_ms BIGINT, trade_seq BIGINT, instrument_name VARCHAR, direction VARCHAR, amount DOUBLE)")
    con.executemany("INSERT INTO option_trades VALUES (?, ?, ?, ?, ?)", [
        (1000, 1, "ETH-1OCT26-2700-C", "buy", 5.0), (2000, 2, "ETH-1OCT26-2700-C", "sell", 2.0),     # 1·2 = 완결
        (3000, 3, "ETH-3OCT26-2700-P", "sell", 4.0),                                                # 첫 건 없음(수집 뒤부터)
        (3100, 1, "ETH-4OCT26-2700-P", "buy", 1.0), (3200, 3, "ETH-4OCT26-2700-P", "buy", 1.0),     # 가운데 2 가 빠짐
        (1500, 1, "BTC-1OCT26-60000-C", "buy", 9.0)])
    f = gex._taker_flow(con, "ETH")
    assert f["net"] == {"ETH-1OCT26-2700-C": 3.0, "ETH-3OCT26-2700-P": -4.0, "ETH-4OCT26-2700-P": 2.0}
    assert f["covered"] == {"ETH-1OCT26-2700-C"}, "첫 건이 없거나 중간이 빠진 종목은 수준을 모른다"

def test_dealer_gex_from_taker_flow(monkeypatch):
    """2026-09-29 체결 기반 딜러 GEX: 테이커가 ATM 콜 50개 순매수 → 딜러 콜 숏 → 음감마(딜러 가정 «콜 +»와 반대 부호).
    곡선 6번째 칸의 지금 가격 점 = dealer_gex_usd · 부호가 한쪽뿐이면 dealer_flip 없음 · flow 가 없으면 키 없음."""
    monkeypatch.setattr(gex, "_pub", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("offline")))
    ch = _chain()
    g = gex.options_summary(ch, "ETH", {"net": {"x2000call": 50.0}, "covered": {"x2000call"}})["gamma"]
    assert g["dealer_gex_usd"] < 0, g["dealer_gex_usd"]
    expect = -50 * gex._bs_gamma(2000.0, 2000.0, 50.0, 2 / 365) * 2000.0 ** 2 * 0.01
    assert abs(g["dealer_gex_usd"] - expect) < 1e-6 * abs(expect)
    mid = g["profile"][12]
    assert abs(mid[0] - 2000.0) < 0.1 and abs(mid[5] - g["dealer_gex_usd"]) < 1e-9
    assert g["dealer_flip"] is None
    assert "dealer_gex_usd" not in gex.options_summary(ch, "ETH")["gamma"]


def test_ladder_dealer_trade_gamma_column(monkeypatch):
    """사다리 행 5번째 칸 = 딜러·체결 순감마: 커버 종목이 있는 행사가만 값(테이커 콜 매수 → 음수), 나머지는 None."""
    monkeypatch.setattr(gex, "_pub", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("offline")))
    o = gex.options_summary(_chain(), "ETH", {"net": {"x2000call": 50.0}, "covered": {"x2000call"}})
    fr = {r[0]: r for r in o["strikes"]["front"]}
    assert fr[2000.0][4] < 0 and fr[1900.0][4] is None and fr[2100.0][4] is None
    assert all(len(r) == 6 and r[4] is None and r[5] == 0 for r in gex.options_summary(_chain(), "ETH")["strikes"]["front"])
    assert fr[2000.0][5] == 0.5 and fr[1900.0][5] == 0, "2000 = 콜(커버)·풋(미커버) 미결제 같음 → 0.5"


def test_dealer_charm_1h_sign_and_scale(monkeypatch):
    """2026-09-29 charm: 시간만 1시간 흐를 때 딜러 델타 변화($, 체결 기반 딜러 포지션). 테이커가 콜 50개 순매수 → 딜러 콜 숏.
    외가격 콜은 델타가 0 쪽으로 줄어 딜러 델타가 올라간다(+, 헤지 매도) · 내가격 콜은 1 쪽으로 늘어 내려간다(−, 헤지 매수).
    크기 = −50 × (Δ(τ−1h) − Δ(τ)) × 가격 -- _bs_delta 와 같은 식. flow 가 없으면 키 없음."""
    monkeypatch.setattr(gex, "_pub", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("offline")))
    ch = _chain()
    otm = gex.options_summary(ch, "ETH", {"net": {"x2100call": 50.0}, "covered": {"x2100call"}})["gamma"]["dealer_charm_1h_usd"]
    itm = gex.options_summary(ch, "ETH", {"net": {"x1900call": 50.0}, "covered": {"x1900call"}})["gamma"]["dealer_charm_1h_usd"]
    assert otm > 0 > itm, (otm, itm)
    t, h = 2 / 365, 1 / 8760
    expect = -50 * (gex._bs_delta(2000.0, 2100.0, 50.0, t - h, True) - gex._bs_delta(2000.0, 2100.0, 50.0, t, True)) * 2000.0
    assert abs(otm - expect) < 1e-6 * abs(expect), (otm, expect)
    assert "dealer_charm_1h_usd" not in gex.options_summary(ch, "ETH")["gamma"]


def test_greeks_use_forward_and_profile_keeps_small_prices(monkeypatch):
    """09-30 검증: ① 그릭스는 만기별 선도가 -- 선도가가 지수보다 높으면 콜 델타(보유자 DEX)가 지수 기준보다 커진다.
    ② XRP 처럼 가격이 1.5 면 곡선 가격 25점이 전부 달라야 한다(소수 1자리 반올림 계단 금지). ③ exps·dealer_n 이 나온다."""
    def pub(method, **k):              # 지수만 2000 으로 고정(선도가 효과만 떼어 본다), 나머지 조회는 실패
        if method == "get_index_price":
            return {"index_price": 2000.0}
        raise RuntimeError("offline")
    monkeypatch.setattr(gex, "_pub", pub)
    ch = _chain()
    calls = ch[ch.option_type == "call"].assign(days_to_expiry=90.0)
    lo = gex.options_summary(calls, "ETH")["gamma_by"]["all"]["dex_usd"]                                   # 선도 = 지수 2000
    hi = gex.options_summary(calls.assign(underlying_price=2040.0), "ETH")["gamma_by"]["all"]["dex_usd"]   # 선도 +2%
    assert hi > lo * 1.02, (lo, hi)     # 지수로 재던 옛 코드는 hi == lo 였다
    x = ch.assign(strike=ch.strike / 1000 * 0.75, underlying_price=1.5, gamma_bs=0.0)
    prof = gex.options_summary(x, "XRP")["gamma_by"]["all"]["profile"]
    assert len({r[0] for r in prof}) == 25, [r[0] for r in prof]
    g = gex.options_summary(ch, "ETH", {"net": {"x2000call": 5.0}, "covered": {"x2000call"}})["gamma_by"]["week"]
    assert g["dealer_n"] == 1 and len(g["exps"]) == 1


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
    # 2026-10-01 커버는 공백 기록이 아니라 trade_seq 완결로 정한다 -- 둘 다 seq 1 한 건뿐이라 완결
    assert gex._taker_flow(con, "ETH")["covered"] == {"ETH-1OCT26-2700-C", "ETH-3OCT26-2700-P"}


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


def test_summarize_gex_spot_is_nearest_expiry_forward():
    """09-30 검증: gex_summary 의 spot 은 API 첫 행이 아니라 가장 가까운 만기의 선도가."""
    ch = _chain()
    far = ch.assign(days_to_expiry=90.0, underlying_price=2100.0)
    mixed = pd.concat([far, ch], ignore_index=True)        # 첫 행이 먼 만기
    assert gex.summarize_gex(mixed, "ETH")["spot_price"] == 2000.0


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
