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
        pay = {"gamma_by": {"week": {"now_usd": 1.0, "dex_usd": dex}, "front": {"now_usd": 1.0, "dex_usd": dex, "exp_ms": 7}}}
        con.execute("INSERT INTO option_summary VALUES (?, 'ETH', 2000, NULL, NULL, NULL, NULL, NULL, NULL, ?)",
                    [now - timedelta(minutes=mins), json.dumps(pay)])
    gex.write_state(con)
    st = json.loads((tmp_path / "st.json").read_text())["currencies"]["ETH"]
    assert st["dex_1h_ago"]["week"]["dex_usd"] == 111.0 and st["dex_1h_ago"]["front"]["exp_ms"] == 7
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
    """option_trades 에서 종목별 테이커 순수량, 상장 시각 ≥ 첫 체결이면 covered. 접두어 밖(BTC) 종목은 안 섞인다."""
    import duckdb
    con = duckdb.connect()
    con.execute("CREATE TABLE option_trades (ts_ms BIGINT, instrument_name VARCHAR, direction VARCHAR, amount DOUBLE)")
    con.executemany("INSERT INTO option_trades VALUES (?, ?, ?, ?)", [
        (1000, "ETH-1OCT26-2700-C", "buy", 5.0), (2000, "ETH-1OCT26-2700-C", "sell", 2.0),
        (3000, "ETH-3OCT26-2700-P", "sell", 4.0), (1500, "BTC-1OCT26-60000-C", "buy", 9.0)])
    gex._CREATED.clear()
    monkeypatch.setattr(gex, "_pub", lambda m, **k: [{"instrument_name": "ETH-1OCT26-2700-C", "creation_timestamp": 500},
                                                      {"instrument_name": "ETH-3OCT26-2700-P", "creation_timestamp": 2500}])
    f = gex._taker_flow(con, "ETH")
    assert f["net"] == {"ETH-1OCT26-2700-C": 3.0, "ETH-3OCT26-2700-P": -4.0}
    assert f["covered"] == {"ETH-3OCT26-2700-P"}, "첫 체결(1000) 전에 상장된 종목은 수준을 모른다"


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
