"""2026-10-31 디스크 정책: DEEP 밖 코인은 가벼운 단계만, 체결 테이프는 코인 수와 무관하게 프로세스 둘."""
import importlib.util

sp = importlib.util.spec_from_file_location("m", "scripts/ops/multicoin_collectors_20260926.py")
m = importlib.util.module_from_spec(sp)
sp.loader.exec_module(m)


def test_light_tier_and_grouped_tape():
    m.DEEP = ("BTC",)
    keys = {s["key"] for s in m.specs(["BTC", "SOL"], {1, 2, 3})}
    assert {"bn_depthdiff_btc", "bn_raster_btc", "hl_trades_btc"} <= keys
    assert not {"bn_depthdiff_sol", "bn_raster_sol", "hl_trades_sol", "hl_bbo_sol"} & keys
    assert {"okx_ctx_all", "bo_all", "hl_ctx_sol"} <= keys and "bo_sol" not in keys and "okx_ctx_sol" not in keys
    assert {"bn_tape_all", "okx_tape_all"} <= keys and not {"bn_tape_sol", "okx_tape_sol"} & keys
    tape = next(s for s in m.specs(["BTC", "SOL"], {2}) if s["key"] == "bn_tape_all")
    assert tape["env"]["TAPE_SYMBOL"] == "ethusdt,btcusdt,solusdt" and len(tape["fresh"]) == 3
    bo = next(s for s in m.specs(["BTC", "SOL"], {3}) if s["key"] == "bo_all")
    assert bo["env"]["BO_SYMBOL"] == "ETHUSDT,SOLUSDT"      # BTC 는 검정 밖 -- 켜지 않는다


def test_group_survives_one_bad_ctval_and_partial_start_is_blocked(monkeypatch, tmp_path, capsys):
    """ctVal 점검이 한 종목만 실패하면 그 종목만 그룹에서 빠지고, 매니페스트와 다른 부분 start 는 보류된다."""
    import argparse
    import json

    monkeypatch.setattr(m, "ROOT", tmp_path)
    monkeypatch.setattr(m, "MANIFEST", tmp_path / "mc.json")
    monkeypatch.setattr(m, "check_okx_ct_vals", lambda insts: {i: ("bad" if i.startswith("HYPE") else None) for i in insts})
    monkeypatch.setattr(m, "raster_bins", lambda coins: {c: 1.0 for c in coins})
    monkeypatch.setattr(m, "pids_of", lambda runner, match: [])
    launched = []
    monkeypatch.setattr(m, "launch", lambda s, py: launched.append(s) or "기동")
    monkeypatch.setattr(m, "mem_available_mb", lambda: 1e6)
    ns = lambda coins: argparse.Namespace(coins=coins, phase="2", force=True, install_cron=False)  # noqa: E731
    m.DEEP = ()
    assert m.cmd_plan_or_start(ns("BTC,HYPE"), True) == 0
    ok = json.loads(m.MANIFEST.read_text())["collectors"]["okx_tape_all"]
    assert ok["env"]["OKX_TAPE_INST"] == "ETH-USDT-SWAP,BTC-USDT-SWAP" and len(ok["fresh"]) == 2
    launched.clear()
    m.cmd_plan_or_start(ns("BTC,HYPE"), True)    # 같은 목록으로 다시 하면 통과
    assert "그룹 종목 목록이" not in capsys.readouterr().out
    m.cmd_plan_or_start(ns("BTC,SOL"), True)
    assert "그룹 종목 목록이" in capsys.readouterr().out


def test_tradfi_stock_gets_tape_and_depth_only():
    """SNDK(TradFi 주식): 체결 테이프·OKX·호가 diff 만. HL·래스터는 없고 야간 신선도 기준이 넓다."""
    m.DEEP = ()
    specs = m.specs(["ZEC", "SNDK"], {1, 2, 3})
    keys = {s["key"] for s in specs}
    assert "bn_depthdiff_sndk" in keys and not {"bn_depthdiff_zec", "bn_raster_sndk", "hl_trades_sndk", "hl_ctx_sndk"} & keys
    assert "hl_ctx_zec" in keys
    tape = next(s for s in specs if s["key"] == "bn_tape_all")
    assert tape["env"]["TAPE_SYMBOL"] == "ethusdt,zecusdt,sndkusdt"
    assert [f["warn"] for f in tape["fresh"]] == [5, 5, 15]
    okx = next(s for s in specs if s["key"] == "okx_tape_all")
    assert [f["warn"] for f in okx["fresh"]] == [10, 10, 20]
    assert {s["key"] for s in m.specs(["SNDK"], {1})} == {"bn_depthdiff_sndk"}


def test_hl_positions_only_for_deep_coins():
    """HL 포지션 대상은 HL 체결 DB 를 쌓는 DEEP 코인뿐 -- 가벼운 코인(ZEC)이 끼면 프로세스 env·파일 이름이 바뀌어 중복 수집이 된다."""
    m.DEEP = ("BTC",)
    pos = [s for s in m.specs(["BTC", "ZEC", "SNDK"], {1}) if s["key"].startswith("hl_positions")]
    assert [s["env"]["HL_POS_COINS"] for s in pos] == ["BTC"]
