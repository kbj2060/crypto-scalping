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
    assert {"okx_ctx_all", "bo_sol", "hl_ctx_sol"} <= keys and "okx_ctx_sol" not in keys
    assert {"bn_tape_all", "okx_tape_all"} <= keys and not {"bn_tape_sol", "okx_tape_sol"} & keys
    tape = next(s for s in m.specs(["BTC", "SOL"], {2}) if s["key"] == "bn_tape_all")
    assert tape["env"]["TAPE_SYMBOL"] == "ethusdt,btcusdt,solusdt" and len(tape["fresh"]) == 3
