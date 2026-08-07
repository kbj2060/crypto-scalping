from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_supervisor_runs_only_non_executing_shadow_services() -> None:
    source = (ROOT / "scripts/supervise_micro_scalp_shadows_20260718.sh").read_text()

    assert "run_eth_micro_scalp_v4_shadow_bot_20260718.py" in source
    assert "run_btc_sol_micro_scalp_shadow_bot_20260718.py" in source
    assert "run_micro_scalp_reuse_shadow_bot_20260718.py" in source
    assert "run_hexa_pulse_formula_shadow_20260718.py" in source
    assert "--interval-seconds 300" in source
    assert "trading_bot.py" not in source
    assert "binance_execution" not in source
