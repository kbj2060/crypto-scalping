from datetime import datetime, timedelta, timezone

import duckdb

from microstructure_scanner import MicrostructureScanner


def test_db_bootstrap_restores_causal_price_warmup(tmp_path, monkeypatch) -> None:
    database = tmp_path / "microstructure.duckdb"
    monkeypatch.setenv("QUANT_MICRO_DB_PATH", str(database))

    first = MicrostructureScanner(symbol="ethusdt")
    first._db_init()

    start = datetime.now(timezone.utc).replace(second=0, microsecond=0) - timedelta(minutes=29)
    connection = duckdb.connect(str(database))
    try:
        connection.executemany(
            """
            INSERT INTO microstructure_1m (
                ts, nif_whale, oi_delta_pct, shadow_absorption_score,
                shadow_toxicity_score, shadow_queue_bias, eai, mark_price
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (start + timedelta(minutes=index), 0.0, 0.0, 0.0, 0.0, 0, 0.0, 1_800.0 + index)
                for index in range(30)
            ],
        )
    finally:
        connection.close()

    restored = MicrostructureScanner(symbol="ethusdt")
    restored._db_init()

    assert len(restored._price_hist) == 30
    assert restored._price_hist[0] == 1_800.0
    assert restored._price_hist[-1] == 1_829.0
    assert restored._warmup_30m_ready is True
