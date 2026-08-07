from __future__ import annotations

import asyncio
import json

from trading_bot_modules.async_jsonl_writer import AsyncJsonlWriter


def test_jsonl_writer_preserves_fifo_order(tmp_path):
    async def scenario():
        target = tmp_path / "journal.jsonl"
        writer = AsyncJsonlWriter()
        for sequence in range(20):
            await writer.append(target, {"sequence": sequence})
        await writer.close()
        return [json.loads(line) for line in target.read_text().splitlines()]

    rows = asyncio.run(scenario())
    assert [row["sequence"] for row in rows] == list(range(20))


def test_jsonl_writer_drains_concurrent_requests_before_close(tmp_path):
    async def scenario():
        target = tmp_path / "journal.jsonl"
        writer = AsyncJsonlWriter()
        await asyncio.gather(
            *(writer.append(target, {"sequence": sequence}) for sequence in range(40))
        )
        await writer.close()
        return [json.loads(line) for line in target.read_text().splitlines()]

    rows = asyncio.run(scenario())
    assert len(rows) == 40
    assert {row["sequence"] for row in rows} == set(range(40))
