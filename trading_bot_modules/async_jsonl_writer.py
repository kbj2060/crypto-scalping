from __future__ import annotations

import asyncio
from dataclasses import dataclass
from pathlib import Path

from trading_bot_modules.live_io import _append_jsonl_many


@dataclass(slots=True)
class _WriteRequest:
    path: str
    rows: list[dict]
    completed: asyncio.Future[None]


class AsyncJsonlWriter:
    """Serialize every JSONL append through one FIFO worker."""

    def __init__(self) -> None:
        self._queue: asyncio.Queue[_WriteRequest | None] = asyncio.Queue()
        self._task: asyncio.Task[None] | None = None
        self._closed = False

    def start(self) -> None:
        if self._closed:
            raise RuntimeError("jsonl writer is closed")
        if self._task is None:
            self._task = asyncio.create_task(self._run(), name="jsonl-writer")

    async def append(self, path: str | Path, row: dict) -> None:
        await self.append_many(path, [row])

    async def append_many(self, path: str | Path, rows: list[dict]) -> None:
        payloads = [dict(row) for row in rows if isinstance(row, dict)]
        if not payloads:
            return
        if self._closed:
            raise RuntimeError("jsonl writer is closed")
        self.start()
        completed = asyncio.get_running_loop().create_future()
        await self._queue.put(_WriteRequest(str(path), payloads, completed))
        await completed

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._task is None:
            return
        await self._queue.put(None)
        await self._task
        self._task = None

    async def _run(self) -> None:
        while True:
            request = await self._queue.get()
            if request is None:
                self._queue.task_done()
                break
            try:
                await asyncio.to_thread(_append_jsonl_many, request.path, request.rows)
            except Exception as exc:
                if not request.completed.done():
                    request.completed.set_exception(exc)
            else:
                if not request.completed.done():
                    request.completed.set_result(None)
            finally:
                self._queue.task_done()
