from __future__ import annotations

import asyncio
from collections.abc import Coroutine
from typing import Any, Callable


class AsyncTaskSupervisor:
    """Own background tasks, observe failures, and await cancellation on shutdown."""

    def __init__(self, on_error: Callable[[str, BaseException], None] | None = None) -> None:
        self._tasks: set[asyncio.Task[Any]] = set()
        self._on_error = on_error
        self.errors: list[tuple[str, BaseException]] = []
        self._closed = False

    def set_on_error(
        self, on_error: Callable[[str, BaseException], None] | None
    ) -> None:
        self._on_error = on_error

    def create(self, coroutine: Coroutine[Any, Any, Any], *, name: str) -> asyncio.Task[Any]:
        if self._closed:
            coroutine.close()
            raise RuntimeError("task supervisor is closed")
        task = asyncio.create_task(coroutine, name=name)
        self._tasks.add(task)
        task.add_done_callback(self._task_done)
        return task

    def _task_done(self, task: asyncio.Task[Any]) -> None:
        self._tasks.discard(task)
        if task.cancelled():
            return
        error = task.exception()
        if error is None:
            return
        name = task.get_name()
        self.errors.append((name, error))
        if self._on_error is not None:
            self._on_error(name, error)

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        tasks = list(self._tasks)
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        self._tasks.clear()
