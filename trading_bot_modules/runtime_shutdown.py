from __future__ import annotations

import asyncio
from collections.abc import Iterable
from typing import Any


async def shutdown_runtime_resources(
    *,
    task_supervisor: Any = None,
    journal_writer: Any = None,
    scanners: Iterable[Any] = (),
    tail_interceptor: Any = None,
    fetchers: Iterable[Any] = (),
    on_error=None,
) -> None:
    async def _await_close(label: str, resource: Any) -> None:
        if resource is None:
            return
        try:
            await resource.close()
        except BaseException as error:
            if on_error is not None:
                on_error(label, resource, error)

    await _await_close("task_supervisor", task_supervisor)
    await _await_close("journal_writer", journal_writer)

    for resource in [*scanners, tail_interceptor]:
        if resource is None:
            continue
        try:
            resource.stop()
        except BaseException as error:
            if on_error is not None:
                on_error("stop", resource, error)

    closeable_fetchers = [resource for resource in fetchers if resource is not None]
    if not closeable_fetchers:
        return
    results = await asyncio.gather(
        *(resource.close() for resource in closeable_fetchers),
        return_exceptions=True,
    )
    for resource, result in zip(closeable_fetchers, results):
        if isinstance(result, BaseException) and on_error is not None:
            on_error("fetcher", resource, result)
