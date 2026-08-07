import asyncio

from trading_bot_modules.runtime_shutdown import shutdown_runtime_resources


class _AsyncResource:
    def __init__(self, *, fail=False):
        self.closed = False
        self.fail = fail

    async def close(self):
        self.closed = True
        if self.fail:
            raise RuntimeError("close failed")


class _SyncResource:
    def __init__(self, *, fail=False):
        self.stopped = False
        self.fail = fail

    def stop(self):
        self.stopped = True
        if self.fail:
            raise RuntimeError("stop failed")


def test_shutdown_continues_after_partial_resource_failures():
    supervisor = _AsyncResource(fail=True)
    writer = _AsyncResource()
    scanner = _SyncResource(fail=True)
    tail = _SyncResource()
    first_fetcher = _AsyncResource(fail=True)
    second_fetcher = _AsyncResource()
    errors = []

    asyncio.run(
        shutdown_runtime_resources(
            task_supervisor=supervisor,
            journal_writer=writer,
            scanners=(scanner,),
            tail_interceptor=tail,
            fetchers=(first_fetcher, second_fetcher),
            on_error=lambda stage, resource, error: errors.append((stage, type(error))),
        )
    )

    assert supervisor.closed is True
    assert writer.closed is True
    assert scanner.stopped is True
    assert tail.stopped is True
    assert first_fetcher.closed is True
    assert second_fetcher.closed is True
    assert [stage for stage, _ in errors] == ["task_supervisor", "stop", "fetcher"]
