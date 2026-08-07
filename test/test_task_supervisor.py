from __future__ import annotations

import asyncio

from trading_bot_modules.task_supervisor import AsyncTaskSupervisor


def test_task_supervisor_awaits_cancellation():
    async def scenario():
        cancelled = asyncio.Event()

        async def worker():
            try:
                await asyncio.Event().wait()
            finally:
                cancelled.set()

        supervisor = AsyncTaskSupervisor()
        supervisor.create(worker(), name="worker")
        await asyncio.sleep(0)
        await supervisor.close()
        return cancelled.is_set()

    assert asyncio.run(scenario()) is True


def test_task_supervisor_observes_background_failure():
    async def scenario():
        supervisor = AsyncTaskSupervisor()

        async def fail():
            raise RuntimeError("injected")

        supervisor.create(fail(), name="failure")
        await asyncio.sleep(0)
        await supervisor.close()
        return supervisor.errors

    errors = asyncio.run(scenario())
    assert len(errors) == 1
    assert errors[0][0] == "failure"
    assert str(errors[0][1]) == "injected"
