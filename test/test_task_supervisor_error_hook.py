from __future__ import annotations

import asyncio
import unittest

from trading_bot_modules.task_supervisor import AsyncTaskSupervisor


class TaskSupervisorErrorHookTests(unittest.IsolatedAsyncioTestCase):
    async def test_error_hook_can_be_attached_after_runtime_initialization(self) -> None:
        observed: list[tuple[str, str]] = []
        supervisor = AsyncTaskSupervisor()
        supervisor.set_on_error(
            lambda name, error: observed.append((name, str(error)))
        )

        async def fail() -> None:
            raise RuntimeError("injected-background-error")

        supervisor.create(fail(), name="background-worker")
        await asyncio.sleep(0)
        await supervisor.close()

        self.assertEqual(
            observed,
            [("background-worker", "injected-background-error")],
        )


if __name__ == "__main__":
    unittest.main()
