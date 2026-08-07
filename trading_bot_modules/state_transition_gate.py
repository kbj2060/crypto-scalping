from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from typing import AsyncIterator


class StateTransitionGate:
    """Serializes the specific call sites that opt into it via `transition(name)`.

    This does NOT gate every mutation of live trading state -- GovernorPositionRouter
    (position_router.py) has no internal locking of its own, so any position-mutating
    code path that isn't wrapped in `async with transition_gate.transition(...)` runs
    unserialized against this gate. As of this writing trading_bot.py wraps exactly 3
    call sites with it: "exchange_reconcile", "pending_next_open", and "bar_cycle".
    Treat this as protecting those three phases from overlapping each other, not as a
    blanket guarantee that position/router state can't be mutated concurrently from
    elsewhere.
    """

    def __init__(self) -> None:
        self._lock = asyncio.Lock()
        self.revision = 0
        self.active_transition = ""

    @asynccontextmanager
    async def transition(self, name: str) -> AsyncIterator[int]:
        async with self._lock:
            self.active_transition = str(name)
            starting_revision = self.revision
            try:
                yield starting_revision
            except Exception:
                raise
            else:
                self.revision += 1
            finally:
                self.active_transition = ""
