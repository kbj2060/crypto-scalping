from __future__ import annotations

import asyncio

from trading_bot_modules.state_transition_gate import StateTransitionGate


def test_transition_gate_serializes_competing_mutations():
    async def scenario():
        gate = StateTransitionGate()
        active = 0
        max_active = 0
        committed = []

        async def mutate(name):
            nonlocal active, max_active
            async with gate.transition(name) as revision:
                active += 1
                max_active = max(max_active, active)
                await asyncio.sleep(0.001)
                committed.append((revision, name))
                active -= 1

        await asyncio.gather(*(mutate(f"event-{index}") for index in range(30)))
        return gate, max_active, committed

    gate, max_active, committed = asyncio.run(scenario())
    assert max_active == 1
    assert gate.revision == 30
    assert [revision for revision, _name in committed] == list(range(30))


def test_failed_transition_does_not_advance_revision():
    async def scenario():
        gate = StateTransitionGate()
        try:
            async with gate.transition("failure"):
                raise RuntimeError("injected")
        except RuntimeError:
            pass
        return gate

    gate = asyncio.run(scenario())
    assert gate.revision == 0
    assert gate.active_transition == ""
