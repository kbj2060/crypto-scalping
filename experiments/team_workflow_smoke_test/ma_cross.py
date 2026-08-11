"""Pure-function moving-average cross detector.

No external I/O, no global state, no imports from the rest of this repo.
Smoke-test utility only (see CONTRACT.md, task SMOKE-1).
"""

from typing import Sequence


def detect_ma_cross(fast: Sequence[float], slow: Sequence[float]) -> str:
    """Classify the moving-average cross state at the last bar.

    Compares the second-to-last and last values of `fast` and `slow` to
    determine whether `fast` crossed above `slow` ("golden_cross"), crossed
    below `slow` ("death_cross"), or neither ("none") at the most recent bar.

    Args:
        fast: Fast moving-average values, oldest to newest.
        slow: Slow moving-average values, oldest to newest, same length as `fast`.

    Returns:
        One of "golden_cross", "death_cross", "none".

    Raises:
        ValueError: If `fast` and `slow` have different lengths, or either
            has fewer than 2 elements.
    """
    if len(fast) != len(slow):
        raise ValueError(
            f"fast and slow must have equal length, got {len(fast)} and {len(slow)}"
        )
    if len(fast) < 2:
        raise ValueError(
            f"fast and slow must have at least 2 elements, got {len(fast)}"
        )

    prev_fast, curr_fast = fast[-2], fast[-1]
    prev_slow, curr_slow = slow[-2], slow[-1]

    was_below_or_equal = prev_fast <= prev_slow
    was_above_or_equal = prev_fast >= prev_slow
    is_above = curr_fast > curr_slow
    is_below = curr_fast < curr_slow

    if was_below_or_equal and is_above:
        return "golden_cross"
    if was_above_or_equal and is_below:
        return "death_cross"
    return "none"
