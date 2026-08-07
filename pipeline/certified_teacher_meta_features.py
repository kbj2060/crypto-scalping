from __future__ import annotations

import pandas as pd

from pipeline.teacher_meta_side_features import append_side_teacher_features


def append_teacher_meta_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Compatibility entry point for the strict Formula Teacher v1 transform."""
    return append_side_teacher_features(frame)
