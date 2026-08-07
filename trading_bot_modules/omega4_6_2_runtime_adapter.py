"""Runtime replay adapter for the selected Omega 4.6.2 validation-only sleeve."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_ID = "omega4_6_2_v5_roll8_side_specific_two_stage_exposure_validation_only_20260701"
DEFAULT_REPORT = ROOT / "tmp/causal_regen_20260516" / DEFAULT_MODEL_ID / "report.json"
EPS = 1.0e-12
LIVE_DECISION_SUPPORTED = False

REQUIRED_COLUMNS = {
    "entry_i",
    "exit_i",
    "entry_timestamp",
    "exit_timestamp",
    "side",
    "reason",
    "trade_return",
    "net_per_notional",
    "notional",
    "margin_fraction",
    "leverage",
    "hold_hours",
    "take_profit",
    "stop_loss",
    "two_stage_exposure_spec",
}

NUMERIC_DECISION_COLUMNS = [
    "entry_i",
    "exit_i",
    "side",
    "trade_return",
    "net_per_notional",
    "notional",
    "margin_fraction",
    "leverage",
    "hold_hours",
    "take_profit",
    "stop_loss",
    "exit_input_notional",
    "exit_input_leverage",
    "exit_input_exposure",
    "two_stage_exposure_long_factor",
    "two_stage_exposure_short_factor",
    "two_stage_exposure_cap_notional",
    "two_stage_exposure_base_notional",
]

STRING_DECISION_COLUMNS = [
    "entry_timestamp",
    "exit_timestamp",
    "reason",
    "two_stage_exposure_spec",
    "feature_veto_spec",
]


@dataclass(frozen=True)
class Omega462Decision:
    split: str
    action: str
    entry_timestamp: str
    exit_timestamp: str
    entry_i: int
    exit_i: int
    side: int
    reason: str
    notional: float
    margin_fraction: float
    leverage: float
    trade_return: float
    net_per_notional: float
    hold_hours: float
    take_profit: float
    stop_loss: float
    payload: dict[str, Any]

    def as_row(self) -> dict[str, Any]:
        row = dict(self.payload)
        row["split"] = self.split
        row["action"] = self.action
        return row


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def _timestamp_key(value: Any) -> str:
    ts = pd.Timestamp(value)
    if pd.isna(ts):
        raise ValueError(f"invalid timestamp: {value!r}")
    return ts.strftime("%Y-%m-%d %H:%M:%S")


def _coerce_payload_value(value: Any) -> Any:
    if pd.isna(value):
        return None
    if hasattr(value, "item"):
        return value.item()
    return value


class Omega462LedgerReplayAdapter:
    """Fail-fast adapter that replays the selected Omega 4.6.2 ledger contract."""

    def __init__(self, *, report_path: Path, report: dict[str, Any], ledgers: dict[str, pd.DataFrame]):
        self.report_path = Path(report_path)
        self.report = report
        self.model_id = str(report.get("model_id", ""))
        if self.model_id != DEFAULT_MODEL_ID:
            raise ValueError(f"unexpected model_id: {self.model_id}")
        selected = dict(report.get("selected_variant", {}) or {})
        if selected.get("oos_used_in_selection") is not False:
            raise ValueError("selected Omega4.6.2 validation-only model must declare oos_used_in_selection=False")
        self.selected_variant = selected
        self.ledgers = {split: self._validate_ledger(split, df) for split, df in ledgers.items()}
        self._index = {
            split: self._build_timestamp_index(split, df)
            for split, df in self.ledgers.items()
        }

    @classmethod
    def from_report(cls, report_path: Path = DEFAULT_REPORT) -> "Omega462LedgerReplayAdapter":
        report_path = Path(report_path)
        report = _read_json(report_path)
        artifacts = report.get("artifacts", {})
        required = {
            "validation": artifacts.get("selected_validation_ledger"),
            "oos": artifacts.get("selected_oos_ledger"),
        }
        missing = {split: path for split, path in required.items() if not path}
        if missing:
            raise KeyError(f"missing ledger artifacts: {missing}")
        ledgers = {split: pd.read_csv(path) for split, path in required.items()}
        return cls(report_path=report_path, report=report, ledgers=ledgers)

    @classmethod
    def from_model_id(cls, model_id: str = DEFAULT_MODEL_ID, root: Path = ROOT) -> "Omega462LedgerReplayAdapter":
        if model_id != DEFAULT_MODEL_ID:
            raise ValueError(f"unsupported Omega4.6.2 runtime adapter model_id: {model_id}")
        return cls.from_report(root / "tmp/causal_regen_20260516" / model_id / "report.json")

    def _validate_ledger(self, split: str, df: pd.DataFrame) -> pd.DataFrame:
        missing = sorted(REQUIRED_COLUMNS.difference(df.columns))
        if missing:
            raise KeyError(f"{split} ledger missing required columns: {missing}")
        out = df.copy()
        out["entry_timestamp_key"] = out["entry_timestamp"].map(_timestamp_key)
        duplicate_keys = out.loc[out["entry_timestamp_key"].duplicated(), "entry_timestamp_key"].tolist()
        if duplicate_keys:
            raise ValueError(f"{split} ledger has duplicate entry timestamps: {duplicate_keys[:5]}")
        return out

    def _build_timestamp_index(self, split: str, df: pd.DataFrame) -> dict[str, int]:
        return {str(row["entry_timestamp_key"]): int(i) for i, row in df.iterrows()}

    def decide(self, *, split: str, timestamp: Any) -> Omega462Decision:
        if split not in self.ledgers:
            raise KeyError(f"unknown split: {split}")
        key = _timestamp_key(timestamp)
        if key not in self._index[split]:
            raise KeyError(f"{split} timestamp outside Omega4.6.2 replay contract: {key}")
        row = self.ledgers[split].iloc[self._index[split][key]]
        notional = float(row["notional"])
        action = "ENTER" if notional > EPS else "SKIP"
        payload: dict[str, Any] = {}
        for col in NUMERIC_DECISION_COLUMNS + STRING_DECISION_COLUMNS:
            if col in row.index:
                payload[col] = _coerce_payload_value(row[col])
        payload["entry_timestamp"] = _timestamp_key(row["entry_timestamp"])
        payload["exit_timestamp"] = _timestamp_key(row["exit_timestamp"])
        payload["reason"] = str(row["reason"])
        return Omega462Decision(
            split=split,
            action=action,
            entry_timestamp=payload["entry_timestamp"],
            exit_timestamp=payload["exit_timestamp"],
            entry_i=int(row["entry_i"]),
            exit_i=int(row["exit_i"]),
            side=int(row["side"]),
            reason=payload["reason"],
            notional=notional,
            margin_fraction=float(row["margin_fraction"]),
            leverage=float(row["leverage"]),
            trade_return=float(row["trade_return"]),
            net_per_notional=float(row["net_per_notional"]),
            hold_hours=float(row["hold_hours"]),
            take_profit=float(row["take_profit"]),
            stop_loss=float(row["stop_loss"]),
            payload=payload,
        )

    def decide_live(self, *args: Any, **kwargs: Any) -> Omega462Decision:
        raise RuntimeError(
            "Omega4.6.2 ledger replay adapter is historical-only and cannot be used "
            "as a live/future timestamp decision provider. Use "
            "Omega462SourceParentLiveAdapter for live-native source-parent inference."
        )

    def replay_split(self, split: str) -> pd.DataFrame:
        if split not in self.ledgers:
            raise KeyError(f"unknown split: {split}")
        rows = [
            self.decide(split=split, timestamp=row["entry_timestamp"]).as_row()
            for _, row in self.ledgers[split].iterrows()
        ]
        return pd.DataFrame(rows)
