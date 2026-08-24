#!/usr/bin/env python3
"""Static triage scan for the train/inference feature-parity bug class confirmed 2026-08-18 in
`research_eth_omega461_exit_head_liveatr_relabel_20260813.py` (see docs/experiments/eth_odyssey4_
exit_head_liveatr_barrier_and_label_reaudit_20260818.md and docs/experiments/eth_odyssey4_exit_head_
tpsl_feature_barrier_mismatch_20260817.md) and independently re-found the same day in 5 more files
by manual grep+read across ~40 scripts. The bug shape, every time: a training-time label/feature
builder recomputes something the live/replay path also computes, and the two silently drift apart.

Three concrete patterns, matched heuristically here (see CLAUDE.md's "Position-Feature Train/
Inference Parity Contract" for the policy this operationalizes):

  A) A price-move-like local (feeding a `*_position_feature_row`/`*_feature_row` call's `unreal`/
     `mfe`/`mae` keyword) gets multiplied by a notional/leverage/exposure-named variable inside the
     SAME function -- but every actual inference/replay path in this repo feeds those features the
     UNSCALED price move. Detected via AST: precise, reported as "confirmed" (still read the code --
     this is a triage aid, not a verdict).
  B) The `notional=`/`leverage=`/`take_profit=`/`stop_loss=` keyword of that same call traces back
     to an assignment textually outside any enclosing loop the call itself sits inside (i.e.
     loop-invariant -- likely a fixed BASE_TEMPLATE-style constant standing in for what should vary
     per row/candidate). Detected via AST + loop-range heuristic: reported as "needs_review" -- a
     real risk-sizing model legitimately used as a per-run constant during early bootstrap (no
     sidecar exists yet) is NOT a bug, see `_build_exit_dataset_entry_label_live_atr_barrier`'s
     documented no-sidecar fallback path. `take_profit`/`stop_loss` were added to this check
     2026-08-18 after finding the exact same stale-constant shape independently: 9 files fed a fixed
     `omega.BASE_TEMPLATE["take_profit"/"stop_loss"]` (2.6%/1.4%) into every candidate's pos_tp/
     pos_sl instead of live's real per-candidate ATR-adaptive value (`omega4_6_1_live.py:91-97,181-
     185`) -- see CLAUDE.md's Position-Feature Train/Inference Parity Contract.
  C) A barrier/exit-resolution loop compares `high`/`low`-indexed arrays against a level to decide
     when a labeled trade "resolves". Detected via regex + proximity: reported as "needs_review" --
     ATR/volatility computation legitimately uses high/low and is explicitly excluded by
     function-name heuristic, but this exclusion is not perfect.
     ⚠️ 2026-08-18 CORRECTION: this pattern does NOT mean "intrabar high/low is always wrong."
     `omega4_6_1_live.py::evaluate_exit`'s TP/SL hard-check for h48qual/zig075 genuinely IS
     intrabar (`bar_high_move`/`bar_low_move`, computed from the just-completed bar's real
     high/low -- `trading_bot.py:9181-9202`; deliberate, documented, no lookahead since the bar is
     already closed). An earlier pass this same day wrongly "fixed" a barrier-resolution function
     toward close-only based on `greedy_replay`/`_price_move` (both close-only) without checking
     `evaluate_exit` directly -- reverted once caught. `greedy_replay` being close-only does NOT
     prove live is close-only; it may simply be a replay tool that predates or never adopted the
     bar_high_move/bar_low_move improvement. **Every Pattern C finding needs the asset's actual
     live `evaluate_exit`-equivalent function read directly (not a replay/backtest script) before
     deciding which direction is the bug** -- see CLAUDE.md's Position-Feature Train/Inference
     Parity Contract for the corrected, non-universal framing.

This is a TRIAGE tool, not a verdict engine: B and C in particular need a human (or Claude session)
to read the surrounding code before calling something a real bug, exactly as happened when this
script's own pattern definitions were derived (several apparent hits during the 2026-08-18 sweep
turned out to be clean on inspection -- e.g. ATR-window bounds keyed `low[c]`/`high[c]` by feature
name, unrelated to price bars). Only scans `scripts/*.py` source text via `ast`/regex -- imports
nothing from this repo, pulls in no torch/pandas, safe to run anytime.
"""
from __future__ import annotations

import argparse
import ast
import json
import re
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

FEATURE_ROW_CALL_SUFFIXES = ("_position_feature_row", "_feature_row")
SCALE_NAME_RE = re.compile(r"\b(notional|leverage|exposure)\b")
BASE_TEMPLATE_RE = re.compile(r"\bBASE_TEMPLATE\s*\[")
SCALED_KWARGS = {"unreal", "mfe", "mae"}
SIZING_KWARGS = {"notional", "leverage", "take_profit", "stop_loss"}

HILO_ARRAY_READ_RE = re.compile(r"(\[\s*[\"']high[\"']\s*\]|\bhigh\s*\[|\[\s*[\"']low[\"']\s*\]|\blow\s*\[)")
RESOLUTION_TOKEN_RE = re.compile(r"\b(hi|lo|high\w*|low\w*|best|worst|mfe|mae)\b")
LEVEL_TOKEN_RE = re.compile(r"(_level\b|\btp_|\bsl_|barrier|take_profit|stop_loss)", re.IGNORECASE)
COMPARISON_RE = re.compile(r"(>=|<=)")
ATR_NAME_RE = re.compile(r"atr", re.IGNORECASE)


@dataclass
class Finding:
    pattern: str  # "A" | "B" | "C"
    confidence: str  # "confirmed" | "needs_review"
    file: str
    line: int
    function: str
    detail: str


@dataclass
class AuditResult:
    audit_id: str
    created_at: str
    files_scanned: int
    findings: list[Finding] = field(default_factory=list)


def _unparse(node: ast.AST | None) -> str:
    if node is None:
        return ""
    try:
        return ast.unparse(node)
    except Exception:  # noqa: BLE001 -- best-effort text for a heuristic report line
        return "<unparsable>"


def _call_name(node: ast.Call) -> str:
    return _unparse(node.func)


def _is_loop(node: ast.AST) -> bool:
    return isinstance(node, (ast.For, ast.AsyncFor, ast.While))


def _scan_function_ast(func: ast.FunctionDef | ast.AsyncFunctionDef, path: str) -> list[Finding]:
    findings: list[Finding] = []
    all_nodes = list(ast.walk(func))
    loop_ranges = [
        (n.lineno, getattr(n, "end_lineno", n.lineno)) for n in all_nodes if _is_loop(n)
    ]

    def _inside_a_loop(lineno: int) -> bool:
        return any(lo <= lineno <= hi for lo, hi in loop_ranges)

    # Pattern A/B setup: map assigned Name -> (line, is_scaled_by_notional_leverage_exposure,
    # is_base_template_constant, inside_loop, source_text). Last assignment above a given line wins.
    assigns: dict[str, list[tuple[int, bool, bool, bool]]] = {}
    for n in all_nodes:
        if not isinstance(n, ast.Assign) or len(n.targets) != 1 or not isinstance(n.targets[0], ast.Name):
            continue
        name = n.targets[0].id
        text = _unparse(n.value)
        is_scaled = isinstance(n.value, ast.BinOp) and isinstance(n.value.op, ast.Mult) and bool(SCALE_NAME_RE.search(text))
        is_base_template = bool(BASE_TEMPLATE_RE.search(text))
        assigns.setdefault(name, []).append((n.lineno, is_scaled, is_base_template, _inside_a_loop(n.lineno)))

    def _latest_assign_before(name: str, before_line: int) -> tuple[int, bool, bool, bool] | None:
        candidates = [a for a in assigns.get(name, []) if a[0] <= before_line]
        return max(candidates, key=lambda a: a[0]) if candidates else None

    for n in all_nodes:
        if not isinstance(n, ast.Call):
            continue
        call_name = _call_name(n)
        if not any(call_name.endswith(suffix) for suffix in FEATURE_ROW_CALL_SUFFIXES):
            continue
        kw_by_name = {kw.arg: kw.value for kw in n.keywords if kw.arg is not None}

        for kwname in SCALED_KWARGS:
            val = kw_by_name.get(kwname)
            if not isinstance(val, ast.Name):
                continue
            assign = _latest_assign_before(val.id, n.lineno)
            if assign is not None and assign[1]:
                findings.append(Finding(
                    pattern="A", confidence="confirmed", file=path, line=n.lineno, function=func.name,
                    detail=(
                        f"{call_name}(...{kwname}={val.id}...) -- {val.id} was assigned at line "
                        f"{assign[0]} via a notional/leverage/exposure multiplication; every actual "
                        f"inference/replay path in this repo feeds {kwname} UNSCALED (see CLAUDE.md's "
                        "Position-Feature Train/Inference Parity Contract)."
                    ),
                ))

        call_inside_loop = _inside_a_loop(n.lineno)
        for kwname in SIZING_KWARGS:
            val = kw_by_name.get(kwname)
            if not isinstance(val, ast.Name):
                continue
            assign = _latest_assign_before(val.id, n.lineno)
            if assign is None:
                continue
            _, _, is_base_template, assign_inside_loop = assign
            if is_base_template and call_inside_loop and not assign_inside_loop:
                findings.append(Finding(
                    pattern="B", confidence="needs_review", file=path, line=n.lineno, function=func.name,
                    detail=(
                        f"{call_name}(...{kwname}={val.id}...) -- {val.id} was assigned at line "
                        f"{assign[0]} from a BASE_TEMPLATE lookup OUTSIDE the loop this call sits "
                        "inside (loop-invariant across every row/candidate). Confirm whether a real "
                        "per-row risk-sizing source exists and is being ignored (bug) or genuinely "
                        "does not exist yet for this candidate (acceptable -- must be explicitly "
                        "labeled, e.g. a risk_sizing_source diagnostic field, not silent)."
                    ),
                ))
    return findings


def scan_pattern_ab(path: Path) -> list[Finding]:
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    except (SyntaxError, UnicodeDecodeError):
        return []
    rel = str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path)
    findings: list[Finding] = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            findings.extend(_scan_function_ast(node, rel))
    return findings


def _has_feature_row_call(tree: ast.AST) -> bool:
    return any(
        isinstance(n, ast.Call) and any(_call_name(n).endswith(suffix) for suffix in FEATURE_ROW_CALL_SUFFIXES)
        for n in ast.walk(tree)
    )


def scan_pattern_c(path: Path) -> list[Finding]:
    """THREE-signal, FILE-scoped heuristic (not same-line): does this file (1) build dense per-bar
    training features via a `*_position_feature_row`/`*_feature_row`-style call (the same signal
    Patterns A/B key off), (2) read intrabar high/low ANYWHERE outside an ATR-named function, and
    (3) separately contain a `>=`/`<=` comparison against a tp/sl/barrier-ish level using an
    hi/lo/best/worst/mfe/mae-named variable? Signal 1 is what actually distinguishes the bug (a
    barrier that anchors DENSE bar-by-bar model-input features) from a completely standard,
    unproblematic single-label-per-candidate triple-barrier LABEL builder (`build_*_tripbarrier_
    *label*.py`, `train_eval_omega3_tabm_4head_triple_barrier_20260618.py`'s "atr_scaled_triple_
    barrier_first_touch" etc. -- both legitimately use high/low, but only ever emit ONE label per
    candidate, not a bar-by-bar feature-generating exit-lifecycle walk). Without signal 1 this
    detector drowned in exactly that false-positive class (60 files, mostly plain label builders) on
    its first pass 2026-08-18 -- signal 1 cut it down to real dense-feature builders only.

    Signals 2/3 are still FILE-scoped, not function-scoped: the real 2026-08-18 finding in
    research_eth_omega461_censored_stopping_value_20260724.py has the high/low read in one function
    (`_bar_moves`) and the resolution comparison in a DIFFERENT function that calls it
    (`_position_path`) -- a same-function requirement would have missed it."""
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except UnicodeDecodeError:
        return []
    rel = str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path)
    try:
        tree = ast.parse("\n".join(lines), filename=str(path))
    except SyntaxError:
        return []
    if not _has_feature_row_call(tree):
        return []
    func_ranges: list[tuple[int, int, str]] = [
        (n.lineno, getattr(n, "end_lineno", n.lineno), n.name)
        for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
    ]

    def _enclosing_func(lineno: int) -> str:
        enclosing = [f for f in func_ranges if f[0] <= lineno <= f[1]]
        return min(enclosing, key=lambda f: f[1] - f[0])[2] if enclosing else "<module>"

    has_intrabar_read = any(
        HILO_ARRAY_READ_RE.search(line) and not ATR_NAME_RE.search(_enclosing_func(i))
        for i, line in enumerate(lines, start=1)
    )
    if not has_intrabar_read:
        return []

    findings: list[Finding] = []
    for i, line in enumerate(lines, start=1):
        if not (RESOLUTION_TOKEN_RE.search(line) and COMPARISON_RE.search(line) and LEVEL_TOKEN_RE.search(line)):
            continue
        func_name = _enclosing_func(i)
        if ATR_NAME_RE.search(func_name):
            continue
        findings.append(Finding(
            pattern="C", confidence="needs_review", file=rel, line=i, function=func_name,
            detail=(
                f"file reads intrabar high/low AND has a resolution-looking comparison here: "
                f"`{line.strip()}`. Read this asset/model's ACTUAL live evaluate_exit-equivalent "
                "function directly (not a replay/backtest script) to determine whether intrabar "
                "or close-only is the real live barrier convention here -- do NOT assume close-only "
                "is correct (h48qual/zig075's real TP/SL hard-check is intrabar, see CLAUDE.md's "
                "Position-Feature Train/Inference Parity Contract, corrected 2026-08-18). Flag a "
                "mismatch only once you know what this specific model's live convention actually is."
            ),
        ))
    return findings


def write_markdown(path: Path, audit: AuditResult) -> None:
    lines = [
        f"# Position-Feature Train/Inference Parity Audit ({audit.created_at})",
        "",
        f"Files scanned: {audit.files_scanned}. Findings: {len(audit.findings)} "
        f"({sum(1 for f in audit.findings if f.confidence == 'confirmed')} confirmed, "
        f"{sum(1 for f in audit.findings if f.confidence == 'needs_review')} needs_review).",
        "",
        "Triage tool, not a verdict -- read each finding's surrounding code before treating it as a "
        "real bug, especially pattern B/C (needs_review). See this script's own module docstring.",
        "",
    ]
    for pattern in ("A", "B", "C"):
        matches = [f for f in audit.findings if f.pattern == pattern]
        if not matches:
            continue
        lines.append(f"## Pattern {pattern} ({len(matches)})")
        lines.append("")
        for f in matches:
            lines.append(f"- `{f.file}:{f.line}` ({f.function}, {f.confidence}) -- {f.detail}")
        lines.append("")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", default="scripts/*.py", help="glob relative to repo root")
    ap.add_argument("--out-dir", type=Path, default=ROOT / "tmp/causal_regen_20260516/position_feature_parity_audit_20260818")
    args = ap.parse_args()

    paths = sorted(ROOT.glob(args.glob))
    findings: list[Finding] = []
    for path in paths:
        if path.name == Path(__file__).name:
            continue
        findings.extend(scan_pattern_ab(path))
        findings.extend(scan_pattern_c(path))

    audit = AuditResult(
        audit_id="position_feature_train_inference_parity_audit_20260818",
        created_at=datetime.now(timezone.utc).isoformat(),
        files_scanned=len(paths),
        findings=findings,
    )
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "audit.json"
    md_path = out_dir / "audit.md"
    json_path.write_text(json.dumps(asdict(audit), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_markdown(md_path, audit)

    confirmed = [f for f in findings if f.confidence == "confirmed"]
    print(json.dumps({
        "files_scanned": len(paths), "findings": len(findings), "confirmed": len(confirmed),
        "needs_review": len(findings) - len(confirmed), "json": str(json_path), "markdown": str(md_path),
    }, ensure_ascii=False, indent=2))
    return 1 if confirmed else 0


if __name__ == "__main__":
    raise SystemExit(main())
