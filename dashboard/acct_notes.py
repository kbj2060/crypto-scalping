"""계좌 카드 날짜별 메모 (2026-11-04 사용자 «메모장은 저장하면 서버에 계속 저장»).
data/live/acct_notes.json = {KST 날짜: {"text", "at"(ms)}}. 빈 글을 저장하면 그 날 메모를 지운다.
python -m pytest -q test/test_acct_notes_20261104.py"""
import json
import os
import re
import time
from pathlib import Path

NOTE_MAX = 4000   # 글자 -- 하루 메모로 충분하고 파일이 무한히 자라지 않게
_DATE = re.compile(r"\d{4}-\d{2}-\d{2}")


def load(path: Path) -> dict:
    try:
        d = json.loads(Path(path).read_text(encoding="utf-8"))
        return d if isinstance(d, dict) else {}
    except (OSError, ValueError):
        return {}   # 파일 없음·깨짐 = 메모 없음 (다음 저장이 새로 쓴다)


def save(path: Path, date, text, now_ms: int | None = None) -> dict:
    """date·text 검증 후 원자적으로 쓴다. 잘못된 입력은 ValueError."""
    if not isinstance(date, str) or not _DATE.fullmatch(date):
        raise ValueError("date must be YYYY-MM-DD")
    if not isinstance(text, str) or len(text) > NOTE_MAX:
        raise ValueError(f"text must be a string up to {NOTE_MAX} chars")
    notes = load(path)
    at = int(time.time() * 1000) if now_ms is None else now_ms
    if text.strip():
        notes[date] = {"text": text, "at": at}
    else:
        notes.pop(date, None)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(notes, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, path)
    return {"ok": True, "at": at if text.strip() else None}
