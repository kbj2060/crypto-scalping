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


# 2026-11-06 오답노트 -- 메모(날짜별 글 하나)와 따로, 한 줄씩 쌓는 목록. data/live/acct_wrong.json = [{id, date, text, tag, at}] 오래된 것부터.
WRONG_MAX = 300    # 글자/줄
WRONG_KEEP = 2000  # 줄 -- 넘으면 가장 오래된 것부터 버린다(파일이 무한히 자라지 않게)
WRONG_TAGS = ("", "진입 성급", "원칙 위반", "가짜 돌파", "사이징", "기타")


def wrong_load(path: Path) -> list:
    try:
        d = json.loads(Path(path).read_text(encoding="utf-8"))
        return d if isinstance(d, list) else []
    except (OSError, ValueError):
        return []


def _wrong_write(path: Path, rows: list) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(rows, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, path)


def wrong_add(path: Path, date, text, tag="", now_ms: int | None = None) -> dict:
    if not isinstance(date, str) or not _DATE.fullmatch(date):
        raise ValueError("date must be YYYY-MM-DD")
    if not isinstance(text, str) or not text.strip() or len(text) > WRONG_MAX:
        raise ValueError(f"text must be 1..{WRONG_MAX} chars")
    if tag not in WRONG_TAGS:
        raise ValueError("unknown tag")
    rows = wrong_load(path)
    at = int(time.time() * 1000) if now_ms is None else now_ms
    row = {"id": max([r.get("id", 0) for r in rows] + [at - 1]) + 1, "date": date, "text": " ".join(text.split()), "tag": tag, "at": at}   # 한 줄: 줄바꿈·연속 공백은 접는다
    _wrong_write(path, (rows + [row])[-WRONG_KEEP:])
    return row


def wrong_del(path: Path, rid) -> dict:
    if not isinstance(rid, int) or isinstance(rid, bool):
        raise ValueError("id must be int")
    rows = wrong_load(path)
    keep = [r for r in rows if r.get("id") != rid]
    if len(keep) != len(rows):
        _wrong_write(path, keep)
    return {"ok": True}
