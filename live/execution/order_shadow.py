"""주문 경로 그림자 기록(2026-11-21 · rtorder/SHADOW.md 8절 제안서) -- Rust 재생(rtorder shadow_replay)이 같은 입력으로 같은 요청을 냈는지 견주는 실기록.

한 줄 = 한 사건 `{"rec", "chain", "t", "mono", ...}` → `data/live/order_shadow.jsonl`(chmod 600 · 50MB 넘으면 `.1` 로 한 번 돌림).
- 기본 켜짐 · 끄기 = 환경 `ORDER_SHADOW_LOG=0`. 경로는 orders.register 가 LIVE_DIR 로 정한다(그 전에는 기록 안 함).
- 🔴주문 동작 불변: 모든 함수가 예외를 삼킨다 · 이벤트 루프에서는 JSON 한 줄을 만들어 메모리 큐에 넣기만 한다(await·잠금·파일 I/O 없음) --
  파일 쓰기·돌리기·무장 파일 읽기(lazy)는 뒤의 스레드 하나가 한꺼번에.
- chain = 작업 하나(접수 요청 · 진입·청산 작업 · 감시 틱) -- contextvars 라 create_task 로 집행까지 따라간다. chain 이 없는 곳(화면 계좌 조회 등)은 적지 않는다.
- 가림: 서명 쿼리 `&signature=` 뒤 · API 키 머리(머리는 아예 안 적음) · listenKey 값. 계좌 값은 그대로(재생에 필요 -- 파일은 서버 밖으로 내지 않는다).
"""
from __future__ import annotations

import contextvars
import json
import os
import queue
import re
import threading
import time
from pathlib import Path
from typing import Any

SHADOW_PATH: Path | None = None
SHADOW_MAX_BYTES = 50 * 1024 * 1024
_CHAIN: contextvars.ContextVar = contextvars.ContextVar("order_shadow_chain", default=None)
_BUF: contextvars.ContextVar = contextvars.ContextVar("order_shadow_buf", default=None)
_TICK: contextvars.ContextVar = contextvars.ContextVar("order_shadow_tick", default=None)
_Q: queue.SimpleQueue = queue.SimpleQueue()
_writer: threading.Thread | None = None
_slot_chain: dict[int, str] = {}   # id(상태 칸 dict) → 그 칸을 도는 작업의 chain(화면 입력을 그 작업에 잇는다 · 칸 dict 에는 아무것도 안 넣는다)
_LK = re.compile(r'("listenKey"\s*:\s*")[^"]*(")')
_SIG = re.compile(r"[&?]signature=[0-9A-Za-z]+")


def enabled() -> bool:
    return SHADOW_PATH is not None and os.getenv("ORDER_SHADOW_LOG", "1") != "0"


def set_path(p: Path) -> None:
    global SHADOW_PATH
    SHADOW_PATH = Path(p)


def new_chain(kind: str) -> str | None:
    try:
        c = f"{kind}-{time.time_ns():x}"
        _CHAIN.set(c)
        return c
    except Exception:  # noqa: BLE001
        return None


def current_chain() -> str | None:
    return _CHAIN.get()


def strip_sig(target: str) -> str:
    return target.split("&signature=", 1)[0].split("?signature=", 1)[0]


def rec(kind: str, *, chain: str | None = None, lazy_dir: Path | None = None, **kw: Any) -> None:
    """사건 하나. chain 이 없으면(맥락 밖) 버린다. lazy_dir = 무장 파일·복구 파일 이름을 쓰는 스레드가 그 순간 읽어 싣는다(루프에서 파일을 안 읽게)."""
    try:
        if not enabled():
            return
        c = chain or _CHAIN.get()
        if c is None:
            return
        row = {"rec": kind, "chain": c, "t": time.time(), "mono": time.monotonic(), **kw}
        buf = _BUF.get()
        if buf is not None:
            buf.append((kind, row, lazy_dir))   # 감시 틱: 남길지 모르니 글자로 바꾸지 않고 들고만 있다(2초마다 계좌 dict 를 JSON 으로 만들지 않게)
            return
        _put(json.dumps(row, ensure_ascii=False, default=str), lazy_dir)
    except Exception:  # noqa: BLE001 -- 기록은 주문을 막지 않는다
        pass


def wire(method: str, url: str, sent_at: float, response: Any, exc: BaseException | None) -> None:
    """서명·공개 왕복 하나. 응답 원문 = aiohttp 가 json() 하며 받아 둔 바이트(response._body) -- 새로 읽지 않는다(json() 동작 그대로)."""
    try:
        if not enabled() or _CHAIN.get() is None:
            return
        target = strip_sig(url.split("://", 1)[-1].split("/", 1)[-1])
        raw = getattr(response, "_body", None) if response is not None else None
        body = raw.decode("utf-8", "replace") if isinstance(raw, (bytes, bytearray)) else None
        if body and "listenKey" in body:
            body = _LK.sub(r"\1<listenKey>\2", body)
        rec("wire", method=method, target="/" + target, sent_at=sent_at, status=getattr(response, "status", None), body=body,
            error=None if exc is None else _SIG.sub("", f"{type(exc).__name__}: {exc}"))   # 예외 문구에 URL(서명)이 실리는 갈래(ContentTypeError 등)
    except Exception:  # noqa: BLE001
        pass


def bind_slot(state: dict, chain: str | None) -> None:
    try:
        if chain:
            _slot_chain[id(state)] = chain
    except Exception:  # noqa: BLE001
        pass


def chain_of(state: Any) -> str | None:
    try:
        return _slot_chain.get(id(state))
    except Exception:  # noqa: BLE001
        return None


async def task(coro, *, op: str, plan: Any, slot: dict, lazy_dir: Path | None, **kw: Any):
    """작업 하나(진입·청산)를 start … end 로 감싼다. 안의 coroutine 은 그대로 기다린다(취소·예외도 그대로 지나간다)."""
    c = None
    try:
        if enabled():
            parent = _CHAIN.get()   # 접수 요청의 chain(접수 기록 줄과 잇는다) -- 작업은 제 chain 을 따로(접수 핸들러의 조회와 섞이지 않게)
            c = new_chain(op)
            bind_slot(slot, c)
            rec("start", op=op, parent=parent, plan=plan, slot=slot, lazy_dir=lazy_dir, **kw)
    except Exception:  # noqa: BLE001
        pass
    raised = None
    try:
        return await coro
    except BaseException as exc:
        raised = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        rec("end", chain=c, slot=slot, raised=raised, lazy_dir=lazy_dir)


def tick_begin(slots: dict) -> Any:
    """감시 틱: 새 chain · 기록을 모아 둔다(요청이 나갔거나 무장 파일을 썼을 때만 남긴다 -- 2초마다 hold 를 쌓지 않게)."""
    try:
        if not enabled():
            return None
        c = new_chain("tick")
        buf: list = []
        _BUF.set(buf)
        tb = {"chain": c, "buf": buf, "slots0": json.loads(json.dumps(slots, ensure_ascii=False, default=str)), "file0": None}
        _TICK.set(tb)
        return tb
    except Exception:  # noqa: BLE001
        return None


def tick_file(st: Any) -> None:
    """틱이 읽은 무장 파일(이미 읽은 dict -- 다시 읽지 않는다). 틱 밖이면 아무것도 안 한다."""
    try:
        tb = _TICK.get()
        if tb is not None and tb["file0"] is None:
            tb["file0"] = json.dumps(st, ensure_ascii=False, default=str)
    except Exception:  # noqa: BLE001
        pass


def saved(st: Any) -> None:
    """bracket_save 가 부른다 -- 틱 안이면 «파일이 바뀜»으로 적어 둔다."""
    try:
        buf = _BUF.get()
        if buf is not None:
            buf.append(("saved", json.dumps(st, ensure_ascii=False, default=str), None))
    except Exception:  # noqa: BLE001
        pass


def tick_end(tb: Any, slots: dict, raised: str | None) -> None:
    try:
        if tb is None:
            return
        _BUF.set(None)
        _TICK.set(None)
        buf = tb["buf"]
        saves = [x[1] for x in buf if x[0] == "saved"]
        if not any(x[0] == "wire" for x in buf) and not saves:
            return
        t = time.time()
        # 시작 = 틱 첫 순간의 칸(slot = 진입 칸 · side = 코인별 청산 칸 · waits = 대기 칸) + 틱이 읽은 무장 파일
        head = {"rec": "start", "chain": tb["chain"], "t": t, "mono": time.monotonic(), "op": "tick", **tb["slots0"],
                "file": json.loads(tb["file0"]) if tb["file0"] else None}
        _put(json.dumps(head, ensure_ascii=False, default=str), None)
        for kind, row, lazy in buf:
            if kind != "saved":
                _put(json.dumps(row, ensure_ascii=False, default=str), lazy)
        end = {"rec": "end", "chain": tb["chain"], "t": time.time(), "mono": time.monotonic(), **json.loads(json.dumps(slots, ensure_ascii=False, default=str)),
               "file": json.loads(saves[-1] if saves else (tb["file0"] or "null")), "raised": raised}
        _put(json.dumps(end, ensure_ascii=False, default=str), None)
    except Exception:  # noqa: BLE001
        pass


# ── 뒤의 쓰기 스레드 ────────────────────────────────────────────────────────
def _put(line: str, lazy_dir: Path | None) -> None:
    global _writer
    _Q.put((SHADOW_PATH, line, lazy_dir))
    if _writer is None or not _writer.is_alive():
        _writer = threading.Thread(target=_write_loop, name="order-shadow-log", daemon=True)
        _writer.start()


def _resolve(line: str, lazy_dir: Path | None) -> str:
    if lazy_dir is None:
        return line
    row = json.loads(line)
    try:
        p = Path(lazy_dir) / "manual_bracket_state.json"
        row["file"] = json.loads(p.read_text("utf-8")) if p.exists() else {}
    except (OSError, ValueError):
        row["file"] = None
    try:
        row["limits"] = sorted(p.name for p in Path(lazy_dir).glob("manual_limit_wait*.json"))
    except OSError:
        row["limits"] = None
    return json.dumps(row, ensure_ascii=False, default=str)


def _write(path: Path, lines: list[str]) -> None:
    try:
        if path.exists() and path.stat().st_size > SHADOW_MAX_BYTES:
            os.replace(path, path.with_name(path.name + ".1"))
        fd = os.open(path, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o600)
        try:
            os.write(fd, "".join(x + "\n" for x in lines).encode("utf-8"))
        finally:
            os.close(fd)
    except OSError:
        pass


def _write_loop() -> None:
    while True:
        try:
            batch = [_Q.get()]
            while True:
                try:
                    batch.append(_Q.get_nowait())
                except queue.Empty:
                    break
            by: dict[Path, list[str]] = {}
            for path, line, lazy in batch:
                if path is not None:
                    by.setdefault(path, []).append(_resolve(line, lazy))
            for path, lines in by.items():
                _write(path, lines)
        except Exception:  # noqa: BLE001
            time.sleep(0.1)


def flush(timeout: float = 2.0) -> None:
    """시험용: 큐가 빌 때까지 기다린다."""
    t0 = time.monotonic()
    while not _Q.empty() and time.monotonic() - t0 < timeout:
        time.sleep(0.01)
    time.sleep(0.05)
