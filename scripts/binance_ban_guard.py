"""바이낸스 IP 차단(418)·한도(429) 공용 가드 (2026-09-26 IP 밴 사고).

같은 공인 IP 를 봇·대시보드·워커·수집기가 같이 쓴다. 차단 중에 누구든 요청을 보내면 해제 시각이 늘어난다
(실측: 14:48:50 → 15:04:59, 워커들이 주기마다 계속 두드렸다). 그래서 한 프로세스가 418/429 를 받으면 해제
시각을 **공용 파일**에 적고, 그 시각까지 가드를 단 모든 프로세스가 바이낸스 REST 를 **네트워크에 내보내지 않고**
즉시 실패시킨다(요청이 안 나가니 차단이 늘지 않는다).

사용: 프로세스 진입점에서 `import scripts.binance_ban_guard  # noqa: F401` 한 줄 -- requests 를 전역으로 감싼다.
aiohttp 를 쓰는 곳(대시보드)은 `ban_remaining()` / `note()` 를 직접 부른다.
ponytail: 파일 하나로 프로세스 간 공유한다(락 없음 -- 쓰기는 드물고 원자적 교체라 읽는 쪽이 반쪽을 못 본다).
"""
from __future__ import annotations

import os
import re
import time
from pathlib import Path
from urllib.parse import urlparse

BAN_FILE = Path(__file__).resolve().parents[1] / "data" / "live" / "binance_ban_until"
_BANNED_UNTIL = re.compile(r"banned until (\d{13})")


def is_binance(url: str) -> bool:
    return (urlparse(str(url)).hostname or "").endswith("binance.com")


def ban_remaining(now: float | None = None) -> float:
    """해제까지 남은 초(없으면 0)."""
    try:
        until = float(BAN_FILE.read_text().strip())
    except (OSError, ValueError):
        return 0.0
    return max(0.0, until - (time.time() if now is None else now))


def note(status: int, body: str = "", retry_after: str | None = None) -> float:
    """418/429 응답을 기록한다. 돌려주는 값 = 새 해제까지 남은 초(기록 안 했으면 0).
    418 은 본문의 «banned until <ms>», 429 는 Retry-After(없으면 30초)."""
    if status not in (418, 429):
        return 0.0
    now = time.time()
    m = _BANNED_UNTIL.search(body or "")
    if m:
        until = int(m.group(1)) / 1000.0
    elif retry_after and str(retry_after).isdigit():
        until = now + int(retry_after)
    else:
        until = now + (120.0 if status == 418 else 30.0)
    if until <= now + ban_remaining(now):
        return 0.0                                   # 이미 더 긴 기록이 있다
    BAN_FILE.parent.mkdir(parents=True, exist_ok=True)
    tmp = BAN_FILE.with_suffix(f".{os.getpid()}.tmp")
    tmp.write_text(f"{until:.3f}\n")
    tmp.replace(BAN_FILE)
    return until - now


def _install_requests_guard() -> None:
    try:
        import requests
    except ImportError:
        return
    orig = requests.Session.request
    if getattr(orig, "_binance_ban_guard", False):
        return

    class BinanceBanned(requests.exceptions.ConnectionError):
        """차단 중이라 **보내지 않은** 요청. 호출부의 기존 연결 오류 처리(재시도·건너뛰기)를 그대로 탄다."""

    def request(self, method, url, *args, **kwargs):
        if is_binance(url):
            left = ban_remaining()
            if left > 0:
                raise BinanceBanned(f"binance IP ban: {left:.0f}s left -- request not sent")
        resp = orig(self, method, url, *args, **kwargs)
        if is_binance(url) and resp.status_code in (418, 429):
            note(resp.status_code, resp.text[:1000], resp.headers.get("Retry-After"))
        return resp

    request._binance_ban_guard = True
    requests.Session.request = request
    requests.BinanceBanned = BinanceBanned


_install_requests_guard()


if __name__ == "__main__":   # 자체점검: 네트워크 없이 «차단 중이면 안 나간다»·기록 규칙
    import tempfile

    import requests
    BAN_FILE = Path(tempfile.mkdtemp()) / "ban"
    assert ban_remaining() == 0.0
    left = note(418, '{"code":-1003,"msg":"Way too many requests; IP(1.2.3.4) banned until %d. ..."}'
                % int((time.time() + 600) * 1000))
    assert 590 < left <= 600 and 590 < ban_remaining() <= 600, left
    assert note(429, "", "5") == 0.0, "더 짧은 기록이 긴 차단을 덮으면 안 된다"
    try:
        requests.get("https://fapi.binance.com/fapi/v1/ping", timeout=1)
        raise AssertionError("차단 중인데 요청이 나갔다")
    except requests.BinanceBanned:
        pass
    assert not is_binance("https://www.okx.com/api/v5/public/time") and is_binance("https://api.binance.com/x")
    print("binance_ban_guard selftest ok")
