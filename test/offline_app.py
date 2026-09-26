"""대시보드 앱을 **백그라운드 작업 없이** 띄운다 -- 라우팅·직렬화만 보는 시험용 (2026-09-26).

make_app() 의 on_startup 에는 수집기(바이낸스·OKX WS, OI 0.25초 폴링, REST 백필)와 주문 브래킷 감시기가
걸려 있다. 시험이 그대로 띄우면 이 PC 에서 거래소로 요청이 나간다 -- 이 PC 는 서버와 **같은 공인 IP**라
그 요청이 서버·봇의 IP 한도를 같이 먹는다(09-26 418 밴 사고). HTTP 세션 하나만 남긴다(연결은 안 연다).
"""
from __future__ import annotations

KEEP = {"start_http_session", "stop_http_session"}


def offline_app(server):
    app = server.make_app()
    for sig in (app.on_startup, app.on_cleanup):
        keep = [f for f in sig if getattr(f, "__name__", "") in KEEP]
        del sig[:]
        sig.extend(keep)
    return app
