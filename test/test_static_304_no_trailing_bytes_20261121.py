"""2026-11-21 실서버: 압축 켠 화면 파일(app.js 등)의 304 머리 뒤에 빈 zlib 흐름 8B(789c030000000001)가 길이 없이 붙었다.

서버 모듈은 import 하지 않는다(무거운 의존·환경 파일) -- dashboard/server.py 원문에서 ASSET_FILES · asset_compress_ok ·
make_app 안의 static_asset_headers 미들웨어만 ast 로 떼어 작은 앱에 붙이고, 127.0.0.1 임시 포트에 원시 소켓으로 묻는다.
실행: python test/test_static_304_no_trailing_bytes_20261121.py
"""
from __future__ import annotations

import ast
import asyncio
import os
import socket
import tempfile
import zlib
from pathlib import Path
from typing import Any

from aiohttp import web

SRC = Path(__file__).resolve().parents[1] / "dashboard" / "server.py"


def load_middleware():
    tree = ast.parse(SRC.read_text(encoding="utf-8"))
    keep = [n for n in tree.body if (isinstance(n, ast.Assign) and any(getattr(t, "id", "") == "ASSET_FILES" for t in n.targets))
            or (isinstance(n, ast.FunctionDef) and n.name == "asset_compress_ok")]
    make_app = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "make_app")
    keep += [n for n in make_app.body if isinstance(n, ast.AsyncFunctionDef) and n.name == "static_asset_headers"]
    ns: dict[str, Any] = {"web": web, "Path": Path, "Any": Any, "os": os}
    exec(compile(ast.Module(body=keep, type_ignores=[]), str(SRC), "exec"), ns)
    return ns["static_asset_headers"]


def recv_until_idle(s: socket.socket) -> bytes:
    buf = b""
    s.settimeout(0.5)
    try:
        while chunk := s.recv(65536):
            buf += chunk
    except socket.timeout:
        pass
    return buf


def split_head(raw: bytes) -> tuple[str, dict[str, str], bytes]:
    head, _, rest = raw.partition(b"\r\n\r\n")
    lines = head.decode("latin-1").split("\r\n")
    return lines[0], {k.lower(): v.strip() for k, _, v in (ln.partition(":") for ln in lines[1:])}, rest


def dechunk(body: bytes) -> tuple[bytes, bytes]:
    out = b""
    while True:
        size_line, _, body = body.partition(b"\r\n")
        n = int(size_line, 16)
        if n == 0:
            return out, body.partition(b"\r\n")[2]
        out, body = out + body[:n], body[n + 2:]


async def main() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        content = b"console.log('x');\n" * 200
        Path(tmp, "app.js").write_bytes(content)
        app = web.Application(middlewares=[load_middleware()])
        app.router.add_static("/dashboard/live/", tmp)
        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, "127.0.0.1", 0)
        await site.start()
        port = site._server.sockets[0].getsockname()[1]
        req = "GET /dashboard/live/app.js HTTP/1.1\r\nHost: x\r\nAccept-Encoding: deflate\r\n{extra}\r\n"
        try:
            # (가) 200 + deflate 본문이 원본으로 풀린다
            with socket.create_connection(("127.0.0.1", port)) as s:
                s.sendall(req.format(extra="").encode())
                status, h, rest = split_head(await asyncio.to_thread(recv_until_idle, s))
            assert status.startswith("HTTP/1.1 200"), status
            assert h.get("content-encoding") == "deflate" and h.get("cache-control") == "no-cache", h
            body, tail = dechunk(rest)
            assert zlib.decompress(body) == content and tail == b"", tail
            etag = h["etag"]
            print(f"(가) 200 deflate {len(body)}B -> {len(content)}B 풀림 OK")

            # (나) If-None-Match 로 304 · 머리 뒤 남은 바이트 0  (다) 같은 연결의 둘째 요청이 정상 파싱
            with socket.create_connection(("127.0.0.1", port)) as s:
                s.sendall(req.format(extra=f"If-None-Match: {etag}\r\n").encode())
                status, h, rest = split_head(await asyncio.to_thread(recv_until_idle, s))
                print(f"(나) {status} · 머리 뒤 {len(rest)}B {rest.hex()} · content-encoding={h.get('content-encoding')}")
                assert status.startswith("HTTP/1.1 304"), status
                assert rest == b"", f"304 뒤 남은 바이트 {rest.hex()}"
                assert "content-encoding" not in h, h
                s.sendall(req.format(extra="").encode())
                status2, h2, rest2 = split_head(await asyncio.to_thread(recv_until_idle, s))
            assert status2.startswith("HTTP/1.1 200"), status2
            assert zlib.decompress(dechunk(rest2)[0]) == content
            print(f"(다) 같은 연결 둘째 요청 {status2} 정상 파싱 OK")

            # (라) 본문 없는 다른 경우: HEAD · If-Modified-Since 304
            for extra, method in (("", "HEAD"), (f"If-Modified-Since: {h['last-modified']}\r\n", "GET")):
                with socket.create_connection(("127.0.0.1", port)) as s:
                    s.sendall(req.format(extra=extra).replace("GET", method, 1).encode())
                    status, h3, rest = split_head(await asyncio.to_thread(recv_until_idle, s))
                print(f"(라) {method} {extra.strip() or '-'} -> {status} · 머리 뒤 {len(rest)}B {rest.hex()}")
                assert rest == b"" and "content-encoding" not in h3, (rest.hex(), h3)
        finally:
            await runner.cleanup()


if __name__ == "__main__":
    asyncio.run(main())
    print("PASS")
