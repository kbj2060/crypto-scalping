from __future__ import annotations

import argparse
import functools
import os
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


class DashboardHandler(SimpleHTTPRequestHandler):
    def do_GET(self) -> None:
        if self.path in {"", "/"}:
            self.send_response(HTTPStatus.FOUND)
            self.send_header("Location", "/dashboard/live/")
            self.end_headers()
            return
        super().do_GET()

    def end_headers(self) -> None:
        # The dashboard polls JSON/JSONL files from the same origin.
        self.send_header("Cache-Control", "no-store")
        super().end_headers()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Serve the repo root and redirect / to /dashboard/live/."
    )
    parser.add_argument("--host", default=os.getenv("DASHBOARD_HOST", "127.0.0.1"))
    parser.add_argument(
        "--port", type=int, default=int(os.getenv("DASHBOARD_PORT", "8787"))
    )
    args = parser.parse_args()

    handler = functools.partial(DashboardHandler, directory=str(REPO_ROOT))
    server = ThreadingHTTPServer((args.host, args.port), handler)

    print(
        f"Serving dashboard at http://{args.host}:{args.port}/dashboard/live/ "
        f"(root={REPO_ROOT})",
        flush=True,
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
