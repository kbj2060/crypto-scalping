from __future__ import annotations

import argparse
import os
import re
import signal
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
TOOLS_DIR = REPO_ROOT / ".tools" / "cloudflared"
CLOUDFLARED_EXE = TOOLS_DIR / "cloudflared.exe"
WINDOWS_AMD64_URL = (
    "https://github.com/cloudflare/cloudflared/releases/latest/download/"
    "cloudflared-windows-amd64.exe"
)
URL_RE = re.compile(r"https://[-a-z0-9]+\.trycloudflare\.com", re.IGNORECASE)


def ensure_cloudflared() -> Path:
    TOOLS_DIR.mkdir(parents=True, exist_ok=True)
    if CLOUDFLARED_EXE.exists():
        return CLOUDFLARED_EXE

    tmp_fd, tmp_name = tempfile.mkstemp(suffix=".exe", dir=str(TOOLS_DIR))
    os.close(tmp_fd)
    tmp_path = Path(tmp_name)
    try:
        print("Downloading cloudflared...", flush=True)
        urllib.request.urlretrieve(WINDOWS_AMD64_URL, tmp_path)
        tmp_path.replace(CLOUDFLARED_EXE)
    finally:
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)
    return CLOUDFLARED_EXE


def start_dashboard_server(host: str, port: int) -> subprocess.Popen[str]:
    return subprocess.Popen(
        [
            sys.executable,
            str(REPO_ROOT / "scripts" / "serve_dashboard.py"),
            "--host",
            host,
            "--port",
            str(port),
        ],
        cwd=str(REPO_ROOT),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        text=True,
        creationflags=subprocess.CREATE_NEW_PROCESS_GROUP,
    )


def wait_for_origin(url: str, timeout: float = 15.0) -> None:
    deadline = time.time() + timeout
    last_error: Exception | None = None
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=2.0) as resp:
                if resp.status < 500:
                    return
        except Exception as exc:  # pragma: no cover - transient network/proc timing
            last_error = exc
            time.sleep(0.4)
    raise RuntimeError(f"Dashboard origin did not become ready: {last_error}")


def stop_process(proc: subprocess.Popen[str] | None) -> None:
    if proc is None or proc.poll() is not None:
        return
    try:
        proc.send_signal(signal.CTRL_BREAK_EVENT)
        proc.wait(timeout=5)
        return
    except Exception:
        proc.kill()
        proc.wait(timeout=5)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Expose the live dashboard through a Cloudflare quick tunnel."
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8787)
    args = parser.parse_args()

    cloudflared = ensure_cloudflared()
    server_proc: subprocess.Popen[str] | None = None
    tunnel_proc: subprocess.Popen[str] | None = None
    origin = f"http://{args.host}:{args.port}/"

    try:
        server_proc = start_dashboard_server(args.host, args.port)
        wait_for_origin(origin)

        tunnel_proc = subprocess.Popen(
            [
                str(cloudflared),
                "tunnel",
                "--no-autoupdate",
                "--url",
                origin,
            ],
            cwd=str(REPO_ROOT),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            creationflags=subprocess.CREATE_NEW_PROCESS_GROUP,
        )

        print(f"Origin ready: {origin}dashboard/live/", flush=True)
        print("Starting Cloudflare quick tunnel...", flush=True)

        public_url: str | None = None
        assert tunnel_proc.stdout is not None
        for line in tunnel_proc.stdout:
            print(line.rstrip(), flush=True)
            m = URL_RE.search(line)
            if m:
                public_url = m.group(0)
                print(f"PUBLIC_URL={public_url}", flush=True)
                break

        if public_url is None:
            raise RuntimeError("Cloudflare tunnel started but no public URL was found.")

        print("Tunnel is running. Press Ctrl+C to stop.", flush=True)
        tunnel_proc.wait()
    except KeyboardInterrupt:
        pass
    finally:
        stop_process(tunnel_proc)
        stop_process(server_proc)


if __name__ == "__main__":
    main()
