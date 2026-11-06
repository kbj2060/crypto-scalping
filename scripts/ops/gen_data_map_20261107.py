#!/usr/bin/env python3
"""데이터 지도 생성 (2026-11-07). 손으로 쓴 인벤토리는 낡는다(09-16 판은 10-01 재설계 뒤 틀렸다) -- 그래서 실물에서 만든다.

디스크 여유 · data/ 항목의 실제 위치(6TB 링크 여부) · 원시 스트림 코인별 크기·최신 파일 · hot 파일 · 수집기 매니페스트를
data/live/ops_watchdog/data_map.md 로 쓴다(추적 안 함 -- 서버가 저장소 파일을 고치면 배포 워처와 충돌한다).
원칙과 규칙은 docs/data_storage_map_20261107.md, 지금 상태는 이 파일이다.

cron(서버, 매일 11:30 KST, 봉인 뒤):  python scripts/ops/gen_data_map_20261107.py
"""
from __future__ import annotations

import json
import os
import shutil
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DISKS = (("C:", Path("/mnt/c")), ("D:", Path("/mnt/d")), ("6TB", Path("/mnt/data2")), ("NVMe VHDX", Path("/mnt/nvme")), ("WSL 루트(가상 크기 -- 한계 아님)", Path("/")))


def _gb(n: float) -> str:
    return f"{n / 2**30:,.1f}GB"


def _where(p: Path) -> str:
    return f"→ {os.readlink(p)}" if p.is_symlink() else "(로컬)"


def _dir_stats(p: Path) -> tuple[int, int, str]:
    """(파일 수, 바이트, 최신 파일 이름). 시각 파일 폴더라 평평하다 -- 한 단계만 본다."""
    n = size = 0
    newest, name = 0.0, "-"
    try:
        entries = list(os.scandir(p))
    except OSError:
        return 0, 0, "(못 읽음)"
    for e in entries:
        if not e.is_file(follow_symlinks=False):
            continue
        st = e.stat()
        n, size = n + 1, size + st.st_size
        if st.st_mtime > newest:
            newest, name = st.st_mtime, e.name
    return n, size, name


def render(root: Path, disks=DISKS, manifest: dict | None = None) -> str:
    out = [f"# 데이터 지도 (자동 생성 {time.strftime('%Y-%m-%d %H:%M %Z')})", "",
           "원칙·규칙은 docs/data_storage_map_20261107.md. 이 파일은 scripts/ops/gen_data_map_20261107.py 가 실물에서 만든다.", "",
           "## 디스크", "| 디스크 | 여유 | 전체 |", "|---|---|---|"]
    for name, p in disks:
        if p.exists() and (p == Path("/") or p == root or os.path.ismount(p)):
            u = shutil.disk_usage(p)
            out.append(f"| {name} | {_gb(u.free)} | {_gb(u.total)} |")
        else:
            out.append(f"| {name} | 🔴 안 붙음 | - |")
    data = root / "data"
    out += ["", "## data/ 항목의 실제 위치", "| 항목 | 위치 |", "|---|---|"]
    for e in sorted(data.iterdir()):
        if e.is_dir():
            out.append(f"| data/{e.name} | {_where(e)} |")
    of = data / "live" / "orderflow"
    out += ["", "## 원시 스트림 (data/live/orderflow)", "| 스트림 | 코인 | 파일 | 크기 | 최신 파일 | 위치 |", "|---|---|---:|---:|---|---|"]
    for s in sorted(of.iterdir()) if of.exists() else []:
        for c in sorted(s.iterdir()) if s.is_dir() else []:
            if c.is_dir():
                n, size, name = _dir_stats(c)
                out.append(f"| {s.name} | {c.name} | {n:,} | {_gb(size)} | {name} | {_where(s)} |")
    out += ["", "## hot (data/hot)", "| 파일 | 크기 |", "|---|---:|"]
    for f in sorted((data / "hot").glob("*.sqlite")) if (data / "hot").exists() else []:
        out.append(f"| {f.name} | {_gb(f.stat().st_size)} |")
    if manifest is None:
        try:
            manifest = json.loads((data / "live" / "multicoin_collectors.json").read_text())
        except (OSError, ValueError):
            manifest = {"collectors": {}}
    out += ["", "## 수집기 매니페스트 (multicoin_collectors.json -- ETH 전용 수집기는 crontab)", "| 키 | 감독 | 환경 |", "|---|---|---|"]
    for k, c in sorted(manifest.get("collectors", {}).items()):
        env = " ".join(f"{a}={b}" for a, b in c.get("env", {}).items())
        out.append(f"| {k} | {c.get('supervisor', '')} | {env} |")
    return "\n".join(out) + "\n"


def main() -> None:
    dst = ROOT / "data" / "live" / "ops_watchdog" / "data_map.md"
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(".md.tmp")
    tmp.write_text(render(ROOT), encoding="utf-8")
    tmp.replace(dst)
    print(dst)


if __name__ == "__main__":
    main()
