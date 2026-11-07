#!/usr/bin/env python3
"""원시 호가 6TB 장애 대피소 (2026-11-07). 1분 cron · 표준 라이브러리만.

원시 호가는 다시 받을 수 없는데 6TB USB 하나에 바로 쓴다. 6TB 가 2회 연속 실패하면 data/live/orderflow/<stream> 링크를
D: 의 /mnt/work/orderflow_spool 로 돌리고 쓰는 수집기를 SIGTERM 한다(감독이 15초 뒤 새 경로로 다시 띄운다).
5회 연속 정상이면 되돌리고, 2분 넘게 안 바뀐 대피 파일을 6TB 로 옮긴다. 같은 시각 파일이 6TB 에 있으면
orderflow_partial/ 로 보낸다(이어 붙이면 파일 중간에 스냅숏이 끼어 읽는 쪽이 틀린다).

ponytail: D 상태로 멈춘 수집기는 SIGTERM 도 안 듣는다. 커널이 입출력 오류를 돌려줄 때까지 그 수집기는 대피소로 못 넘어간다.
ponytail: 읽는 쪽(ob5m·bookdepth·iceberg)은 일부러 안 죽인다(12시간 warm 이 날아간다). 다음 시각 파일부터 링크 너머로 넘어간다.
ponytail: 대피소와 연구 폴더가 같은 /mnt/work 라 연구가 꽉 채우면 대피소도 찬다. 감시기가 여유 100GiB 아래에서 WARN 한다.

  python3 scripts/ops/raw_failover_20261107.py             # 1회 판정
  python3 scripts/ops/raw_failover_20261107.py --selftest
"""
from __future__ import annotations

import fcntl
import json
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
LINKS = ROOT / "data" / "live" / "orderflow"
DATA2 = Path("/mnt/data2")
WORK = Path("/mnt/work")
SPOOL = WORK / "orderflow_spool"
STATE = ROOT / "data" / "live" / "ops_watchdog" / "raw_failover.json"
# depthdiff 를 마지막에 돌리고 그것으로 현재 위치를 판정한다 -- 도중에 죽으면 다음 회차가 이어서 돌린다.
STREAMS = ("bookticker", "hyperliquid", "hyperliquid_bookticker", "okx_bookticker", "raster_archive", "depthdiff")
FAIL_RUNS = 2
OK_RUNS = 5
SETTLE_S = 120


def log(msg: str) -> None:
    print(f"{time.strftime('%F %T')} {msg}", flush=True)


def is_mounted(path: Path) -> bool:
    """/proc/mounts 만 본다. os.path.ismount 는 멈춘 디스크를 건드려 이 프로세스까지 멈출 수 있다."""
    with open("/proc/mounts", encoding="utf-8", errors="replace") as fh:
        return any(line.split()[1] == str(path) for line in fh)


def healthy(data2: Path = DATA2, seconds: float = 30.0) -> bool:
    """쓰기 시험. 앞선 시험이 D 상태로 남아 있으면 새로 쌓지 않고 실패로 본다. 30초 근거 = ops_watchdog._probe_write."""
    if not is_mounted(data2):
        return False
    probe = str(data2 / ".raw_failover_probe")
    for pid in filter(str.isdigit, os.listdir("/proc")):
        try:
            if probe in Path(f"/proc/{pid}/cmdline").read_bytes().decode(errors="ignore"):
                return False
        except OSError:
            continue
    p = subprocess.Popen(["touch", probe], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        return p.wait(seconds) == 0
    except subprocess.TimeoutExpired:
        p.kill()
        return False


def point(links: Path, target_root: Path) -> None:
    for stream in STREAMS:
        link = links / stream
        if not link.is_symlink():      # 실물 폴더를 덮어쓰지 않는다
            log(f"🔴 not a symlink, left alone: {link}")
            continue
        tmp = links / f".{stream}.lnk"
        tmp.unlink(missing_ok=True)
        os.symlink(target_root / stream, tmp)
        os.replace(tmp, link)


def kill_writers(prefix: str) -> list[int]:
    """prefix 아래 파일을 쓰기로 연 수집기(scripts/live_*)만 끈다. 읽는 쪽과 그 밖의 프로세스는 건드리지 않는다."""
    killed = []
    for pid in filter(str.isdigit, os.listdir("/proc")):
        try:
            if "scripts/live_" not in Path(f"/proc/{pid}/cmdline").read_bytes().decode(errors="ignore"):
                continue
            for fd in os.listdir(f"/proc/{pid}/fd"):
                if not os.readlink(f"/proc/{pid}/fd/{fd}").startswith(prefix):
                    continue
                flags = int(Path(f"/proc/{pid}/fdinfo/{fd}").read_text().split("flags:")[1].split()[0], 8)
                if flags & (os.O_WRONLY | os.O_RDWR):
                    os.kill(int(pid), signal.SIGTERM)
                    killed.append(int(pid))
                    break
        except (OSError, IndexError, ValueError):
            continue
    return killed


def mirror_dirs(raw: Path, spool: Path) -> None:
    """심볼 폴더를 미리 만들어 둔다. 수집기는 시작할 때만 mkdir 하고, 장애가 나면 6TB 를 읽어 만들 수 없다."""
    for stream in STREAMS:
        (spool / stream).mkdir(parents=True, exist_ok=True)
        if (raw / stream).is_dir():
            for e in os.scandir(raw / stream):
                if e.is_dir():
                    (spool / stream / e.name).mkdir(exist_ok=True)


def drain(raw: Path, spool: Path, partial: Path, now: float, settle: float = SETTLE_S) -> tuple[int, list[str]]:
    moved, errors = 0, []
    for f in spool.rglob("*"):
        try:
            if not f.is_file():
                continue
            st = f.stat()
            if now - st.st_mtime < settle:
                continue
            rel = f.relative_to(spool)
            dest = raw / rel
            if dest.exists():
                dest = partial / rel
                if dest.exists():
                    dest = dest.with_name(f"{dest.name}.{int(now)}")
            dest.parent.mkdir(parents=True, exist_ok=True)
            tmp = dest.parent / f".draining_{os.getpid()}"
            shutil.copy2(f, tmp)
            fd = os.open(tmp, os.O_RDONLY)
            try:
                os.fsync(fd)           # 원본을 지우기 전에 사본이 디스크에 있어야 한다
            finally:
                os.close(fd)
            if tmp.stat().st_size != st.st_size:
                tmp.unlink()
                errors.append(f"size mismatch {rel}")
                continue
            os.replace(tmp, dest)
            f.unlink()
            moved += 1
        except OSError as exc:
            errors.append(f"{f.name}: {type(exc).__name__}: {exc}"[:200])
    return moved, errors


def _save(state_path: Path, st: dict) -> None:
    state_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = state_path.with_suffix(".tmp")
    tmp.write_text(json.dumps(st, ensure_ascii=False))
    tmp.replace(state_path)


def tick(links: Path, data2: Path, spool: Path, state_path: Path, data2_ok: bool, spool_ok: bool, kill=kill_writers, now: float | None = None) -> dict:
    now = time.time() if now is None else now
    try:
        st = json.loads(state_path.read_text())
    except (OSError, ValueError):
        st = {}
    st["ok_runs"] = st.get("ok_runs", 0) + 1 if data2_ok else 0
    st["fail_runs"] = 0 if data2_ok else st.get("fail_runs", 0) + 1
    raw = data2 / "orderflow"
    on_spool = os.readlink(links / "depthdiff").startswith(f"{spool}/")
    st["error"] = ""
    if not on_spool and st["fail_runs"] >= FAIL_RUNS:
        if spool_ok:
            point(links, spool)
            on_spool = True
            log(f"🔴 6TB failed {st['fail_runs']} runs: raw -> spool, restarted writers {kill(f'{raw}/')}")
        else:
            st["error"] = "6TB failed and spool not mounted"
    elif on_spool and st["ok_runs"] >= OK_RUNS:
        point(links, raw)
        on_spool = False
        log(f"6TB back {st['ok_runs']} runs: raw -> 6TB, restarted writers {kill(f'{spool}/')}")
    st.update(on_spool=on_spool, at=now)
    _save(state_path, st)              # 옮겨 담기가 멈춰도 상태는 남긴다
    if data2_ok and not on_spool and spool_ok:
        with open(state_path.with_suffix(".lock"), "w") as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError:
                return st              # 앞 회차가 옮기는 중이거나 멈춰 있다 -- 판정만 하고 끝낸다
            mirror_dirs(raw, spool)
            moved, errors = drain(raw, spool, data2 / "orderflow_partial", now)
        if moved or errors:
            log(f"drained {moved} files, errors {errors[:3]}")
        st["error"] = "; ".join(errors[:3])
        _save(state_path, st)
    return st


def selftest() -> None:
    with tempfile.TemporaryDirectory() as d:
        r = Path(d)
        links, data2, spool, state = r / "links", r / "data2", r / "spool", r / "state.json"
        raw = data2 / "orderflow"
        links.mkdir()
        for s in STREAMS:
            (raw / s / "ETH").mkdir(parents=True)
            os.symlink(raw / s, links / s)
        kills: list[str] = []
        args = dict(links=links, data2=data2, spool=spool, state_path=state, kill=lambda p: kills.append(p) or [])

        # 정상: 그대로 + 폴더 미리 생성
        assert not tick(**args, data2_ok=True, spool_ok=True)["on_spool"]
        assert (spool / "depthdiff" / "ETH").is_dir() and not kills
        # 1회 실패는 안 돌린다(부팅 직후 마운트 전 등)
        assert not tick(**args, data2_ok=False, spool_ok=True)["on_spool"]
        # 대피소가 없으면 돌리지 않고 오류를 남긴다
        st = tick(**args, data2_ok=False, spool_ok=False)
        assert not st["on_spool"] and "spool" in st["error"] and not kills
        # 대피소가 있으면 돌리고 6TB 쪽 수집기를 재시작한다
        assert tick(**args, data2_ok=False, spool_ok=True)["on_spool"]
        assert kills == [f"{raw}/"]
        assert os.readlink(links / "depthdiff") == str(spool / "depthdiff")
        assert os.readlink(links / "bookticker") == str(spool / "bookticker")
        # 링크로 쓰면 대피소에 들어간다: 겹치는 시각 1개 + 안 겹치는 시각 1개 + 아직 쓰는 파일 1개
        (raw / "depthdiff/ETH/T10.jsonl").write_text("before\n")
        for name in ("T10.jsonl", "T11.jsonl.gz"):
            (links / "depthdiff/ETH" / name).write_text(f"spool {name}\n")
            os.utime(spool / "depthdiff/ETH" / name, (time.time() - 600,) * 2)
        (links / "depthdiff/ETH/T12.jsonl").write_text("still writing\n")
        # 4회 정상까지는 그대로, 옮기지도 않는다
        for _ in range(4):
            assert tick(**args, data2_ok=True, spool_ok=True)["on_spool"]
        assert (spool / "depthdiff/ETH/T11.jsonl.gz").exists()
        # 다시 실패하면 정상 횟수가 초기화된다(USB 가 깜빡일 때 왕복 금지)
        tick(**args, data2_ok=False, spool_ok=True)
        for _ in range(4):
            assert tick(**args, data2_ok=True, spool_ok=True)["on_spool"]
        # 5회 연속 정상: 되돌리고 대피소 쪽 수집기를 재시작하고 옮겨 담는다
        st = tick(**args, data2_ok=True, spool_ok=True)
        assert not st["on_spool"] and kills[-1] == f"{spool}/", (st, kills)
        assert os.readlink(links / "depthdiff") == str(raw / "depthdiff")
        assert (raw / "depthdiff/ETH/T11.jsonl.gz").read_text() == "spool T11.jsonl.gz\n"
        assert (raw / "depthdiff/ETH/T10.jsonl").read_text() == "before\n", "6TB 파일을 덮어쓰면 안 된다"
        assert (data2 / "orderflow_partial/depthdiff/ETH/T10.jsonl").read_text() == "spool T10.jsonl\n"
        assert (spool / "depthdiff/ETH/T12.jsonl").exists(), "방금 쓴 파일은 아직 옮기면 안 된다"
        assert not (spool / "depthdiff/ETH/T11.jsonl.gz").exists()
        # 같은 이름이 partial 에도 있으면 덮어쓰지 않는다
        (spool / "depthdiff/ETH/T10.jsonl").write_text("second\n")
        os.utime(spool / "depthdiff/ETH/T10.jsonl", (time.time() - 600,) * 2)
        tick(**args, data2_ok=True, spool_ok=True)
        assert (data2 / "orderflow_partial/depthdiff/ETH/T10.jsonl").read_text() == "spool T10.jsonl\n"
        assert [p.read_text() for p in (data2 / "orderflow_partial/depthdiff/ETH").glob("T10.jsonl.*")] == ["second\n"]
        # 실물 폴더는 링크로 덮어쓰지 않는다
        real = links / "okx_bookticker"
        real.unlink()
        real.mkdir()
        point(links, spool)
        assert real.is_dir() and not real.is_symlink()

        # kill_writers: 쓰는 수집기만 끈다. 양성·음성 대조는 실제 프로세스로 한다.
        (r / "scripts").mkdir()
        target = raw / "depthdiff/ETH/live.jsonl"
        target.write_text("")
        body = "import sys,time\nf=open(sys.argv[1],sys.argv[2]);print('up',flush=True);time.sleep(30)\n"
        (r / "scripts/live_w.py").write_text(body)
        (r / "scripts/other.py").write_text(body)
        procs = {name: subprocess.Popen([sys.executable, str(r / "scripts" / script), str(target), mode], stdout=subprocess.PIPE)
                 for name, script, mode in (("writer", "live_w.py", "a"), ("reader", "live_w.py", "r"), ("non_live", "other.py", "a"))}
        for p in procs.values():
            assert p.stdout.readline().strip() == b"up"
        killed = kill_writers(f"{raw}/")
        assert killed == [procs["writer"].pid], killed
        assert procs["writer"].wait(10) == -signal.SIGTERM
        assert procs["reader"].poll() is None and procs["non_live"].poll() is None
        for p in procs.values():
            p.kill()
            p.wait()
            p.stdout.close()
    assert not healthy(Path("/nonexistent/mount"))
    print("selftest OK")


def main() -> int:
    if "--selftest" in sys.argv:
        selftest()
        return 0
    tick(LINKS, DATA2, SPOOL, STATE, healthy(), is_mounted(WORK))
    return 0


if __name__ == "__main__":
    sys.exit(main())
