#!/usr/bin/env python3
"""같은 시작점 재생 대조(2026-11-22): ob5m · 호가 깊이 10열(bookdepth)은 값이 «사슬을 어디서 시작했나·이력(z 점수 288봉·1시간 변화)»에 달려
실시간 그림자로는 같은 값을 기대할 수 없다(그림자는 따로 뜬 때부터 센다). 그래서 두 구현을 **같은 h0 · 같은 시 파일 · 같은 «지금»** 으로 한 번씩 돌려 봉을 견준다.

  1. depthdiff 시 파일 [끝 시 − 25, 끝 시] 26개를 임시 폴더로 복사(실시간으로 자라는 파일을 두 쪽이 다른 길이로 읽지 않게)
  2. 파이썬 원본 main() -- time.time() = (끝 시 + 1)시 + 180초로 멈춤 · 기다려야 하면(파일 끝) 멈춤 · DB = 임시 파일(빈 DB → h0 = 지금 − 26시)
  3. Rust 같은 종류 -- RTCOLLECT_FAKE_NOW 같은 값 · OB_DB 임시 · DD_ROOT 임시
  4. shadow_tape 로 bars·marks 대조(warm 봉 빼고 · 호가 깊이 원값은 상대 1e-9) → «요약: 견준 N · 설명 못 함 M» · M>0 이거나 N=0 이면 종료코드 1
읽기 전용(원본 시 파일·hot 테이프는 읽기만 · 쓰는 것은 --tmp 아래뿐 · 끝나면 지운다). 거래소 접속 없음.

    nice -n 10 python3 rtcollect/tools/replay_book.py bookdepth --end-hour 2026-11-21T14 [--tmp 6TB 폴더] [--keep]
    nice -n 10 python3 rtcollect/tools/replay_book.py ob5m --symbol BTCUSDT --end-hour 2026-11-21T14
"""
from __future__ import annotations

import argparse
import calendar
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))

PY_RUNNER = r'''
import functools, sys, time
root, kind, sym, dd, db, now = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5], float(sys.argv[6])
sys.path.insert(0, root)
sys.argv = ["x", "--symbol", sym] if kind == "ob5m" else ["x"]   # live_ob5m 은 import 때 --symbol 을 읽는다
class Stop(BaseException):
    pass
def _sleep(s):
    raise Stop
time.time, time.sleep = (lambda: now), _sleep
from pathlib import Path
import scripts.live_ob5m_20261026 as ob
ob.DD = Path(dd)
m = ob if kind == "ob5m" else __import__("scripts.live_bookdepth_feat_20261028", fromlist=["x"])
assert Path(db).resolve() != Path(m.DB).resolve(), "진짜 DB"
m.db_open = functools.partial(m.db_open, Path(db))   # 기본 인자는 정의 때 굳었다 -- 바꿔 끼운다
try:
    m.main()
except Stop:
    pass
'''


def stage(name: str, cmd: list, deadline: float, **kw) -> tuple:
    """하위 프로세스 하나 -- 줄이 나오는 대로 바로 찍는다(파일로 보내도 버퍼에 갇히지 않게 flush) · 전체 상한(deadline)을 넘으면 죽이고 «시간 초과».
    11-22 서버: check 가 이 도구를 파일로 돌리자 40분 동안 0바이트였다 -- 출력이 블록 버퍼에 갇혀 어디까지 갔는지 남지 않았다."""
    import threading
    t0 = time.time()
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1, **kw)
    tail: list = []

    def pump():
        for ln in p.stdout:
            tail.append(ln.rstrip("\n"))
            del tail[:-40]
            print(f"    [{name} {time.time() - t0:.0f}초] {ln.rstrip()[:200]}", flush=True)
    th = threading.Thread(target=pump, daemon=True)
    th.start()
    try:
        p.wait(timeout=max(1.0, deadline - time.time()))
    except subprocess.TimeoutExpired:
        p.kill()
        p.wait()
        th.join(5)
        print(f"🔴시간 초과({name} 단계 · 그 단계 {time.time() - t0:.0f}초 · 전체 상한을 넘었다)", flush=True)
        return None, tail
    th.join(5)
    print(f"  {name} {time.time() - t0:.0f}초 · 종료 {p.returncode}", flush=True)
    return p.returncode, tail


def hour_name(h: int) -> str:
    return time.strftime("%Y-%m-%dT%H", time.gmtime(h * 3600))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("kind", choices=["ob5m", "bookdepth"])
    ap.add_argument("--symbol", default="ETHUSDT")
    ap.add_argument("--end-hour", required=True, help="마지막으로 넣을 온전한 시(UTC, YYYY-MM-DDTHH)")
    ap.add_argument("--dd-root", type=Path, default=ROOT / "data/live/orderflow/depthdiff")
    ap.add_argument("--tmp", type=Path, default=ROOT / "data/live/rtcollect", help="복사본·임시 DB 자리(6TB 쪽)")
    ap.add_argument("--bin", type=Path, default=ROOT / "tmp/rtbin/rtcollect")
    ap.add_argument("--keep", action="store_true")
    ap.add_argument("--limit", type=int, default=int(os.environ.get("REPLAY_LIMIT_S", "1800")), help="전체 상한(초) -- 넘으면 «시간 초과(단계)» 로 종료코드 2")
    a = ap.parse_args()
    sys.stdout.reconfigure(line_buffering=True)   # 파일로 보내도 줄마다 바로
    T0 = time.time()
    deadline = T0 + a.limit
    sym = a.symbol.upper()
    # 11-22 서버: 이 변수를 모르는 옛 실행 파일은 벽시계로 돌고 파일 끝에서 영원히 기다린다(시작 시 h0 가 파이썬과 한 시 어긋남 · 상한까지 침묵) -- 먼저 거른다
    if b"RTCOLLECT_FAKE_NOW" not in a.bin.read_bytes():
        print(f"🔴{a.bin} 는 RTCOLLECT_FAKE_NOW 를 모르는 옛 실행 파일 -- 다시 빌드(be4ab731 뒤)한 것으로")
        return 2
    if a.kind == "bookdepth" and sym != "ETHUSDT":
        ap.error("bookdepth 는 ETHUSDT 만")
    h_end = calendar.timegm(time.strptime(a.end_hour, "%Y-%m-%dT%H")) // 3600
    now = (h_end + 1) * 3600 + 180.0          # 끝 시 다음 시 3분 -- 끝 시 파일은 «닫힌 것», 다음 시는 «아직 안 생김»으로 보인다
    h0 = h_end - 25                            # 빈 DB → h0 = 지금 시 − 26 (두 구현 같은 식)
    src = a.dd_root / sym
    a.tmp.mkdir(parents=True, exist_ok=True)
    d = Path(tempfile.mkdtemp(prefix=".replay_book_", dir=a.tmp))
    try:
        (d / "dd" / sym).mkdir(parents=True)
        n = 0
        tc = time.time()
        for h in range(h0, h_end + 1):
            for ext in (".jsonl", ".jsonl.gz"):    # 원본 hour_path 와 같은 차례(.jsonl 먼저) -- 있는 것을 그대로
                p = src / f"{hour_name(h)}{ext}"
                if p.exists():
                    shutil.copy2(p, d / "dd" / sym / p.name)
                    n += 1
                    break
        print(f"{a.kind} {sym}: 시 파일 {n}개 복사({hour_name(h0)} ~ {hour_name(h_end)}) {time.time() - tc:.0f}초 · 지금 = {time.strftime('%F %T', time.gmtime(now))}Z · 상한 {a.limit}초")
        if n == 0:
            print("🔴시 파일이 없다")
            return 1
        runner = PY_RUNNER if not os.environ.get("REPLAY_FAKE_PY_SLEEP") else (   # 자체 시험: 느린 가짜 파이썬
            "import sys, time\nfor i in range(int(sys.argv[7])):\n    print('봉', i, flush=True); time.sleep(1)\n")
        rc, tail = stage("파이썬", [sys.executable, "-u", "-c", runner, str(ROOT), a.kind, sym, str(d / "dd" / sym), str(d / "py.sqlite"), str(now),
                                    os.environ.get("REPLAY_FAKE_PY_SLEEP", "0")], deadline, cwd=ROOT)
        if rc is None:
            return 2
        if rc:
            print("\n".join(tail[-20:]))
            return 1
        env = {k: v for k, v in os.environ.items() if not k.startswith("RTCOLLECT_")}
        env.update(RTCOLLECT_KIND=a.kind, OB_SYMBOL=sym, DD_ROOT=str(d / "dd"), OB_DB=str(d / "rs.sqlite"), RTCOLLECT_FAKE_NOW=repr(now),
                   TAPE_DB_PATH=str(ROOT / "data/hot/binance_tape.sqlite"))
        rc, tail = stage("Rust", [str(a.bin)], deadline, cwd=ROOT, env=env)
        if rc is None:
            return 2
        if rc:
            print("\n".join(tail[-20:]))
            return 1
        import shadow_tape as S
        S.TALLY[:] = [0, 0]
        S.INTENDED_EXTRA.pop("bars", None)   # 같은 시작점이면 «Rust 에만 봉»도 설명이 안 된다
        out = S.run(d / "py.sqlite", d / "rs.sqlite", show=5)
        print("\n".join("  " + x for x in out))
        print(f"요약: 견준 {S.TALLY[0]:,} · 설명 못 함 {S.TALLY[1]:,} · 전체 {time.time() - T0:.0f}초")
        return 1 if S.TALLY[1] or not S.TALLY[0] else 0
    finally:
        if a.keep:
            print(f"  남김: {d}")
        else:
            shutil.rmtree(d, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
