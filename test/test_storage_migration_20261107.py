"""2026-11-07 저장 이전: 실제 디스크 감시 · 백업 결과 감시 · 백업 스크립트 가드 · 이동 스크립트 · 데이터 지도."""
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
import ops_watchdog as w  # noqa: E402


def _tmp() -> Path:
    return Path(tempfile.mkdtemp())


# ── 감시기 ───────────────────────────────────────────────────────────
def test_unmounted_data2_is_critical():
    c = w.check_host_disks(disks=(), data2=_tmp(), fast=None)
    assert c.status == "CRITICAL" and "not mounted" in c.summary


def test_real_disk_thresholds_not_wsl_virtual_free():
    d = _tmp()
    with mock.patch.object(w.os.path, "ismount", return_value=True):
        ok = w.check_host_disks(disks=(("X", d, 0.001, 0.0001),), data2=d, data2_warn=0.001, data2_crit=0.0001, fast=None)
        warn = w.check_host_disks(disks=(("X", d, 1e12, 0.0001),), data2=d, data2_warn=0.001, data2_crit=0.0001, fast=None)
        crit = w.check_host_disks(disks=(("X", d, 1e12, 1e12),), data2=d, data2_warn=0.001, data2_crit=0.0001, fast=None)
    assert (ok.status, warn.status, crit.status) == ("OK", "WARN", "CRITICAL")


def test_probe_reports_failure_and_hang_without_blocking():
    assert w._probe_write(Path("/proc/no/such/file")) is not None
    assert w._probe_write(_tmp() / "ok") is None
    fake = mock.Mock()
    fake.poll.return_value = None                # D 상태 프로세스 흉내: 끝나지 않는다
    with mock.patch.object(w.subprocess, "Popen", return_value=fake):
        t = time.monotonic()
        err = w._probe_write(Path("/x"), seconds=0.3)
    assert "hung" in err and time.monotonic() - t < 2 and fake.kill.called


def _status(d: Path, age_h: float, **targets) -> Path:
    p = d / "backup_status.json"
    p.write_text(json.dumps({"finished_at": time.time() - age_h * 3600,
                             "targets": {"hot_6tb": 0, "cold_d": "skipped_vhdx_on_d", "dev": 0, **targets}}))
    return p


def test_backup_check():
    d, none = _tmp(), _tmp() / "none.json"
    assert w.check_backup(_status(d, 1), none).status == "OK"
    assert w.check_backup(_status(d, 30), none).status == "WARN"
    assert w.check_backup(_status(d, 60), none).status == "CRITICAL"
    assert w.check_backup(_status(d, 1, dev=1), none).status == "CRITICAL"
    assert w.check_backup(d / "missing.json", none).status == "CRITICAL"
    r = d / "restore.json"
    r.write_text(json.dumps({"ok": False, "finished_at": time.time()}))
    assert w.check_backup(_status(d, 1), r).status == "CRITICAL"
    r.write_text(json.dumps({"ok": True, "finished_at": time.time() - 41 * 86400}))
    assert w.check_backup(_status(d, 1), r).status == "WARN"


def test_fast_disk_missing_is_critical():
    d = _tmp()
    with mock.patch.object(w.os.path, "ismount", side_effect=lambda p: Path(p) == d):
        c = w.check_host_disks(disks=(), data2=d, data2_warn=0.001, data2_crit=0.0001, fast=_tmp())
    assert c.status == "CRITICAL" and "not mounted" in c.summary


def test_burn_rate_warns_but_one_off_move_does_not():
    a = _tmp()
    now = time.mktime((2026, 11, 20, 12, 0, 0, 0, 0, -1))
    def rec(days, gb):
        d = time.strftime("%Y%m%d", time.localtime(now - days * 86400))
        (a / f"{d}.json").write_text(json.dumps({"disk": {"D_drive": {"free_gb": gb}}}))
    rec(7, 156); rec(1, 108)                # 하루 8GB 씩 꾸준히
    assert 12 < w._days_left("D_drive", 100, a, now) < 13
    rec(7, 200); rec(1, 100.1)              # 어제 크게 옮긴 뒤로는 안 줄어든다
    assert w._days_left("D_drive", 100, a, now) > 14
    rec(1, 99)                              # 늘었다 = 소진 없음
    assert w._days_left("D_drive", 100, a, now) is None
    assert w._days_left("nvme", 100, a, now) is None, "기록 없는 디스크"
    assert w._days_left("D_drive", 100, _tmp(), now) is None, "기록 없음"
    rec(7, 156); rec(1, 108)
    d = _tmp()
    with mock.patch.object(w.shutil, "disk_usage", return_value=mock.Mock(free=100 * 2**30)), \
            mock.patch.object(w.os.path, "ismount", return_value=False):
        c = w.check_host_disks(disks=(("D_drive", d, 50, 20),), data2=d, fast=None, audit=a, now=now)
    assert "D_drive 12 days left" in c.summary and c.details["D_drive_days_left"] == 12.5


def test_seal_check():
    d, now = _tmp(), time.time()
    p = d / "seal.json"
    assert w.check_seal(p, now).status == "WARN", "기록 없음"
    p.write_text(json.dumps({"finished_at": now - 3600, "failed": 0, "written": 9}))
    assert w.check_seal(p, now).status == "OK"
    p.write_text(json.dumps({"finished_at": now - 3600, "failed": 1}))
    assert w.check_seal(p, now).status == "WARN"
    p.write_text(json.dumps({"finished_at": now - 30 * 3600, "failed": 0}))
    assert w.check_seal(p, now).status == "WARN"
    p.write_text(json.dumps({"finished_at": now - 60 * 3600, "failed": 0}))
    assert w.check_seal(p, now).status == "CRITICAL"


def test_seal_survives_a_stream_that_cannot_be_read():
    import datetime as dt
    import sqlite3
    from scripts import seal_to_lake as seal
    root, lake = _tmp(), _tmp() / "lake"
    (root / "data").mkdir()
    con = sqlite3.connect(root / "data/ok.sqlite")
    con.execute("create table t(symbol text, ts_ms integer)")
    now = dt.datetime(2026, 11, 7, 3, tzinfo=dt.timezone.utc)
    ms = int((now - dt.timedelta(days=1)).timestamp() * 1000)
    con.execute("insert into t values ('ETH', ?)", [ms])
    con.commit()
    con.close()
    # 열 이름이 틀린 시각식 = 읽다가 나는 duckdb 오류 대신 쓰는 재현(뒤 스트림이 그래도 돌아야 한다)
    specs = [("x", "bad", [("data/ok.sqlite", "t", "symbol")], "1 / (select 1 from range(2))"),
             ("x", "good", [("data/ok.sqlite", "t", "symbol")], "ts_ms")]
    s = seal.seal(specs, root=root, lake=lake, now=now)
    assert s["failed"] == 1 and s["written"] == 1
    assert list((lake / "x/good").rglob("part.parquet")) and not (lake / "x/bad").exists()


# ── 백업 스크립트 ─────────────────────────────────────────────────────
def _repo() -> Path:
    r = _tmp() / "repo"
    (r / "scripts/ops").mkdir(parents=True)
    shutil.copy(ROOT / "scripts/ops/backup_live_data.sh", r / "scripts/ops/")
    (r / "data/live/orderflow/raster/ETHUSDT").mkdir(parents=True)
    (r / "data/live/orderflow/raster/ETHUSDT/a.f32").write_text("raster")
    (r / "data/live/state.json").write_text("{}")
    (r / "data/models").mkdir()
    (r / "data/models/m.bin").write_text("m")
    (r / "data/binance_vision").mkdir()
    (r / "data/binance_vision/z.zip").write_text("z")
    return r


def _backup(r: Path, **env) -> tuple[int, dict]:
    p = subprocess.run(["bash", str(r / "scripts/ops/backup_live_data.sh")], capture_output=True, text=True,
                       env={**os.environ, **env})
    return p.returncode, json.loads((r / "data/live/ops_watchdog/backup_status.json").read_text())


def test_backup_never_writes_into_unmounted_data2():
    r, data2 = _repo(), _tmp()
    rc, st = _backup(r, BACKUP_DATA2=str(data2), BACKUP_VHDX_PATH="/nonexistent")
    assert rc != 0 and st["targets"]["hot_6tb"] == 1
    assert not list(data2.iterdir()), "안 붙은 6TB 폴더에 쓰면 루트 디스크가 찬다"


def test_backup_follows_live_symlink_to_nvme():
    r, data2, nvme = _repo(), _tmp(), _tmp()
    shutil.move(str(r / "data/live"), str(nvme / "live"))
    os.symlink(nvme / "live", r / "data/live")          # 11-07 전환 뒤의 모양
    os.symlink(_tmp(), nvme / "live/orderflow/depthdiff")
    rc, st = _backup(r, BACKUP_DATA2=str(data2), BACKUP_SKIP_MOUNT_CHECK="1", BACKUP_VHDX_PATH="/nonexistent",
                     BACKUP_COLD_DEST=str(_tmp() / "cold"))
    dest = data2 / "crypto-scalping-backups/data"
    assert st["targets"]["hot_6tb"] == 0 and (dest / "live/state.json").is_file(), "링크라고 data/live 를 건너뛰면 안 된다"
    assert not (dest / "live").is_symlink() and not (dest / "live/orderflow/raster").exists()
    assert not (dest / "live/orderflow/depthdiff").exists(), "6TB 링크는 cold 단계가 실물로 보낸다"


def test_backup_scope_excludes_and_skips_d_while_vhdx_there():
    r, data2 = _repo(), _tmp()
    vhdx = _tmp() / "ext4.vhdx"
    vhdx.write_text("")
    rc, st = _backup(r, BACKUP_DATA2=str(data2), BACKUP_SKIP_MOUNT_CHECK="1", BACKUP_VHDX_PATH=str(vhdx),
                     BACKUP_COLD_DEST=str(_tmp() / "cold"), BACKUP_D_MIN_FREE_GIB="999999999")
    assert st["targets"] == {"hot_6tb": 0, "cold_d": "skipped_vhdx_on_d", "dev": 1}   # 이 시험엔 dev 설정이 없다
    dest = data2 / "crypto-scalping-backups/data"
    assert (dest / "live/state.json").exists() and (dest / "models/m.bin").exists()
    assert not (dest / "live/orderflow/raster").exists(), "raster 는 14일 보존 -- 사본에 쌓지 않는다"
    assert not (dest / "binance_vision").exists()


def test_cold_runs_on_d_when_there_is_room_even_with_vhdx_there():
    r, data2, cold = _repo(), _tmp(), _tmp() / "cold"
    (data2 / "orderflow/depthdiff/ETHUSDT").mkdir(parents=True)
    (data2 / "orderflow/depthdiff/ETHUSDT/a.jsonl.gz").write_text("z")
    vhdx = _tmp() / "ext4.vhdx"
    vhdx.write_text("")
    rc, st = _backup(r, BACKUP_DATA2=str(data2), BACKUP_SKIP_MOUNT_CHECK="1", BACKUP_VHDX_PATH=str(vhdx),
                     BACKUP_COLD_DEST=str(cold), BACKUP_D_MIN_FREE_GIB="0")
    # 여기서는 마운트 확인을 건너뛴 시험이라 cold 단계의 mountpoint 에서 멈춘다 -- «건너뜀» 표시만 안 나오면 된다
    assert st["targets"]["cold_d"] != "skipped_vhdx_on_d"


def test_backup_seals_stale_hour_files():
    r, data2 = _repo(), _tmp()
    d = data2 / "orderflow/depthdiff/ETHUSDT"
    d.mkdir(parents=True)
    old = time.time() - 3 * 3600
    for name in ("alone.jsonl", "shipped.jsonl", "shipped.jsonl.gz", "fresh.jsonl"):
        (d / name).write_text(name)
        if name != "fresh.jsonl":
            os.utime(d / name, (old, old))
    rc, st = _backup(r, BACKUP_DATA2=str(data2), BACKUP_SKIP_MOUNT_CHECK="1", BACKUP_VHDX_PATH="/nonexistent",
                     BACKUP_COLD_DEST=str(_tmp() / "cold"))
    assert (d / "alone.jsonl.gz").is_file() and not (d / "alone.jsonl").exists()
    assert (d / "shipped.jsonl.gz").read_text() == "shipped.jsonl.gz", "Pi 완본을 덮으면 안 된다"
    assert not (d / "shipped.jsonl").exists() and (data2 / "orderflow_partial/depthdiff/ETHUSDT/shipped.jsonl").is_file()
    assert (d / "fresh.jsonl").is_file(), "쓰는 중인 시각 파일은 건드리지 않는다"
    assert st["targets"]["hot_6tb"] == 0


# ── 이동 스크립트 ─────────────────────────────────────────────────────
def test_move_swap_and_fail_closed_link():
    base = _tmp()
    src, dst = base / "src", base / "6tb" / "dst"
    (src / "BTC").mkdir(parents=True)
    (src / "BTC/a.gz").write_text("x" * 10)
    sh = str(ROOT / "scripts/ops/move_to_data2_20261107.sh")
    env = {**os.environ, "MOVE_SKIP_MOUNT_CHECK": "1"}
    for cmd in (["copy", str(src), str(dst)], ["swap", str(src), str(dst)]):
        assert subprocess.run(["bash", sh, *cmd], env=env, capture_output=True).returncode == 0
    assert src.is_symlink() and (src / "BTC/a.gz").read_text() == "x" * 10 and (base / "src.moved").is_dir()
    assert subprocess.run(["bash", sh, "clean", str(src)], env=env, capture_output=True).returncode == 0
    assert not (base / "src.moved").exists()
    # 6TB 가 안 붙으면 링크가 깨진다 -- 수집기의 mkdir 은 조용히 새 폴더를 만들지 않고 실패해야 한다
    shutil.rmtree(base / "6tb")
    try:
        (src / "BTC").mkdir(parents=True, exist_ok=True)
    except OSError:
        pass
    else:
        raise AssertionError("깨진 링크 아래에 폴더가 만들어졌다")
    assert not (base / "6tb").exists()


def test_move_refuses_short_copy_and_missing_mount():
    base = _tmp()
    (base / "src").mkdir()
    sh = str(ROOT / "scripts/ops/move_to_data2_20261107.sh")
    p = subprocess.run(["bash", sh, "copy", str(base / "src"), str(base / "dst")], capture_output=True, text=True,
                       env={**os.environ, "MOVE_DATA2": str(base)})
    assert p.returncode != 0 and "not mounted" in p.stderr


# ── 데이터 지도 ───────────────────────────────────────────────────────
def test_data_map_shows_real_location_of_links():
    sys.path.insert(0, str(ROOT / "scripts" / "ops"))
    import gen_data_map_20261107 as g
    r = _tmp()
    (r / "data/live/orderflow/raster/ETHUSDT").mkdir(parents=True)
    (r / "data/live/orderflow/raster/ETHUSDT/a.f32").write_text("x")
    target = _tmp()
    (target / "BTCUSDT").mkdir()
    (target / "BTCUSDT/2026-11-07T00.jsonl.gz").write_text("y")
    os.symlink(target, r / "data/live/orderflow/depthdiff")
    (r / "data/hot").mkdir()
    (r / "data/hot/a.sqlite").write_text("s")
    text = g.render(r, disks=(("tmp", r),), manifest={"collectors": {"bn_x": {"supervisor": "s.sh", "env": {"A": "1"}}}})
    assert str(target) in text and "BTCUSDT" in text and "a.sqlite" in text and "bn_x" in text
    assert "raster" in text and "(로컬)" in text


# ── 재시작 복원 풀 ────────────────────────────────────────────────────
def test_restore_call_runs_on_its_own_small_pool():
    import asyncio
    import threading
    from dashboard.coin_flow import _RESTORE_POOL, restore_call
    assert _RESTORE_POOL._max_workers == 3

    async def go():
        got = await restore_call(lambda a, k=0: (a, k, threading.current_thread().name), 1, k=2)
        # 기본 풀은 복원과 별개라 복원이 다 차지해도 막히지 않는다
        blocker = threading.Event()
        jobs = [restore_call(blocker.wait, 5) for _ in range(3)]
        tasks = [asyncio.ensure_future(j) for j in jobs]
        await asyncio.sleep(0.05)
        other = await asyncio.wait_for(asyncio.to_thread(lambda: "free"), 1)
        blocker.set()
        await asyncio.gather(*tasks)
        return got, other
    (a, k, name), other = asyncio.run(go())
    assert (a, k) == (1, 2) and name.startswith("restore") and other == "free"


def test_wal_writers_do_not_fsync_every_commit():
    import data_store
    import live_trade_tape_collector_20260916 as tape
    p = _tmp() / "t.sqlite"
    with data_store.rw_connect(p) as con:
        assert con.execute("PRAGMA synchronous").fetchone()[0] == 1
    with tape.sqlite_shared(p) as con:
        assert con.execute("PRAGMA synchronous").fetchone()[0] == 1


def test_baseline_reads_one_text_block_and_matches_rows():
    import sqlite3
    import warnings
    import numpy as np
    from dashboard import micro_ref
    p = _tmp() / "t.sqlite"
    con = sqlite3.connect(p)
    con.execute("CREATE TABLE trade_tape_1s(symbol TEXT, ts_sec INTEGER, price_bin INTEGER, buy_qty REAL, sell_qty REAL)")
    now = int(time.time())
    rows = [("ethusdt", now - i * 10, b, 0.1 + i % 7, 1 / (1 + i)) for i in range(4000) for b in (1, 2)]
    con.executemany("INSERT INTO trade_tape_1s VALUES (?,?,?,?,?)", rows)
    con.commit()
    want = {}
    for _, t, _, b, s in rows:
        want[t] = want.get(t, 0.0) + b + s
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        got = micro_ref.baseline_from_tape(p, symbol="ethusdt")
    assert \
        abs(sum(sum(v) for v in got["minute_vol_sorted"].values()) / sum(want.values()) - 1) < 1e-12
    ts = np.array(list(want)); v = np.array(list(want.values())); h = (ts % 86400) // 3600
    for k, q in got["sec_vol_p99"].items():
        assert abs(q - np.quantile(v[h == k], 0.99)) < 1e-9
    assert micro_ref.baseline_from_tape(p, symbol="btcusdt") is None


def test_cross_coin_klines_fetched_once_per_bar_and_not_shared_by_reference():
    import pandas as pd
    import live_eth_dir4h_20261015 as d4
    calls = []
    def fake(sym, n, now_ms):
        calls.append((sym, now_ms))
        return pd.DataFrame({"timestamp": [pd.Timestamp(now_ms // 300_000 * 300_000, unit="ms")], "close": [1.0]})
    d4._SHORT.clear()
    t = 1_793_400_000_000
    with mock.patch.object(d4, "_klines", fake):
        a = d4._short_klines("ADA", 340, t)
        a["x"] = 1                                    # rest_frames 가 열을 붙인다
        b = d4._short_klines("ADA", 340, t + 60_000)
        c = d4._short_klines("ADA", 340, t + 300_000)  # 다음 봉은 다시 받는다
        d = d4._short_klines("APT", 340, t)
    assert "x" not in b and len(calls) == 3 and c.close[0] == 1.0 and d.close[0] == 1.0
    assert c.timestamp[0] > b.timestamp[0], "낡은 봉을 새 봉에 돌려주면 안 된다"


def test_tunnel_start_waits_for_nvme_mount():
    def tree(strip_wait: bool):
        r, nvme, bin_ = _tmp(), _tmp() / "nvme", _tmp()
        d = r / "dashboard/scripts"
        d.mkdir(parents=True)
        src = (ROOT / "dashboard/scripts/start_public_dashboard.sh").read_text(encoding="utf-8")
        if strip_wait:
            src = "\n".join(ln for ln in src.splitlines() if "until mountpoint" not in ln) + "\n"
        (d / "start_public_dashboard.sh").write_text(src, encoding="utf-8")
        for name in ("start_external.sh", "supervise_cloudflare_tunnel.sh"):
            (d / name).write_text("#!/bin/sh\nexit 0\n")
            (d / name).chmod(0o755)
        (r / "data").mkdir()
        os.symlink(nvme / "live", r / "data/live")      # 부팅 직후 모양: 링크는 있는데 디스크가 아직 안 붙었다
        # 첫 확인은 «안 붙음», 둘째 확인에서 디스크가 붙는다
        (bin_ / "mountpoint").write_text(f'#!/bin/sh\nif [ -e "{bin_}/seen" ]; then mkdir -p "{nvme}/live"; exit 0; fi\ntouch "{bin_}/seen"; exit 1\n')
        (bin_ / "mountpoint").chmod(0o755)
        return subprocess.run(["bash", str(d / "start_public_dashboard.sh")], capture_output=True, text=True, timeout=30,
                              env={**os.environ, "PATH": f"{bin_}:{os.environ['PATH']}"})
    assert tree(strip_wait=True).returncode != 0, "대기가 없으면 11-07 처럼 mkdir 에서 실패해야 재현이다"
    ok = tree(strip_wait=False)
    assert ok.returncode == 0 and "PUBLIC_URL=" in ok.stdout, ok.stderr
