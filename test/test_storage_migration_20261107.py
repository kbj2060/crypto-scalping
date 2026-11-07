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
                     BACKUP_COLD_DEST=str(_tmp() / "cold"))
    assert st["targets"] == {"hot_6tb": 0, "cold_d": "skipped_vhdx_on_d", "dev": 1}   # 이 시험엔 dev 설정이 없다
    dest = data2 / "crypto-scalping-backups/data"
    assert (dest / "live/state.json").exists() and (dest / "models/m.bin").exists()
    assert not (dest / "live/orderflow/raster").exists(), "raster 는 14일 보존 -- 사본에 쌓지 않는다"
    assert not (dest / "binance_vision").exists()


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
