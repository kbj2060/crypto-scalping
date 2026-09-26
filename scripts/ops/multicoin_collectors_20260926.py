#!/usr/bin/env python3
"""BTC·SOL·XRP·HYPE 실시간 수집기 확장 -- 기동·재기동·점검 (2026-09-26).

ETH 에서 이미 도는 수집기들을 코인별로 더 띄운다. 코드 쪽 전제(코인별 duckdb 파일, HL 포지션 다코인
단일 프로세스, OKX 청산 중복 방지, 래스터 보관 뷰의 symbol·bin_size)는 같은 날 커밋에 들어 있다.
설계와 용량 판단: docs/multicoin_collector_expansion_20260926.md

  plan    무엇을 띄울지·사전점검·crontab 줄만 보여준다(아무것도 안 띄운다)
  start   사전점검을 통과한 것만 띄우고 이 호스트의 매니페스트에 적는다
  resume  매니페스트에 있는 것 중 죽은 것만 다시 띄운다(워치독 recommended_action)
  check   매니페스트의 수집기별 생존·신선도·증가량·RSS 와 이 호스트의 여유(4단계 점검)

단계(--phase):
  1  소급이 **불가능한** 것 -- 바이낸스 bookTicker·depthDiff, HL 체결·호가·포지션, OKX 호가·컨텍스트
  2  소급이 가능하거나 덜 급한 것 -- 바이낸스 래스터(보관 parquet)·체결 테이프, OKX 체결 테이프

🔴사전점검이 실패하면 그 수집기는 **띄우지 않는다**:
  · OKX ctVal: XRP·HYPE 값은 거래소 REST 로 실측하지 못한 채 커밋됐다. 수집기 자체의 대조
    (`assert_ct_val`)는 «REST 가 안 되면 통과»라 여기서 **REST 성공 + 일치**를 요구한다.
  · 래스터 빈 폭: ETH 의 상대 폭(0.5 / ETH 가격)과 같게 코인별로 계산해 crontab 줄에 **고정**한다.
    재부팅마다 다시 계산하면 파일마다 빈 폭이 달라진다(헤더에 적히긴 하지만 비교가 번거롭다).
  · 메모리: 새로 뜰 수집기들의 실측 RSS 합(RSS_MB)을 빼고도 512MB 가 남아야 한다.
🔴crontab 은 `--install-cron` 을 줄 때만 고친다(기존 crontab 을 logs/ 에 백업하고 줄을 덧붙이기만 한다).
🔴이 스크립트는 주문을 내지 않고 봇을 건드리지 않는다. 바이낸스 REST 는 사전점검에서 가격·틱 조회
  몇 번(가중치 < 50)뿐이다.

사용(수집 호스트에서, quant_ai 파이썬으로 -- 그 파이썬이 supervisor 의 PYTHON_BIN 이 된다):
  python scripts/ops/multicoin_collectors_20260926.py plan --phase 1
  python scripts/ops/multicoin_collectors_20260926.py start --phase 1 --install-cron
  python scripts/ops/multicoin_collectors_20260926.py start --phase 2 --install-cron
  python scripts/ops/multicoin_collectors_20260926.py check --hours 24
"""
from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import socket
import subprocess
import sys
import time
import urllib.request
from datetime import datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

LIVE = ROOT / "data" / "live"
MANIFEST = LIVE / "multicoin_collectors.json"
OPS = "scripts/ops"
COINS = ("BTC", "SOL", "XRP", "HYPE")
# 수집기별 RSS(MB) 추정 -- 2026-09-26 이 세션 실측(x86_64, 네트워크 없이 기동 후 15초). duckdb 를 import
# 하는 수집기는 그것만으로 ~150MB 다. 래스터는 북이 차면 도크스트링 실측 67MB 까지 는다.
RSS_MB = {"bn_bookticker": 26, "bn_depthdiff": 26, "bn_raster": 67, "bn_tape": 155, "hl_trades": 64,
          "hl_bbo": 151, "hl_positions": 151, "okx_bbo": 37, "okx_ctx": 155, "okx_tape": 156,
          "tail_risk": 150}
MEM_FLOOR_MB = 512
DISK_FLOOR_GB = 100       # 09-16 인벤토리의 재검토선
ETH_RASTER_BIN = 0.5      # ETH 래스터의 현재 빈 폭(OF_BIN_SIZE 기본값). 다른 코인은 같은 상대 폭으로 맞춘다
UA = {"User-Agent": "Mozilla/5.0 (compatible; crypto-scalping-collector/1.0)"}   # OKX 는 UA 없으면 403


def _dir(name: str, path: str, warn: float = 5, critical: float = 15, glob: str = "*") -> dict:
    return {"name": name, "kind": "dir", "path": path, "glob": glob, "warn": warn, "critical": critical}


def _db(name: str, path: str, table: str, ts: str, warn: float = 5, critical: float = 10) -> dict:
    return {"name": name, "kind": "duckdb", "path": path, "table": table, "ts": ts,
            "warn": warn, "critical": critical}


def specs(coins: list[str], phases: set[int], tail_btc_sol: bool) -> list[dict]:
    """띄울 수집기 목록. `env` 는 supervisor 에 넘기는 값, `match` 는 이미 도는지 판별할 값이다."""
    out: list[dict] = []

    def add(kind, sfx, phase, sup, runner, env, fresh, grace=15, eth=None):
        if phase in phases:
            out.append({"key": f"{kind}_{sfx}", "rss": RSS_MB[kind], "phase": phase, "supervisor": sup,
                        "runner": runner, "env": env,
                        "match": {k: v for k, v in env.items() if k != "OF_BIN_SIZE"},
                        "fresh": fresh, "grace_minutes": grace, "eth_path": eth})

    for c in coins:
        sym, lc, inst = f"{c}USDT", c.lower(), f"{c}-USDT-SWAP"
        of = "data/live/orderflow"
        add("bn_bookticker", lc, 1, "supervisor_book_ticker.sh", "live_book_ticker_collector_20260914.py",
            {"BT_SYMBOL": sym.lower()}, [_dir(f"bn_bookticker_{lc}", f"{of}/bookticker/{sym}")],
            eth=f"{of}/bookticker/ETHUSDT")
        add("bn_depthdiff", lc, 1, "supervisor_depth_diff.sh", "live_depth_diff_collector_20260915.py",
            {"DD_SYMBOL": sym.lower()}, [_dir(f"bn_depthdiff_{lc}", f"{of}/depthdiff/{sym}")],
            eth=f"{of}/depthdiff/ETHUSDT")
        add("hl_trades", lc, 1, "supervisor_hyperliquid_trades.sh", "live_hyperliquid_trade_collector_20260916.py",
            {"HL_COINS": c}, [_dir(f"hl_trades_{lc}", f"{of}/hyperliquid/{c}", 15, 60)],
            eth=f"{of}/hyperliquid/ETH")
        add("hl_bbo", lc, 1, "supervisor_hyperliquid_book_ticker.sh",
            "live_hyperliquid_book_ticker_collector_20260923.py", {"HL_BT_COIN": c},
            [_dir(f"hl_bbo_{lc}", f"{of}/hyperliquid_bookticker/{c}"),
             _db(f"hl_ctx_{lc}", f"data/live/hyperliquid_context_{lc}.duckdb", "hl_asset_ctx",
                 "to_timestamp(recv_ms / 1000)")],
            eth=f"{of}/hyperliquid_bookticker/ETH")
        add("okx_bbo", lc, 1, "supervisor_okx_book_ticker.sh", "live_okx_book_ticker_collector_20260923.py",
            {"OKX_BT_INST": inst}, [_dir(f"okx_bbo_{lc}", f"{of}/okx_bookticker/{inst}")],
            eth=f"{of}/okx_bookticker/ETH-USDT-SWAP")
        add("okx_ctx", lc, 1, "supervisor_okx_context.sh", "live_okx_context_collector_20260923.py",
            {"OKX_CTX_INST": inst},
            [_db(f"okx_ctx_{lc}", f"data/live/okx_context_{lc}.duckdb", "okx_mark", "to_timestamp(ts_ms / 1000)")])
        add("bn_raster", lc, 2, "supervisor_orderflow_raster.sh", "live_orderflow_raster_collector_20260914.py",
            {"OF_SYMBOL": sym.lower()}, [_dir(f"bn_raster_{lc}", f"{of}/raster/{sym}", glob="*.f32")],
            eth=f"{of}/raster/ETHUSDT")
        add("bn_tape", lc, 2, "supervisor_trade_tape.sh", "live_trade_tape_collector_20260916.py",
            {"TAPE_SYMBOL": sym.lower()},
            [_db(f"bn_tape_{lc}", f"data/live/trade_tape_{lc}.duckdb", "trade_tape_1s", "to_timestamp(ts_sec)")])
        add("okx_tape", lc, 2, "supervisor_okx_trade_tape.sh", "live_okx_trade_tape_collector_20260923.py",
            {"OKX_TAPE_INST": inst},
            [_db(f"okx_tape_{lc}", f"data/live/okx_trade_tape_{lc}.duckdb", "trade_tape_1s",
                 "to_timestamp(ts_sec)", 10, 30)])
    # HL 포지션은 코인마다가 아니라 **하나**다(한 번의 조회가 전 코인 -- 수집기 도크스트링).
    # 대상 주소를 HL 체결 DB 에서 뽑으므로 체결이 쌓인 뒤 첫 바퀴(최대 ~8분)가 돈다 -> 유예 30분.
    tag = "_".join(c.lower() for c in coins)
    add("hl_positions", tag, 1, "supervisor_hyperliquid_positions.sh",
        "live_hyperliquid_positions_collector_20260924.py", {"HL_POS_COINS": ",".join(coins)},
        [_db(f"hl_positions_{tag}", f"data/live/hyperliquid_positions_{tag}.duckdb", "hl_cycles",
             "to_timestamp(ts_ms / 1000)", 30, 60)], grace=30)
    if tail_btc_sol and 1 in phases and {"BTC", "SOL"} & set(coins):
        # 🔴서버 전용: 기존 파일(tail_risk_btc_sol.duckdb, 09-19 정지)이 서버에 있다. 다른 호스트에서
        #   띄우면 같은 이름의 새 파일이 생겨 이력이 두 곳으로 갈린다. supervisor 가 BTC·SOL 을 함께 켠다.
        add("tail_risk", "btc_sol", 1, "supervisor_tail_risk_btc_sol_worker.sh", "duckdb_persist_worker.py", {},
            [_db(f"tail_risk_{c.lower()}", "data/live/tail_risk_btc_sol.duckdb", f"tail_risk_1m_{c.lower()}", "ts")
             for c in ("BTC", "SOL")])
        out[-1]["match"] = {"BOT_SYMBOLS": "BTCUSDT,SOLUSDT", "COLLECT_TAIL_RISK": "true"}
    return out


# ── 사전점검 ───────────────────────────────────────────────────────────────────────────────
def _get_json(url: str, timeout: float = 10.0):
    with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=timeout) as r:
        return json.loads(r.read())


def check_okx_ct_vals(insts: list[str]) -> dict[str, str | None]:
    """{inst: None(통과) | 실패 사유}. REST 가 안 되면 **실패**다(수집기 자체 대조와 반대 -- 도크스트링)."""
    from scripts.live_okx_trade_tape_collector_20260923 import CT_VALS, INSTRUMENTS_URL
    out: dict[str, str | None] = {}
    for inst in insts:
        want = CT_VALS.get(inst)
        if want is None:
            out[inst] = "CT_VALS 에 없다"
            continue
        try:
            d = _get_json(f"{INSTRUMENTS_URL}?instType=SWAP&instId={inst}")["data"][0]
            got = float(d["ctVal"])
        except Exception as exc:  # noqa: BLE001
            out[inst] = f"REST 대조 실패 {type(exc).__name__}: {exc}"
            continue
        out[inst] = None if abs(got - want) < 1e-12 else f"ctVal 코드 {want} vs 거래소 {got} {d.get('ctValCcy', '')}"
    return out


def nice_bin(raw: float, tick: float) -> float:
    """raw 에 가장 가까운 1·2·2.5·5 x 10^k 를 틱의 배수로. 파일 헤더에 적히는 값이라 읽기 좋은 수로."""
    k = math.floor(math.log10(raw))
    cands = [m * 10 ** e for e in (k - 1, k, k + 1) for m in (1, 2, 2.5, 5)]
    b = min(cands, key=lambda x: abs(math.log(x / raw)))
    b = max(tick, round(b / tick) * tick)
    return float(f"{b:.10g}")


def raster_bins(coins: list[str]) -> dict[str, float | str]:
    """{코인: 빈 폭 | 실패 사유}. ETH 와 같은 상대 폭(= 같은 가격 범위 ±~2.2%, 240빈)."""
    try:
        eth = float(_get_json("https://fapi.binance.com/fapi/v1/ticker/price?symbol=ETHUSDT")["price"])
        info = _get_json("https://fapi.binance.com/fapi/v1/exchangeInfo", timeout=20)
    except Exception as exc:  # noqa: BLE001
        return {c: f"바이낸스 REST 실패 {type(exc).__name__}: {exc}" for c in coins}
    ticks = {s["symbol"]: float(f["tickSize"]) for s in info.get("symbols", [])
             for f in s.get("filters", []) if f.get("filterType") == "PRICE_FILTER"}
    out: dict[str, float | str] = {}
    for c in coins:
        sym = f"{c}USDT"
        try:
            px = float(_get_json(f"https://fapi.binance.com/fapi/v1/ticker/price?symbol={sym}")["price"])
            out[c] = nice_bin(px * ETH_RASTER_BIN / eth, ticks[sym])
        except Exception as exc:  # noqa: BLE001
            out[c] = f"가격/틱 조회 실패 {type(exc).__name__}: {exc}"
    return out


def mem_available_mb() -> float | None:
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) / 1024
    except OSError:
        pass
    return None


# ── 프로세스 ───────────────────────────────────────────────────────────────────────────────
def pids_of(runner: str, match: dict) -> list[int]:
    """supervisor 들과 같은 방식: 명령줄에 수집기 파일명 + /proc/<pid>/environ 의 값 일치."""
    out = []
    for p in Path("/proc").iterdir():
        if not p.name.isdigit():
            continue
        try:
            cmd = [a.decode(errors="ignore") for a in (p / "cmdline").read_bytes().split(b"\0")]
            # supervisor 의 bash(_supervise.sh ... python -u .../runner)도 인자에 러너가 있다 -- 뺀다.
            if not any(a.endswith(runner) for a in cmd) or any(a.endswith("_supervise.sh") for a in cmd):
                continue
            env = dict(kv.split("=", 1) for kv in (p / "environ").read_bytes().decode(errors="ignore").split("\0")
                       if "=" in kv)
        except OSError:
            continue
        if all(env.get(k) == v for k, v in match.items()):
            out.append(int(p.name))
    return out


def rss_mb(pid: int) -> float:
    try:
        for line in Path(f"/proc/{pid}/status").read_text().splitlines():
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) / 1024
    except OSError:
        pass
    return 0.0


def cron_line(s: dict, py: str) -> str:
    env = " ".join(f"{k}={v}" for k, v in {**s["env"], "PYTHON_BIN": py}.items())
    return (f"@reboot sleep 60 && cd {ROOT} && {env} bash {OPS}/{s['supervisor']} "
            f">> logs/supervisor/multicoin_{s['key']}_reboot.log 2>&1 &")


def launch(s: dict, py: str) -> str:
    log = ROOT / "logs" / "supervisor" / f"multicoin_{s['key']}_start.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    with open(log, "ab") as fh:
        proc = subprocess.Popen(["bash", f"{OPS}/{s['supervisor']}"], cwd=ROOT,
                                env={**os.environ, **s["env"], "PYTHON_BIN": py},
                                stdin=subprocess.DEVNULL, stdout=fh, stderr=subprocess.STDOUT,
                                start_new_session=True)
    time.sleep(3)
    if proc.poll() is None:
        return "기동"
    tail = log.read_text(errors="ignore").strip().splitlines()[-1:] or ["(로그 없음)"]
    return f"종료 code={proc.returncode}: {tail[0]}"


def load_manifest() -> dict:
    try:
        return json.loads(MANIFEST.read_text())
    except (OSError, ValueError):
        return {"collectors": {}}


def save_manifest(m: dict) -> None:
    m["updated_at"] = datetime.now().isoformat(timespec="seconds")
    m["host"] = socket.gethostname()
    tmp = MANIFEST.with_suffix(".json.part")
    tmp.write_text(json.dumps(m, ensure_ascii=False, indent=1))
    tmp.replace(MANIFEST)


def install_cron(lines: list[str]) -> None:
    cur = subprocess.run(["crontab", "-l"], capture_output=True, text=True)
    before = cur.stdout if cur.returncode == 0 else ""
    new = [ln for ln in lines if ln not in before.splitlines()]
    if not new:
        print("crontab: 추가할 줄 없음(이미 있다)")
        return
    backup = ROOT / "logs" / f"crontab_before_multicoin_{datetime.now():%Y%m%d_%H%M%S}.txt"
    backup.parent.mkdir(parents=True, exist_ok=True)
    backup.write_text(before)
    body = before.rstrip("\n") + ("\n" if before.strip() else "") + "\n".join(new) + "\n"
    subprocess.run(["crontab", "-"], input=body, text=True, check=True)
    print(f"crontab: {len(new)}줄 추가 (백업 {backup.relative_to(ROOT)})")


# ── 명령 ───────────────────────────────────────────────────────────────────────────────────
def cmd_plan_or_start(a, start: bool) -> int:
    coins = [c.strip().upper() for c in a.coins.split(",") if c.strip()]
    bad = [c for c in coins if c not in COINS]
    if bad:
        print(f"🔴 모르는 코인 {bad} -- 이 스크립트는 {COINS} 만 안다(OKX 계약 단위·빈 폭 표가 그 넷뿐이다)")
        return 2
    phases = {1, 2} if a.phase == "all" else {int(a.phase)}
    todo = specs(coins, phases, a.tail_risk_btc_sol)
    py = os.getenv("PYTHON_BIN") or sys.executable
    print(f"호스트 {socket.gethostname()} · 파이썬 {py} · 코인 {coins} · 단계 {sorted(phases)}")

    blocked: dict[str, str] = {}
    okx = sorted({s["env"][k] for s in todo for k in ("OKX_BT_INST", "OKX_CTX_INST", "OKX_TAPE_INST") if k in s["env"]})
    if okx:
        for inst, why in check_okx_ct_vals(okx).items():
            print(f"  OKX ctVal {inst}: {'OK' if why is None else '🔴 ' + why}")
            if why:
                for s in todo:
                    if inst in s["env"].values():
                        blocked[s["key"]] = f"OKX ctVal {why}"
    if any(s["key"].startswith("bn_raster_") for s in todo):
        bins = raster_bins(coins)
        for s in todo:
            if s["key"].startswith("bn_raster_"):
                c = s["key"].rsplit("_", 1)[1].upper()
                if isinstance(bins[c], float):
                    s["env"]["OF_BIN_SIZE"] = repr(bins[c])
                    print(f"  래스터 빈 {c}: {bins[c]} (240빈 = ETH 와 같은 상대 폭)")
                else:
                    blocked[s["key"]] = f"래스터 빈 {bins[c]}"
                    print(f"  래스터 빈 {c}: 🔴 {bins[c]}")

    running = {s["key"]: pids_of(s["runner"], s["match"]) for s in todo}
    new = [s for s in todo if not running[s["key"]] and s["key"] not in blocked]
    avail = mem_available_mb()
    need = sum(s["rss"] for s in new)
    free_gb = shutil.disk_usage(ROOT).free / 1e9
    print(f"  메모리: 가용 {avail:,.0f}MB · 새 프로세스 {len(new)}개 실측 RSS 합 {need}MB · 남길 최소 {MEM_FLOOR_MB}MB"
          if avail is not None else "  메모리: /proc/meminfo 를 못 읽었다")
    print(f"  디스크: 여유 {free_gb:,.0f}GB (재검토선 {DISK_FLOOR_GB}GB)")
    mem_short = avail is not None and avail - need < MEM_FLOOR_MB
    if mem_short:
        print("  🔴 메모리 부족 -- 코인이나 단계를 줄이거나, 그래도 띄우려면 --force")

    for s in todo:
        state = ("실행 중 pid " + ",".join(map(str, running[s["key"]]))) if running[s["key"]] else \
                (f"🔴 보류: {blocked[s['key']]}" if s["key"] in blocked else "새로 띄움")
        print(f"  [{s['phase']}] {s['key']:<26} {state}")
    lines = [cron_line(s, py) for s in todo if s["key"] not in blocked]
    print("\ncrontab @reboot 줄:" + "".join(f"\n  {ln}" for ln in lines))
    if not start:
        return 0
    if mem_short and not a.force:
        return 1

    m = load_manifest()
    for s in todo:
        if s["key"] in blocked:
            continue
        result = "이미 실행 중" if running[s["key"]] else launch(s, py)
        print(f"  {s['key']:<26} {result}")
        if result in ("기동", "이미 실행 중"):
            prev = m["collectors"].get(s["key"], {})
            m["collectors"][s["key"]] = {
                "phase": s["phase"], "supervisor": s["supervisor"], "runner": s["runner"], "env": s["env"],
                "match": s["match"], "fresh": s["fresh"], "grace_minutes": s["grace_minutes"],
                "eth_path": s["eth_path"], "python": py,
                "started_at": prev.get("started_at", time.time()) if result == "이미 실행 중" else time.time()}
    save_manifest(m)
    print(f"매니페스트 {MANIFEST.relative_to(ROOT)} · {len(m['collectors'])}개")
    if a.install_cron:
        install_cron(lines)
    return 1 if blocked else 0


def cmd_resume(_a) -> int:
    m = load_manifest()
    for key, c in sorted(m["collectors"].items()):
        if pids_of(c["runner"], c["match"]):
            continue
        s = {"key": key, "supervisor": c["supervisor"], "env": c["env"]}
        print(f"  {key:<26} {launch(s, c.get('python') or sys.executable)}")
        c["started_at"] = time.time()
    save_manifest(m)
    return 0


def _dir_bytes_since(path: Path, since: float) -> tuple[int, float | None]:
    total, newest = 0, None
    for f in path.glob("*"):
        try:
            st = f.stat()
        except OSError:
            continue
        newest = st.st_mtime if newest is None else max(newest, st.st_mtime)
        if st.st_mtime >= since:
            total += st.st_size
    return total, newest


def _duckdb_latest(path: Path, table: str, ts: str):
    import duckdb
    for _ in range(10):
        try:
            con = duckdb.connect(str(path), read_only=True)
        except duckdb.IOException:
            time.sleep(0.3)      # 수집기가 쓰는 중 -- 잠깐 뒤 다시(붙들지 않고 바로 닫는다)
            continue
        try:
            return con.execute(f"SELECT max(cast({ts} AS timestamp)) FROM {table}").fetchone()[0]
        finally:
            con.close()
    return "잠김"


def cmd_check(a) -> int:
    """4단계 점검: 하루 돌린 뒤 실제 용량·RSS·신선도로 확장을 계속할지 본다."""
    m = load_manifest()
    if not m["collectors"]:
        print(f"매니페스트가 비었다({MANIFEST}) -- 이 호스트에서 start 한 적이 없다")
        return 1
    now = time.time()
    since = now - a.hours * 3600
    per_day = 24.0 / a.hours
    print(f"호스트 {socket.gethostname()} · 창 {a.hours}시간 · 기준 {datetime.now():%Y-%m-%d %H:%M}")
    print(f"{'수집기':<26}{'pid':>8}{'RSS MB':>8}{'CPU%':>6}  {'마지막 쓰기':<22}{'MB/일':>9}{'ETH 대비':>9}")
    total_rss = total_mb_day = 0.0
    bad = 0
    for key, c in sorted(m["collectors"].items()):
        pids = pids_of(c["runner"], c["match"])
        rss = sum(rss_mb(p) for p in pids)
        cpu = 0.0
        if pids:
            out = subprocess.run(["ps", "-o", "pcpu=", "-p", ",".join(map(str, pids))], capture_output=True, text=True)
            cpu = sum(float(x) for x in out.stdout.split() or [0])
        total_rss += rss
        for f in c["fresh"]:
            p = ROOT / f["path"]
            ratio = ""
            if f["kind"] == "dir":
                b, newest = _dir_bytes_since(p, since)
                last = "-" if newest is None else f"{(now - newest) / 60:.1f}분 전"
                stale = newest is None or (now - newest) / 60 >= f["critical"]
                mb_day = b / 1e6 * per_day
                if c.get("eth_path") and (ROOT / c["eth_path"]).exists():
                    eb, _ = _dir_bytes_since(ROOT / c["eth_path"], since)
                    ratio = f"{b / eb:.2f}x" if eb else ""
            else:
                latest = _duckdb_latest(p, f["table"], f["ts"]) if p.exists() else None
                last = str(latest)[:19] if latest else "-"
                stale = latest is None or (isinstance(latest, datetime)
                                           and datetime.now() - latest >= timedelta(minutes=f["critical"]))
                age_d = max((now - float(c.get("started_at", now))) / 86400, 1 / 24)
                mb_day = (p.stat().st_size / 1e6 / age_d) if p.exists() else 0.0   # 파일 크기 / 가동 일수
            total_mb_day += mb_day
            bad += stale or not pids
            flag = " 🔴" if stale or not pids else ""
            print(f"{f['name']:<26}{(pids[0] if pids else '-'):>8}{rss:>8.0f}{cpu:>6.1f}  {last:<22}"
                  f"{mb_day:>9.1f}{ratio:>9}{flag}")
            rss = cpu = 0.0       # 한 프로세스가 여러 산출물을 가지면 첫 줄에만 센다
    avail = mem_available_mb()
    free_gb = shutil.disk_usage(ROOT).free / 1e9
    days = (free_gb - DISK_FLOOR_GB) * 1e3 / total_mb_day if total_mb_day > 0 else float("inf")
    print(f"\n합계: RSS {total_rss:,.0f}MB · 증가 {total_mb_day / 1e3:,.2f}GB/일 · 메모리 가용 "
          + (f"{avail:,.0f}MB" if avail is not None else "?"))
    print(f"디스크 여유 {free_gb:,.0f}GB -> 이 증가율만으로 재검토선({DISK_FLOOR_GB}GB)까지 {days:,.0f}일"
          " (ETH 수집분·다른 증가는 뺀 값이다. Pi 는 48시간 뒤 서버로 옮기므로 서버 여유로 다시 볼 것)")
    try:
        print(f"부하 {Path('/proc/loadavg').read_text().split()[:3]} · 코어 {os.cpu_count()}")
    except OSError:
        pass
    print("문제 없음" if not bad else f"🔴 {bad}건 -- 죽었거나(pid -) 신선도 critical 을 넘었다")
    return 1 if bad else 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("plan", "start"):
        p = sub.add_parser(name)
        p.add_argument("--coins", default=",".join(COINS))
        p.add_argument("--phase", choices=("1", "2", "all"), default="1")
        p.add_argument("--tail-risk-btc-sol", action="store_true",
                       help="서버 전용: 09-19 에 멈춘 BTC·SOL 청산(forceOrder) 1분 수집 워커를 되살린다")
        if name == "start":
            p.add_argument("--install-cron", action="store_true")
            p.add_argument("--force", action="store_true", help="메모리 부족 경고를 무시한다")
    sub.add_parser("resume")
    c = sub.add_parser("check")
    c.add_argument("--hours", type=float, default=24.0)
    a = ap.parse_args()
    if a.cmd in ("plan", "start"):
        return cmd_plan_or_start(a, a.cmd == "start")
    return cmd_resume(a) if a.cmd == "resume" else cmd_check(a)


if __name__ == "__main__":
    sys.exit(main())
