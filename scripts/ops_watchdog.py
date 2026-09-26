#!/usr/bin/env python3
"""24/7 live operations watchdog. It observes only; it never submits orders or repairs data."""
from __future__ import annotations

import argparse
import html
import json
import os
import shutil
import sqlite3
import subprocess
import time
import urllib.request

import duckdb
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
# trading_bot.py loads .env the same way; this script never did, so
# TELEGRAM_BOT_TOKEN/TELEGRAM_CHAT_ID were unset in its process environment and
# every alert has been silently swallowed (telegram_sent=false on every event)
# regardless of severity. Explicit path so it doesn't depend on invocation cwd.
load_dotenv(ROOT / ".env")
LIVE = ROOT / "data" / "live"
RESEARCH = ROOT / "data" / "research"
OUT = LIVE / "ops_watchdog"
HISTORY = OUT / "history"
KST = ZoneInfo("Asia/Seoul")
SEVERITY = {"OK": 0, "WARN": 1, "CRITICAL": 2, "BLOCKED": 3}


@dataclass
class Check:
    component: str
    status: str
    summary: str
    details: dict[str, Any]


def now_kst() -> datetime:
    return datetime.now(KST)


def iso_now() -> str:
    return now_kst().isoformat()


def parse_kst(value: Any) -> datetime | None:
    if not value:
        return None
    text = str(value).strip().replace("Z", "+00:00")
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    return parsed.replace(tzinfo=KST) if parsed.tzinfo is None else parsed.astimezone(KST)


def age_minutes(value: Any) -> float | None:
    stamp = parse_kst(value)
    return None if stamp is None else max(0.0, (now_kst() - stamp).total_seconds() / 60.0)


def age_minutes_utc_naive(value: Any) -> float | None:
    stamp = parse_utc_naive(value)
    return None if stamp is None else max(0.0, (now_kst() - stamp).total_seconds() / 60.0)


def parse_utc_naive(value: Any) -> datetime | None:
    if not value:
        return None
    text = str(value).strip().replace("Z", "+00:00")
    try:
        stamp = datetime.fromisoformat(text)
    except ValueError:
        return None
    if stamp.tzinfo is None:
        stamp = stamp.replace(tzinfo=timezone.utc)
    return stamp.astimezone(KST)


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, default=str), encoding="utf-8")
    os.replace(tmp, path)


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str) + "\n")


def retain_history(days: int = 30) -> None:
    cutoff = time.time() - days * 86400
    for path in HISTORY.glob("*.jsonl"):
        try:
            if path.stat().st_mtime < cutoff:
                path.unlink()
        except OSError:
            pass


def recommended_action(component: str) -> str:
    # 2026-08-03: migrated to systemd (scripts/ops/systemd/*.service) -- journalctl
    # replaces the old logs/supervisor/*.log tail, which no longer receives output.
    if component in {"trading_bot_process", "trading_bot_heartbeat", "decision_snapshot", "data_pipeline", "pipeline_contract"}:
        return "scripts/ops/botctl.sh status; journalctl -u trading-bot.service -n 100 --no-pager"
    if component in {"market_data_sources", "dashboard_state", "execution_contract"}:
        return "scripts/ops/triage.sh"
    if component in {"runtime_resources", "watchdog_storage"}:
        return "scripts/ops/triage.sh; df -h ."
    if component.startswith("shadow_"):
        # 섀도우 러너는 systemd가 아니라 scripts/ops/supervisor_*.sh + crontab @reboot로 뜬다.
        # pgrep/pkill 패턴은 자기 handoff 잡 명령줄에도 매칭되므로 grep -v로 걸러서 본다(2026-09-05).
        return ("ps -eo pid,args | grep -v handoff_jobs | grep -E 'live_.*shadow.*runner'; "
                "ls -l data/live/*shadow*state*.json")
    if component == "duckdb_trade_tape_eth":
        # supervisor 가 중복 실행 가드를 갖고 있어 그냥 다시 켜도 안전하다(이미 돌면 스스로 exit 1).
        return ("ps -eo pid,args | grep -v handoff_jobs | grep live_trade_tape_collector; "
                "nohup setsid bash scripts/ops/supervisor_trade_tape.sh "
                ">> logs/supervisor/trade_tape_manual.log 2>&1 < /dev/null &")
    if component.startswith("duckdb_"):
        return "scripts/ops/botctl.sh status; journalctl -u trading-bot.service -n 100 --no-pager"
    if component.startswith("multicoin_"):
        # 다시 띄우는 것도 같은 스크립트다(supervisor 가 중복을 막아 이미 돌면 건너뛴다).
        return ("python scripts/ops/multicoin_collectors_20260926.py check; "
                "python scripts/ops/multicoin_collectors_20260926.py resume")
    return "scripts/ops/triage.sh"


def load_json(path: Path) -> tuple[dict[str, Any] | None, str | None]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None, "missing"
    except (OSError, json.JSONDecodeError) as exc:
        return None, f"invalid_json:{type(exc).__name__}"
    return (data, None) if isinstance(data, dict) else (None, "json_not_object")


def tail_jsonl(path: Path) -> tuple[dict[str, Any] | None, str | None]:
    try:
        with path.open("rb") as fh:
            fh.seek(max(0, path.stat().st_size - 65536))
            lines = fh.read().decode("utf-8", errors="replace").splitlines()
    except FileNotFoundError:
        return None, "missing"
    except OSError as exc:
        return None, f"read_error:{type(exc).__name__}"
    for line in reversed(lines):
        if line.strip():
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                return None, "invalid_tail_jsonl"
            return (data, None) if isinstance(data, dict) else (None, "tail_not_object")
    return None, "empty"


def stale_status(age: float | None, warn: float, critical: float) -> str:
    if age is None:
        return "BLOCKED"
    if age >= critical:
        return "CRITICAL"
    if age >= warn:
        return "WARN"
    return "OK"


def process_args() -> str:
    try:
        return subprocess.check_output(["ps", "-eo", "args="], text=True, timeout=5)
    except (OSError, subprocess.SubprocessError):
        return ""


def check_process(component: str, signature: str, required: bool = True) -> Check:
    found = signature in process_args()
    if found:
        return Check(component, "OK", "registered process is present", {"signature": signature})
    status = "CRITICAL" if required else "WARN"
    return Check(component, status, "registered process is absent", {"signature": signature})


def check_snapshot() -> Check:
    path = LIVE / "decision_feature_snapshot.jsonl"
    row, error = tail_jsonl(path)
    if error:
        return Check("decision_snapshot", "BLOCKED", "decision snapshot cannot be read", {"path": str(path), "error": error})
    values = row.get("values") if isinstance(row, dict) else None
    market_ts = values.get("timestamp") if isinstance(values, dict) else None
    # The market bar timestamp belongs to the completed bar and is therefore
    # expected to trail the write that records its decision.  Measure the
    # artifact's creation time for liveness; retain market_ts as diagnostics.
    created_at = row.get("created_at") if isinstance(row, dict) else None
    age = age_minutes(created_at)
    return Check("decision_snapshot", stale_status(age, 12, 18), "market snapshot freshness", {
        "path": str(path), "market_ts": market_ts, "created_at": created_at,
        "age_minutes": age, "warn_minutes": 12, "critical_minutes": 18,
    })


def check_heartbeat() -> Check:
    path = LIVE / "trading_bot_decision_heartbeat.json"
    state, error = load_json(path)
    if error:
        return Check("trading_bot_heartbeat", "BLOCKED", "decision heartbeat cannot be read", {"path": str(path), "error": error})
    recorded = state.get("recorded_at_kst")
    age = age_minutes(recorded)
    return Check("trading_bot_heartbeat", stale_status(age, 6, 10), "decision heartbeat freshness", {
        "recorded_at_kst": recorded, "decision_bar_ts": state.get("decision_bar_ts"), "age_minutes": age,
        "warn_minutes": 6, "critical_minutes": 10,
    })


def check_pipeline() -> Check:
    path = LIVE / "data_pipeline_health.json"
    state, error = load_json(path)
    if error:
        return Check("data_pipeline", "BLOCKED", "pipeline health cannot be read", {"path": str(path), "error": error})
    raw_eth = state.get("raw_eth") if isinstance(state.get("raw_eth"), dict) else {}
    last_ts = raw_eth.get("last_ts")
    age = age_minutes(last_ts)
    freshness = stale_status(age, 12, 18)
    reported = str(state.get("status", "OK")).upper()
    status = freshness if SEVERITY[freshness] >= SEVERITY.get(reported, 1) else reported
    if status not in SEVERITY:
        status = "WARN"
    return Check("data_pipeline", status, "pipeline report and raw ETH freshness", {
        "reported_status": reported, "raw_eth_last_ts": last_ts, "age_minutes": age,
        "warnings": state.get("warnings", []), "warn_minutes": 12, "critical_minutes": 18,
    })


def check_dashboard() -> Check:
    path = LIVE / "dashboard_state.json"
    state, error = load_json(path)
    if error:
        return Check("dashboard_state", "BLOCKED", "dashboard state cannot be read", {"path": str(path), "error": error})
    # `cycle_timestamp_kst` changes only with a 5-minute decision cycle. The
    # dashboard shadow loop updates independently every 10 seconds, so using
    # the decision timestamp here creates false dashboard outage alerts.
    stamp = state.get("shadow_updated_at") or state.get("updated_at") or state.get("cycle_timestamp_kst")
    age = age_minutes(stamp)
    return Check("dashboard_state", stale_status(age, 2, 5), "dashboard shadow refresh freshness", {
        "timestamp": stamp, "age_minutes": age, "warn_minutes": 2, "critical_minutes": 5,
    })


def check_pipeline_contract() -> Check:
    path = LIVE / "data_pipeline_health.json"
    state, error = load_json(path)
    if error:
        return Check("pipeline_contract", "BLOCKED", "pipeline contract cannot be read", {"path": str(path), "error": error})
    ai = state.get("ai") if isinstance(state.get("ai"), dict) else {}
    groups = ai.get("groups") if isinstance(ai.get("groups"), list) else []
    errors = ai.get("errors") if isinstance(ai.get("errors"), list) else []
    missing = ai.get("missing_cols") if isinstance(ai.get("missing_cols"), list) else []
    nonfinite = ai.get("nonfinite_cols") if isinstance(ai.get("nonfinite_cols"), list) else []
    valid = (
        state.get("pipeline_stage") == "final_governor_success"
        and groups == ["tide", "dlinear", "patchtst"]
        and not errors and not missing and not nonfinite
        and state.get("signal_align_ok") is True
    )
    return Check(
        "pipeline_contract",
        "OK" if valid else "BLOCKED",
        "pipeline AI and bar contract" if valid else "pipeline AI or bar contract mismatch",
        {
            "pipeline_stage": state.get("pipeline_stage"), "ai_groups": groups,
            "ai_errors": errors, "missing_cols": missing, "nonfinite_cols": nonfinite,
            "signal_align_ok": state.get("signal_align_ok"),
        },
    )


def check_data_sources() -> Check:
    path = LIVE / "dashboard_state.json"
    state, error = load_json(path)
    if error:
        return Check("market_data_sources", "BLOCKED", "market data state cannot be read", {"path": str(path), "error": error})
    micro = state.get("microstructure") if isinstance(state.get("microstructure"), dict) else {}
    tail = state.get("tail_risk") if isinstance(state.get("tail_risk"), dict) else {}
    sources = {
        "depth_websocket": micro.get("depth_connected"),
        "trade_websocket": micro.get("trade_connected"),
        "rest_poll": micro.get("poll_connected"),
        "liquidation_websocket": tail.get("ws_connected"),
    }
    failed = [name for name, connected in sources.items() if connected is not True]
    stale = bool(micro.get("data_stale"))
    try:
        trade_age = max(0.0, float(micro.get("trade_age_sec")))
    except (TypeError, ValueError):
        trade_age = None
    # Depth and REST polling are the core price/contract feeds. A lone stale
    # trade-flow stream is significant, but short exchange stream gaps must not
    # page as CRITICAL. WARN debounce absorbs brief gaps; sustained loss escalates.
    core_failed = any(sources[name] is not True for name in ("depth_websocket", "rest_poll"))
    if core_failed or len(failed) >= 2:
        status = "CRITICAL"
    elif stale:
        status = "CRITICAL" if trade_age is None or trade_age >= 600.0 else "WARN"
    elif failed:
        status = "WARN"
    else:
        status = "OK"
    return Check("market_data_sources", status, "market source connectivity", {
        "sources": sources, "data_stale": stale, "failed_sources": failed,
        "trade_age_sec": trade_age, "trade_critical_after_sec": 600,
    })


_DUCKDB_CHECK_CACHE: dict[str, tuple[float, Check]] = {}


def _duckdb_check_ttl(warn_minutes: float) -> float:
    """실제 DB 를 여는 주기. **자기 임계값의 1/5**(최소 60초·최대 600초).

    30초 간격으로 도는 루프가 임계 5분짜리를 찌르는 건 **10배 과표집**이다. duckdb 는 단일
    writer 라 read_only 연결조차 writer 를 막으므로, 그 과표집이 그대로 **봇의 쓰기 실패**가
    된다(2026-09-17 `MS duckdb insert failed ... PID 308` 8회). 임계의 1/5 면 판정이 늦어질
    여지는 최대 임계의 20% 이고, 그 대신 DB 접촉이 컴포넌트별로 2~20배 줄어든다.
    """
    return min(600.0, max(60.0, warn_minutes * 60.0 / 5.0))


def check_duckdb_table_freshness(component: str, db_path: Path, table: str, ts_column: str,
                                  warn_minutes: float, critical_minutes: float) -> Check:
    """TTL 캐시 껍데기 -- 실제 판정은 아래 `_uncached` 가 한다. 캐시가 살아 있으면 **DB 를 아예
    열지 않는다**(락 경합의 근원을 줄이는 게 목적이므로 «열지 않는 것» 자체가 핵심이다)."""
    ttl = _duckdb_check_ttl(warn_minutes)
    hit = _DUCKDB_CHECK_CACHE.get(component)
    if hit is not None and (time.monotonic() - hit[0]) < ttl:
        return hit[1]
    check = _check_duckdb_table_freshness_uncached(
        component, db_path, table, ts_column, warn_minutes, critical_minutes)
    _DUCKDB_CHECK_CACHE[component] = (time.monotonic(), check)
    return check


def _check_duckdb_table_freshness_uncached(component: str, db_path: Path, table: str, ts_column: str,
                                            warn_minutes: float, critical_minutes: float) -> Check:
    """Freshness of a specific DuckDB table's latest row -- catches a live-but-silently-not-writing
    collector that a process-liveness check (check_process) cannot see. Read-only, so this can
    never corrupt whatever process is writing the same file -- but DuckDB briefly refuses a new
    read-only connection while a writer holds its own connection open across an insert (observed
    2026-08-17: trading_bot.py's 5-minute decision cycle), so a lock-conflict IOException gets a
    few short retries before being treated as a real failure."""
    if not db_path.is_file():
        return Check(component, "BLOCKED", "duckdb file is missing", {"path": str(db_path)})
    last_error: duckdb.Error | None = None
    # 2026-09-17: 재시도 사다리를 (0, 0.4, 0.8, 1.6)=최대 2.8초에서 (0, 0.5)=최대 0.5초로 줄였다.
    # 긴 사다리는 «거짓 BLOCKED 를 피하려고» 있었는데, 이제 락 충돌은 아래에서 WARN(120초
    # 디바운스)으로 내려가므로 그 역할이 사라졌다. 반면 사다리가 길수록 **writer 와 더 오래
    # 싸운다** -- 그게 봇의 `MS duckdb insert failed` 였다. 짧게 포기하고 다음 TTL 을 기다린다.
    for attempt, delay in enumerate((0.0, 0.5)):
        if delay:
            time.sleep(delay)
        try:
            con = duckdb.connect(str(db_path), read_only=True)
            try:
                max_ts = con.execute(f"select max(cast({ts_column} as timestamp)) from {table}").fetchone()[0]
            finally:
                con.close()
            last_error = None
            break
        except duckdb.Error as exc:
            last_error = exc
    if last_error is not None:
        # 🔴2026-09-17: **락 충돌은 «고장»이 아니라 «지금 누가 쓰는 중»이다.**
        # duckdb 는 단일 writer 라 read_only 연결조차 writer 를 막고 그 반대도 마찬가지다.
        # 체결 테이프는 초당 ~3행을 쓰므로 30초 폴링이 쓰기 순간과 자주 겹친다. 실제로
        # `[alert] BLOCKED -> [recovered] OK` 가 09-16 이후 **59회** 반복됐는데 그동안
        # 테이블은 내내 멀쩡했다(trade_tape_1s 265,962행, 수집기 정상 가동).
        # BLOCKED 는 debounce_seconds()가 **0**(즉시 호출)이라 순간 충돌이 곧 텔레그램이 된다.
        # => 락 충돌만 WARN 으로 낮춰 **기존 120초 디바운스**를 타게 한다. 2분 넘게 지속되면
        #    그때 울리므로 진짜 교착은 여전히 잡히고, 스쳐가는 충돌은 조용히 지나간다.
        #    파일 없음 / 행 없음 / 그 밖의 오류는 지금처럼 **즉시 BLOCKED** 다.
        # ⚠️이건 알림 소음만 고친다. 봇의 `MS duckdb insert failed`(실제 쓰기 실패)는 별건이고
        #    감시자가 라이브 DB 락을 아예 안 건드리게 해야 없어진다.
        busy = "conflicting lock" in str(last_error).lower()
        return Check(component, "WARN" if busy else "BLOCKED",
                     "duckdb table busy: writer holds the lock" if busy
                     else "duckdb table cannot be read", {
            "path": str(db_path), "table": table, "error": f"{type(last_error).__name__}: {last_error}",
            "attempts": attempt + 1, "lock_conflict": busy,
        })
    if max_ts is None:
        return Check(component, "BLOCKED", "duckdb table has no rows", {"path": str(db_path), "table": table})
    # every timestamp column checked here is KST wall time whether or not duckdb attaches
    # tzinfo (VARCHAR-cast columns come back naive but are KST strings at the source) --
    # age_minutes()/parse_kst() already treats a naive value as KST, which is correct here.
    age = age_minutes(max_ts)
    return Check(component, stale_status(age, warn_minutes, critical_minutes), "duckdb table freshness", {
        "path": str(db_path), "table": table, "latest_ts": str(max_ts), "age_minutes": age,
        "warn_minutes": warn_minutes, "critical_minutes": critical_minutes,
    })


# ── 섀도우 러너 생존 (2026-09-06) ──────────────────────────────────────────────────────────
# 90일 판정을 쌓고 있는 가상매매 러너가 조용히 멈추면 그 구간의 일손익 시계열이 영구히 빈다
# (사후 복구 불가). 2026-09-05 원장 전수점검에서 **그 사망을 아무도 알리지 않는다**가 확인됐다 --
# check_process는 trading_bot.py 하나만 보고, 푸시 알림(live_push_notifier_20260904.py)은
# V자반등 원장의 포지션 개시/청산만 본다. 러너는 전부 5분 주기로 save_state()를 하므로 상태파일
# mtime이 생존 신호이고, last_decided_bar_utc(가진 러너)는 "프로세스는 살아있는데 봉 결정을 못 하는"
# 상태까지 잡는다. ⚠️러너를 은퇴시킬 때는 이 표에서도 줄을 지운다 -- 안 지우면 영구 BLOCKED다
# (2026-09-05 진입 지정가 페이드 v4 제거가 그런 사례였다).
# 2026-09-07 은퇴: shadow_fire_cont_{eth,xrp,sol} · shadow_retail_shift_b2 (4줄) 제거.
# 트레일링 청산의 "걸 수 없는 스톱" 결함으로 경제성 근거가 무효화됐고, 청산과 무관한 경로 측정에서도
# 방향 정보가 없었다(지속 규칙 H200 0.5054 [0.4981,0.5127] · B2 H200 VAL 0.5029/OOS 0.5186 -- 둘 다 동전).
# 러너 정지 + crontab @reboot + supervisor까지 함께 제거했다. 되살리려면 그 판정부터 다시 세울 것.
# 2026-09-14 은퇴: shadow_evidence_chip_{btc,xrp} (2줄) 제거. 러너를 **의도적으로** 정지시킨
# 것(09-14 01:00)이라 사망 자체가 알릴 일이 아닌데, 표에 줄이 남아 CRITICAL 이 계속 나갔다.
# 위 경고의 재발이다 -- 은퇴는 러너 정지와 이 표에서 줄 지우기가 **한 쌍**이다.
# 2026-09-15 은퇴: shadow_v_rebound_econ (마지막 1줄) 제거. 섀도우 원장 139건(09-05 리셋 이후
# 단일 설정)이 기대값 -20.68bp · t -2.71 · 95%CI [-35.6,-5.7]로 **0을 배제**했다. 09-07 회계 수정
# 후의 백테스트 HOLDOUT은 이미 -8.42bp/건이었고(hold_cap_removal_20260908.md), 라이브가 그보다
# 나쁘게 그 예고를 확인한 것이다 -- 즉 기각이 아니라 확인이다. ⚠️러너 `--report`가 찍는 +6.09bp는
# 09-08에 폐기된 legacy sim_exit 상수이니 기준선으로 쓰지 말 것.
# 한 건짜리 사고가 아니다: 최악 10건을 빼도 -2.68bp, 전·후반·양 측면·일별 6/8일 모두 음수.
# 러너 정지 + crontab @reboot 제거와 한 쌍.
# 2026-09-18 등재: Zeus Baseline v3 섀도우(주문 없음). 동결 docs/zeus/README.md §2 ·
# 사전등록 docs/zeus/shadow_prereg_v3_20260918.md. 수익은 체결 436건까지 «판정하지 않는다»
# -- 0.8건/일에서 MDE 가 기대 엣지(+12.56bp)보다 커서 중간 판정이 불가능하기 때문이다.
# 은퇴할 때는 러너 정지 + @reboot 제거 + 이 줄 삭제가 **한 쌍**이다(위 경고 참조).
# 2026-09-18 추가 등재: Zeus Baseline v4(157열·edge 게이트). v3 과 **같은 러너**가 돌린다
# (판본별 러너를 두면 한쪽만 고쳐진다 -- v3 전용 러너에 원천 병합 버그가 있어 최근 4일
#  OI 결측 43.4% 였다). v4 는 2.17건/일이라 436건 판정 지평이 1.5년 -> 200일로 내려온다.
# 2026-09-19 은퇴: zeus_v3_shadow · zeus_v4_shadow (2줄) 제거 -> 표가 **비었다**.
# 사용자 지시로 러너를 정지시켰다(실시간 스캘핑으로 무게가 옮겨가 수요가 줄었다). 라이브 표본은
# v3 0건 · v4 2건뿐이었다(나머지는 백필) -- 사전등록 판정선 436건 대비 사실상 0이라 잃은 누적은
# 없고, 포기한 것은 앞으로의 누적(현 속도로 ~267일)이다. 원장은 보존했다.
# 러너 정지 + crontab @reboot 제거 + 이 줄 삭제가 **한 쌍**이다(위 09-14 사고 참조).
# 표가 비어도 안전하다 -- 아래 소비처가 제너레이터 언패킹이라 0줄이면 검사를 안 만든다.
SHADOW_RUNNERS: tuple[tuple[str, str], ...] = ()


def check_shadow_runner(component: str, filename: str, warn_minutes: float = 15,
                        critical_minutes: float = 30) -> Check:
    """섀도우 원장의 쓰기 신선도. 러너가 5분 봉마다 쓰므로 warn 15분(3주기)·critical 30분(6주기)."""
    path = LIVE / filename
    state, error = load_json(path)
    if error:
        return Check(component, "BLOCKED", "shadow ledger cannot be read", {"path": str(path), "error": error})
    try:
        write_age = max(0.0, (now_kst() - datetime.fromtimestamp(path.stat().st_mtime, KST)).total_seconds() / 60.0)
    except OSError as exc:
        return Check(component, "BLOCKED", "shadow ledger stat failed", {"path": str(path), "error": type(exc).__name__})
    decided = state.get("last_decided_bar_utc")
    decided_age = age_minutes_utc_naive(decided)
    age = write_age if decided_age is None else max(write_age, decided_age)
    ledger = state.get("ledger") if isinstance(state.get("ledger"), list) else []
    positions = state.get("positions") if isinstance(state.get("positions"), list) else []
    pending = state.get("pending") if isinstance(state.get("pending"), list) else []
    return Check(component, stale_status(age, warn_minutes, critical_minutes), "shadow runner ledger freshness", {
        "path": str(path), "age_minutes": round(age, 1), "write_age_minutes": round(write_age, 1),
        "last_decided_bar_utc": decided,
        "decided_age_minutes": (None if decided_age is None else round(decided_age, 1)),
        "closed_trades": len(ledger), "open_positions": len(positions), "pending": len(pending),
        "started_utc": state.get("started_utc"), "rule": state.get("rule"),
        "warn_minutes": warn_minutes, "critical_minutes": critical_minutes,
    })


# ── 다코인 수집기 (2026-09-26) ─────────────────────────────────────────────────────────────
# BTC·SOL·XRP·HYPE 수집기는 `scripts/ops/multicoin_collectors_20260926.py start` 가 띄우고, 띄운 것만
# 이 호스트의 매니페스트에 적는다. 감시는 **매니페스트에 있는 것만** 본다 -- 코드에 박으면
#   ① 서버/Pi 중 안 띄운 호스트에서 «파일 없음» BLOCKED 가 영구히 울리고(위 SHADOW_RUNNERS 사고),
#   ② 배포(머지)가 기동보다 먼저 닿는 순간 울린다.
# 은퇴는 러너 정지 + @reboot 제거 + 매니페스트에서 항목 삭제가 **한 쌍**이다.
MULTICOIN_MANIFEST = LIVE / "multicoin_collectors.json"


def check_file_dir_freshness(component: str, directory: Path, pattern: str,
                             warn_minutes: float, critical_minutes: float) -> Check:
    """시각별 파일(.bt/.jsonl/.f32/날짜별 duckdb)을 쓰는 수집기 -- 가장 최근에 쓴 파일의 mtime.
    진행 중인 시각 파일에 계속 append 하므로 mtime 이 곧 생존 신호다."""
    try:
        newest = max((f.stat().st_mtime for f in directory.glob(pattern) if f.is_file()), default=None)
    except OSError as exc:
        return Check(component, "BLOCKED", "collector directory cannot be read",
                     {"path": str(directory), "error": type(exc).__name__})
    if newest is None:
        return Check(component, "BLOCKED", "collector directory has no files", {"path": str(directory)})
    age = max(0.0, (time.time() - newest) / 60.0)
    return Check(component, stale_status(age, warn_minutes, critical_minutes), "collector file freshness", {
        "path": str(directory), "age_minutes": round(age, 1),
        "warn_minutes": warn_minutes, "critical_minutes": critical_minutes,
    })


def check_multicoin_collectors() -> list[Check]:
    manifest, error = load_json(MULTICOIN_MANIFEST)
    if manifest is None:
        # 파일이 없으면 이 호스트엔 다코인 수집기가 없다 -- 알릴 일이 아니다. 깨졌으면 알린다.
        return [] if not MULTICOIN_MANIFEST.exists() else [
            Check("multicoin_manifest", "BLOCKED", "multicoin manifest cannot be read",
                  {"path": str(MULTICOIN_MANIFEST), "error": error})]
    out = []
    for c in (manifest.get("collectors") or {}).values():
        # 막 띄운 수집기는 첫 쓰기 전이다(파일 없음 = 즉시 BLOCKED = 즉시 텔레그램). 유예를 둔다.
        if time.time() - float(c.get("started_at", 0)) < 60.0 * float(c.get("grace_minutes", 15)):
            continue
        for f in c.get("fresh") or []:
            component = f"multicoin_{f['name']}"
            if f.get("kind") == "duckdb":
                out.append(check_duckdb_table_freshness(component, ROOT / f["path"], f["table"], f["ts"],
                                                        f["warn"], f["critical"]))
            elif f.get("kind") == "dir":
                out.append(check_file_dir_freshness(component, ROOT / f["path"], f["glob"],
                                                    f["warn"], f["critical"]))
    return sorted(out, key=lambda ch: ch.component)


def check_runtime_resources() -> Check:
    usage = shutil.disk_usage(ROOT)
    free_gib = usage.free / (1024 ** 3)
    try:
        meminfo = dict(line.split(":", 1) for line in Path("/proc/meminfo").read_text().splitlines() if ":" in line)
        available_kib = int(meminfo.get("MemAvailable", "0 kB").split()[0])
        memory_available_gib = available_kib / (1024 ** 2)
    except (OSError, ValueError):
        memory_available_gib = None
    if free_gib < 10 or (memory_available_gib is not None and memory_available_gib < 2):
        status = "CRITICAL"
    elif free_gib < 20 or (memory_available_gib is not None and memory_available_gib < 4):
        status = "WARN"
    else:
        status = "OK"
    return Check("runtime_resources", status, "disk and memory headroom", {
        "disk_free_gib": round(free_gib, 2), "memory_available_gib": None if memory_available_gib is None else round(memory_available_gib, 2),
        "disk_warn_gib": 20, "disk_critical_gib": 10, "memory_warn_gib": 4, "memory_critical_gib": 2,
    })


def check_raster_archive() -> Check:
    """만료 래스터가 parquet 으로 **실제로 보관되는가**. (2026-09-20)

    왜 필요한가: 래스터는 보존 14일이고 만료분을 `_archive_then_unlink` 가 parquet 으로
    떠낸다. 그게 조용히 실패하면 **원본이 그대로 쌓이거나**(디스크) 최악엔 보관 없이
    사라진다. 아카이버는 실패해도 수집을 안 멈추므로(그게 설계다) 로그를 안 보면 모른다 --
    위 duckdb 신선도 검사들과 같은 부류의 실패 모드다.

    판정:
      · 살아있는 .f32 의 가장 오래된 것이 보존일+2 보다 오래됐다 -> 아카이버가 안 돈다
      · 보관본에 .gz 가 섞여 있다 -> parquet 변환이 실패해 폴백으로 떨어졌다
    🔴첫 보관 예정일(가장 오래된 파일 + 14일) **전에는 «대기»** 다. 파일이 없다고 경보를
      울리면 09-28 까지 열흘간 거짓 경보가 된다.
    """
    live = LIVE / "orderflow" / "raster"
    arc = LIVE / "orderflow" / "raster_archive"
    retention = int(os.getenv("OF_RETENTION_DAYS", "14"))
    f32 = sorted(live.glob("*/*.f32"))
    if not f32:
        return Check("raster_archive", "OK", "raster collector idle (no live files)",
                     {"live_files": 0})
    def _stamp(path: Path) -> datetime | None:
        try:
            return datetime.strptime(path.stem, "%Y-%m-%dT%H").replace(tzinfo=timezone.utc)
        except ValueError:
            return None
    stamps = [t for t in (_stamp(x) for x in f32) if t]
    oldest = min(stamps) if stamps else None
    age_days = (datetime.now(timezone.utc) - oldest).total_seconds() / 86400 if oldest else 0.0
    pq = list(arc.glob("*/*/*.parquet")) if arc.is_dir() else []
    gz = list(arc.glob("*/*/*.gz")) if arc.is_dir() else []
    due = oldest + timedelta(days=retention) if oldest else None

    if gz:
        status, summary = "WARN", "parquet conversion fell back to gzip (unreadable by duckdb)"
    elif age_days > retention + 2:
        status, summary = "WARN", "expired rasters are not being archived"
    elif not pq and due and datetime.now(timezone.utc) < due:
        status, summary = "OK", "archive pending first run"
    else:
        status, summary = "OK", "raster archive healthy"
    return Check("raster_archive", status, summary, {
        "live_files": len(f32), "oldest_live_age_days": round(age_days, 2),
        "retention_days": retention,
        "first_archive_due": due.date().isoformat() if due else None,
        "parquet_files": len(pq), "gzip_fallback_files": len(gz),
        "archive_db": str(arc / "orderbook.duckdb"),
    })


def check_watchdog_storage() -> Check:
    required = [OUT / "state.json", OUT / "incidents.sqlite", OUT / "watchdog_heartbeat.json"]
    missing = [str(path) for path in required if not path.is_file()]
    writable = os.access(OUT, os.W_OK)
    status = "OK" if writable and not missing else "BLOCKED"
    return Check("watchdog_storage", status, "watchdog state and incident storage", {
        "directory": str(OUT), "writable": writable, "missing_files": missing,
    })


def check_execution_contract() -> Check:
    path = LIVE / "dashboard_state.json"
    state, error = load_json(path)
    if error:
        return Check("execution_contract", "BLOCKED", "execution contract cannot be read", {"path": str(path), "error": error})
    account = state.get("account") if isinstance(state.get("account"), dict) else {}
    alert = state.get("execution_alert") if isinstance(state.get("execution_alert"), dict) else {}
    valid = account.get("enabled") is False and account.get("testnet") is True and alert.get("status") == "disabled"
    return Check("execution_contract", "OK" if valid else "BLOCKED", "shadow execution safety contract", {
        "account_enabled": account.get("enabled"), "testnet": account.get("testnet"), "execution_alert_status": alert.get("status"),
    })


def init_db(path: Path) -> None:
    with sqlite3.connect(path) as con:
        con.execute("""CREATE TABLE IF NOT EXISTS incidents (
            id INTEGER PRIMARY KEY, observed_at_kst TEXT NOT NULL, component TEXT NOT NULL,
            status TEXT NOT NULL, summary TEXT NOT NULL, details_json TEXT NOT NULL,
            notification_kind TEXT NOT NULL, telegram_sent INTEGER NOT NULL)""")


def load_state(path: Path) -> dict[str, Any]:
    state, _ = load_json(path)
    return state or {"schema_version": "ops_watchdog.state.v1", "checks": {}}


def debounce_seconds(status: str) -> float:
    # CRITICAL/BLOCKED must page immediately -- never sit in a pending window.
    # Everything else (entering OR leaving WARN) must hold for a sustained period
    # before it's treated as real, otherwise a metric that oscillates around its
    # own threshold every polling cycle pages an alert+recovered pair every cycle.
    return 0.0 if status in {"CRITICAL", "BLOCKED"} else 120.0


def apply_debounce(previous: dict[str, Any], raw_status: str) -> tuple[str, dict[str, Any]]:
    confirmed = previous.get("status") or raw_status
    if raw_status == confirmed:
        return confirmed, {"pending_status": None, "pending_since_kst": None}
    pending_status = previous.get("pending_status")
    pending_since = parse_kst(previous.get("pending_since_kst"))
    if raw_status != pending_status or pending_since is None:
        pending_status, pending_since = raw_status, now_kst()
    elapsed = (now_kst() - pending_since).total_seconds()
    if elapsed >= debounce_seconds(raw_status):
        return raw_status, {"pending_status": None, "pending_since_kst": None}
    return confirmed, {"pending_status": pending_status, "pending_since_kst": pending_since.isoformat()}


def notification_kind(previous: str, current: str, last_notified: str | None) -> str | None:
    if current == "OK":
        return "recovered" if previous and previous != "OK" else None
    if current != previous:
        return "alert"
    last = parse_kst(last_notified)
    if last is None:
        return "alert"
    repeat_minutes = 30 if current in {"CRITICAL", "BLOCKED"} else 120
    return "reminder" if (now_kst() - last).total_seconds() >= repeat_minutes * 60 else None


def telegram_message(check: Check, kind: str) -> str:
    icon = {"OK": "🟢", "WARN": "🟠", "CRITICAL": "🔴", "BLOCKED": "⛔"}[check.status]
    title = "RECOVERED" if kind == "recovered" else check.status
    fields = check.details
    lines = [f"{icon} <b>[{title}] {html.escape(check.component)}</b>", html.escape(check.summary), f"감지: {iso_now()}"]
    for key in ("market_ts", "recorded_at_kst", "raw_eth_last_ts", "timestamp", "last_processed_bar_ts", "age_minutes", "error", "equity_curve_error"):
        value = fields.get(key)
        if value not in (None, "", []):
            label = "지연(분)" if key == "age_minutes" else key
            lines.append(f"{label}: <code>{html.escape(str(round(value, 1) if isinstance(value, float) else value))}</code>")
    return "\n".join(lines)


def send_telegram(message: str) -> bool | None:
    # None = not configured (no token/chat_id -- retrying would never help, so
    # this must not be treated the same as a genuine failure). False = configured
    # but the send itself failed (network/API error) -- this case should be retried.
    token, chat_id = os.getenv("TELEGRAM_BOT_TOKEN", ""), os.getenv("TELEGRAM_CHAT_ID", "")
    if not token or not chat_id:
        return None
    body = json.dumps({"chat_id": chat_id, "text": message, "parse_mode": "HTML"}).encode()
    request = urllib.request.Request(f"https://api.telegram.org/bot{token}/sendMessage", data=body, headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(request, timeout=8) as response:
            response.read()
        return True
    except OSError:
        return False


def ping_deadman_switch() -> None:
    # Optional complementary signal to scripts/ops/watchdog_deadman.sh: an external
    # dead-man's-switch service (e.g. https://healthchecks.io) that pages independently
    # of this process/host if the ping stops arriving. No-op unless the user has set up
    # their own account and put the ping URL in .env -- we never create that account.
    url = os.getenv("HEALTHCHECK_PING_URL", "")
    if not url:
        return
    try:
        with urllib.request.urlopen(urllib.request.Request(url, method="GET"), timeout=5) as response:
            response.read()
    except OSError:
        pass


def run_once(dry_run: bool) -> list[Check]:
    OUT.mkdir(parents=True, exist_ok=True)
    state_path, db_path = OUT / "state.json", OUT / "incidents.sqlite"
    init_db(db_path)
    micro_db = LIVE / "microstructure.duckdb"
    tape_db = LIVE / "trade_tape.duckdb"
    tail_db = LIVE / "tail_risk.duckdb"
    tail_btc_sol_db = LIVE / "tail_risk_btc_sol.duckdb"
    gex_db = LIVE / "deribit_gex.duckdb"
    altdata_db = RESEARCH / "altdata.duckdb"
    checks = [
        check_process("trading_bot_process", "trading_bot.py"),
        check_snapshot(), check_heartbeat(), check_pipeline(), check_pipeline_contract(),
        check_data_sources(), check_dashboard(), check_execution_contract(), check_runtime_resources(),
        check_watchdog_storage(), check_raster_archive(),
        # DuckDB write-freshness (2026-08-17): the checks above watch process liveness and
        # dashboard-reported connectivity flags, neither of which catches a process that stays
        # alive but silently stops persisting rows -- exactly the failure mode found in this
        # session's dev-collector audit. These read the tables directly instead.
        check_duckdb_table_freshness("duckdb_orderbook_l2_eth", micro_db, "orderbook_decision_snapshots", "timestamp_kst", 15, 30),
        # 2026-09-19 은퇴: duckdb_orderbook_l2_{btc,sol} (2줄). BTC/SOL 결정 루프를 껐더니
        # (.env SHADOW_ASSETS_ENABLE=False) 이 두 테이블에 쓰던 orderbook recorder 도 함께
        # 멈췄다 -- 실측: 봇이 05:18 에 OFF 로 기동한 뒤 최신 행이 05:10 에서 안 늘어난다.
        # ⚠️`microstructure_1m_{btc,sol}` 은 **안 지웠다** -- 같은 플래그에 안 묶여 있고 실제로
        #   계속 쓰이고 있다(05:20 확인). 둘을 한 덩어리로 보면 살아있는 검사를 죽인다.
        # 러너 정지 + @reboot 제거 + 이 줄 삭제가 **한 쌍**이다.
        check_duckdb_table_freshness("duckdb_microstructure_1m_btc", micro_db, "microstructure_1m_btc", "ts", 5, 10),
        check_duckdb_table_freshness("duckdb_microstructure_1m_sol", micro_db, "microstructure_1m_sol", "ts", 5, 10),
        check_duckdb_table_freshness("duckdb_tail_risk_1m_eth", tail_db, "tail_risk_1m", "ts", 5, 10),
        # 2026-08-17: BTC/SOL liquidation tracking, dev-to-server migration -- deliberately its
        # own duckdb file, not data/live/tail_risk.duckdb (see supervisor_tail_risk_btc_sol_worker.sh
        # for why: a shared-file writer caused two real trading_bot.py write failures on first try).
        # 2026-09-19 은퇴: duckdb_tail_risk_1m_{btc,sol} (2줄) 제거. 사용자 지시로 trading_bot 의
        # BTC/SOL 결정 루프를 껐고(.env FINAL_GOVERNOR_OMEGA4_6_1_SHADOW_ASSETS_ENABLE=False),
        # 그래서 이 표를 읽던 hexa-pulse 인터셉터(진입차단·강제청산)가 더는 호출되지 않는다.
        # 쓰던 워커(supervisor_tail_risk_btc_sol_worker.sh)도 같은 날 정지 + @reboot 제거했다.
        # 러너 정지 + @reboot 제거 + 이 줄 삭제가 **한 쌍**이다(SHADOW_RUNNERS 주석의 사고).
        # 되살리려면: .env 플래그 True -> 워커 재기동 -> 이 두 줄 복원, 순서로.
        # hourly cron; one missed run is normal, two in a row is not.
        check_duckdb_table_freshness("duckdb_deribit_gex", gex_db, "gex_summary", "recorded_at_utc", 90, 150),
        # daily cron (0 1 * * *); warn/critical give ~1 and ~2 missed days of slack.
        check_duckdb_table_freshness("duckdb_altdata_fear_greed", altdata_db, "fear_greed_index", "recorded_at_utc", 1800, 2880),
        check_duckdb_table_freshness("duckdb_altdata_funding_spread", altdata_db, "cross_exchange_funding_spread", "recorded_at_utc", 1800, 2880),
        # 2026-09-16: 체결 테이프 수집기. 5초마다 완결된 초를 쓰므로 1분이면 이미 늦은 것이지만,
        # 조용한 새벽에도 ETH 는 초당 100건 넘게 체결되므로 «행이 없다 = 수집기가 죽었다»가
        # 성립한다. warn 5분·critical 10분은 다른 1분 수집기들과 같은 값이다.
        # ⚠️ts_sec 는 epoch 정수라 그대로 cast 하면 안 된다 -- to_timestamp 로 감싸 넘긴다.
        # ⚠️수집기를 **은퇴시키면 이 줄도 지운다**. 안 지우면 파일이 안 늘어나(또는 지워져)
        #   영구 BLOCKED/CRITICAL 이 된다 -- SHADOW_RUNNERS 주석의 사고가 세 번 반복된 자리다.
        check_duckdb_table_freshness("duckdb_trade_tape_eth", tape_db, "trade_tape_1s",
                                     "to_timestamp(ts_sec)", 5, 10),
        # 2026-09-06: 섀도우 러너 7종의 원장 쓰기 신선도(SHADOW_RUNNERS 주석 참고).
        *(check_shadow_runner(component, filename) for component, filename in SHADOW_RUNNERS),
        *check_multicoin_collectors(),
    ]
    state = load_state(state_path)
    stored = state.setdefault("checks", {})
    effective_checks: list[Check] = []
    with sqlite3.connect(db_path) as con:
        for check in checks:
            previous = stored.get(check.component, {})
            confirmed_status, debounce_fields = apply_debounce(previous, check.status)
            effective = check if confirmed_status == check.status else replace(
                check, status=confirmed_status, details={**check.details, "raw_status": check.status},
            )
            effective_checks.append(effective)
            kind = notification_kind(str(previous.get("status", "")), confirmed_status, previous.get("last_notified_at_kst"))
            sent = None
            if kind:
                message = telegram_message(effective, kind)
                sent = None if dry_run else send_telegram(message)
                print(f"[{kind}] {effective.component} {effective.status}: {effective.summary}")
                con.execute("INSERT INTO incidents(observed_at_kst, component, status, summary, details_json, notification_kind, telegram_sent) VALUES (?, ?, ?, ?, ?, ?, ?)",
                            (iso_now(), effective.component, effective.status, effective.summary, json.dumps(effective.details, default=str), kind, int(bool(sent))))
                append_jsonl(HISTORY / f"events_{now_kst():%Y%m%d}.jsonl", {
                    "observed_at_kst": iso_now(), "kind": kind, "component": effective.component,
                    "status": effective.status, "summary": effective.summary, "telegram_sent": bool(sent),
                    "recommended_action": recommended_action(effective.component), "details": effective.details,
                })
            stored[check.component] = {
                "status": confirmed_status, "raw_status": check.status, "summary": check.summary, "last_seen_at_kst": iso_now(),
                **debounce_fields,
                # Record every attempted notification as the dedupe point, EXCEPT a
                # genuine send failure (sent is False -- Telegram configured but the
                # request itself failed): that case must retry next poll instead of
                # silently losing the alert. "Not configured" (sent is None) still
                # advances normally since retrying it every cycle would never help.
                "last_notified_at_kst": previous.get("last_notified_at_kst") if sent is False else iso_now(),
            }
    observed_at = iso_now()
    snapshot = {"schema_version": "ops_watchdog.health.v1", "updated_at_kst": observed_at, "checks": [asdict(c) for c in effective_checks]}
    append_jsonl(HISTORY / f"watchdog_{now_kst():%Y%m%d}.jsonl", snapshot)
    retain_history()
    atomic_json(state_path, state)
    atomic_json(OUT / "health_snapshot.json", snapshot)
    atomic_json(OUT / "watchdog_heartbeat.json", {"recorded_at_kst": iso_now(), "status": "ok", "check_count": len(checks)})
    ping_deadman_switch()
    return effective_checks


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--interval-seconds", type=float, default=30.0)
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Never send Telegram messages")
    args = parser.parse_args()
    while True:
        run_once(args.dry_run)
        if args.once:
            return
        time.sleep(max(5.0, args.interval_seconds))


if __name__ == "__main__":
    main()
