#!/usr/bin/env python3
"""대시보드 웹푸시 알림 데몬, 2026-09-04.

사용자 요청: "다른 작업하다가 계속 신호를 놓친다. 요약된 정보를 데스크톱/폰 알림으로 받고 싶다."

왜 별도 데몬인가
----------------
대시보드 서버는 **조회가 있을 때만** 신호를 계산한다(dashboard/server.py의 load_evidence_signals()
는 60초 TTL 캐시 뒤에 있고, 요청이 없으면 아무것도 돌지 않는다). 즉 아무도 안 보고 있으면 —
정확히 이 기능이 필요한 그 상황에서 — 트리거가 될 계산 자체가 일어나지 않는다. 그래서 자기
폴링 루프를 가진 프로세스가 필요하다.

계산을 여기서 새로 하지 않고 로컬 대시보드 API를 폴링하는 이유는 두 가지다. (1) **알림 숫자와
화면 숫자가 반드시 같아야 한다** — 같은 공식을 두 군데서 계산하면 언젠가 갈라지고, 그때 어느
쪽이 맞는지 알 수 없다. (2) 부수효과로 캐시가 데워져서 폰으로 대시보드를 열 때 오히려 빨라진다.

무엇을 보내는가 (사용자가 고른 범위)
------------------------------------
T1 즉시(소리 O)   실제로 일어난 사건 — 섀도우/라이브 포지션 개시·종료, 운영 헬스 이상 전환.
T2 즉시(무음)     드물고 강한 컨텍스트 — net_score |>=3|, 청산 버스트 발생, 세션 변동성 창 진입.
다이제스트        변화가 있을 때만 — 발동 신호 집합/레짐/포지션 수가 직전 발송과 달라졌을 때.

⚠️ T2와 다이제스트는 **매매 트리거가 아니다.** 증거신호 8종은 ETH/BTC/XRP 전부에서 경제성 게이트를
전수 실패했고(docs/homer/README.md 5.21절, memory: 증거신호 경제성 종결), 대시보드 자신도
"probability-shift CONTEXT ONLY -- Not a trade trigger"라고 명시한다. 알림 문구가 행동을
지시하지 않도록 일부러 서술형으로 쓴다 — 푸시로 오면 "뭔가 해야 한다"는 압력이 생기고, 검증되지
않은 신호가 그렇게 사실상의 매매 트리거가 되는 것이 이 기능의 가장 큰 위험이다.

재시작/장애 후 폭주 방지
------------------------
두 겹으로 막는다. (1) 상태파일이 없는 **최초 실행은 현재 상태를 baseline으로 기록만 하고 아무것도
보내지 않는다**. (2) 그 이후에도 EVENT_MAX_AGE_SEC보다 오래된 사건은 seen으로만 표시하고 보내지
않는다 — 데몬이 6시간 죽어 있었다면 복구 시점에 필요한 건 그동안의 전부가 아니라 "지금"이다.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(REPO_ROOT / ".env")

from scripts.push_webpush_20260904 import broadcast, load_subscriptions  # noqa: E402

STATE_PATH = REPO_ROOT / "data" / "live" / "push_notifier_state.json"
POLL_SECONDS = 45
# 이보다 오래된 사건은 "지금"이 아니므로 조용히 seen 처리한다(재시작 폭주 방지 2단계).
EVENT_MAX_AGE_SEC = 30 * 60
# 같은 key가 이 시간 안에 다시 떠도 재발송하지 않는다. 플래핑하는 헬스체크 하나가 알림을
# 도배해서 사용자가 전체를 음소거해버리는 것을 막는 장치다.
COOLDOWN_SECONDS = {"t1": 300, "t2": 1800}
# 다이제스트 최소 간격. 5분봉이라 신호 집합은 5분마다 바뀔 수 있는데, 그대로 내보내면
# "변화가 있을 때만"이 사실상 5분 주기 알림이 된다.
DIGEST_MIN_INTERVAL_SEC = 15 * 60
# seen 딕셔너리가 무한히 자라지 않도록 이 나이가 지난 항목은 버린다.
SEEN_TTL_SEC = 24 * 3600


def log(msg: str) -> None:
    print(f"[{datetime.now(timezone.utc):%Y-%m-%dT%H:%M:%SZ}] {msg}", flush=True)


# ------------------------------------------------------------------------------------------
# 상태
# ------------------------------------------------------------------------------------------
def load_state() -> dict[str, Any]:
    try:
        with open(STATE_PATH, encoding="utf-8") as fh:
            state = json.load(fh)
    except (FileNotFoundError, json.JSONDecodeError):
        return {"seen": {}, "digest_fingerprint": None, "digest_sent_at": 0, "baseline_done": False}
    state.setdefault("seen", {})
    state.setdefault("digest_fingerprint", None)
    state.setdefault("digest_sent_at", 0)
    state.setdefault("baseline_done", False)
    return state


def save_state(state: dict[str, Any]) -> None:
    cutoff = time.time() - SEEN_TTL_SEC
    state["seen"] = {k: v for k, v in state["seen"].items() if v >= cutoff}
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = STATE_PATH.with_suffix(".json.tmp")
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(state, fh, ensure_ascii=False, indent=2)
    tmp.replace(STATE_PATH)


def parse_utc(value: Any) -> float | None:
    """ISO8601 -> epoch seconds. 이 저장소의 시각 필드는 'Z' 접미사와 '+00:00'이 섞여 있고
    타임존이 아예 없는 것도 있다(그 경우 UTC로 읽는다)."""
    if not value:
        return None
    try:
        text = str(value).replace("Z", "+00:00")
        dt = datetime.fromisoformat(text)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.timestamp()
    except (ValueError, TypeError):
        return None


# ------------------------------------------------------------------------------------------
# 알림 한 건
# ------------------------------------------------------------------------------------------
class Note:
    __slots__ = ("key", "tier", "title", "body", "tag", "url", "event_ts")

    def __init__(self, key: str, tier: str, title: str, body: str,
                 *, tag: str | None = None, url: str = "/dashboard/live/",
                 event_ts: float | None = None) -> None:
        self.key = key
        self.tier = tier
        self.title = title
        self.body = body
        # tag가 같으면 브라우저가 이전 알림을 대체한다. 서로 다른 사건은 서로 다른 tag를 써야
        # 하나가 다른 하나를 지우지 않는다.
        self.tag = tag or key
        self.url = url
        self.event_ts = event_ts

    def payload(self) -> dict[str, Any]:
        return {"tier": self.tier, "title": self.title, "body": self.body,
                "tag": self.tag, "url": self.url,
                "ts": datetime.now(timezone.utc).isoformat()}


# ------------------------------------------------------------------------------------------
# T1 감지기 -- 실제로 일어난 사건
# ------------------------------------------------------------------------------------------
def detect_shadow_positions(shadow: dict[str, Any]) -> list[Note]:
    """V자반등 경제라벨 섀도우 러너의 포지션 개시/종료.

    현재 유일하게 가동 중인 진입모델 섀도우다(memory: homer_entry_v2 / v_rebound_econ). 판정에는
    1~2개월이 필요한 상태라, 알림은 '지금 뭘 하라'가 아니라 '지금 원장에 뭐가 찍혔다'를 전한다."""
    notes: list[Note] = []
    for pos in shadow.get("open_positions") or []:
        opened = pos.get("opened_utc")
        if not opened:
            continue
        side = "롱" if str(pos.get("side", "")).lower() in ("long", "buy") else "숏"
        entry = pos.get("entry")
        proba = pos.get("proba")
        detail = f"진입 {entry}" if entry is not None else ""
        if isinstance(proba, (int, float)):
            detail += f" · p={proba:.3f}"
        notes.append(Note(
            f"shadow_open:{opened}", "t1",
            f"V자반등 섀도우 {side} 진입",
            detail or "포지션이 열렸습니다.",
            event_ts=parse_utc(opened),
        ))
    for trade in shadow.get("recent_trades") or []:
        exit_utc = trade.get("exit_utc")
        if not exit_utc:
            continue
        side = "롱" if str(trade.get("side", "")).lower() in ("long", "buy") else "숏"
        pnl = trade.get("pnl_bp")
        pnl_txt = f"{float(pnl):+.1f}bp" if isinstance(pnl, (int, float)) else "-"
        reason = trade.get("reason") or ""
        notes.append(Note(
            f"shadow_close:{exit_utc}", "t1",
            f"V자반등 섀도우 {side} 청산 {pnl_txt}",
            f"사유 {reason}" if reason else "포지션이 닫혔습니다.",
            event_ts=parse_utc(exit_utc),
        ))
    return notes


def detect_trades(trades: dict[str, Any]) -> list[Note]:
    """실거래 원장(trade_journal.jsonl)의 신규 이벤트.

    봇은 현재 페이퍼 모드(BINANCE_EXECUTION_ENABLED=False)라 실제 체결은 나지 않지만 원장에는
    결정/청산이 계속 기록된다. 실행이 켜지면 같은 경로로 자동으로 실체결 알림이 된다."""
    notes: list[Note] = []
    for row in (trades.get("rows") or [])[-20:]:
        ts = row.get("ts") or row.get("closed_at") or row.get("opened_at")
        event = str(row.get("event") or row.get("kind") or "")
        trade_id = row.get("trade_id") or ts
        if not ts or not trade_id:
            continue
        symbol = row.get("symbol") or row.get("asset") or ""
        side = str(row.get("side") or "").upper()
        pnl = row.get("pnl_pct")
        pnl_txt = f" {float(pnl):+.2f}%" if isinstance(pnl, (int, float)) else ""
        dry = row.get("exchange_execution_dry_run")
        prefix = "[페이퍼] " if dry else ""
        notes.append(Note(
            f"trade:{trade_id}:{event}", "t1",
            f"{prefix}{symbol} {side} {event}{pnl_txt}".strip(),
            f"전략 {row.get('source') or '-'}",
            url="/dashboard/live/",
            event_ts=parse_utc(ts),
        ))
    return notes


def detect_ops_health(ops: dict[str, Any]) -> list[Note]:
    """운영 헬스가 정상->이상으로 넘어간 순간.

    key에 상태값을 넣어 '전환'만 새 알림이 되게 한다 -- CRITICAL이 계속 유지되는 동안 같은 key가
    반복되므로 seen에 걸려 한 번만 나간다. 회복(OK 복귀)도 같은 방식으로 한 번 알린다."""
    notes: list[Note] = []
    for check in (ops.get("health") or {}).get("checks") or []:
        status = str(check.get("status") or "").upper()
        component = check.get("component") or "?"
        if status in ("CRITICAL", "WARN"):
            notes.append(Note(
                f"ops:{component}:{status}", "t1",
                f"운영 이상 · {component}",
                f"{status} — {check.get('summary') or ''}".strip(),
                url="/dashboard/live/#ops",
            ))
    for proc in ops.get("supervisors") or []:
        status = str(proc.get("status") or "").upper()
        if status and status != "RUNNING":
            notes.append(Note(
                f"supervisor:{proc.get('name')}:{status}", "t1",
                f"프로세스 {status} · {proc.get('name')}",
                "supervisor가 관리하는 프로세스가 떠 있지 않습니다.",
                url="/dashboard/live/#ops",
            ))
    return notes


# ------------------------------------------------------------------------------------------
# T2 감지기 -- 드물고 강한 컨텍스트 (매매 트리거 아님, 모듈 docstring 참고)
# ------------------------------------------------------------------------------------------
NET_SCORE_THRESHOLD = 3


def detect_net_score(evidence: dict[str, Any]) -> list[Note]:
    """증거신호 합의가 |net_score| >= 3인 봉.

    ⚠️ 발동 여부는 반드시 `*_last_fired_ts`(발동한 봉)로 판단해야 한다. payload의
    `bottom_fired`/`top_fired`는 sustain window(신호별 8~72봉) 동안 계속 True이므로, 그것으로
    key를 만들면 사건 하나에 최대 72번 알림이 나간다."""
    net = evidence.get("net_score")
    bar = evidence.get("latest_bar_utc")
    if not isinstance(net, int) or abs(net) < NET_SCORE_THRESHOLD or not bar:
        return []
    side = "바닥측" if net > 0 else "천장측"
    firing = []
    for sig in evidence.get("signals") or []:
        want = "bottom" if net > 0 else "top"
        if sig.get(f"{want}_last_fired_ts") == bar:
            firing.append(sig.get("name"))
    names = ", ".join(n for n in firing if n) or "-"
    price = evidence.get("price")
    price_txt = f"ETH {price:,.0f} · " if isinstance(price, (int, float)) else ""
    return [Note(
        f"net_score:{bar}:{net}", "t2",
        f"{side} 증거 {abs(net)}표 합의",
        f"{price_txt}{names}\n※ 방향 정보 아님 — 참고용 컨텍스트입니다.",
        tag="net-score",
        event_ts=parse_utc(bar),
    )]


def detect_liq_burst(burst: dict[str, Any]) -> list[Note]:
    """청산 버스트(Hawkes) 발생. 상태가 켜져 있는 동안 updated_at은 계속 갱신되므로 key는
    'hawkes가 켜진 그 시각'으로 고정해 한 번만 나가게 한다."""
    if not burst.get("available") or not burst.get("hawkes_active"):
        return []
    updated = burst.get("updated_at")
    z_long, z_short = burst.get("z_long"), burst.get("z_short")
    detail = []
    if isinstance(z_long, (int, float)):
        detail.append(f"롱청산 z={z_long:+.1f}")
    if isinstance(z_short, (int, float)):
        detail.append(f"숏청산 z={z_short:+.1f}")
    return [Note(
        f"liq_burst:{updated}", "t2",
        f"청산 버스트 {burst.get('crisis_type') or ''}".strip(),
        " · ".join(detail) or "청산이 군집 발생 중입니다.",
        tag="liq-burst",
        event_ts=parse_utc(updated),
    )]


def detect_session_window(alerts: dict[str, Any]) -> list[Note]:
    """세션 개장 변동성 창 진입. 창 안에 있는 동안 매 폴링마다 active로 보이므로, key를
    (시장, 그날 날짜)로 만들어 창당 한 번만 알린다."""
    notes: list[Note] = []
    today = datetime.now(timezone.utc).strftime("%Y%m%d")
    active = ((alerts.get("session_volatility_alert") or {}).get("active")) or []
    for market in active:
        code = market.get("code")
        notes.append(Note(
            f"session:{code}:{today}", "t2",
            f"{market.get('label')} 변동성 창",
            f"개장 {market.get('minutes_from_open'):+.0f}분 — 실현변동성이 평소보다 높은 구간입니다.",
            tag=f"session-{code}",
        ))
    macro = (alerts.get("macro_event_alert") or {}).get("active") or []
    for event in macro:
        title = event.get("title") or event.get("event") or "경제지표"
        notes.append(Note(
            f"macro:{title}:{today}", "t2",
            f"경제지표 발표 임박 · {title}",
            str(event.get("detail") or event.get("when") or ""),
            tag="macro-event",
        ))
    return notes


# ------------------------------------------------------------------------------------------
# 다이제스트 -- "변화가 있을 때만"
# ------------------------------------------------------------------------------------------
def _regime_label(regime: dict[str, Any]) -> str:
    if not regime.get("warmed_up"):
        return "-"
    probs = {"상승": regime.get("bull_prob"), "하락": regime.get("bear_prob"),
             "횡보": regime.get("chop_prob")}
    probs = {k: v for k, v in probs.items() if isinstance(v, (int, float))}
    if not probs:
        return "-"
    name = max(probs, key=probs.__getitem__)
    return f"{name} {probs[name]:.0%}"


def _active_signals(evidence: dict[str, Any]) -> tuple[list[str], list[str]]:
    """지금 점등돼 있는 신호. 여기서는 sustain window 기준(`*_fired`)이 맞다 -- 다이제스트는
    '사건이 방금 일어났다'가 아니라 '화면이 지금 이렇다'를 요약하는 것이고, 화면의 칩도 같은
    `_active` 규칙으로 켜진다. T2의 발동 감지와는 의도적으로 다른 기준이다."""
    bottom = [s.get("name") for s in evidence.get("signals") or [] if s.get("bottom_fired")]
    top = [s.get("name") for s in evidence.get("signals") or [] if s.get("top_fired")]
    return [n for n in bottom if n], [n for n in top if n]


def build_digest(evidence: dict[str, Any], regime: dict[str, Any],
                 shadow: dict[str, Any]) -> tuple[str, Note] | None:
    """(fingerprint, Note)를 돌려준다. fingerprint가 직전 발송과 같으면 호출자가 버린다."""
    if not evidence.get("warmed_up"):
        return None
    bottom, top = _active_signals(evidence)
    net = evidence.get("net_score")
    regime_txt = _regime_label(regime)
    n_open = shadow.get("n_open") or 0

    fingerprint = json.dumps(
        {"b": sorted(bottom), "t": sorted(top), "net": net, "regime": regime_txt, "open": n_open},
        ensure_ascii=False, sort_keys=True,
    )

    price = evidence.get("price")
    head = f"ETH {price:,.0f}" if isinstance(price, (int, float)) else "ETH"
    lines = [f"net {net:+d} · 레짐 {regime_txt}" if isinstance(net, int) else f"레짐 {regime_txt}"]
    lines.append(f"↓바닥측: {', '.join(bottom) if bottom else '—'}")
    lines.append(f"↑천장측: {', '.join(top) if top else '—'}")
    if n_open:
        lines.append(f"섀도우 포지션 {n_open}건 보유 중")
    return fingerprint, Note(
        "digest", "digest", head, "\n".join(lines),
        tag="digest",  # 항상 같은 tag -> 이전 다이제스트를 대체하고 알림함에 쌓이지 않는다
    )


# ------------------------------------------------------------------------------------------
# 루프
# ------------------------------------------------------------------------------------------
ENDPOINTS = {
    "evidence": "/api/evidence-signals",
    "regime": "/api/regime-wide24",
    "shadow": "/api/v-rebound-econ-shadow",
    "trades": "/api/trades",
    "ops": "/api/ops-status",
    "burst": "/api/liq-burst-state",
    "alerts": "/api/session-alerts",
}


async def fetch_all(session, base_url: str) -> dict[str, dict[str, Any]]:
    """엔드포인트 하나가 죽어도 나머지는 살린다 -- 예를 들어 바이낸스 klines가 일시적으로
    실패하면 /api/evidence-signals는 502를 내는데, 그것 때문에 운영 헬스 알림까지 멈추면 안 된다."""
    async def one(path: str) -> dict[str, Any]:
        try:
            async with session.get(base_url + path) as resp:
                if resp.status != 200:
                    return {}
                return await resp.json()
        except Exception:
            return {}

    results = await asyncio.gather(*(one(p) for p in ENDPOINTS.values()))
    return dict(zip(ENDPOINTS.keys(), results))


def collect_notes(data: dict[str, dict[str, Any]]) -> list[Note]:
    notes: list[Note] = []
    notes += detect_shadow_positions(data.get("shadow") or {})
    notes += detect_trades(data.get("trades") or {})
    notes += detect_ops_health(data.get("ops") or {})
    notes += detect_net_score(data.get("evidence") or {})
    notes += detect_liq_burst(data.get("burst") or {})
    notes += detect_session_window(data.get("alerts") or {})
    return notes


async def run_cycle(session, base_url: str, state: dict[str, Any],
                    *, private: str, subject: str, dry_run: bool) -> None:
    data = await fetch_all(session, base_url)
    now = time.time()
    seen = state["seen"]
    baseline = not state["baseline_done"]

    for note in collect_notes(data):
        last_sent = seen.get(note.key)
        cooldown = COOLDOWN_SECONDS.get(note.tier, 1800)
        if last_sent is not None and now - last_sent < cooldown:
            continue
        seen[note.key] = now
        if baseline:
            continue  # 최초 실행: 현재 상태를 기준선으로만 기록
        if note.event_ts is not None and now - note.event_ts > EVENT_MAX_AGE_SEC:
            continue  # 지난 사건 -- seen 처리만 하고 보내지 않는다
        if dry_run:
            log(f"DRY [{note.tier}] {note.title} | {note.body.splitlines()[0] if note.body else ''}")
            continue
        result = await broadcast(note.payload(), private_b64=private, subject=subject,
                                 ttl=3600 if note.tier == "t1" else 900,
                                 urgency="high" if note.tier == "t1" else "normal")
        log(f"[{note.tier}] {note.title} -> {result}")

    digest = build_digest(data.get("evidence") or {}, data.get("regime") or {},
                          data.get("shadow") or {})
    if digest:
        fingerprint, note = digest
        changed = fingerprint != state["digest_fingerprint"]
        due = now - (state["digest_sent_at"] or 0) >= DIGEST_MIN_INTERVAL_SEC
        if changed and due:
            state["digest_fingerprint"] = fingerprint
            state["digest_sent_at"] = now
            if baseline:
                pass
            elif dry_run:
                log(f"DRY [digest] {note.title} | {note.body!r}")
            else:
                result = await broadcast(note.payload(), private_b64=private, subject=subject,
                                         ttl=600, urgency="low")
                log(f"[digest] {note.title} -> {result}")
        elif changed:
            # 바뀌었지만 최소 간격 전 -- fingerprint를 갱신하지 않고 두면 간격이 지난 뒤
            # 그때의 최신 상태로 나간다.
            pass

    if baseline:
        state["baseline_done"] = True
        log(f"기준선 기록 완료 -- {len(seen)}개 항목을 seen 처리(발송 없음).")
    save_state(state)


async def main_async(args: argparse.Namespace) -> int:
    from aiohttp import ClientSession, ClientTimeout

    private = os.getenv("VAPID_PRIVATE_KEY", "")
    subject = os.getenv("VAPID_SUBJECT", "mailto:kbj2060@gmail.com")
    if not private and not args.dry_run:
        log("VAPID_PRIVATE_KEY가 없습니다 -- .env를 확인하세요. (--dry-run은 키 없이 됩니다)")
        return 1

    state = load_state()
    log(f"시작 -- base={args.base_url} poll={args.poll}s 구독={len(load_subscriptions())}대 "
        f"dry_run={args.dry_run}")

    async with ClientSession(timeout=ClientTimeout(total=30)) as session:
        while True:
            try:
                await run_cycle(session, args.base_url, state,
                                private=private, subject=subject, dry_run=args.dry_run)
            except Exception as exc:  # noqa: BLE001 -- 한 사이클 실패로 데몬이 죽으면 안 된다
                log(f"사이클 실패(다음 주기에 재시도): {exc!r}")
            if args.once:
                return 0
            await asyncio.sleep(args.poll)


def main() -> int:
    parser = argparse.ArgumentParser(description="대시보드 웹푸시 알림 데몬")
    parser.add_argument("--base-url", default=f"http://127.0.0.1:{os.getenv('DASHBOARD_PORT', '8787')}")
    parser.add_argument("--poll", type=float, default=POLL_SECONDS)
    parser.add_argument("--once", action="store_true", help="한 사이클만 돌고 종료")
    parser.add_argument("--dry-run", action="store_true",
                        help="실제 발송 없이 무엇이 나갈지만 로그로 출력")
    args = parser.parse_args()
    try:
        return asyncio.run(main_async(args))
    except KeyboardInterrupt:
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
