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

무엇을 보내는가 (2026-10-05 재배선 — PUSH_KINDS 표가 원본, 알림 센터가 그대로 보여 준다)
------------------------------------
T1 즉시(소리 O)   내 포지션 위험(청산까지 5% 미만 · 명목 6배 초과) · ETH 보유 중 청산 급증.
T2 즉시(무음)     사전등록 판정일(09:00 KST) · 보유 코인의 «전환 예고».

⚠️ 알림은 **매매 트리거가 아니다.** 문구는 서술형으로 쓴다 — 푸시로 오면 "뭔가 해야 한다"는 압력이
생기고, 검증되지 않은 신호가 그렇게 사실상의 매매 트리거가 되는 것이 이 기능의 가장 큰 위험이다.
그래서 화면에 근거(연구 통과·원장 재현)가 있는 것만 보낸다.

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
from datetime import datetime, timedelta, timezone
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
        return {"seen": {}, "baseline_done": False}
    state.setdefault("seen", {})
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
# 감지기 (2026-10-05 재배선, 사용자 «푸시 알림 + 판정 예정일을 알림 아이콘 하나로»)
#   옛 감지기 7종(증거신호 net_score · V자 · 섀도우 · 거래 · 운영 · Hawkes 청산 · 세션)과 다이제스트를 걷어냈다.
#   켜져 있던 둘(net_score·v_rebound)이 읽던 /api/evidence-signals·/api/v-rebound-signal 이 지워져 404 → 09-24 재시작
#   이후 발송 0건(구독 2대)이었고, 나머지는 꺼진 채였다. 지금은 근거가 있는 넷만 보낸다:
#   판정 예정일 · 내 포지션 위험(원장 재현) · 보유 중 청산 급증(H2) · 보유 코인 전환 예고(경보기 정밀도 77.8%).
# ------------------------------------------------------------------------------------------
CALENDAR_PATH = REPO_ROOT / "dashboard" / "verdict_calendar.json"
SENT_LOG_PATH = REPO_ROOT / "data" / "live" / "push_sent_log.jsonl"   # 알림 센터 «최근 보낸 알림»의 원천
KST = timezone(timedelta(hours=9))
VERDICT_HOUR_KST = 9                       # = 판정일(UTC) 00시
LIQ_WARN_PCT, LIQ_REARM_PCT = 5.0, 7.0     # 청산까지 거리 -- 5% 미만으로 들어설 때 한 번, 7% 밖으로 나가면 다시 무장
EXPO_WARN_X, EXPO_REARM_X = 6.0, 5.0       # 명목 ÷ 순자산 -- 6배 = 명목 상한 보험 기준(avgdown_gate_notional_cap_20261004)
COINS = ("eth", "sol", "xrp")

# 화면(알림 센터)이 이 표를 그대로 보여 준다 -- 이름·설명이 두 군데서 갈라지지 않게 여기 하나만 둔다.
PUSH_KINDS = {
    "verdict": {"name": "판정 예정일", "when": "사전등록 판정일 아침 9시(KST)에 한 번"},
    "risk": {"name": "포지션 위험", "when": f"청산까지 {LIQ_WARN_PCT:g}% 미만 · 명목이 순자산의 {EXPO_WARN_X:g}배 초과로 들어설 때 한 번"},
    "burst_hold": {"name": "보유 중 청산 급증", "when": "ETH 보유 중 직전 60초 한쪽 청산이 학습 7일 상위 0.5%를 넘을 때 — 그 순간 시장가로 닫으면 평균 −17bp였다"},
    "prewarn_hold": {"name": "보유 코인 전환 예고", "when": "보유 중인 코인에 «전환 예고»가 켜질 때 — 30분 안 거래대금·체결속도 급증 확률 상위 10%, 방향은 모른다"},
}
ENABLED_DETECTORS = set(PUSH_KINDS)


def coin_of(symbol: Any) -> str:
    return str(symbol or "").replace("USDC", "").replace("USDT", "").lower()


def held_coins(account: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    out: dict[str, list[dict[str, Any]]] = {}
    for p in (account or {}).get("positions") or []:
        if float(p.get("qty") or 0) > 0:
            out.setdefault(coin_of(p.get("symbol")), []).append(p)
    return out


def load_calendar() -> dict[str, Any]:
    try:
        return json.loads(CALENDAR_PATH.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def detect_verdicts(cal: dict[str, Any], now: float) -> list[Note]:
    """판정일 09:00 KST 부터 그날 한 번. key 에 날짜가 있어 seen 이 24시간 뒤 지워져도 다음 날은 안 나간다.
    event_ts 를 안 단다 -- 데몬이 아침에 죽어 있다 오후에 살아나도 그날 안이면 보내야 한다."""
    t = datetime.fromtimestamp(now, KST)
    if t.hour < VERDICT_HOUR_KST:
        return []
    today = t.strftime("%Y-%m-%d")
    return [Note(f"verdict:{it['id']}:{today}", "t2", f"판정일 · {it['title']}",
                 f"{it.get('what', '')}\n{it.get('how', '')}".strip(), tag=f"verdict-{it['id']}",
                 url="/dashboard/live/#notify-center")
            for it in cal.get("items") or []
            if it.get("date") == today and not it.get("approx") and not it.get("done")]


def detect_risk(account: dict[str, Any], state: dict[str, Any], now: float) -> list[Note]:
    """청산까지 거리·명목 배수가 선을 **넘어 들어설 때** 한 번. 히스테리시스(다시 무장 선)로 경계 진동을 막는다.
    계좌를 못 읽으면(ok=False) 아무것도 바꾸지 않는다 -- «못 읽음»은 «위험 없음»이 아니다."""
    if not (account or {}).get("ok"):
        return []
    fired = state.setdefault("risk_fired", {})
    notes: list[Note] = []
    positions = [p for ps in held_coins(account).values() for p in ps]
    for p in positions:
        mark, liq = float(p.get("mark_price") or 0), float(p.get("liquidation_price") or 0)
        if mark <= 0 or liq <= 0:
            continue
        d = abs(mark - liq) / mark * 100
        k = f"liq:{p.get('symbol')}:{p.get('side')}"
        if d < LIQ_WARN_PCT and k not in fired:
            fired[k] = now
            side = "롱" if p.get("side") == "LONG" else "숏"
            notes.append(Note(f"risk:{k}:{int(now)}", "t1", f"청산까지 {d:.1f}% · {coin_of(p.get('symbol')).upper()} {side}",
                              f"마크 {mark:,.2f} → 청산가 {liq:,.2f}. 명목을 줄이거나 증거금을 더할 자리", tag=f"risk-{k}"))
        elif d > LIQ_REARM_PCT:
            fired.pop(k, None)
    for k in [k for k in fired if k.startswith("liq:") and k not in {f"liq:{p.get('symbol')}:{p.get('side')}" for p in positions}]:
        fired.pop(k)                       # 포지션이 닫히면 다시 무장
    eq = float((account.get("balance") or {}).get("margin") or 0)
    tot = sum(float(p.get("notional") or 0) for p in positions)
    x = tot / eq if eq > 0 else 0.0
    if x > EXPO_WARN_X and "expo" not in fired:
        fired["expo"] = now
        notes.append(Note(f"risk:expo:{int(now)}", "t1", f"명목 {x:.1f}배 — 순자산의 {EXPO_WARN_X:g}배를 넘었다",
                          f"명목 ${tot:,.0f} ÷ 순자산 ${eq:,.0f}. 원장 재현: 배수 4.1배 고정이면 손익은 같고 최대 낙폭 −45→−14%",
                          tag="risk-expo"))
    elif x < EXPO_REARM_X:
        fired.pop("expo", None)
    return notes


def detect_burst_hold(mc: dict[str, Any], account: dict[str, Any], state: dict[str, Any], now: float) -> list[Note]:
    """대시보드 청산 급증 배지와 같은 판정(시장 맥락 burst: 직전 60초 > 학습 7일 상위 0.5%) · ETH 보유 중 · 켜지는 순간 한 번."""
    bu = (mc or {}).get("burst") or {}
    thr = bu.get("thr") or [None, None]
    prev = state.setdefault("burst_on", {})
    held = bool(held_coins(account).get("eth"))
    notes: list[Note] = []
    for i, side in enumerate(("long", "short")):
        v = bu.get(f"{side}_usd_60s")
        on = v is not None and thr[i] is not None and float(v) > float(thr[i])
        if on and not prev.get(side) and held:
            notes.append(Note(f"burst:{side}:{int(now)}", "t1",
                              f"ETH {'롱' if side == 'long' else '숏'} 청산 급증 ${float(v) / 1e3:,.0f}k/60초 · 보유 중",
                              "급히 시장가로 닫지 말 것 — 이런 버스트를 만난 보유 포지션을 그 순간 시장가로 닫으면 평균 −17bp(35건)",
                              tag=f"burst-{side}"))
        prev[side] = on
    return notes


def detect_prewarn_hold(bo_by_coin: dict[str, dict[str, Any]], account: dict[str, Any],
                        state: dict[str, Any]) -> list[Note]:
    """보유 중인 코인의 «전환 예고»(경보기 원시 판정 prewarn.on)가 꺼짐→켜짐일 때 한 번."""
    prev = state.setdefault("prewarn_on", {})
    notes: list[Note] = []
    for coin in held_coins(account):
        bo = bo_by_coin.get(coin) or {}
        if not bo.get("available", True) or "prewarn" not in bo:
            continue                         # 못 읽은 주기는 상태를 안 바꾼다
        on = bool((bo.get("prewarn") or {}).get("on"))
        if on and not prev.get(coin):
            notes.append(Note(f"prewarn:{coin}:{bo.get('timestamp')}", "t2", f"{coin.upper()} 전환 예고 · 보유 중",
                              "30분 안 거래대금·체결속도가 함께 급증할 확률이 상위 10% — 방향은 말하지 않는다(정밀도 실측 77.8%)",
                              tag=f"prewarn-{coin}"))
        prev[coin] = on
    return notes


async def fetch_all(session, base_url: str) -> dict[str, Any]:
    """대시보드 API 를 읽는다(화면과 같은 숫자). 하나가 죽어도 나머지는 산다 -- 실패는 빈 dict.
    시장 맥락·경보기는 **보유 중인 코인만** 읽는다(안 쓰는 값을 45초마다 덥히지 않는다)."""
    async def get(path: str) -> dict[str, Any]:
        try:
            async with session.get(base_url + path) as resp:
                return await resp.json() if resp.status == 200 else {}
        except Exception:
            return {}

    account = await get("/api/binance-account")
    held = held_coins(account)
    return {"account": account, "calendar": load_calendar(),
            "mc_eth": await get("/api/market-context?asset=eth") if "eth" in held else {},
            "bo": {c: await get(f"/api/breakout-detector?asset={c}") for c in held if c in COINS}}


def collect_notes(data: dict[str, Any], state: dict[str, Any], now: float | None = None) -> list[Note]:
    now = time.time() if now is None else now
    acct = data.get("account") or {}
    plan = {"verdict": lambda: detect_verdicts(data.get("calendar") or {}, now),
            "risk": lambda: detect_risk(acct, state, now),
            "burst_hold": lambda: detect_burst_hold(data.get("mc_eth") or {}, acct, state, now),
            "prewarn_hold": lambda: detect_prewarn_hold(data.get("bo") or {}, acct, state)}
    return [n for k, fn in plan.items() if k in ENABLED_DETECTORS for n in fn()]


def append_sent(note: Note, result: Any) -> None:
    row = {"ts": datetime.now(timezone.utc).isoformat(), "kind": note.key.split(":")[0], "tier": note.tier,
           "title": note.title, "body": note.body, "sent": (result or {}).get("sent") if isinstance(result, dict) else None}
    try:
        SENT_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        with SENT_LOG_PATH.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    except OSError as exc:
        log(f"보낸 알림 기록 실패: {exc!r}")


async def run_cycle(session, base_url: str, state: dict[str, Any],
                    *, private: str, subject: str, dry_run: bool) -> None:
    data = await fetch_all(session, base_url)
    now = time.time()
    seen = state["seen"]
    baseline = not state["baseline_done"]

    for note in collect_notes(data, state, now):
        # 🔴2026-09-09 회귀 복구: 한 key 는 **한 번만** 보낸다(쿨다운이 아니라 seen). seen 은 나이로 정리된다.
        if note.key in seen:
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
        append_sent(note, result)
        log(f"[{note.tier}] {note.title} -> {result}")

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
