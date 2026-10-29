"""계좌·주문 -- 거래소 계좌 조회·왕복 원장, 수동 진입/청산/기다리기, 브래킷(SL/TP) 감시, 사이징 상한.

2026-10-19 server.py 에서 내용 그대로 옮겼다(함수 본문 바이트 동일).
  · 모듈 수준 도우미(원장·day_start·습관·사이징·MANUAL_EXEC)는 이 파일 맨 위 수준에 그대로.
  · make_app() 안 클로저였던 핸들러는 register() 안에 들여쓰기까지 그대로 -- 코어·시장 쪽이 소유하고 같이 쓰는 객체는
    register 의 키워드 인자로 받는다(참조 공유, 같은 dict·Lock·함수).
🔴이 모듈은 dashboard.server 를 import 하면 안 된다 -- 서버는 `python dashboard/server.py`(__main__)로 돌아서
  import 하면 server.py 가 한 번 더 실행된다(load_dotenv·스레드·FLOW_SPECS).
🔴시험은 주문 기본 함수(place_bracket·run_entry·run_exit·place_stop·clear_bracket·exec_enabled·fetch_account·
  load_filters·…)를 **이 모듈**에서 patch 해야 한다. server.py 는 그 이름을 갖고 있지 않다(옛 patch 는 AttributeError 로 깨진다).
"""
from __future__ import annotations

import asyncio
from types import SimpleNamespace
import bisect
import functools
import json
import math
import os
import statistics
import time
from collections import deque
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from aiohttp import web

from dashboard.trend_rule import trend_payload
from dashboard.coin_flow import FOOTPRINT_BAR_SECONDS
from dashboard.worker_state import worker_payload
# 2026-09-10: 거래소 계정 자체(수동 매매 포함)를 읽는 유일한 경로. trade_journal.jsonl은
# trading_bot.py가 스스로 결정한 것만 담고, 그 봇은 지금 account.enabled=false(페이퍼)다.
from scripts.live_binance_account_20260910 import fetch_account  # noqa: E402
from scripts.live_manual_peg_entry_20260912 import (  # noqa: E402
    EXIT_VOL_WINDOW, bracket_action, bracket_key, bracket_merge, bracket_rekey, build_entry_plan, build_exit_plan,
    exec_enabled, load_filters, realized_vol_bpm, resolve_exit_position,
    COMMIT_CAP_X, HOLD_CAP_X, RULE_STOP_PCT, commit_rule_l, line_bracket, rec_rule_add_order, rec_rule_bracket, rec_rule_size)   # 2026-10-09 자동 규칙 진입(재생 ×15.2 크기식) · 10-14 자동 물타기 지정가
from live.execution.live_manual_peg_execute_20260912 import (  # noqa: E402
    WAIT_CHOICES_MIN, cancel_wait_orphans, clear_bracket, commission_bp, margin_type, place_add_order, place_bracket, place_stop, run_entry, run_entry_wait, run_exit)
# 2026-09-13 보유시간 조건부 위험 사이징. 계산은 사이징 워커가 하고 여기서는 상태파일만
# 읽는다(요청 경로 계산 금지 -- 2026-09-10 스레드 풀 고갈 실장애).
from scripts.live_eth_risk_sizing_policy_20260913 import (  # noqa: E402
    HARD_CAP_X, entry_notional, exit_fraction_required)
from scripts.live_eth_trade_plan_20260913 import (  # noqa: E402
    EXCHANGE_MAX_LEVERAGE, LEVERAGE_STEPS, PRESCRIBE_ACC, leverage_setting, plan_now,
    recommend_hold)

# server.py 와 같은 값(두 파일 다 dashboard/ 아래).
REPO_ROOT = Path(__file__).resolve().parents[1]
LIVE_DIR = REPO_ROOT / "data" / "live"


# 2026-09-12: 바이낸스 userTrades 는 **7일 롤링**이라 그 앞의 왕복은 API 에서 그냥 사라진다
# (실측: 09-11 에 보이던 2026-09-03~04 건이 09-12 조회에는 없다). 이 왕복이 «크기 배분이
# 손익을 얼마나 갈랐나»를 재는 유일한 표본인데, 원천이 스스로를 지우고 있어서 표본이 15건
# 근처에 정체한다. 보이는 동안 디스크에 붙여 둔다. 화면에는 쓰지 않는다 -- 축적이 목적이다.
ACCOUNT_TRIP_LEDGER_PATH = LIVE_DIR / "account_round_trips.jsonl"
ACCOUNT_TRIP_RECORD_SECONDS = 300.0
ACCOUNT_TRIP_CHECK_TOL_BP = 0.5   # 올바른 폴딩은 0.0000 이다. 0.5 는 부동소수 여유일 뿐이다


ACCOUNT_TRIP_CARD_ROWS = 400      # 카드가 그릴 최대 왕복 수. 하루 ~20건이면 20일치다


def load_account_trip_rows() -> list[dict]:
    """카드가 그릴 원장 꼬리. **화면이 거래소 payload 대신 이걸 그린다**(2026-09-22).

    userTrades 는 7일 롤링이라, 창 앞에서 열린 포지션은 진입 체결이 사라져 폴딩이 포지션
    한가운데서 시작한다. 그때의 왕복은 수량·진입가가 틀리고 없던 왕복까지 생긴다(실측:
    ETHUSDT 8건 중 3건이 유령, 6건이 항등식 불일치, 승률 50% vs 실제 80%). 원장은 그날그날
    항등식·겹침을 통과한 것만 담고 7일이 지나도 남으므로, 화면이 믿을 수 있는 유일한 판이다.

    ponytail: 꼬리만 읽는다(liq_events_load 와 같은 패턴). 매 조회마다 파일을 훑지만 30초
    캐시 뒤이고 지금 40KB 다. 수 MB 가 되면 바이트 tail 로 바꾼다."""
    try:
        with open(ACCOUNT_TRIP_LEDGER_PATH, encoding="utf-8") as fh:
            tail = deque(fh, maxlen=ACCOUNT_TRIP_CARD_ROWS)
    except OSError:
        return []
    keep = ("symbol", "side", "entry_time", "exit_time", "entry_price", "exit_price",
            "max_qty", "net_pnl")
    try:
        habits = json.loads(ACCOUNT_TRIP_HABITS_PATH.read_text())
    except (OSError, ValueError):
        habits = {}
    rows = []
    for line in tail:
        try:
            trip = json.loads(line)
        except ValueError:
            continue      # 마지막 줄이 쓰다 만 상태일 수 있다
        row = {k: trip.get(k) for k in keep}
        row.update(trip_habit(trip, habits.get(trip_key(trip))))
        rows.append(row)
    return rows


# 2026-10-14 계좌 카드 «오늘 손익»의 기준 = 오늘(KST) 첫 계좌 조회의 순자산(지갑 + 미실현). 원장 루프(300초)·화면(10초)이 이 함수를 지난다.
ACCOUNT_DAY_START_PATH = LIVE_DIR / "account_day_start.json"
KST_TZ = timezone(timedelta(hours=9))


def account_day_start(payload: dict, now: float | None = None, path: Path = ACCOUNT_DAY_START_PATH) -> dict | None:
    """날이 바뀐 뒤 첫 조회가 {date, equity, wallet, ts, late_s}를 남기고, 같은 날엔 그대로 돌려준다.

    ponytail: 입출금이 섞인다. late_s(0시 뒤 몇 초에 잰 값인가)가 크면 화면이 «지갑 − 오늘 실현» 추정으로 돌아간다."""
    bal = payload.get("balance") if isinstance(payload, dict) and payload.get("ok") else None
    try:
        wallet = float(bal["wallet"])
        equity = wallet + float(bal.get("unrealized") or 0.0)
    except (KeyError, TypeError, ValueError):
        return None
    now = time.time() if now is None else now
    kst = datetime.fromtimestamp(now, KST_TZ)
    day = kst.strftime("%Y-%m-%d")
    try:
        cur = json.loads(path.read_text())
    except (OSError, ValueError):
        cur = {}
    if cur.get("date") != day:
        midnight = kst.replace(hour=0, minute=0, second=0, microsecond=0).timestamp()
        cur = {"date": day, "equity": equity, "wallet": wallet, "ts": now, "late_s": round(now - midnight)}
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(cur))
        tmp.replace(path)
    return cur


# 2026-10-20 계좌 카드 달력·«오늘 손익» = 그날(KST) 지갑에 실제로 들어온 돈(/fapi/v1/income). 원장(왕복)은 «청산한 날»에
#   왕복 전체를 몰아 넣어(전날 부분 익절·수수료까지) 날이 어긋났고, 펀딩만 있는 날은 비었다.
#   서명 GET 하나(무게 30) · 60초에 한 번 · 첫 기동은 88일(바이낸스는 최근 3개월만 준다)을 7일 창으로 메운다.
ACCOUNT_INCOME_PATH = LIVE_DIR / "account_income_daily.json"
INCOME_REFRESH_SECONDS = 60.0
INCOME_BACKFILL_DAYS = 88
INCOME_WINDOW_MS = 7 * 86400_000
INCOME_LIMIT = 1000
INCOME_OVERLAP_MS = 600_000      # 바로 앞 10분은 다시 묻는다 -- 늦게 적히는 줄(펀딩 정산 등) 대비, 중복은 키로 거른다
INCOME_MAX_CALLS = 8             # 한 번 갱신에 최대 호출(밀린 창은 다음 갱신이 이어 받는다)
INCOME_STABLE = {"USDT", "USDC"}
INCOME_TRANSFER_TYPES = {"TRANSFER", "INTERNAL_TRANSFER", "CROSS_COLLATERAL_TRANSFER", "COIN_SWAP_DEPOSIT",
                         "COIN_SWAP_WITHDRAW", "AUTO_EXCHANGE", "STRATEGY_UMFUTURES_TRANSFER"}
INCOME_FIELDS = {"REALIZED_PNL": "realized", "FUNDING_FEE": "funding", "COMMISSION": "commission"}


def income_fold(state: dict, rows: list[dict]) -> None:
    """income 줄 → state["days"][KST 날짜] 합. 같은 줄(tranId·incomeType·time)은 한 번만(state["recent"]).
    $ 칸은 USDT·USDC 만 -- 다른 자산(BNB 수수료 등)은 그날 other_assets 에 따로(단위를 섞지 않는다)."""
    parsed = [(int(r["time"]), f"{r['tranId']}|{r['incomeType']}|{r['time']}", str(r["incomeType"]),
               str(r.get("asset") or ""), float(r["income"])) for r in rows]     # 먼저 다 읽는다 -- 깨진 줄이면 아무것도 안 바뀐다
    recent, days = state.setdefault("recent", {}), state.setdefault("days", {})
    for t, key, typ, asset, amt in parsed:
        if key in recent:
            continue
        recent[key] = t
        d = days.setdefault(datetime.fromtimestamp(t / 1000, KST_TZ).strftime("%Y-%m-%d"),
                            {"pnl": 0.0, "realized": 0.0, "funding": 0.0, "commission": 0.0, "other": 0.0, "transfer": 0.0})
        if asset not in INCOME_STABLE:
            oa = d.setdefault("other_assets", {})
            oa[asset] = oa.get(asset, 0.0) + amt
        elif typ in INCOME_TRANSFER_TYPES:
            d["transfer"] += amt
        else:
            d[INCOME_FIELDS.get(typ, "other")] += amt
            d["pnl"] += amt


async def income_refresh(state: dict, fetch, now_ms: int, max_calls: int = INCOME_MAX_CALLS) -> None:
    """state 를 제자리에서 앞으로 민다. fetch(start, end) → income 줄 목록(실패면 raise). 매 창 뒤 state 는 일관된다
    (예외가 나도 거기까지 저장해도 된다). 꽉 찬(1000줄) 창은 반으로 줄여 다시 묻는다 -- 응답 정렬 방향에 기대지 않는다."""
    if "cursor" not in state:
        day0 = (datetime.fromtimestamp(now_ms / 1000, KST_TZ) - timedelta(days=INCOME_BACKFILL_DAYS)).replace(
            hour=0, minute=0, second=0, microsecond=0)
        state.update({"from": day0.strftime("%Y-%m-%d"), "cursor": int(day0.timestamp() * 1000), "days": {}, "recent": {}})
    end = min(state["cursor"] + INCOME_WINDOW_MS, now_ms)
    for _ in range(max_calls):
        rows = await fetch(state["cursor"], end)
        if len(rows) >= INCOME_LIMIT and end > state["cursor"]:
            end = state["cursor"] + (end - state["cursor"]) // 2
            continue
        income_fold(state, rows)    # ponytail: 1ms 안에 1000줄이면 넘친 줄은 잃는다 -- 현실에선 없다
        if end >= now_ms:
            state["cursor"], state["synced_at"] = max(state["cursor"], now_ms - INCOME_OVERLAP_MS), now_ms
            break
        state["cursor"], end = end + 1, min(end + 1 + INCOME_WINDOW_MS, now_ms)
    state["recent"] = {k: t for k, t in state["recent"].items() if t >= state["cursor"]}


# 2026-10-05 계좌 카드 «습관 분해»(추가 0회/≥1 · 명목 배수 ≤6/>6 · 추세 쪽/역추세). 10-05 이후 줄은 원장이 직접
#   adds·peak_notional·equity_at_record 를 싣고, 그 전 줄은 scripts/backfill_account_trip_habits_20261005.py 가 만든 이 파일로.
ACCOUNT_TRIP_HABITS_PATH = LIVE_DIR / "account_trip_habits.json"


def trip_habit(trip: dict, backfill: dict | None) -> dict[str, Any]:
    """원장 줄 하나 → {adds, peak_x}. 모르면 None(화면은 «미분류»로 센다).

    ponytail: 새 줄의 진입 시 자본 = 기록 시 순자산 − 이 왕복 손익. 그 사이 다른 포지션 손익·입출금은 섞인다
    (기록은 청산 후 ≤5분). 정확히 하려면 income 누적을 서버가 들고 있어야 한다 -- 과거 줄은 그 방식(backfill)."""
    src = trip if trip.get("peak_notional") is not None else (backfill or {})
    eq = src.get("equity_entry")
    if eq is None and trip.get("equity_at_record") is not None:
        eq = float(trip["equity_at_record"]) - float(trip.get("net_pnl") or 0.0)
    peak = src.get("peak_notional")
    return {"adds": src.get("adds"),
            "peak_x": round(float(peak) / float(eq), 2) if peak and eq and float(eq) > 0 else None}


def annotate_trip_trend(rows: list[dict], daily: dict[str, tuple[list[float], list[int]]]) -> None:
    """각 왕복에 진입 시 1~4주 추세 부호(+1 위 · −1 아래)를 붙인다 -- 진입 전 완결 일봉만(trend_rule 과 같은 식).
    코인 = 심볼에서 USDT/USDC 를 뗀 것. 일봉이 모자라거나 신호가 0 이면 None."""
    for r in rows:
        coin = str(r.get("symbol") or "").replace("USDC", "").replace("USDT", "").lower()
        closes, ends = daily.get(coin) or ([], [])
        k = bisect.bisect_left(ends, int(r.get("entry_time") or 0))      # ends[k-1] < 진입
        tp = trend_payload(closes[:k], ends[:k]) if k else {"ok": False}
        r["trend"] = (1 if tp["signal"] > 0 else -1) if tp.get("ok") and tp["signal"] else None


def trip_key(trip: dict) -> str:
    """왕복 하나의 신원. **이 계좌는 헤지 모드**라 같은 심볼에 LONG/SHORT 가 동시에 열린다 --
    side 를 빼면 서로 다른 두 왕복이 한 건으로 뭉개진다."""
    return f"{trip.get('symbol')}|{trip.get('side')}|{trip.get('entry_time')}"


def trip_span(trip: dict) -> tuple[str, str, int, int]:
    """(심볼, 측면, 진입ms, 청산ms). 청산이 없으면 진입 한 점으로 본다."""
    entry = int(trip.get("entry_time") or 0)
    exit_ms = trip.get("exit_time")
    return (str(trip.get("symbol")), str(trip.get("side")), entry,
            int(exit_ms) if exit_ms else entry)


def load_account_trip_keys() -> dict[str, tuple[str, str, int, int]]:
    """이미 적어 둔 왕복의 신원 → 구간. 깨진 줄 하나가 원장 전체를 버리게 두지 않는다 --
    그 줄의 왕복은 seen 에 안 들어가므로 아직 API 창 안에 있으면 다음 주기에 다시 적힌다.

    2026-09-12: 키만 담던 것을 **구간까지** 담게 바꿨다. 키(진입시각)만으로는 중복을 못 막는다 --
    아래 record_account_trips 주석 참조."""
    try:
        lines = ACCOUNT_TRIP_LEDGER_PATH.read_text().splitlines()
    except FileNotFoundError:
        return {}
    except Exception as exc:
        print(f"account_round_trips read failed: {exc}", flush=True)
        return {}
    index: dict[str, tuple[str, str, int, int]] = {}
    for line in lines:
        try:
            trip = json.loads(line)
            index[trip_key(trip)] = trip_span(trip)
        except Exception:
            continue
    return index


def overlaps_recorded(trip: dict, seen: dict[str, tuple[str, str, int, int]]) -> bool:
    """같은 심볼·측면으로 **시간이 겹치는** 왕복이 이미 있는가.

    한 방향 포지션은 같은 시각에 둘일 수 없으므로 겹침은 «같은 왕복» 이라는 뜻이다.
    """
    symbol, side, lo, hi = trip_span(trip)
    for sym2, side2, lo2, hi2 in seen.values():
        if sym2 == symbol and side2 == side and lo <= hi2 and lo2 <= hi:
            return True
    return False


def record_account_trips(payload: dict, seen: dict[str, tuple[str, str, int, int]]) -> int:
    """종료된 왕복 중 처음 보는 것만 덧붙인다(append-only). 반환값은 새로 적은 건수.

    미청산 왕복은 적지 않는다 -- 나중에 청산되면 exit/net_pnl 이 채워지므로 지금 적으면
    같은 왕복의 반쪽 판이 원장에 영구히 남는다.

    🔴키(진입시각)만으로는 중복을 못 막는다. 거래소 조회가 체결 스트림의 **시작을 자르면**
    폴딩이 포지션 한가운데서 시작해 같은 포지션을 «더 늦게 진입한 더 작은 왕복» 으로 만든다.
    키가 다르니 그대로 또 적히고, 원장이 같은 거래를 두 번 센다. 2026-09-12 실측으로 28줄 중
    10줄이 그런 조각이었고, 그 조각들의 +355.02 USDT 가 원장 합계 +232.32 를 만들고 있었다
    (검증된 18줄만 보면 −122.70). 그래서 **시간 겹침**으로도 막는다."""
    if not (isinstance(payload, dict) and payload.get("ok")):
        return 0
    fresh = []
    for trip in payload.get("trades") or []:
        if not trip.get("closed") or trip_key(trip) in seen:
            continue
        # 🔴2026-10-17 실계좌 대조: 7일 창이 열린 롱 한가운데서 시작하면 그 (심볼, 측면) 스트림 **전체**의 0 교차가 어긋난다
        #   (_fold 가 스트림 모든 왕복에 truncated_open 을 단다). 어긋난 조각 중 항등식을 우연히 통과한 것(0.0095bp)이 적혔고,
        #   창이 평탄해진 뒤 나온 진짜 왕복은 아래 겹침 가드에 막혀 영영 못 들어왔다(10-10 −$27.36 · 10-12 −$3.56, 달력이 거래소보다 +$30.92).
        #   잘린 스트림은 안 적는다 -- 진짜 왕복은 창 시작이 평탄 구간을 지날 때(청산 뒤 7일 안) 다시 접혀 들어온다.
        if trip.get("truncated_open"):
            print(f"account_round_trips: 잘린 체결 스트림이라 건너뜀 {trip_key(trip)}", flush=True)
            continue
        if overlaps_recorded(trip, seen):
            print(f"account_round_trips: 겹침으로 건너뜀 {trip_key(trip)} "
                  f"(절단된 체결 스트림의 조각으로 보인다)", flush=True)
            continue
        # 🔴회계 항등식이 안 맞으면 적지 않는다. 완전히 닫힌 왕복은
        #   realizedPnl 합 = (청산VWAP − 진입VWAP) × 수량 × 방향부호 가 **정확히** 성립한다
        # (2026-09-12 재구성한 67건 전부 ±0.0000bp). 그래서 0 이 아닌 값 자체가 폴딩이 잘못됐다는
        # 신호다 -- 겹치지 않는 «빈 구간» 에 생긴 유령 왕복은 겹침 가드가 못 잡는데 이게 잡는다
        # (실측: 재시작 직후 09-10 04:53 SHORT 가 −1.04bp 로 다시 생겼고, 거래소 전체 폴딩에는 없다).
        check = trip.get("pnl_check_bp")
        if check is None or abs(float(check)) > ACCOUNT_TRIP_CHECK_TOL_BP:
            print(f"account_round_trips: 항등식 불일치로 건너뜀 {trip_key(trip)} "
                  f"(check={check}bp) -- 체결 스트림이 잘린 것으로 보인다", flush=True)
            continue
        fresh.append(trip)
    if not fresh:
        return 0
    now = datetime.now(timezone.utc).isoformat()
    equity = (payload.get("balance") or {}).get("margin")     # 습관 분해의 «진입 시 자본» 근사(trip_habit)
    try:
        with ACCOUNT_TRIP_LEDGER_PATH.open("a") as handle:
            for trip in fresh:
                handle.write(json.dumps({**trip, "recorded_at": now, "equity_at_record": equity},
                                        ensure_ascii=False) + "\n")
    except Exception as exc:
        print(f"account_round_trips append failed: {exc}", flush=True)
        return 0
    seen.update({trip_key(t): trip_span(t) for t in fresh})
    return len(fresh)


# 2026-09-11 크기 가늠자 -- 역변동성 사이징의 실시간 눈금
# (scripts/live_eth_position_sizing_worker_20260911.py). 읽기만 한다.
# 🔴수익을 예측하지 않는다. 검정된 것은 위험 축뿐이다(무작위 진입 82,167건에서 평균 명목 동일
#   조건에 표준편차 -18% · 50배 청산 도달률 13.9%->9.1%). 카드 문구도 그렇게 쓴다.
POSITION_SIZING_STATE_PATH = REPO_ROOT / "data" / "live" / "eth_position_sizing_state.json"
POSITION_SIZING_MAX_AGE_MIN = 30.0         # 워커 주기 300초 x 6

MARKET_SYMBOLS = {"eth": "ETHUSDT", "sol": "SOLUSDT", "btc": "BTCUSDT", "xrp": "XRPUSDT", "hype": "HYPEUSDT"}
# 2026-09-19 수동 진입이 나갈 심볼. **차트·시세·청산맵과 분리한다** -- MARKET_SYMBOLS 는
# 화면 전체가 공유하므로 그걸 뒤집으면 데이터 출처까지 통째로 바뀐다.
# ⭐왜 USDC 인가: 계정 실요율(/fapi/v1/commissionRate) ETHUSDT 메이커 2.0bp vs **ETHUSDC 0bp**.
#   peg-maker 섀도우 실측 전량 왕복 ETHUSDC 1.36bp vs ETHUSDT 5.77bp (2,056 leg, 09-16~).
# 🔴켜기 전 필수: 계정이 단일자산 담보(multiAssetsMargin=False)면 **USDC 잔고가 있어야** 한다.
#   2026-09-19 실측 USDT 1,355 / USDC 0 -- 이 상태로 ETHUSDC 주문은 증거금 부족으로 거절된다.
MANUAL_EXEC_SYMBOL = (os.getenv("DASHBOARD_MANUAL_EXEC_SYMBOL", "").strip().upper()
                      or MARKET_SYMBOLS["eth"])
# 2026-09-26 SOL·XRP 주문(사용자 지시, 단순 규칙). 🔴데이터는 USDT · 주문만 USDC -- SOLUSDC·XRPUSDC 는 계정 요율
#   메이커 0bp·테이커 4bp(ETHUSDC 와 같음, 09-26 commissionRate 조회). SOL·XRP 는 ETH 위험 모델을 안 쓰고
#   상한은 증거금 50% 하나뿐이다(assemble_entry_plan 의 asset 분기).
MANUAL_EXEC_SYMBOLS = {"eth": MANUAL_EXEC_SYMBOL, "sol": "SOLUSDC", "xrp": "XRPUSDC"}


def exec_asset_of(symbol: str) -> str | None:
    """주문/시장 심볼 → 코인. 브래킷 감시가 걸린 심볼에서 청산 경로를 고를 때 쓴다."""
    for a, sym in MANUAL_EXEC_SYMBOLS.items():
        if symbol in (sym, MARKET_SYMBOLS.get(a)):
            return a
    return None


# 3 + N개의 서명 GET(weight 5씩)이라 폴링 자체는 싸다. 포지션은 실시간성이 필요하고
# 체결내역은 안 변하지만, 캐시를 둘로 쪼개는 값어치는 없어서 한 페이로드 30초로 묶었다.
BINANCE_ACCOUNT_CACHE_SECONDS = 10   # 2026-10-02 30 -> 10(사용자 지시). 조회 1회 ≈ 무게 50(계좌·포지션·심볼별 체결) → 분당 ~300 = IP 한도 2,400 의 ~12%


def position_sizing_payload() -> dict[str, Any]:
    """크기 가늠자 상태. **계좌 포지션과의 결합은 프런트가 한다** -- 프런트는 이미
    `/api/binance-account` 를 들고 있어(app.js latestBinanceAccount) 서버에 비동기 의존을
    새로 만들 이유가 없다. 여기서는 파일 하나만 읽는다(요청 경로 계산 금지, 2026-09-10 실장애)."""
    if not SIZING_RISK_MODEL_ENABLED:
        return dict(SIMPLE_SIZING)
    out = worker_payload(POSITION_SIZING_STATE_PATH, POSITION_SIZING_MAX_AGE_MIN,
                         ts_field="generated_at", require_ok=True, stamp_available=True,
                         bare_missing=True)
    return {**out, "cap": sizing_cap()} if out.get("available") else out


# 2026-09-12 단건 상한. ⚠️이 값은 처음 19왕복에서 골랐는데 **그 원장이 틀려 있었다** --
# 진입/청산가가 첫/마지막 체결가였고(다리별 VWAP 아님) 절단된 체결 스트림이 만든 중복 왕복이
# 섞여 있었다. 수리·재구성 후 68왕복으로 다시 쟀고(research_sizing_cap_mult_68trips_20260912.py)
# **2.0x 를 유지**한다. 근거가 바뀌었으므로 옛 숫자는 쓰지 말 것:
#   · 🔴크기 매칭 없이 비교하면 답이 뒤집힌다 -- 미매칭 최고는 4.0x(누적 483.80)인데 평균 노출을
#     맞추면 4.0x 는 531.55 로 하위권이고 0.5x 가 637.59 로 최고다(2026-09-06 짝비교 규율).
#   · 크기 매칭 후 곡선은 **단조**(조일수록 누적↑·t↑·MDD↓)라 내부 최적점이 없다. 다만 1.0~2.5x 는
#     565~577 로 사실상 동률 고원이고 2.0x 가 그 안에 있다.
#   · 2.0x 를 쓰는 이유는 최고점이어서가 아니라 **부트스트랩 2.5% 하한이 양수로 남는 가장 느슨한
#     값**이기 때문이다(2.0x +22.48 / 2.5x −62.00, B=2000 왕복 재표집).
#   · 시간 앞뒤 절반 **양방향 이월** 확인(앞→뒤 451.96 vs 그쪽 상한없음 146.53).
# 더 조이면 기대값은 낫지만 그건 상한이 아니라 사실상 균일 크기다(68건 중 34~42건을 자른다).
# 진짜 지렛대는 배수가 아니라 균일성이고, 재량을 얼마나 남길지는 사람이 정한다.
SIZING_CAP_MULT = 2.0
SIZING_CAP_MIN_TRIPS = 10   # 이보다 적으면 중앙값이 표본 하나에 휘둘린다 -- 상한을 만들지 않는다
SIZING_CAP_WINDOW = 30      # 최근 N 왕복만 본다(2026-09-13). 아래 sizing_cap 주석 참조

# 🔴2026-09-12 저녁 수정: 원장 기반 상한만 두면 세 가지가 깨진다(실제로 터졌다).
#   ① **기준이 움직인다** — 왕복이 19→68 건이 되자 중앙 명목이 6,790→4,346 으로 내려가
#      상한이 13,579→8,692 가 됐고, 낮에 정당하게 잡은 포지션이 **소급해서 초과**가 됐다.
#   ② **자기참조다** — 상한을 «사용자가 해온 크기의 중앙값»에 묶었는데 고치려는 대상이
#      바로 그 습관이다.
#   ③ **자산과 무관하다** — 계좌가 반토막 나도 상한은 그대로인데, 실제 위험은 명목이 아니라
#      **명목 ÷ 순자산**이다.
# ⇒ 교차 마진에서 `청산까지 거리 ≈ 순자산 / 총명목` 이므로(실측 1,065/7,642=13.9% vs
#   거래소 13.48%), 총명목을 순자산의 N 배로 캡하면 **청산 거리에 1/N 하한**이 생긴다.
# 🔴2026-09-13 하향 12.5 -> 6.7 (사용자 결정). 12.5 는 «역행폭 **95분위**»로 골랐는데
#   그게 틀린 잣대였다 — 청산은 꼬리 사건이라 95분위가 아니라 꼬리를 봐야 하고,
#   보유시간도 «24시간»으로 가정했지 실제 분포를 안 봤다(실제: 중앙 1.12시간이지만
#   **최대 9일**, 1일 초과가 7.4%).
#   실측 재계산 — **상한이 실제로 묶이는 봉**(배포 모델의 권고 > 상한)에서 5분봉 MAE 를
#   **원장 68왕복의 실제 보유시간에 대입**한 «68건 중 최소 1회 이상 청산»:
#        청산거리  레버리지   묶임비율   건당    최소1회
#           8%      12.5배     9.9%    1.70%    85.9%   <- 옛 값
#          10%      10.0배    21.7%    1.19%    66.9%
#        12.5%       8.0배    39.5%    0.76%    46.3%
#        14.9%       6.7배    55.0%    0.52%    33.0%
#          20%       5.0배    77.5%    0.29%    18.9%
#   ⭐2026-09-13 재결정(사용자): **수익 우선**으로 목적을 바꿔 8.0배 채택.
#   파산을 «계좌 0» 으로 두고 실제 복리를 시뮬레이션한 결과(research_sizing_growth_optimal_leverage):
#        L    68왕복 중앙배수 / 파산율      200왕복 중앙배수 / 파산율
#        3       1.44 /  2.4%                2.89 /  7.8%
#        5       1.77 / 11.2%                4.77 / 30.4%
#        8       2.12 / 30.2%  <- 채택       0.00 / 65.3%
#       12       0.00 / 57.1%                0.00 / 92.0%
#   🔴**8배는 «한 달 승부» 값이지 지속 가능한 값이 아니다.** 68왕복(≈한 달)에서 중앙배수가
#   최대지만 200왕복이면 파산 65%·중앙 0 이다. 거래를 계속하면 5배가 최적이다.
#   ⚠️엣지 민감도: μ 를 95% 하한(8.57bp)으로 낮추면 8배 1.30 vs 5배 1.27 로 **구분되지 않는다**.
#   8배의 우위는 μ=18.37bp 점추정이 맞을 때만 나온다. μ=0 이면 모든 L 에서 중앙 < 1.
#   🔴같은 날 **초판 계산이 틀렸다**(6.7배를 19.0% 로 봤다). 두 가지였다:
#     ① 저변동을 후행 atr_pct 로 정의 — 배포 경로는 `live_eth_sizing_vol_model_20260912`
#        의 **전방 변동성 예측**이다. 후행으로 재면 «저변동일수록 꼬리가 두껍다»가 나오는데
#        (1일 MAE 99분위 21.66%) 모델 기준으로는 **7.83% 로 오히려 얇다** — 모델이 고치려던 게 그거다.
#     ② 조건을 «하위 5.68%» 로 걸었다 — 맞는 조건은 «상한이 묶이는 상태»다(6.7배에서 55%).
#   ①은 부풀리고 ②는 줄이는데 ②가 커서 합치면 과소평가였다.
#   근거 스크립트: research_sizing_cap_liquidation_risk_20260913.py (두 함정을 assert 로 고정)
# 두 상한의 **작은 쪽**을 쓴다 — 순자산 연동이 주 방어선, 원장 기반은 보조.
# 🔴2026-09-13 8.0 -> 6.0 (사용자 결정). 파산 포함 1년 복리(735건)를 다시 재니 중앙값
# 최대가 6배였다(68건이면 12배 · 180건이면 10배 — **거래를 계속할수록 최적이 내려간다**).
# 엣지를 95% 하한으로 낮춰도 6배가 최적이라 결론이 엣지 추정에 민감하지 않다.
# ⭐지금 처방(1일 보유)에는 영향이 없다 — 생존 제약 4.30배가 이미 더 작게 묶는다.
#   줄어드는 건 짧은 보유(8->6, −25%)뿐이고 1년 중앙 계좌배수는 226 -> 230 으로 오히려 올랐다.
#   근거: scratchpad kelly_cap / tradeoff 계산, research_sizing_growth_optimal_leverage_20260913.
SIZING_CAP_EQUITY_X = 6.0
# 🔴2026-09-25 사용자 지시: «증거금이 작아 위험 베팅을 해야 한다 -- 위험모델 상한만 남기고
#   나머지(원장·순자산)는 해제, 나중에 다시 적용». True 면 상한 = 위험모델(자체 천장 HARD_CAP_X
#   25배 포함) 하나이고, 지평·거래소 레버리지를 고르는 정책 천장도 HARD_CAP_X 가 된다(6배로
#   두면 레버리지가 7배로 걸려 그 위 주문을 거래소가 거부한다). 모델이 없으면(워커 낡음) 옛
#   두 상한으로 떨어진다 -- 기준이 없으면 수량을 못 만든다. 되돌리기 = False.
SIZING_CAP_MODEL_ONLY = True
# 🔴2026-09-27 사용자 지시 «변동성 모델이 필요없다고 판명 났으니 제거»: 위험(MAE 분위) 모델을 끈다.
#   False 면 ETH 도 SOL·XRP 와 같은 단순 규칙이다 -- 위험모델·원장·순자산 상한 없이 **증거금 50%
#   상한 하나**, 권고 수량·청산 권고 하한·보유시간 처방 없음. 사이징 워커도 서버에서 멈췄다.
#   근거: 새 실거래 38건에서 E|r| 크기 기울기 재현 실패(ρ +0.298→−0.075) — 메모리
#   direction_evr_gate_real_ledger_20260915 · vol-risk-upgrade-audit-20260927.
#   되돌리기 = True + 워커 재기동(scripts/ops/supervisor_position_sizing_worker.sh).
SIZING_RISK_MODEL_ENABLED = False
# 2026-10-20 자동 규칙(규칙 B)을 켜는 코인 -- 그 코인의 권장 배수(flow_read.VOLM_BY)가 있어야 실제로 열린다(없으면 rule_vol_unavailable).
RULE_ASSETS = frozenset({"eth", "xrp"})
# 위험 모델 없이 주문할 때의 사이징 입력(SOL·XRP 는 원래 이것, ETH 는 위 스위치가 꺼지면 이것).
SIMPLE_SIZING = {"available": True, "cap": {}, "risk_mae": {}, "vol_equivalent_qty": 0.0,
                 "atr_pct": None, "source": "simple_rule"}
# 2026-09-25 사용자 지시 «증거금 사용을 50% 상한으로 막아줘». 이 주문 뒤 증거금(총 명목 ÷ 주문이 걸
#   레버리지 -- 게이지 값, «자동»이면 처방 값)이 순자산의 이 %를 넘지 않는다. 위 스위치와 무관하게
#   항상 경쟁한다. 진입에만 건다 -- 증거금 초과는 닫기가 아니라 레버리지로도 풀리므로 청산 카드의
#   «최소 청산 비율»에 넣으면 틀린 처방이 된다. 0 = 끔.
# ponytail: 계좌 전체 명목을 이 심볼 레버리지 하나로 나눈다(다른 심볼 다리는 자기 레버리지를 쓴다) --
#   봇이 실주문을 내기 시작하면 balance.initial_margin 으로 바꾼다.
# 🔴2026-10-09 사용자 지시 «수량 제한 때문에 진입 불가 -- 주문·물타기·청산 규칙을 모두 제거해 수동으로, 규칙 확정되면 다시 적용»:
#   증거금 상한 해제(되돌리기 = 50.0). 크기는 진입비율 게이지(순자산 × 비율 × 레버리지)뿐이고, 넘치면 거래소가 거부한다.
SIZING_MARGIN_CAP_PCT = 0.0
# 같은 지시: 자동 규칙(rule=c 크기·물타기·손절)과 진입 때 자동 SL/TP(청산맵 익절·비상 스탑·SL 감시 무장)를 끈다.
#   이미 무장된 측면의 감시(포지션 사라지면 우리 주문 정리)는 그대로 돈다. 되돌리기 = True(app.js MANUAL_RULES 도 같이).
# 🔴2026-10-09 사용자 «1번(재생 크기식)으로 배포»: 다시 켬. 증거금 50% 상한은 꺼진 채(자동 모드는 원래 규칙 상한 하나 · 직접 모드는 게이지만).
MANUAL_RULES_ENABLED = True
# 2026-10-14 사용자 «−0.75% 물타기 지정가도 자동으로»: 자동 규칙 첫 진입이 체결되면 첫 진입가 ∓0.75% 에 남은 여유 지정가를 건다. 되돌리기 = False.
REC_AUTO_ADD = True
# 🔴상한 무시 스위치 (2026-09-20 사용자 지시 «우리 사이징 코드는 잠깐 꺼줘»).
# 세 상한(원장·순자산·위험모델)을 **전부 건너뛰고** 「순자산 × 이 배수」 하나만 기준으로 쓴다.
# 🔴상한을 «없애는» 게 아니다. 진입 비율 슬라이더의 100% 가 **상한 명목에 대한 비율**이라
#   기준이 없으면 수량 자체를 만들 수 없다 -- 끄기는 실제로 «기준을 하나로 바꾸기»다.
# 기본값 0 = 꺼짐. 커밋된 코드의 동작은 그대로이고, 서버 환경변수로만 열리며 재기동으로
# 되돌아온다(사용자 선택: 환경변수 스위치).
#   DASHBOARD_SIZING_CAP_EQUITY_X=20 python dashboard/server.py
# ⚠️이 배수를 거래소 레버리지와 **같게** 두면(둘 다 20) 슬라이더 100% 에서 필요 증거금이
#   순자산의 100% 가 되어 수수료 몫이 없다 -- 거래소가 거부한다(leverage_setting 의
#   ENFORCE_SLACK 주석이 같은 실패를 기록한다). 끝까지 쓸 생각이면 배수를 조금 낮춰라.
# ⚠️이 값이 0 보다 크면 크기를 막는 것이 **아무것도 남지 않는다** -- 이 화면이 존재하는
#   이유(재량 사이징이 실계좌 손실의 출처, 크기-수익 상관 -0.494)가 잠시 꺼진 상태다.
try:
    SIZING_CAP_OVERRIDE_X = max(0.0, float(os.getenv("DASHBOARD_SIZING_CAP_EQUITY_X", "0") or 0))
except ValueError:                 # 오타를 «무제한»으로 읽지 않는다 -- 꺼진 쪽으로 떨어뜨린다
    SIZING_CAP_OVERRIDE_X = 0.0


def notional_sum(positions: list[dict[str, Any]] | None, symbol: str | None = None) -> float:
    """|명목| 합. `symbol` 을 주면 그 심볼만, 없으면 **계좌 전체**.

    헤지 모드라 롱·숏이 동시에 열리므로 상쇄를 가정하지 않고 절대값을 더한다 -- 두 다리 다
    증거금을 먹고 둘 다 청산될 수 있다.
    🔴범위를 틀리면 조용히 위험해진다: 상한(교차 마진의 청산거리 = 순자산/총명목)은 계좌
    전체여야 하고, 청산거리 투영은 심볼 안에서만 성립한다. 2026-09-19 에 상한이 심볼 하나만
    세고 있었다(봇 ETHUSDT + 수동 ETHUSDC). 자체점검 test/test_notional_sum_scope_20260919.py.
    """
    return sum(abs(float(p.get("notional") or 0.0)) for p in (positions or [])
               if symbol is None or p.get("symbol") == symbol)


def entry_projection(plan: dict, account: dict, positions: list, existing: float,
                     equity: float) -> dict[str, Any]:
    """«이 주문을 넣으면 계좌 카드가 어떻게 바뀌나» -- 사용자 요청(2026-09-12).

    계좌 카드가 실제로 띄우는 **세 타일과 같은 정의**를 쓴다(app.js 의 tiles 참조):
      · 증거금 사용 = 사용증거금 ÷ 순자산     · 계좌 노출 = 명목 ÷ 순자산
      · 청산까지    = |마크 − 청산가| ÷ 마크
    정의가 어긋나면 «미리보기에서 본 숫자»와 «진입 후 카드»가 달라져 신뢰를 잃는다.

    청산 후 거리는 거래소가 준 **현재 청산가에서 비례 축소**한다 -- 교차 마진에서 거리는
    대략 순자산÷총명목이므로 명목이 커진 비율만큼 좁아진다. 순수 근사식(순자산÷총명목)보다
    실측값에 앵커된 이 쪽이 낫다(실측 13.48% vs 근사 13.9%). 포지션이 없으면 근사로 떨어진다.
    """
    balance = account.get("balance") or {}
    used = balance.get("initial_margin")
    used = float(used) if used is not None else max(
        0.0, float(balance.get("margin") or 0.0) - float(balance.get("available") or 0.0))
    available = float(balance.get("available") or 0.0)
    add_margin = float(plan.get("margin_usdt") or 0.0)
    total = float(plan.get("total_notional_usdt") or 0.0)

    liq_before = None
    for position in positions:
        mark, liq = float(position.get("mark_price") or 0.0), float(position.get("liquidation_price") or 0.0)
        if mark > 0 and liq > 0:
            distance = abs(mark - liq) / mark * 100.0
            liq_before = distance if liq_before is None else min(liq_before, distance)
    if liq_before is not None and total > 0 and existing > 0:
        liq_after = liq_before * existing / total
    elif equity > 0 and total > 0:
        liq_after = 100.0 * equity / total
    else:
        liq_after = None

    def side(margin_used: float, notional: float, liq: float | None) -> dict[str, Any]:
        return {
            "margin_usdt": round(margin_used, 2),
            "margin_used_pct": round(100 * margin_used / equity, 1) if equity else None,
            "available_usdt": round(equity - margin_used, 2) if equity else None,
            "notional_usdt": round(notional, 2),
            "exposure_x": round(notional / equity, 2) if equity else None,
            "liq_pct": round(liq, 2) if liq is not None else None,
        }

    return {
        "equity_usdt": round(equity, 2),
        "available_before_usdt": round(available, 2),
        "before": side(used, existing, liq_before),
        "after": side(used + add_margin, total, liq_after),
    }


def sizing_cap() -> dict[str, Any]:
    """왕복 원장에서 중앙 명목을 읽어 상한을 낸다. **현재가는 곱하지 않는다** --
    ETH 수량 환산은 가격을 이미 들고 있는 프런트가 한다(요청 경로 계산 금지 원칙)."""
    try:
        lines = ACCOUNT_TRIP_LEDGER_PATH.read_text().splitlines()
    except Exception:
        return {"available": False, "reason": "ledger_missing"}
    trips = []
    for line in lines:
        try:
            trip = json.loads(line)
            trips.append((int(trip.get("entry_time") or 0),
                          abs(float(trip["max_qty"]) * float(trip["entry_price"]))))
        except Exception:
            continue          # 깨진 줄 하나가 상한을 통째로 못 내게 하지 않는다
    if len(trips) < SIZING_CAP_MIN_TRIPS:
        return {"available": False, "reason": "not_enough_trips",
                "trips": len(trips), "need": SIZING_CAP_MIN_TRIPS}
    # 🔴2026-09-13: **최근 창**만 본다. 이력 전체에 묶으면 상한이 톱니처럼 계속 조여진다 --
    #   원장은 자라기만 하고, 과거 거래가 지금보다 작으면 중앙값이 계속 내려가 결국 정상
    #   매매까지 막는다. 실제로 왕복이 19→68건이 되자(백필) 중앙 명목 6,790→4,346,
    #   상한 13,579→8,692 로 떨어져 **낮에 정당하게 잡은 포지션이 소급 초과**가 됐다.
    #   실측(68왕복): 전체 중앙 4,346 vs 최근 30건 6,774 -- 옛 거래가 지금 규모와 무관하다
    #   (하위 12건이 646~1,194 USDT). 최근 30건 ×2 = 13,549 는 순자산 ×12.5 = 13,083 과
    #   2% 안에서 만난다 -- 독립인 두 기준이 같은 값을 가리킨다.
    trips.sort()
    notionals = [v for _, v in trips[-SIZING_CAP_WINDOW:]]
    median = statistics.median(notionals)
    return {"available": True, "trips": len(notionals), "trips_total": len(trips),
            "window": SIZING_CAP_WINDOW, "mult": SIZING_CAP_MULT,
            "median_notional_usdt": round(median, 2),
            "cap_notional_usdt": round(SIZING_CAP_MULT * median, 2)}


def register(*, account_trip_state, refresh_tasks, swr_locks, binance_session, fetch_binance_json, swr_cached,
             _timed_produce, load_market_history, situation_by, NOCACHE, STALE_GRACE_SECONDS, sec_prices=lambda _a: None):
    """make_app() 이 부른다. 인자 = 코어·시장 쪽 객체(server.py make_app 지역 · 모듈 상수). sec_prices(asset) = 1초 마지막 체결가 [(초, 가격)] · 없는 코인 None(선 손절 감시). 돌려주는 것 = 라우트·기동/정리 훅이 쓸 함수들.
    라우트 등록·on_startup 순서는 server.py 에 그대로 둔다(기동 순서 불변)."""
    binance_account_cache: dict[str, Any] = {"ts": 0.0, "payload": None}
    # 진행 중인 수동 주문 하나. 동시에 둘을 두지 않는다 -- 겹치면 단건 상한의 뜻이 흐려진다.
    manual_entry_state: dict[str, Any] = {"phase": "idle"}
    # 2026-10-20 사용자 «청산만 코인별 허용»: 위 칸을 **다른 코인**이 쓰는 동안(예: ETH 손절 청산 최대 2분) 이 코인 청산은 자기 칸으로 나간다.
    #   진입·스위칭은 여전히 위 칸 하나(상한이 계좌 전체라 진입끼리는 겹치지 않게).
    side_exit_states: dict[str, dict[str, Any]] = {}

    def _busy(st: dict | None) -> bool:
        return bool(st) and st.get("phase") in ("working", "submitting")

    def exit_slot(asset: str) -> dict | None:
        """청산이 쓸 상태 칸. 위 칸이 비었으면 위 칸 · 다른 코인이 쓰는 중이면 이 코인 청산 칸 · 같은 코인이 쓰는 중이면 None(409)."""
        side = side_exit_states.setdefault(asset, {"phase": "idle"})
        if _busy(side):
            return None
        if not _busy(manual_entry_state):
            return manual_entry_state
        return None if (manual_entry_state.get("asset") or "eth") == asset else side
    # 2026-10-16 접수 검사 ~ 상태 점유 사이 await(계획 조립 0.5~2초) 동안 두 번째 접수(폰+데스크탑)가 검사를 빠져나가 두 번 나갔다
    manual_order_lock = asyncio.Lock()
    # 2026-10-10 진입 «기다리기»(최대 60분)는 **따로** 둔다 -- 위 칸을 60분 잡으면 그동안 청산·SL 감시가 막힌다.
    manual_wait_state: dict[str, Any] = {"phase": "idle"}

    async def produce_account() -> dict[str, Any]:
        """계좌 조회는 여기 한 곳뿐이다 -- 화면 요청이든 원장 주기든 같은 캐시를 통과하므로
        조회가 두 벌로 갈라지지 않는다. 원장 기록을 이 안에 둔 이유도 같다."""
        # 🔴수동 주문 심볼(ETHUSDC)을 같이 넣는다. 이게 빠져 있어서 그 심볼의 **최근 거래**가
        #   통째로 안 보였다(2026-09-19 사용자 보고). 포지션 자체는 /fapi/v2/positionRisk 를
        #   심볼 필터 없이 부르므로 원래 다 들어온다 -- 빠지는 건 userTrades(왕복) 쪽뿐이다.
        # 🔴잔고를 어느 «자산 지갑»에서 읽을지 정한다. 단일자산 담보 모드에서 최상위 total*
        #   합계는 **USDT 전용**이라, 현금이 USDC 에만 있으면 전부 0 으로 내려온다(2026-09-19
        #   실측: USDC 1,472 인데 잔고 0). 수동 주문이 나가는 심볼의 담보 자산을 따른다 --
        #   증거금을 실제로 먹는 지갑이 거기다.
        quote = "USDC" if MANUAL_EXEC_SYMBOL.endswith("USDC") else "USDT"
        payload = await fetch_account(
            binance_session(),
            list(dict.fromkeys([*MARKET_SYMBOLS.values(), *MANUAL_EXEC_SYMBOLS.values()])),
            quote_asset=quote, income=account_income)
        # 화면이 «이 코인의 포지션»을 찾을 때 USDT 심볼 하나만 보면 USDC 포지션을 못 본다.
        # 서버가 실제로 쓰는 심볼을 payload 에 실어 보내 화면이 하드코딩하지 않게 한다 --
        # 환경변수(DASHBOARD_MANUAL_EXEC_SYMBOL)로 바뀌는 값이다.
        if isinstance(payload, dict):
            payload["exec_symbol"] = MANUAL_EXEC_SYMBOL
            payload["exec_symbols"] = dict(MANUAL_EXEC_SYMBOLS)   # 코인 -> 주문 심볼(화면이 탭마다 고른다)
        added = record_account_trips(payload, account_trip_state["seen"])
        if added:
            print(f"account_round_trips: +{added}건 (누적 {len(account_trip_state['seen'])}건)", flush=True)
        # 기록 **뒤에** 싣는다 -- 방금 닫힌 왕복이 같은 응답에 들어간다(원장 주기 300초는
        # 브라우저가 닫혀 있을 때의 보험일 뿐, 화면 요청도 이 함수를 지난다).
        if isinstance(payload, dict):
            payload["ledger"] = load_account_trip_rows()
            daily = {}
            for coin in {str(r.get("symbol") or "").replace("USDC", "").replace("USDT", "").lower() for r in payload["ledger"]}:
                if coin in MARKET_SYMBOLS:
                    try:
                        daily[coin] = await swr_cached(f"trip_trend_1d_{coin}", 3600.0,
                                                       functools.partial(load_daily_closes, coin), max_stale=86400.0)
                    except Exception as exc:  # noqa: BLE001 -- 추세 칸만 «미분류»가 된다
                        print(f"trip trend {coin}: {type(exc).__name__}: {exc}", flush=True)
            annotate_trip_trend(payload["ledger"], daily)
            try:
                payload["day_start"] = account_day_start(payload)
            except OSError as exc:   # 기록 실패는 화면 추정으로 돌아갈 뿐 -- 계좌 응답을 막지 않는다
                print(f"account_day_start: {type(exc).__name__}: {exc}", flush=True)
        return payload

    income_cache: dict[str, Any] = {"ts": 0.0, "state": None}

    async def account_income(get) -> dict[str, Any]:
        """fetch_account 가 성공 끝에 부른다(get = 서명 GET /fapi/v1/income). {income_days, income_from, income_synced_at} --
        10분 넘게 못 따라잡았으면 {}(화면은 원장 추정으로 돌아간다). 실패는 로그만 -- 계좌 응답을 막지 않는다."""
        now = time.time()
        if now - income_cache["ts"] >= INCOME_REFRESH_SECONDS:
            income_cache["ts"] = now     # await 전에 -- 겹친 호출이 같은 갱신을 두 번 하지 않게
            if income_cache["state"] is None:
                try:
                    income_cache["state"] = json.loads(ACCOUNT_INCOME_PATH.read_text())
                except (OSError, ValueError):
                    income_cache["state"] = {}
            state = income_cache["state"]

            async def fetch(start: int, end: int) -> list[dict]:
                rows = await get({"startTime": start, "endTime": end, "limit": INCOME_LIMIT})
                if not isinstance(rows, list):
                    raise RuntimeError(rows.get("__error__") if isinstance(rows, dict) else rows)
                return rows
            try:
                await income_refresh(state, fetch, int(now * 1000))
            except Exception as exc:  # noqa: BLE001 -- 키·밴·네트워크 전부. 거기까지 민 상태는 일관되다(income_refresh)
                print(f"account_income: {type(exc).__name__}: {exc}", flush=True)
            try:
                tmp = ACCOUNT_INCOME_PATH.with_suffix(".tmp")
                tmp.write_text(json.dumps(state))
                tmp.replace(ACCOUNT_INCOME_PATH)
            except OSError as exc:
                print(f"account_income save: {type(exc).__name__}: {exc}", flush=True)
        state = income_cache["state"] or {}
        if now * 1000 - (state.get("synced_at") or 0) > 600_000:
            return {}
        return {"income_days": state["days"], "income_from": state["from"], "income_synced_at": state["synced_at"]}

    async def load_daily_closes(coin: str) -> tuple[list[float], list[int]]:
        """완결 일봉 종가·종가 시각(ms, 봉 끝+1) 300개 -- 원장 첫 왕복(08월) 진입 전 90일 + 여유. 가중치 2, 1시간 캐시."""
        raw = await fetch_binance_json("https://fapi.binance.com/fapi/v1/klines",
                                       {"symbol": MARKET_SYMBOLS[coin], "interval": "1d", "limit": 300},
                                       error_reason="trip_trend_upstream_error")
        done = [r for r in raw if int(r[6]) < time.time() * 1000]
        return [float(r[4]) for r in done], [int(r[6]) + 1 for r in done]

    async def keep_trip_ledger() -> None:
        """브라우저가 닫혀 있어도 돌아야 한다 -- 사라지는 쪽은 거래소의 7일 창이지 화면이 아니다.

        별도 태스크로 떼는 이유: 부르는 쪽(publish_dashboard_events)은 모든 탭의 SSE 유일
        공급원이라, 서명 요청 3~4개를 그 안에서 기다리면 화면 갱신이 함께 밀린다."""
        try:
            await swr_cached("binance_account", BINANCE_ACCOUNT_CACHE_SECONDS, produce_account,
                             max_stale=STALE_GRACE_SECONDS, cache=binance_account_cache)
        except Exception as exc:  # noqa: BLE001 -- 키 만료·네트워크·시계드리프트 전부 여기로 온다
            print(f"account_round_trips cycle failed (will retry next cycle): {exc}", flush=True)

    async def api_binance_account(request: web.Request) -> web.Response:
        """실계좌 잔고/포지션/왕복거래(진입·청산 시각). 키에 Futures 읽기 권한이 없으면
        ok=false + hint로 내려가고, 프런트는 그 문구를 그대로 보여준다.
        `?fresh=1` = 캐시를 건너뛰고 지금 다시 조회(2026-10-06 사용자 «주문이 완료되면 플로팅 버튼 계좌 조회를 바로») --
        주문 체결 직후 화면이 한 번 부른다. 1초 안에 이미 새로 받은 값이 있으면 그걸 준다(연달아 눌려도 거래소는 한 번)."""
        if request.query.get("fresh") == "1":
            async with swr_locks.setdefault("binance_account", asyncio.Lock()):
                if time.monotonic() - binance_account_cache.get("ts", 0.0) > 1.0:
                    binance_account_cache["payload"] = await _timed_produce("binance_account", produce_account)
                    binance_account_cache["ts"] = time.monotonic()
            return web.json_response(with_bracket(binance_account_cache["payload"]), headers=NOCACHE)
        payload = await swr_cached(
            "binance_account", BINANCE_ACCOUNT_CACHE_SECONDS, produce_account,
            cache=binance_account_cache,
            max_stale=STALE_GRACE_SECONDS,
        )
        return web.json_response(with_bracket(payload), headers=NOCACHE)

    def with_bracket(payload: Any) -> Any:
        """2026-10-15 계좌 카드가 «서버가 무장한 손절»을 보여 주도록(10-14 사고: 바이낸스에서 스탑만 지웠는데 5분봉 종가 감시가 청산) --
        무장 파일의 화면용 몇 칸만 얹는다(캐시된 계좌 payload 는 건드리지 않는다)."""
        if not isinstance(payload, dict):
            return payload
        armed = {k: {f: v.get(f) for f in ("sl_price", "sl_name", "sl_off", "rule", "lines", "tp_level", "sl_level", "tp_resting")} for k, v in bracket_load().items()}
        # 2026-10-20 두 코인 보유(사용자): 손절 무장이 없는 열린 포지션 -- 화면이 «손절 없음»을 크게 말한다(10-20 XRP 가 조용히 무방비였다)
        unprotected = [k for k in (bracket_key(p.get("symbol"), str(p.get("side")).upper()) for p in payload.get("positions") or []
                                   if float(p.get("qty") or 0)) if not (armed.get(k) or {}).get("sl_price")]
        return {**payload, "bracket_armed": armed, "unprotected": unprotected}

    def effective_cap(sizing: dict[str, Any], equity: float,
                      safe_mae_pct: float | None) -> tuple[float | None, str | None, dict]:
        """세 상한(원장·순자산·위험모델)의 **작은 쪽**과 어느 쪽이 묶었는지.

        🔴진입과 청산이 이 함수를 **같이** 쓴다(2026-09-13). 예전에는 진입만 세 상한을 쓰고
        청산의 «최소 청산 비율»은 정책상한 25배를 기준으로 계산해서, 20배 포지션에서 같은
        카드가 «최소 청산 0%»와 «예산 사다리 60%»를 나란히 띄웠다."""
        cap = dict(sizing.get("cap") or {})
        # 🔴상한 무시 스위치 -- assemble_entry_plan 의 인라인 블록과 **같은 규칙**이어야 한다.
        #   두 곳이 갈리면 «추천 지평은 옛 상한, 실제 주문은 새 상한»이 된다.
        if SIZING_CAP_OVERRIDE_X > 0 and equity > 0:
            return equity * SIZING_CAP_OVERRIDE_X, "override", cap
        cands = []
        if cap.get("available") and cap.get("cap_notional_usdt"):
            cands.append((float(cap["cap_notional_usdt"]), "ledger"))
        if equity > 0:
            cands.append((equity * SIZING_CAP_EQUITY_X, "equity"))
            if safe_mae_pct and safe_mae_pct > 0:
                cands.append((entry_notional(equity, safe_mae_pct)["total_notional"], "model"))
        if SIZING_CAP_MODEL_ONLY and equity > 0:
            # 모델 후보가 있으면 그것만, 정책 천장 질의(safe_mae_pct=None)면 모델의 자체 천장.
            model = [c for c in cands if c[1] == "model"]
            cands = model or ([(equity * HARD_CAP_X, "model")] if safe_mae_pct is None else cands)
        if not cands:
            return None, None, cap
        # 🔴값만 비교한다 -- 튜플 min 은 값이 같을 때 이름 알파벳순으로 갈린다.
        notional, who = min(cands, key=lambda t: t[0])
        return notional, who, cap

    async def realized_vol_now(symbol: str) -> float | None:
        """청산 마감·기대 체결시간을 정하는 1분봉 실현변동성. 실패해도 주문을 막지 않는다 --
        None 이면 exit_deadline_sec 이 보수적으로 최대(120초)를 쓴다."""
        try:
            kl = await fetch_binance_json(
                "https://fapi.binance.com/fapi/v1/klines",
                {"symbol": symbol, "interval": "1m", "limit": EXIT_VOL_WINDOW + 2},
                timeout=5.0, error_reason=None)
            if kl:
                # 마지막 봉은 **미완결**이라 버린다. 종가는 인덱스 4.
                return realized_vol_bpm([float(row[4]) for row in kl[:-1]])
        except Exception:  # noqa: BLE001 -- 변동성은 있으면 좋은 값이지 필수가 아니다
            pass
        return None

    async def basis_for(book: dict, asset: str = "eth") -> float:
        """주문 심볼(USDC) 중간가 ÷ 화면 심볼(USDT) 중간가 -- 손절가(주문 심볼)를 5분봉 감시용 USDT 레벨로 옮긴다.
        ponytail: 못 읽으면 1(USDC/USDT 수 bp 오차) -- 거래소 스탑은 주문 심볼 가격이라 영향 없고 5분봉 감시만 수 bp 어긋난다."""
        try:
            ref = await fetch_binance_json("https://fapi.binance.com/fapi/v1/ticker/bookTicker",
                                           {"symbol": MARKET_SYMBOLS[asset]}, error_reason="book_ticker_failed")
            return ((float(book["bidPrice"]) + float(book["askPrice"])) / (float(ref["bidPrice"]) + float(ref["askPrice"]))) or 1.0
        except Exception:  # noqa: BLE001
            return 1.0

    # ── 청산맵 TP/SL 감시 (2026-09-25) ────────────────────────────────────────────
    # 무장 상태는 **파일**에 둔다 -- 옛 3% 손절의 실패가 메모리 state 에만 있어 재시작하면 사라졌다.
    #   {측면: {symbol, sl_price, tp_price, backstop_price, armed_at, placed}}
    def bracket_path() -> Path:
        return LIVE_DIR / "manual_bracket_state.json"

    def bracket_load() -> dict[str, Any]:
        try:
            raw = json.loads(bracket_path().read_text("utf-8"))
        except (OSError, ValueError):
            return {}
        return bracket_rekey(raw)

    def bracket_save(st: dict[str, Any]) -> None:
        tmp = bracket_path().with_suffix(".tmp")
        tmp.write_text(json.dumps(st, ensure_ascii=False), "utf-8")
        tmp.replace(bracket_path())

    async def entry_then_bracket(plan: dict, state: dict, wait_sec: float | None = None) -> None:
        """진입 → 체결이 있으면(부분 포함, 모든 종료 경로) TP·비상 스탑을 걸고 감시를 무장한다.
        wait_sec(2026-10-10 «기다리기»)면 첫 체결을 본 순간 한 번 더 건다 -- 60분 동안 든 몫을 무방비로 두지 않는다."""
        if wait_sec:
            async def first_fill() -> None:
                try:
                    await arm_bracket(plan, state, add=False)   # 2026-10-20 첫 부분 체결로 물타기 크기를 재면 남은 여유 거의 전부가 걸린다
                except Exception as exc:  # noqa: BLE001 -- 대기는 계속(마감 때 다시 건다)
                    state["bracket"] = {"placed": False, "reason": f"{type(exc).__name__}: {exc}"}
            await settle(run_entry_wait(binance_session(), plan, state, wait_sec, on_fill=first_fill), state)
        else:
            await settle(run_entry(binance_session(), plan, state), state)
        if float(state.get("filled") or 0.0) > 0:
            await arm_bracket(plan, state)

    async def settle(coro, state: dict) -> None:
        """집행 예외가 새면 phase 가 working 에 영영 묶인다 -- 모든 접수가 409, 서버 손절 감시도 쏘지 않는다(10-16 감사)."""
        try:
            await coro
        except Exception as exc:  # noqa: BLE001
            state.update(phase="error", error=f"집행 오류 {type(exc).__name__}: {exc} — 거래소에서 직접 확인하세요",
                         done_at=datetime.now(timezone.utc).isoformat())

    async def arm_bracket(plan: dict, state: dict, add: bool = True) -> None:
        b, pside = plan.get("bracket") or {}, plan["positionSide"]
        if not b.get("available"):
            state["bracket"] = {"placed": False, "reason": b.get("reason") or "청산맵 레벨 없음"}
            return
        add = add and not b.get("lines")   # 2026-10-28 선 진입 = 물타기 지정가 없음(익절 수량이 포지션 전량이라 크기가 늘면 어긋난다)
        if add and REC_AUTO_ADD and (plan.get("rule") or {}).get("first"):
            state["add_order"] = {"placed": False, "pending": True}   # 화면 폴링이 물타기 지정가 결과까지 기다리게(손절 먼저 건 뒤 arm_rule_add)
        try:
            state["bracket"] = await place_bracket(binance_session(), b, plan["symbol"], pside)
        except Exception as exc:  # noqa: BLE001 -- 결과 칸은 반드시 채운다(화면 폴링이 이걸 기다린다)
            state["bracket"] = {"placed": False, "reason": f"{type(exc).__name__}: {exc}"}
        add = await arm_rule_add(plan, state) if add else None
        st = bracket_load()
        st[bracket_key(plan["symbol"], pside)] = {"symbol": plan["symbol"], "side": pside, "sl_price": b.get("sl_price"),
                     "sl_level": b.get("sl_level"),   # 5분봉 종가 판정은 USDT 레벨로(bracket_sl_hit)
                     "tp_price": b.get("tp_price"), "backstop_price": b.get("backstop_price"),
                     "rule": bool(b.get("rule")), "sl_name": b.get("sl_name"),   # 2026-10-09 권고 규칙 손절 = 물타기해도 유지
                     # 2026-10-24 직접 진입도 같은 −5% 손절(rule True)이라 «자동 규칙 크기로 연 포지션»은 따로 남긴다 -- 계좌 3배 공유 상한(rule_shared)이 이것만 본다
                     "auto": bool(plan.get("rule")),
                     "lines": bool(b.get("lines")), "tp_level": b.get("tp_level"),   # 2026-10-28 선: 1초 체결가 닿음 감시
                     "tp_resting": bool(((state["bracket"] or {}).get("tp") or {}).get("placed")),   # 익절 지정가가 못 걸렸으면 닿을 때 서버가 peg 로
                     "armed_at": time.time(), "placed": state["bracket"], "add_order": add}
        bracket_save(st)

    async def arm_rule_add(plan: dict, state: dict) -> dict | None:
        """2026-10-14 사용자 «−0.75% 물타기 지정가도 자동으로»: 자동 규칙 **첫 진입**이 체결되면(손절을 건 바로 뒤) 첫 진입가 ∓0.75% 에
        남은 여유를 지정가로 건다. 크기 = 실제 체결 수량으로 다시 잰 rec_rule_add_order(수동 물타기와 같은 손실 한도).
        지우기 = cancel_bracket(ADD_TAG): 포지션이 닫히면(감시 루프 clear) · 손으로 물타기·새 진입(place_bracket 이 먼저 지운다).
        ponytail: 포지션을 손으로 닫고 감시 루프가 지우기 전(≤ 계좌 조회 주기) 그 가격에 닿으면 다시 열린다 -- 그때도 «포지션 전량» 손절 스탑이 같이 남아 있다."""
        r = plan.get("rule") or {}
        filled = float(state.get("filled") or 0.0)
        if not (REC_AUTO_ADD and r.get("first") and filled > 0 and (plan.get("bracket") or {}).get("sl_price")):
            return None
        if not ((state.get("bracket") or {}).get("backstop") or {}).get("placed"):
            # 🔴손절 스탑이 안 걸렸으면 크기를 늘리지 않는다 -- 무방비 포지션을 키우는 게 가장 비싼 실패다(시험에서 가짜 place_bracket 이 실패해도 여기서 멈춘다)
            state["add_order"] = {"placed": False, "reason": "손절 스탑이 안 걸려 물타기 지정가를 걸지 않았습니다"}
            return state["add_order"]
        try:
            spec = rec_rule_add_order(position_side=plan["positionSide"], sl_price=float(plan["bracket"]["sl_price"]),
                                      stop_pct=float(r["stop_pct"]) / 100.0, l_eff=float(r["l"]), cap_notional=float(r["cap_notional"]),
                                      room_before=float(r["room"]), filled_qty=filled, order_leverage=float(r.get("lev") or 0.0),
                                      filters=r.get("filters") or {"tick": 0.01, "step": 0.001})
            if not spec:
                state["add_order"] = {"placed": False, "reason": "남은 여유가 없거나 최소 주문보다 작습니다"}
                return state["add_order"]
            state["add_order"] = await place_add_order(binance_session(), plan["symbol"], plan["positionSide"], spec["price"], spec["qty"])
        except Exception as exc:  # noqa: BLE001 -- 진입·손절은 이미 됐다, 결과 칸만 채운다
            state["add_order"] = {"placed": False, "reason": f"{type(exc).__name__}: {exc}"}
        return state["add_order"]

    async def exit_then_switch(plan: dict, state: dict, sw: dict, asset: str) -> None:
        """스위칭(2026-10-10 서버로 옮김): 전량 청산이 **다 닫힌 뒤에만** 반대 진입. 10-10 부터 두 다리 다 시장가(즉시 체결). 진입 계획·게이트·규칙 검사는 진입 버튼과 같은
        assemble_entry_plan(fresh=True). 막히면 주문 없이 사유를 state["switch"] 에 남긴다 -- 진입으로 넘어가도 «스위칭에서 왔다»는 남는다."""
        try:
            await run_exit(binance_session(), plan, state)
            info = state["switch"]
            if not (state.get("phase") in ("filled_maker", "filled_taker")
                    and float(state.get("filled") or 0.0) >= float(state.get("quantity") or 0.0) * 0.999):
                info.update(phase="aborted", error=f"청산이 다 닫히지 않아({state.get('phase')}) 반대 진입을 내지 않았습니다")
                return
            info = {**info, "phase": "entry", "exit": {k: state.get(k) for k in ("phase", "filled", "quantity", "done_at")}}
            state.clear()       # 🔴await 없이 바로 진입 상태로 -- 사이에 다른 주문이 already_working 검사를 빠져나가지 않게
            state.update(phase="submitting", side=sw["to"], asset=asset, switch=info,
                         started_at=datetime.now(timezone.utc).isoformat())
            why = None if exec_enabled() else "DASHBOARD_MANUAL_EXEC_ENABLED 가 꺼져 있습니다"
            if not why:
                plan2, _, _, err = await assemble_entry_plan(sw["to"], sw["frac"], sw["lev"], asset, fresh=True, rule_l=sw["rule"])
                why = (err[0].get("detail") or err[0].get("error")) if err else plan2.get("blocked")
            if why:
                state.update(phase="error", error=f"스위칭 반대 진입 막힘 — {why}", done_at=datetime.now(timezone.utc).isoformat())
                info.update(phase="aborted", error=str(why))
                return
            plan2["type"] = "MARKET"   # 2026-10-10 스위칭 = 즉시 체결 -- run_entry 가 지정가 없이 시장가로 낸다
            state["plan"] = plan2
            info["plan"] = {k: plan2.get(k) for k in ("quantity", "notional_usdt", "price", "target_leverage")}
            await entry_then_bracket(plan2, state)
            ok = str(state.get("phase") or "").startswith("filled")
            info.update(phase="done" if ok else "aborted", error=None if ok else state.get("error") or state.get("phase"))
        except Exception as exc:  # noqa: BLE001 -- 사유를 남긴다(화면 폴링이 «진행 중»에 멈추지 않게)
            state.update(phase="error", error=f"스위칭 오류 {type(exc).__name__}: {exc}")
            (state.get("switch") or {}).update(phase="aborted", error=str(exc))

    async def bracket_tick() -> None:
        """무장된 측면마다: 포지션이 사라졌으면(TP·비상 스탑·수동 청산) 남은 우리 주문을 지우고 내린다.
        SL 을 넘었으면 **메이커 추격 청산**(청산 버튼과 같은 run_exit)을 돌린다."""
        st = bracket_load()
        if not st or not exec_enabled():
            return
        account = await swr_cached("binance_account", BINANCE_ACCOUNT_CACHE_SECONDS,
                                   produce_account, max_stale=STALE_GRACE_SECONDS,
                                   cache=binance_account_cache)
        try:
            acct_ts = datetime.fromisoformat(str(account.get("generated_at"))).timestamp()
        except ValueError:
            acct_ts = 0.0
        changed = False
        seen_at = {k: v.get("armed_at") for k, v in st.items()}
        for key, b in list(st.items()):
            pside = b.get("side") or key.rsplit(":", 1)[-1]
            # 2026-09-27 SL = 마감된 USDT 5분봉 종가(사용자 지시). 차트가 쓰는 60초 캐시라 REST 가 늘지 않는다
            #   (옛 판은 측면마다 2초에 한 번 bookTicker). 못 읽으면 None -> hold -- «정리»는 계속 판단한다.
            try:
                now_s = time.time()
                closed = [c for c in await load_market_history(exec_asset_of(b["symbol"]) or "eth")
                          if int(c["time"]) + FOOTPRINT_BAR_SECONDS <= now_s]
                last_bar = (int(closed[-1]["time"]) + FOOTPRINT_BAR_SECONDS, float(closed[-1]["close"])) if closed else None
            except Exception as exc:  # noqa: BLE001
                last_bar = None
                print(f"bracket {pside}: 5분봉 못 읽음 {type(exc).__name__}: {exc}", flush=True)
            touch = None
            if b.get("lines"):   # 2026-10-28 무장 뒤 1초 마지막 체결가 범위(USDT) -- 감시 2초 틱 사이도 빠짐없이
                since = max(float(b["armed_at"]), float(b.get("watch_from") or 0))   # 손절 해제 → 다시 켜기 사이에 지나간 가격은 안 본다
                pts = [px for t, px in sec_prices(exec_asset_of(b["symbol"]) or "eth") or [] if t > since]
                touch = (min(pts), max(pts)) if pts else None
            act = bracket_action(b, pside, account.get("positions") or [], bool(account.get("ok")),
                                 acct_ts, last_bar, touch)
            if (act in ("clear", "fire") and wait_pending() and manual_wait_state.get("side") == pside
                    and (manual_wait_state.get("plan") or {}).get("symbol") == b["symbol"]):   # 2026-10-20 다른 코인 대기는 두다
                # 2026-10-10 손절·청산으로 포지션이 닫히는데 같은 쪽 대기 진입이 걸려 있으면 그것도 거둔다(나중에 체결돼 손절 없이 다시 열리지 않게)
                manual_wait_state.update(action="cancel", action_reason="포지션이 닫힘(손절·청산) — 같은 쪽 대기 진입 취소")
            if act == "clear":
                if time.time() < float(b.get("retry_after") or 0):
                    continue
                r = await clear_bracket(binance_session(), b["symbol"], pside)
                print(f"bracket {pside}: 포지션 없음 -> 정리 {r}", flush=True)
                if r.get("errors"):   # 🔴못 지운 물타기 지정가(dbad)가 남으면 손절 없이 다시 열린다 -- 무장을 남겨 30초 뒤 다시 지운다
                    b["retry_after"] = time.time() + 30
                else:
                    st.pop(key)
                changed = True
                continue
            b_asset = exec_asset_of(b["symbol"]) or "eth"
            if act != "fire" or exit_slot(b_asset) is None or time.time() < float(b.get("retry_after") or 0):
                continue
            if b.get("lines") and not b.get("fired"):
                # 2026-10-28 검증: 걸린 익절 지정가를 **먼저** 지운다 -- 청산 수량을 그 뒤에 읽어야 겹쳐 넘기지 않고, 못 지우면 계좌 조회(무게 ~50)를
                #   매 2초 반복하지 않게 10초 쉰다. 지웠으면 fired 로 굳혀 닫힐 때까지 쏜다(익절이 없는데 닿음 창이 지나 조용히 멈추지 않게).
                if manual_order_lock.locked() or (bracket_load().get(key) or {}).get("sl_off"):
                    continue
                r = await clear_bracket(binance_session(), b["symbol"], pside)
                if r.get("errors"):
                    b["retry_after"] = time.time() + 10
                    b["last_error"] = f"익절 지정가 취소 실패: {'; '.join(map(str, r['errors']))}"
                    changed = True
                    continue
                b.update(fired=True, tp_resting=False)
                changed = True
            plan, err = await assemble_exit_plan(pside, 1.0, fresh=True, asset=exec_asset_of(b["symbol"]) or "eth")
            # 🔴검사는 조립(await) **뒤**에 -- 그 사이 «손절 해제»나 사용자 청산이 들어오면 쏘지 않는다(남의 상태를 clear 하고 두 번 청산하지 않게)
            if (bracket_load().get(key) or {}).get("sl_off"):
                continue
            slot = exit_slot(b_asset)   # 2026-10-20 다른 코인 주문이 도는 중이면 이 코인 칸으로 청산(옛 판: 그 주문이 끝날 때까지 손절을 미뤘다)
            if manual_order_lock.locked() or slot is None:
                continue
            if err and err[0].get("error") == "no_position":     # 이미 닫혔다 -- 정리하고 내린다(못 지웠으면 다음 틱 «clear» 가 다시)
                if not (await clear_bracket(binance_session(), b["symbol"], pside)).get("errors"):
                    st.pop(key)
                changed = True
                continue
            if err or plan.get("blocked"):
                # 🔴조회 실패는 «닫혔다»가 아니다 -- 비상 스탑을 두고 30초 뒤 다시 본다.
                b["retry_after"] = time.time() + 30
                b["last_error"] = (err[0] if err else {}).get("detail") or plan.get("blocked")
                changed = True
                continue
            slot.clear()
            slot.update(phase="submitting", kind="exit", trigger="bracket_sl",
                        side=pside, plan=plan, asset=b_asset,
                        started_at=datetime.now(timezone.utc).isoformat())
            await settle(run_exit(binance_session(), plan, slot), slot)
            # 🔴다 닫혔을 때만 비상 스탑을 지운다 -- 청산이 실패했는데 지우면 무방비가 된다.
            if slot.get("phase") in ("filled_maker", "filled_taker"):
                slot["bracket_cleanup"] = await clear_bracket(
                    binance_session(), b["symbol"], pside)
                if not slot["bracket_cleanup"].get("errors"):   # 못 지웠으면 남겨 둔다 -- 다음 틱 «clear» 가 다시 지운다
                    st.pop(key)
            else:
                b["retry_after"] = time.time() + 30
                b["last_error"] = slot.get("error")
            changed = True
            print(f"bracket {pside}: SL {b.get('sl_price')} (레벨 {b.get('sl_level')}) 5분봉 종가 {last_bar} · 1초 범위 {touch} 이탈 -> "
                  f"{slot.get('phase')}",
                  flush=True)
        if changed:
            # 🔴다시 읽고 **내가 본 무장(armed_at)** 에만 적용한다 -- 위 await 동안 새 진입이 다른
            #   측면(또는 같은 측면 물타기)을 무장했으면 통째로 쓰면 그걸 지운다.
            bracket_save(bracket_merge(bracket_load(), st, seen_at))

    async def watch_brackets(app: web.Application) -> None:
        while True:
            await asyncio.sleep(2.0)
            try:
                await bracket_tick()
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001 -- 감시는 죽지 않는다(한 틱 실패는 다음 틱에)
                print(f"bracket watch: {type(exc).__name__}: {exc}", flush=True)

    async def start_bracket_watcher(app: web.Application) -> None:
        app["bracket_task"] = asyncio.create_task(watch_brackets(app))

    async def stop_bracket_watcher(app: web.Application) -> None:
        app["bracket_task"].cancel()
        try:
            await app["bracket_task"]
        except asyncio.CancelledError:
            pass

    async def assemble_entry_plan(side: str, fraction: float = 1.0,
                                  want_lev: int | None = None, asset: str = "eth", fresh: bool = False,
                                  rule_l: str | None = None, lines: tuple[float, float] | None = None):
        """계획 조립은 **여기 한 곳뿐**이다 -- 미리보기와 실주문이 같은 입력·같은 함수를 지난다.
        두 곳에 복사해 두면 언젠가 한쪽만 고쳐져 «미리보기와 다른 주문»이 나간다.

        반환: (plan, cap, sizing, error). error 는 (본문, HTTP상태) 또는 None.
        계좌를 같이 읽는 이유 둘: (1) 상한을 **합산 포지션**에 걸어야 하고
        (2) 화면이 «현금 얼마·레버리지 몇 배·청산까지 몇 %»를 말할 수 있어야 한다."""
        symbol = MANUAL_EXEC_SYMBOLS[asset]   # 2026-09-19 차트 심볼과 분리(위 상수 주석) · 09-26 코인별
        try:
            # 🔴SOL·XRP 는 ETH 위험 모델(보유시간·안전 역행폭·권고 수량)을 쓰지 않는다 -- ETH 로 학습된 값이다.
            #   빈 사이징을 넘겨 모델 상한·권고가 «없음»으로 흐르게 하고, 상한은 아래 증거금 50% 하나만 남긴다.
            sizing = (await asyncio.to_thread(position_sizing_payload) if asset == "eth"
                      else dict(SIMPLE_SIZING))
            if not sizing.get("available"):
                return None, {}, {}, ({"error": "sizing_unavailable",
                                       "detail": sizing.get("error")}, 503)
            # 2026-10-05 fresh=1(떠 있는 버튼 «스위칭»의 반대 진입): 방금 닫은 포지션이 30초 캐시에 남아 노출로 잡히면
            #   증거금 50% 상한이 반대 진입을 깎는다 -- 청산(assemble_exit_plan fresh)처럼 계좌를 새로 읽는다.
            account = (await produce_account() if fresh else
                       await swr_cached("binance_account", BINANCE_ACCOUNT_CACHE_SECONDS,
                                        produce_account, max_stale=STALE_GRACE_SECONDS,
                                        cache=binance_account_cache))   # 계좌 카드와 같은 캐시 -- 없으면 두 벌을 따로 받는다
            if not account.get("ok"):   # 못 읽은 계좌로 상한(합산 노출)을 계산하면 상한이 헐거워진다
                return None, {}, {}, ({"error": "account_unavailable",
                                       "detail": f"계좌 조회 실패: {account.get('error')}"}, 503)
            positions = [p for p in (account.get("positions") or []) if p.get("symbol") == symbol]
            # 헤지 모드라 롱·숏이 동시에 열린다. 위험 상쇄를 가정하지 않고 **절대값 합**으로 본다
            # -- 두 다리 다 증거금을 먹고, 둘 다 청산될 수 있다.
            existing = notional_sum(positions)
            # 🔴상한은 **계좌 전체**에 걸린다. 교차 마진이라 청산거리 = 순자산 / **총명목**이고,
            #   그 총명목에는 다른 심볼도 들어간다 -- 봇은 ETHUSDT 로, 수동 주문은 ETHUSDC 로
            #   나가므로(2026-09-19 MANUAL_EXEC_SYMBOL) 심볼 하나만 세면 봇 포지션이 열려 있는
            #   동안 상한이 그만큼 헐거워진다. 위 `existing`(이 심볼)은 아래 청산거리 투영에만
            #   쓴다 -- 그 계산은 심볼 안에서만 성립한다(liq_after = liq_before × existing/total).
            exposure = notional_sum(account.get("positions"))
            # 2026-10-20 두 코인 보유: **다른 심볼**에 걸린 늘리는 지정가(자동 물타기 dbad 등, reduce-only 아님)는 체결되면 같은 증거금을 먹는다 --
            #   상한 여유에서 미리 뺀다(10-20 감사: ETH 물타기 지정가 ~$7.4k 가 XRP 진입 뒤에도 그 크기로 남았다).
            #   ponytail: clientOrderId 없이 심볼로만 가른다 -- 같은 심볼의 걸린 지정가는 새 진입의 place_bracket 이 지우는 몫이라 뺀다.
            exposure += sum(abs(float(o.get("price") or 0) * float(o.get("qty") or 0)) for o in account.get("open_orders") or []
                            if o.get("symbol") != symbol and not o.get("reduce_only"))
            # 추가 진입 맥락은 **같은 방향**만 본다 -- 헤지 모드에서 반대 다리는 다른 결정이다.
            same = [p for p in positions if p.get("side") == side]
            same_unrealized = sum(float(p.get("unrealized_pnl") or 0.0) for p in same)
            equity = float((account.get("balance") or {}).get("margin") or 0.0)
            # 포지션이 없으면 positions 가 비어 있다 -- 그때도 설정 레버리지는 알아야
            # «증거금 얼마»를 말할 수 있다(교차 마진에서 증거금 = 명목/레버리지).
            leverage = (max((float(p.get("leverage") or 0.0) for p in positions), default=0.0)
                        or float((account.get("leverage_by_symbol") or {}).get(symbol) or 0.0))
            book = await fetch_binance_json("https://fapi.binance.com/fapi/v1/ticker/bookTicker",
                                            {"symbol": symbol}, error_reason="book_ticker_failed")
            filters = await load_filters(binance_session(), symbol)
            cap = dict(sizing.get("cap") or {})
            # 두 상한의 **작은 쪽**. 순자산 연동은 원장이 자라도 안 변하고 자산이 줄면 같이
            # 줄어든다 -- 소급 초과가 생기지 않는다. 어느 쪽이 묶었는지 화면에 남긴다.
            cap_ledger = cap.get("cap_notional_usdt") if cap.get("available") else None
            cap_equity = equity * SIZING_CAP_EQUITY_X if equity > 0 else None
            # 🔴거래소 레버리지의 기준은 **지평과 무관한** 정책 천장이다 -- 모델 상한을 빼고
            # 원장·순자산만 본다. 모델 상한을 넣으면 보유시간 선택마다 설정이 움직인다
            # (2026-09-13 라이브에서 실제로 그랬다: 설정이 4.49 를 따라 8배로 내려앉았다).
            # ⭐이 값이 **지평 선택의 순환을 끊는다**: 지평에는 상한이, 상한에는 지평이 필요한데
            #   정책 천장은 지평과 무관하므로 먼저 정해진다(planning_hold 주석 참조).
            policy_only = [v for v in (cap_ledger, cap_equity) if v]
            if SIZING_CAP_MODEL_ONLY and equity > 0:      # effective_cap 과 같은 규칙
                policy_only = [equity * HARD_CAP_X]
            policy_cap_x = (min(policy_only) / equity) if policy_only and equity > 0 else None
            # 🔴상한을 계산할 지평 = 처방이 고를 지평(2026-09-13 감사). 같은 방향 포지션이
            # 있으면 그 포지션의 **남은 시간**이 이긴다 -- 추가한다고 시계가 새로 생기지 않는다
            # (2026-09-13 사용자 질문: "물타기하면 시간이 늘어나나?").
            # 🔴그 시계의 **길이**도 처방 지평이다(2026-09-14). 240분 상수를 쓰면 24시간
            # 처방으로 들어간 포지션의 추가분이 4시간 예산으로 계산된다.
            plan_hold = planning_hold(sizing, side, policy_cap_x)
            hold_min = (remaining_hold(same[0].get("entry_at"), fixed=plan_hold)
                        if same else plan_hold)
            # 2026-09-13 세 번째 상한: 보유시간 조건부 위험 모델. 있으면 같이 경쟁시킨다.
            risk = risk_sizing(sizing, hold_min, side)
            cap_model = None
            if risk.get("available") and equity > 0:
                e = entry_notional(equity, risk["safe_mae_pct"], existing_notional=exposure)
                cap_model = e["total_notional"]
                risk.update(leverage=round(e["leverage"], 2), binding=e["binding"],
                            survival_x=round(e["survival_x"], 2),
                            growth_x=round(e["growth_x"], 2) if e["growth_x"] else None)
            binding = [(v, k) for v, k in ((cap_ledger, "ledger"), (cap_equity, "equity"),
                                           (cap_model, "model")) if v]
            if SIZING_CAP_MODEL_ONLY and cap_model:
                binding = [(cap_model, "model")]
            if asset != "eth" or not SIZING_RISK_MODEL_ENABLED:
                binding = []          # 단순 규칙(사용자 지시): 원장·순자산 6배·모델 상한 없음 -- 증거금 50% 만
            # 증거금 상한. 레버리지는 주문이 실제로 걸 값 -- «자동»이면 처방(plan_now→prescribe)과
            #   **같은 입력**으로 같은 함수를 부른다(그래야 target_leverage 와 같은 값이다).
            #   화면은 «자동»이어도 10배를 보낸다(app.js MANUAL_LEV_AUTO, 2026-10-13 20→10) -- 이 경로는 lev 없는 요청뿐.
            margin_lev = want_lev or (leverage_setting(
                cap_notional=equity * policy_cap_x, equity=equity, existing_notional=exposure,
                current_notional=exposure).get("setting") if policy_cap_x else None) or leverage
            if SIZING_MARGIN_CAP_PCT > 0 and equity > 0 and margin_lev:
                binding.append((equity * margin_lev * SIZING_MARGIN_CAP_PCT / 100.0, "margin"))
            # 2026-10-20 사용자 «규칙 상한 공유»: 자동 규칙 손절이 무장된 포지션이 있으면 수동 진입(어느 코인이든)도 계좌 명목 ≤ 순자산 × 규칙 상한.
            #   교차 마진이라 규칙의 «손절보다 청산이 멀다» 보장은 계좌 전체 명목에 걸린다 -- 수동으로 다른 코인을 크게 넣으면 그 보장이 깨진다(10-20 감사).
            #   ponytail: 무장 파일엔 «보유 1.5×» 인지가 안 남아 늘 COMMIT_CAP_X(10-25 부터 2) -- 1.5 로 묶으려면 arm_bracket 이 rule_l 을 남기면 된다.
            if equity > 0 and any(v.get("auto", v.get("rule")) and v.get("sl_price") for v in bracket_load().values()):   # 10-24 전 무장 기록엔 auto 가 없다 -- 그땐 rule 이 곧 자동
                binding.append((equity * COMMIT_CAP_X, "rule_shared"))
            cap_notional = min(v for v, _ in binding) if binding else None
            # 🔴상한 무시 스위치. 위 SIZING_CAP_OVERRIDE_X 주석이 계약이다.
            overridden = SIZING_CAP_OVERRIDE_X > 0 and equity > 0
            if overridden:
                cap_notional = equity * SIZING_CAP_OVERRIDE_X
            # 화면이 «무엇이 묶었나»를 말할 때 쓰는 값. policy_leverage 의 binding 은
            # 순자산·원장 상한을 **모르므로** 그대로 보여주면 «정책상한 25배»라고 거짓말한다.
            # 🔴키로 **값만** 비교한다 -- 튜플 min 은 값이 같을 때 이름 알파벳순으로 갈린다.
            #   (아래 cap.update 도 같은 값을 쓴다. 옛 판은 여기만 고치고 그쪽은 `min(binding)`
            #    이라 동점에서 두 줄이 서로 다른 이름을 말할 수 있었다.)
            who = ("override" if overridden
                   else min(binding, key=lambda t: t[0])[1] if binding else None)
            if risk.get("available") and equity > 0 and cap_notional:
                risk["effective_x"] = round(cap_notional / equity, 2)
                risk["applied_binding"] = who
            if cap_notional is not None:
                cap.update(available=True, cap_notional_usdt=round(cap_notional, 2),
                           cap_equity_usdt=round(cap_equity, 2) if cap_equity else None,
                           cap_ledger_usdt=round(cap_ledger, 2) if cap_ledger else None,
                           cap_model_usdt=round(cap_model, 2) if cap_model else None,
                           margin_cap_pct=SIZING_MARGIN_CAP_PCT or None,
                           risk=risk,
                           equity_x=SIZING_CAP_EQUITY_X,
                           liq_floor_pct=round(100.0 / SIZING_CAP_EQUITY_X, 1),
                           # 🔴꺼져 있다는 사실을 payload 에 **명시**한다. 조용히 흘려보내면
                           #   화면이 «원장 상한이 묶었다»고 거짓말한다.
                           override_x=SIZING_CAP_OVERRIDE_X if overridden else None,
                           binding=who)
            # ⭐권고 수량의 기준점이 «습관 중앙값»에서 «위험»으로 바뀌었다(2026-09-13).
            # 모델이 있으면 순자산 × 허용배수 ÷ 가격, 없으면 옛 1/변동성 경로.
            price_ref = float(book["askPrice"]) if side == "LONG" else float(book["bidPrice"])
            rec_qty = float(sizing.get("vol_equivalent_qty") or 0.0)
            rec_src = "vol_equivalent"
            if risk.get("available") and equity > 0 and price_ref > 0:
                # 🔴E|r| 배수(`× evr_q`)를 여기 얹었다가 **켜기 전에 철회했다**(2026-09-15 당일).
                #   5주 원장 순차 검정에서 증분이 앞뒤로 갈렸다 -- 전반 36건 +11.01bp/건,
                #   후반 36건 **+0.73**(Δ>0 47.2% = 동전). 상위 10건을 빼면 합계 부호가
                #   뒤집힌다(+422.5 → −89.7bp). 두 CI 가 0 을 배제하지만 그 유의성을 만드는 건
                #   전반부다. 원장이 ~100왕복 될 때 다시 본다. 상세 docs/homer §5.36-S.
                rec_qty = (entry_notional(equity, risk["safe_mae_pct"])
                           ["total_notional"] / price_ref)
                rec_src = "risk_model"
            # 2026-10-09 권고 규칙(사용자 «방향만 정하면 크기·레버리지·손절 자동»): 상한 = L × 순자산(계좌 전체 명목),
            #   첫 진입 = 그 75% · 물타기 = 남은 여유 -- 기존 상한(증거금 50% 등)과 작은 쪽. 손절은 아래 bracket 에서 ∓5%.
            # 2026-10-09 자동 규칙(사용자 «인용하신 재생(×15.2)으로 넣어줘») → 10-10 L = min(0.075 ÷ 손절폭(동일위험), 청산 안전, 10) · 손절 3σ24 → 10-11 재생 식 복원 L = min(12.4 × 권장배수, 청산 안전, 10) · 손절 ∓3% ·
            #   첫 진입 1/3(10-14, 옛 75%) · 물타기 = 남은 여유. 상한은 이 규칙 하나 -- 증거금 50% 상한은 자동 모드에서 뺀다(20배면 같은 10배라 겹칠 뿐).
            # 🔴2026-10-24 사용자 «주문 규칙 정리 · 손절은 −5% 하나로»: 모든 진입(자동·보유 1.5×·직접·스위칭)에 같은 손절 = 첫 진입가 ∓5% 거래소 스탑 · 익절 없음.
            #   옛 직접 모드의 청산맵 지지2 손절·저항3 익절·«SL/TP» 끄기는 지웠다(청산맵 손절은 10-11 검정에서 같은 폭 무작위와 동률).
            #   물타기면 이미 무장된 규칙 손절을 그대로(물러나지 않는다). 손절 없이 연 포지션(바이낸스 앱 등)에 처음 거는 거면 기준 = 평단.
            prev = bracket_load().get(bracket_key(symbol, side)) if same else None
            if prev and prev.get("lines") and not lines:   # 2026-10-28 검증: 일반 진입이 선 익절을 지우고 −5% 거래소 스탑을 건다(사용자 «비상 스탑 없음»)
                return None, {}, {}, ({"error": "lines_armed",
                                       "detail": "선 주문 포지션입니다 — 같은 쪽 추가 진입은 막습니다(선 익절·손절이 지워지고 거래소 스탑이 걸리게 됩니다). 청산 뒤 다시"}, 409)
            p_in = float(book["bidPrice"] if side == "LONG" else book["askPrice"])   # 이번 주문 지정가(메이커 쪽)
            anchor, kind = ((float(same[0].get("entry_price") or 0.0), "평단") if same else (p_in, "첫 진입가"))
            stop_bracket = rec_rule_bracket({"position_side": side}, position_side=side, anchor_price=anchor or p_in,
                                            basis=await basis_for(book, asset), stop_pct=RULE_STOP_PCT, filters=filters, prev=prev, anchor_kind=kind)
            rule = cr = None
            if rule_l:
                if asset not in RULE_ASSETS:
                    return None, {}, {}, ({"error": "rule_asset_unsupported", "detail": f"자동 규칙은 {'·'.join(sorted(RULE_ASSETS)).upper()} 만 켭니다"}, 400)
                vm = (situation_by.get(asset) or {}).get("vol_mult") or {}   # 그 코인의 권장 배수(ETH 값을 쓰지 않는다)
                if not (vm.get("sigma_bp") and vm.get("mult")) or time.time() - float(vm.get("bar") or 0) > 15 * 60:
                    return None, {}, {}, ({"error": "rule_vol_unavailable",
                                           "detail": "권장 배수(직전 24시간 변동성)가 없거나 15분 넘게 묵었습니다 -- 자동 진입을 막습니다"}, 503)
                mt = await margin_type(binance_session(), symbol)   # 교차 전제 -- 격리면 손절 전에 청산된다(10-09 실사고)
                if mt != "cross":
                    return None, {}, {}, ({"error": "rule_not_cross",
                                           "detail": ("격리 마진이라 자동 진입을 막습니다 -- 손절보다 청산이 먼저 옵니다. 포지션·주문을 정리하고 바이낸스에서 교차로 바꾸세요"
                                                      if mt == "isolated" else "마진 방식을 못 읽어 자동 진입을 막습니다(교차 확인 필요)")}, 409)
                cr = commit_rule_l(position_side=side, sigma24_bp=float(vm["sigma_bp"]) * math.sqrt(6),   # 2026-10-11 재생 ×15.2 식 복원
                                   vol_mult=float(vm["mult"]), cap=HOLD_CAP_X if rule_l == "h" else COMMIT_CAP_X)   # 10-16 «보유 1.5×» = rule=h
                sl = float(stop_bracket["sl_price"])   # 크기(손절 시 손실 한도)가 손절선을 알아야 한다
                sgn = 1.0 if side == "LONG" else -1.0
                q_same = sum(abs(float(p.get("qty") or 0.0)) for p in same)
                rule = rec_rule_size(rule_l=cr["l"], equity=equity, existing_notional=exposure,
                                     order_leverage=float(margin_lev or 0.0), first=not same,   # 첫 진입 = 이 방향 포지션 없음
                                     risk_used=q_same * max(0.0, sgn * (anchor - sl)) if same else 0.0,   # 이미 든 몫이 손절에서 잃을 돈
                                     add_stop_frac=sgn * (p_in - sl) / p_in)
                if rule.get("l_eff", cr["l"]) < cr["l"]:   # 2026-10-13 레버리지가 L 을 묶음(증거금 90%) -- 화면·손절 손실 한도가 실제 배수를 말하게
                    cr = {**cr, "l": rule["l_eff"], "binding": "레버리지 증거금"}
                cap_notional, who = rule["cap_notional"], "rule"
                fraction = rule["fraction"]
            plan = build_entry_plan(
                side=side, best_bid=float(book["bidPrice"]), best_ask=float(book["askPrice"]),
                recommended_qty=rec_qty,
                cap_notional=cap_notional,
                filters=filters, symbol=symbol, existing_notional=exposure,
                equity=equity, leverage=leverage, fraction=fraction,
                # 2026-09-25 사용자 지시: 비율 = 순자산 대비 증거금, 상한 넘으면 진입 불가
                fraction_of_equity=True, order_leverage=margin_lev or 0.0,
                cap_label={"margin": f"증거금 상한 {SIZING_MARGIN_CAP_PCT:g}%",
                           "model": "위험모델 상한",
                           "rule": f"자동 규칙 상한(순자산 {cr['l'] if cr else 0:.2f}배)",
                           "rule_shared": f"자동 규칙 포지션이 있어 계좌 상한(순자산 {COMMIT_CAP_X:g}배)"}.get(who, "상한"))
            plan["recommended_source"] = rec_src
            plan["recommended_qty"] = round(rec_qty, 8)
            plan["projection"] = entry_projection(plan, account, positions, existing, equity)
            # 2026-09-13 «지금 상황» 플랜: 보유시간 권고·집행·분할·예산 사다리. 상한은 적용된 실효 배수.
            plan["trade_plan"] = plan_now(
                side=side, equity=equity, existing_notional=exposure,
                unrealized_pnl=same_unrealized,
                risk_table=sizing.get("risk_mae") or {},
                vol_bpm=await realized_vol_now(symbol),
                cap_x=(cap_notional / equity) if cap_notional and equity > 0 else SIZING_CAP_EQUITY_X,
                atr_pct=sizing.get("atr_pct"), hold_min=hold_min,
                policy_cap_x=policy_cap_x,
                funding_bp_8h=await funding_now(symbol),
                fees=await commission_bp(binance_session(), symbol))   # 표시용 예상 비용만 -- 크기·손절은 표준 상수
            # 집행기는 계획 dict 하나만 받는다. 처방 깊숙이 손을 넣게 하지 않고 여기서 꺼내 준다.
            _rx = (plan["trade_plan"] or {}).get("prescription") or {}
            _lv = (_rx.get("exchange_leverage") or {}) if _rx.get("available") else {}
            plan["bracket"] = stop_bracket
            if lines:   # 2026-10-28 차트 선(익절, 손절 · USDT): 거래소 스탑 없이 익절 = post-only 지정가 · 손절 = 1초 체결가 닿음 → peg 청산
                if same:
                    return None, {}, {}, ({"error": "lines_has_position",
                                           "detail": "같은 쪽 포지션이 이미 있어 선 진입을 막습니다 — 그 포지션의 손절을 바꾸지 않게(닫은 뒤 다시)"}, 409)
                if sec_prices(asset) is None:
                    return None, {}, {}, ({"error": "lines_no_feed", "detail": "이 코인은 1초 체결가가 없어 선 손절을 감시할 수 없습니다"}, 400)
                b_ = await basis_for(book, asset)
                try:
                    plan["bracket"] = line_bracket(position_side=side, ref_level=p_in / b_, tp_level=lines[0], sl_level=lines[1],
                                                   basis=b_, filters=filters)
                except ValueError as exc:
                    return None, {}, {}, ({"error": "bad_lines", "detail": str(exc)}, 400)
            if rule:
                plan["rule"] = {"l": round(cr["l"], 3), "binding": cr["binding"], "stop_pct": round(100 * cr["stop_pct"], 2),
                                "vol_mult": round(float(vm["mult"]), 3), "cap_notional": round(rule["cap_notional"], 2), "first": rule["first"],
                                "room": round(rule["room"], 2), "target_notional": round(rule["target_notional"], 2),
                                "loss_at_stop_pct": round(100 * cr["l"] * cr["stop_pct"], 2),   # 손절 시 손실 한도 = L × 3%
                                "sl_price": plan["bracket"].get("sl_price"), "sl_name": plan["bracket"].get("sl_name"),
                                # 2026-10-14 자동 물타기 지정가(첫 진입일 때만) -- 체결 뒤 arm_rule_add 가 실제 체결 수량으로 다시 계산해 건다
                                "lev": float(margin_lev or 0.0),
                                "filters": {k: float(filters.get(k) or 0.0) for k in ("tick", "step", "min_qty", "min_notional")}}
                if rule["first"] and REC_AUTO_ADD and plan.get("quantity") and plan["bracket"].get("sl_price") and not lines:
                    plan["rule"]["add"] = rec_rule_add_order(
                        position_side=side, sl_price=float(plan["bracket"]["sl_price"]), stop_pct=cr["stop_pct"], l_eff=float(cr["l"]),
                        cap_notional=rule["cap_notional"], room_before=rule["room"], filled_qty=float(plan["quantity"]),
                        order_leverage=float(margin_lev or 0.0), filters=plan["rule"]["filters"])
            # 게이지가 값을 주면 그걸 쓰고, «자동»이면 모델 추천을 쓴다. 어느 쪽인지 남긴다 --
            # 안 남기면 나중에 «왜 30배로 걸렸지»를 못 푼다.
            plan["target_leverage"] = want_lev or _lv.get("setting")
            plan["leverage_source"] = "manual" if want_lev else "model"
            plan["leverage_model"] = _lv.get("setting")
            # «상한만큼 열려면 최소 몇 배»는 **적용된** 상한으로 잰다(2026-09-25). 정책 천장(25배)으로
            #   재면 증거금 50% 가 이미 10배로 묶었는데도 «최소 25배» 경고가 떴다.
            plan["leverage_min_feasible"] = (round(cap_notional / equity, 2)
                                             if cap_notional and equity > 0
                                             else _lv.get("min_feasible"))
            # 열린 포지션이 만드는 바닥. 이 아래를 고르면 거래소가 -2028 로 거부한다.
            plan["leverage_position_floor"] = _lv.get("position_floor")
            # 🔴화면이 띄울 **계획 지평**. 상수가 아니라 크기를 실제로 정한 그 값이다
            # (2026-09-14). 상수를 보내면 화면엔 4시간인데 1440분 셀로 크기가 나간다.
            plan["hold_planned_min"] = plan_hold
            plan["hold_remaining_min"] = hold_min
            plan["leverage_steps"] = list(LEVERAGE_STEPS)
        except Exception as exc:  # noqa: BLE001 -- 여기서 터져도 주문은 아직 안 나갔다
            return None, {}, {}, ({"error": f"{type(exc).__name__}: {exc}"}, 502)
        return plan, cap, sizing, None

    async def api_manual_entry_preview(request: web.Request) -> web.Response:
        """«이 버튼을 누르면 나갈 주문». 1단계에서는 게이트가 닫혀 있어 **보내지 않는다**.

        미리보기가 실주문과 **같은 함수**(build_entry_plan)를 통과한다 -- 다른 경로로 만든
        미리보기는 실주문을 검증하지 못한다(대조군이 안 덮는 경로는 검증 안 된 것)."""
        side = (request.query.get("side") or "").upper()
        if side not in ("LONG", "SHORT"):
            return web.json_response({"ok": False, "error": "side must be LONG or SHORT"}, status=400)
        if query_asset(request) is None:
            return web.json_response({"ok": False, "error": "bad_asset",
                                      "detail": "주문할 수 없는 코인입니다"}, status=400)
        frac = query_fraction(request)
        if frac is None:
            return web.json_response({"ok": False, "error": "bad_pct",
                                      "detail": "진입 비율은 0 초과 100 이하여야 합니다"}, status=400)
        lines = query_lines(request)
        if lines == "bad":
            return web.json_response({"ok": False, "error": "bad_lines", "detail": "익절·손절 선(tp·sl)은 둘 다 양수여야 합니다"}, status=400)
        rule_l = query_rule(request)
        if rule_l == "bad":
            return web.json_response({"ok": False, "error": "bad_rule",
                                      "detail": "자동 규칙이 꺼져 있습니다(수동 모드)" if not MANUAL_RULES_ENABLED else "자동 규칙 값은 c 하나뿐입니다"}, status=400)
        plan, cap, sizing, error = await assemble_entry_plan(side, frac,
                                                             query_leverage(request), query_asset(request),
                                                             fresh=request.query.get("fresh") == "1", rule_l=rule_l,
                                                             lines=lines)
        if error:
            return web.json_response({"ok": False, **error[0]}, status=error[1])
        return web.json_response({"ok": True, "plan": plan, "cap": cap,
                                  "recommended_qty": plan.get("recommended_qty"),
                                  "recommended_source": plan.get("recommended_source"),
                                  "exec_enabled": exec_enabled()},
                                 headers=NOCACHE)

    async def api_manual_entry_submit(request: web.Request) -> web.Response:
        """실주문. **POST 전용 + confirm=1 필수 + 게이트가 켜져 있어야** 나간다.

        POST 인 이유: GET 이면 링크 미리보기·프리페치·주소창 재방문이 그대로 주문이 된다.
        confirm 을 받는 이유: 프런트의 2단 확인을 서버에서 한 번 더 건다 -- 프런트만 믿으면
        프런트 버그가 곧 오발주다."""
        side = (request.query.get("side") or "").upper()
        if side not in ("LONG", "SHORT"):
            return web.json_response({"ok": False, "error": "side must be LONG or SHORT"}, status=400)
        if request.query.get("confirm") != "1":
            return web.json_response({"ok": False, "error": "confirm=1 required"}, status=400)
        if query_asset(request) is None:
            return web.json_response({"ok": False, "error": "bad_asset",
                                      "detail": "주문할 수 없는 코인입니다"}, status=400)
        frac = query_fraction(request)
        if frac is None:
            return web.json_response({"ok": False, "error": "bad_pct",
                                      "detail": "진입 비율은 0 초과 100 이하여야 합니다"}, status=400)
        # 2026-10-10 실행 «기다리기»(mode=wait&wait_min=15|60): 앞질러도 안 따라가고 마감 뒤 잔량 시장가. 없으면 현행 «따라가기».
        wait_min = None
        if request.query.get("mode") == "wait":
            try:
                wait_min = int(request.query.get("wait_min") or 0)
            except ValueError:
                wait_min = 0
            if wait_min not in WAIT_CHOICES_MIN:
                return web.json_response({"ok": False, "error": "bad_wait",
                                          "detail": f"기다리기는 {'·'.join(map(str, WAIT_CHOICES_MIN))}분만 됩니다"}, status=400)
        if not exec_enabled():
            return web.json_response({"ok": False, "error": "exec_disabled",
                                      "detail": "DASHBOARD_MANUAL_EXEC_ENABLED 가 꺼져 있습니다"},
                                     status=403)
        if _busy(manual_entry_state) or _busy(side_exit_states.get(query_asset(request))):
            return web.json_response({"ok": False, "error": "already_working",
                                      "state": manual_entry_state}, status=409)
        if wait_pending():   # 대기 몫은 아직 명목에 안 잡힌다 -- 그 위에 또 넣으면 상한이 두 번 쓰인다
            return web.json_response({"ok": False, "error": "wait_pending",
                                      "detail": "기다리는 진입 주문이 있습니다 — 취소하거나 체결된 뒤에 넣으세요(청산은 됩니다)",
                                      "wait": manual_wait_state}, status=409)
        # 비율은 **여기서 다시** 적용한다 -- 기존 포지션도 다시 읽으므로, 앞 칸이 이미
        # 들어가 있으면 상한 여유가 그만큼 줄어든 상태에서 계산된다.
        lines = query_lines(request)
        if lines == "bad":
            return web.json_response({"ok": False, "error": "bad_lines", "detail": "익절·손절 선(tp·sl)은 둘 다 양수여야 합니다"}, status=400)
        rule_l = query_rule(request)
        if rule_l == "bad":
            return web.json_response({"ok": False, "error": "bad_rule",
                                      "detail": "자동 규칙이 꺼져 있습니다(수동 모드)" if not MANUAL_RULES_ENABLED else "자동 규칙 값은 c 하나뿐입니다"}, status=400)
        plan, cap, sizing, error = await assemble_entry_plan(side, frac,
                                                             query_leverage(request), query_asset(request),
                                                             fresh=request.query.get("fresh") == "1", rule_l=rule_l,
                                                             lines=lines)
        if error:
            return web.json_response({"ok": False, **error[0]}, status=error[1])
        if plan.get("blocked"):
            return web.json_response({"ok": False, "error": "blocked", "detail": plan["blocked"]},
                                     status=400)
        if wait_min:
            manual_wait_state.clear()
            manual_wait_state.update(phase="submitting", mode="wait", wait_min=wait_min, side=side, plan=plan,
                                     asset=query_asset(request), started_at=datetime.now(timezone.utc).isoformat())
            refresh_tasks["manual_wait"] = asyncio.create_task(
                entry_then_bracket(plan, manual_wait_state, wait_sec=wait_min * 60.0))
            return web.json_response({"ok": True, "plan": plan, "state": manual_wait_state, "wait": True},
                                     headers=NOCACHE)
        manual_entry_state.clear()
        manual_entry_state.update(phase="submitting", side=side, plan=plan, asset=query_asset(request),
                                  started_at=datetime.now(timezone.utc).isoformat())
        # refresh_tasks 에 넣어 두면 stop_http_session 이 세션을 닫기 전에 취소해 준다.
        refresh_tasks["manual_entry"] = asyncio.create_task(
            entry_then_bracket(plan, manual_entry_state))
        return web.json_response({"ok": True, "plan": plan, "state": manual_entry_state},
                                 headers=NOCACHE)

    HOLD_CHOICES = (60, 120, 240, 480, 1440)
    # 🔴2026-09-13 사용자 결정: 보유시간을 **4시간 고정**. 화면 선택지를 없앤다.
    # 근거: 생존 상한이 보유시간에 가파르게 반응한다(1일 4.30배 -> 4시간 8배 이상).
    # «지렛대는 크기가 아니라 보유시간»(09-13 반사실)의 직접 적용이다.
    async def funding_now(symbol: str) -> float | None:
        """현재 펀딩률(bp/8h). 🔴비용 모델에 **없던 항목**이다(2026-09-13 감사) -- 8시간마다
        정산되므로 보유시간에 비례하고 롱/숏 부호가 반대다. 1440분 보유면 왕복비용의 28%다.
        못 읽으면 None -- 그때는 최근 30일 중앙값 폴백을 쓴다(0 으로 두면 긴 지평이 공짜가 된다)."""
        try:
            r = await fetch_binance_json("https://fapi.binance.com/fapi/v1/premiumIndex",
                                         {"symbol": symbol}, timeout=5.0, error_reason=None)
            if r and r.get("lastFundingRate") is not None:
                return float(r["lastFundingRate"]) * 1e4
        except Exception:  # noqa: BLE001 -- 있으면 좋은 값이지 필수가 아니다
            pass
        return None

    # 🔴지평의 단일 출처는 `planning_hold()` 다(2026-09-14 사용자 결정). 이 상수는 그 함수가
    # 고르지 못할 때(워커 없음·표 없음)의 **폴백**일 뿐, 정책 지평이 아니다 -- 상수로 쓰면
    # 화면엔 4시간인데 크기는 1440분 셀로 나간다(실측 허용 배수 6.0 vs 4.3배).
    HOLD_FIXED_MIN = 240

    def remaining_hold(entry_at: str | None, fixed: int = HOLD_FIXED_MIN) -> int:
        """**남은** 보유시간을 모델 지평에 맞춰 올림한다.

        🔴물타기를 해도 시계는 **안 늘어난다**. 약속은 포지션에 걸린 것이지 칸마다 새로 생기지
        않는다. `entry_at` 은 첫 체결 시각이라(positionRisk 의 updateTime 과 달리 물타기에
        안 움직인다) 그대로 쓸 수 있다.
        ⚠️올림은 **보수적**이다 -- 남은 90분이면 120분 표를 본다(더 큰 역행폭 = 더 작은 크기).
        """
        if not entry_at:
            return fixed
        try:
            age = (datetime.now(timezone.utc)
                   - datetime.fromisoformat(entry_at)).total_seconds() / 60.0
        except (TypeError, ValueError):
            return fixed
        left = fixed - max(0.0, age)
        if left <= 0:                      # 약속한 시간을 넘겼다 -- 가장 짧은 표를 쓴다
            return min(HOLD_CHOICES)
        return next((h for h in sorted(HOLD_CHOICES) if h >= left), fixed)

    def planning_hold(sizing: dict[str, Any], side: str, policy_cap_x: float | None) -> int:
        """**상한을 계산할 지평**. 처방이 고를 지평과 같아야 한다.

        🔴예전에는 상한을 240분 셀로 계산하면서 화면은 1440분을 권고했다(2026-09-13 감사).
        1440분의 허용 배수는 240분의 3분의 1 수준이라, 권고대로 들면 진입 직후 예산 사다리를
        35% 위반했다. 여기서 한 번 고르고 그 값을 상한·처방·손절 표시가 **공유**한다.

        지평 선택에는 상한이 필요하고 상한에는 지평이 필요한 순환이 있다 -- **지평과 무관한**
        정책 천장(원장 ∧ 순자산)으로 끊는다. 위험모델 상한은 그 뒤에 min 으로 들어간다."""
        table = (sizing or {}).get("risk_mae") or {}
        atr = (sizing or {}).get("atr_pct")
        if not table or not atr:
            return HOLD_FIXED_MIN
        h = recommend_hold(table, side, policy_cap_x or SIZING_CAP_EQUITY_X, atr,
                           acc=PRESCRIBE_ACC)
        return int(h["recommended_min"]) if h.get("available") else HOLD_FIXED_MIN

    def risk_sizing(sizing: dict[str, Any], hold_min: int, side: str) -> dict[str, Any]:
        """이 보유시간에서 각오할 역행폭. **사이징 워커가 300초마다 계산해 둔 값을 읽는다.**

        🔴요청 경로에서 모델을 돌리지 않는다 -- 대시보드는 60초마다 이 미리보기를 폴링하므로
        그러면 to_thread 풀이 고갈된다(2026-09-10 실장애, 워커가 애초에 존재하는 이유).
        워커 상태가 낡았으면 available=False 로 내려가고 호출부는 기존 상한만 쓴다."""
        cell = ((sizing.get("risk_mae") or {}).get(str(hold_min)) or {}).get(side.upper())
        if not cell or not (cell.get("safe_mae_pct", 0) > 0):
            return {"available": False, "hold_min": hold_min,
                    "reason": "worker_no_risk" if sizing.get("available") else "sizing_unavailable"}
        return {"available": True, "hold_min": hold_min,
                "safe_mae_pct": cell["safe_mae_pct"]}

    def query_asset(request: web.Request) -> str | None:
        """주문 코인(`?asset=`, 기본 eth). 주문 심볼 표에 없는 코인이면 None -- 호출부가 400 으로 막는다."""
        a = str(request.query.get("asset") or "eth").lower()
        return a if a in MANUAL_EXEC_SYMBOLS else None

    def query_leverage(request: web.Request, key: str = "lev") -> int | None:
        """화면 게이지가 고른 거래소 레버리지. 없으면 None -- 그때는 **모델 추천**을 쓴다.

        범위를 벗어나면 조용히 자르지 않고 None 을 돌려 모델값으로 떨어뜨린다.
        (자르면 «10 을 눌렀는데 1 이 걸림» 같은 일이 생긴다.)"""
        raw = request.query.get(key)
        if raw in (None, ""):
            return None
        try:
            v = int(float(raw))
        except (TypeError, ValueError):
            return None
        return v if 1 <= v <= EXCHANGE_MAX_LEVERAGE else None

    def query_rule(request: web.Request, key: str = "rule") -> str | None:
        """자동 규칙 = "c"(재생 크기식, 상한 3×) · "h"(같은 식, 상한 1.5× -- 10-16 추세 보유형). 없으면 None(직접 모드). 그 밖이면 "bad" -- 호출부가 400(직접 모드로 조용히 떨어지지 않는다)."""
        raw = request.query.get(key)
        if raw in (None, ""):
            return None
        return raw if raw in ("c", "h") and MANUAL_RULES_ENABLED else "bad"   # 꺼져 있으면 400(직접 모드로 조용히 떨어지지 않는다)

    def query_lines(request: web.Request):
        """2026-10-28 차트 선 = (익절, 손절) USDT. 둘 다 없으면 None · 하나만이거나 이상하면 "bad"(호출부 400)."""
        tp, sl = request.query.get("tp"), request.query.get("sl")
        if tp in (None, "") and sl in (None, ""):
            return None
        try:
            t, s_ = float(tp), float(sl)
        except (TypeError, ValueError):
            return "bad"
        return (t, s_) if t > 0 and s_ > 0 and math.isfinite(t) and math.isfinite(s_) else "bad"

    def query_fraction(request: web.Request, key: str = "pct") -> float | None:
        """쿼리의 비율(%)을 0<f<=1 로 바꾼다. 진입 분할과 부분 청산이 **같은 함수**를 쓴다.
        이상하면 None -- 호출부가 400 을 낸다.
        **조용히 1.0 으로 떨어뜨리지 않는다**: 일부만 하려던 요청이 전량이 되면 안 된다."""
        raw = request.query.get(key)
        if raw in (None, ""):
            return 1.0
        try:
            pct = float(raw)
        except (TypeError, ValueError):
            return None
        return pct / 100.0 if 0.0 < pct <= 100.0 else None

    async def assemble_exit_plan(position_side: str, fraction: float = 1.0,
                                 fresh: bool = False, asset: str = "eth", market: bool = False):
        """청산 계획 조립. 진입과 같은 이유로 **여기 한 곳뿐**이다.

        수량은 반드시 **방금 읽은 포지션**에서 온다 -- 헤지 모드라 reduceOnly 를 못 써서
        (-1106) 과청산을 막는 게 수량밖에 없다. 캐시된 값을 쓰면 이미 닫힌 포지션을
        다시 닫으려다 반대 방향으로 열릴 수 있다.

        🔴2026-09-19 심볼도 **설정이 아니라 실제 포지션**에서 온다. 진입 심볼을 USDC 로
        바꿔도 그 전에 연 ETHUSDT 포지션은 계속 닫을 수 있어야 하고, 반대로 설정값을 믿고
        보내면 «없는 포지션을 닫는» 주문이 헤지 모드에서 **반대 방향 신규 진입**이 된다."""
        candidates = list(dict.fromkeys([MANUAL_EXEC_SYMBOLS[asset], MARKET_SYMBOLS[asset]]))   # 그 코인만(09-26)
        try:
            # 🔴실주문(fresh=True)은 30초 캐시를 **우회**한다. 헤지 모드라 reduceOnly 가 없어
            # 과청산 방어가 수량뿐인데, 그 수량이 30초 묵으면 방어가 30초 묵는다.
            account = (await produce_account() if fresh else
                       await swr_cached("binance_account", BINANCE_ACCOUNT_CACHE_SECONDS,
                                        produce_account, max_stale=STALE_GRACE_SECONDS,
                                        cache=binance_account_cache))
            # 결정 규칙과 자체점검은 live_manual_peg_entry 모듈에 있다(85/85).
            if not account.get("ok"):
                # 🔴«못 읽음»을 no_position 으로 돌려주면 브래킷 감시가 정리(=익절·비상 스탑 삭제)로 간다.
                return None, ({"error": "account_unavailable",
                               "detail": f"계좌 조회 실패: {account.get('error')}"}, 503)
            position, symbol, leftover = resolve_exit_position(
                account.get("positions") or [], position_side, candidates)
            if position is None:
                return None, ({"error": "no_position",
                               "detail": f"{position_side} 포지션이 없습니다"}, 400)
            book = await fetch_binance_json("https://fapi.binance.com/fapi/v1/ticker/bookTicker",
                                            {"symbol": symbol}, error_reason="book_ticker_failed")
            filters = await load_filters(binance_session(), symbol)
            vol_bpm = await realized_vol_now(symbol)
            plan = build_exit_plan(
                position_side=position_side, position_qty=float(position.get("qty") or 0.0),
                best_bid=float(book["bidPrice"]), best_ask=float(book["askPrice"]),
                filters=filters, symbol=symbol,
                entry_price=float(position.get("entry_price") or 0.0),
                mark_price=float(position.get("mark_price") or 0.0),
                vol_bpm=vol_bpm, fraction=fraction)
            if market and plan["type"] != "MARKET":   # 2026-10-10 «즉시»(청산 칩·스위칭) -- run_exit 의 시장가 분기로 간다
                plan.pop("timeInForce", None)
                plan.pop("price", None)
                plan.update(type="MARKET", repeg=False,
                            market_reason="즉시 체결 — 지정가를 걸지 않고 바로 시장가로 닫습니다(테이커 수수료)")
            plan["unrealized_pnl"] = position.get("unrealized_pnl")
            if leftover:                      # 같은 방향이 다른 심볼에도 열려 있다
                plan["other_symbol_open"] = leftover
            # 남은 보유시간 기준 위험 한도. 넘었으면 «최소 이만큼은 닫아야 한다»를 준다.
            # 🔴위에서 읽은 **그 계좌**를 쓴다(2026-09-13 병합). 같은 함수 안에서 캐시를 또
            # 조회하면 fresh=True(실주문)일 때 수량은 새 값인데 순자산·명목은 30초 묵은 값이
            # 되어, 청산 한도가 실제와 어긋난 채로 «최소 몇 % 닫아라»가 나간다.
            eq = float((account.get("balance") or {}).get("margin") or 0.0)
            cur_notional = sum(abs(float(p.get("notional") or 0.0))
                               for p in (account.get("positions") or [])
                               if p.get("symbol") == symbol)
            sz = await asyncio.to_thread(position_sizing_payload)
            # 🔴지평을 고를 정책 천장은 **지평과 무관**해야 순환이 끊긴다(진입과 같은 구조).
            # safe_mae_pct=None 이면 effective_cap 이 모델 후보를 빼고 원장∧순자산만 본다.
            policy_n, _, _ = effective_cap(sz, eq, None)
            policy_cap_x = (policy_n / eq) if policy_n and eq > 0 else None
            # 🔴이미 든 시간만큼 깎은 **남은** 보유시간으로 잰다. 물타기로 칸이 늘어도 시계는
            # 그대로라, 늦게 추가할수록 남은 시간이 짧아 허용 배수가 커진다 -- 그건 «그 시각에
            # 실제로 닫는다»는 전제 위에서만 맞다. 화면이 남은 시간을 같이 띄운다.
            # 🔴시계의 **길이**는 진입이 처방한 지평이다(2026-09-14). 예전에는 240분 상수라
            # 24시간 처방으로 들어간 포지션을 4시간 예산으로 재고, 4시간이 지나면 60분 셀로
            # 떨어져 «여유가 더 생겼다»고 말했다 -- 지평이 짧을수록 허용 배수가 커지기 때문이다.
            plan_hold = planning_hold(sz, position_side, policy_cap_x)
            hold_min = remaining_hold(position.get("entry_at"), fixed=plan_hold)
            risk = risk_sizing(sz, hold_min, position_side)
            # 위험모델이 없으면 모델 후보가 빠진 정책 천장이 그대로 실효 상한이다.
            eff_x = policy_cap_x or SIZING_CAP_EQUITY_X
            if risk.get("available") and eq > 0 and cur_notional > 0:
                # 🔴진입과 **같은 상한**을 쓴다. 정책상한 25배로 재면 진입이 8배에서 막은
                # 포지션을 청산은 «닫을 필요 없음»이라고 말한다(같은 카드에 모순된 두 숫자).
                cap_n, who, _ = effective_cap(sz, eq, risk["safe_mae_pct"])
                eff_x = (cap_n / eq) if cap_n else HARD_CAP_X
                r = exit_fraction_required(eq, risk["safe_mae_pct"], cur_notional,
                                           hard_cap=eff_x)
                risk["effective_x"] = round(eff_x, 2)
                risk["applied_binding"] = who
                plan["risk"] = {**risk, "required_fraction": round(r["required_fraction"], 4),
                                "allowed_notional": round(r["allowed_notional"], 2),
                                "excess_notional": round(r["excess_notional"], 2),
                                "current_notional": round(cur_notional, 2),
                                "leverage": round(r["leverage"], 2), "binding": r["binding"]}
            else:
                plan["risk"] = risk
            plan["hold_planned_min"] = plan_hold
            plan["hold_remaining_min"] = hold_min
            # 🔴상한은 **위에서 이미 고른 실효 상한**이다(2026-09-14). 상수 6.0 을 쓰면 같은
            # 카드가 «상한 넘었으니 닫아라»(risk, effective_cap 기준)와 «15,000 더 넣을 여유
            # 있음»(trade_plan, 6.0배 기준)을 나란히 띄운다 -- effective_cap 이 막으려고 생긴
            # 바로 그 모순인데, 74441ad 감사가 risk 블록에만 적용되고 이 줄을 지나쳤다.
            plan["trade_plan"] = plan_now(
                side=position_side, equity=eq, existing_notional=cur_notional,
                unrealized_pnl=float(position.get("unrealized_pnl") or 0.0),
                risk_table=sz.get("risk_mae") or {}, vol_bpm=vol_bpm,
                cap_x=eff_x, atr_pct=sz.get("atr_pct"), hold_min=hold_min,
                policy_cap_x=policy_cap_x or SIZING_CAP_EQUITY_X,
                funding_bp_8h=await funding_now(symbol),
                fees=await commission_bp(binance_session(), symbol))   # 청산은 실제 포지션 심볼의 요율
        except Exception as exc:  # noqa: BLE001 -- 여기서 터져도 주문은 아직 안 나갔다
            return None, ({"error": f"{type(exc).__name__}: {exc}"}, 502)
        return plan, None

    async def api_manual_exit_preview(request: web.Request) -> web.Response:
        """«이 버튼을 누르면 나갈 청산 주문». 진입과 같은 함수를 지나므로 미리보기가
        실주문을 실제로 검증한다."""
        side = (request.query.get("side") or "").upper()
        if side not in ("LONG", "SHORT"):
            return web.json_response({"ok": False, "error": "side must be LONG or SHORT"}, status=400)
        if query_asset(request) is None:
            return web.json_response({"ok": False, "error": "bad_asset",
                                      "detail": "주문할 수 없는 코인입니다"}, status=400)
        frac = query_fraction(request)
        if frac is None:
            return web.json_response({"ok": False, "error": "bad_pct",
                                      "detail": "청산 비율은 0 초과 100 이하여야 합니다"}, status=400)
        # 🔴청산 미리보기도 **fresh** 다(2026-09-14, 사용자 요청 «청산 누르면 강제 조회부터»).
        # 사람이 버튼을 눌러야만 오는 경로라 호출이 잦지 않고, 30초 캐시로 그리면 화면이
        # 「2.754 닫는다」고 말한 뒤 submit(이미 fresh)이 다른 수량을 내보낼 수 있다.
        # 미리보기와 실주문이 **같은 수량을 보는 것**이 이 화면의 존재 이유다.
        ex = request.query.get("exec", "limit")   # 2026-10-10 청산 체결 방식: limit(현행 peg) · market(즉시)
        if ex not in ("limit", "market"):
            return web.json_response({"ok": False, "error": "bad_exec", "detail": "exec 는 limit 또는 market"}, status=400)
        plan, error = await assemble_exit_plan(side, frac, fresh=True, asset=query_asset(request), market=ex == "market")
        if error:
            return web.json_response({"ok": False, **error[0]}, status=error[1])
        return web.json_response({"ok": True, "plan": plan, "exec_enabled": exec_enabled()},
                                 headers=NOCACHE)

    async def api_manual_exit_submit(request: web.Request) -> web.Response:
        """실제 청산. 진입과 **같은 방어**를 건다: POST 전용 + confirm=1 + 게이트.
        상태 dict 도 진입과 공유한다 -- 수동 주문이 동시에 둘 나가지 않는 보호가 따라온다."""
        side = (request.query.get("side") or "").upper()
        if side not in ("LONG", "SHORT"):
            return web.json_response({"ok": False, "error": "side must be LONG or SHORT"}, status=400)
        if request.query.get("confirm") != "1":
            return web.json_response({"ok": False, "error": "confirm=1 required"}, status=400)
        if query_asset(request) is None:
            return web.json_response({"ok": False, "error": "bad_asset",
                                      "detail": "주문할 수 없는 코인입니다"}, status=400)
        frac = query_fraction(request)
        if frac is None:
            return web.json_response({"ok": False, "error": "bad_pct",
                                      "detail": "청산 비율은 0 초과 100 이하여야 합니다"}, status=400)
        # 2026-10-10 «스위칭» = 전량 청산 + 반대 진입을 **서버가 한 작업으로**(옛 판은 브라우저 3초 폴링이 이어 줘서 화면이 꺼지면
        #   반대 진입이 안 나갔다 -- 10-10 10:51 숏 2.563 청산 뒤 롱 미도착). 반대 진입 값 = 클라이언트가 보내던 그대로(sw_*).
        ex = request.query.get("exec", "limit")   # 2026-10-10 청산 체결 방식: limit(현행 peg) · market(즉시)
        if ex not in ("limit", "market"):
            return web.json_response({"ok": False, "error": "bad_exec", "detail": "exec 는 limit 또는 market"}, status=400)
        sw, to = None, (request.query.get("switch_to") or "").upper()
        if to:
            sw_frac, sw_rule = query_fraction(request, "sw_pct"), query_rule(request, "sw_rule")
            if to != ("SHORT" if side == "LONG" else "LONG") or frac != 1.0 or sw_frac is None or sw_rule == "bad":
                return web.json_response({"ok": False, "error": "bad_switch",
                                          "detail": "스위칭은 전량(100%) 청산 + 반대 방향 진입만 됩니다"}, status=400)
            sw = {"to": to, "frac": sw_frac, "lev": query_leverage(request, "sw_lev"), "rule": sw_rule}
        if not exec_enabled():
            return web.json_response({"ok": False, "error": "exec_disabled",
                                      "detail": "DASHBOARD_MANUAL_EXEC_ENABLED 가 꺼져 있습니다"},
                                     status=403)
        st = exit_slot(query_asset(request))
        if st is None or (sw and st is not manual_entry_state):   # 스위칭은 반대 진입이 따라오므로 위 칸만
            return web.json_response({"ok": False, "error": "already_working",
                                      "state": manual_entry_state}, status=409)
        if sw and wait_pending():
            return web.json_response({"ok": False, "error": "wait_pending",
                                      "detail": "기다리는 진입 주문이 있어 스위칭을 막습니다 — 먼저 취소하세요(그냥 청산은 됩니다)"}, status=409)
        # 비율은 **여기서 다시** 적용한다 -- 포지션도 다시 읽으므로 미리보기 이후에 포지션이
        # 줄었으면 그만큼 줄어든 수량이 나간다(프런트가 계산한 수량을 받지 않는 이유).
        # 2026-10-10 사용자 결정: 스위칭은 즉시 체결(청산·반대 진입 둘 다 시장가) -- 떨어지는 장에서 매도 지정가가 따라가기만 했다.
        plan, error = await assemble_exit_plan(side, frac, fresh=True, asset=query_asset(request),
                                               market=bool(sw) or ex == "market")
        if error:
            return web.json_response({"ok": False, **error[0]}, status=error[1])
        if plan.get("blocked"):
            return web.json_response({"ok": False, "error": "blocked", "detail": plan["blocked"]},
                                     status=400)
        if exit_slot(query_asset(request)) is not st:   # 계획 조립(await) 사이에 칸이 바뀌었으면(손절 청산이 잡음) 두 번 내지 않는다
            return web.json_response({"ok": False, "error": "already_working",
                                      "state": manual_entry_state}, status=409)
        st.clear()
        st.update(phase="submitting", kind="exit", side=side, plan=plan, asset=query_asset(request),
                  started_at=datetime.now(timezone.utc).isoformat())
        if sw:
            st["switch"] = {"from": side, "to": sw["to"], "phase": "exit"}
        refresh_tasks["manual_entry" if st is manual_entry_state else f"manual_exit_{query_asset(request)}"] = asyncio.create_task(
            exit_then_switch(plan, st, sw, query_asset(request)) if sw
            else settle(run_exit(binance_session(), plan, st), st))
        return web.json_response({"ok": True, "plan": plan, "state": st},
                                 headers=NOCACHE)

    async def api_manual_exit_all_submit(request: web.Request) -> web.Response:
        """2026-10-22 사용자 «코인 여러 개를 한꺼번에 청산 -- 지금은 다른 청산을 기다려야 한다».
        방금 읽은 포지션 전부를 **전량** 청산으로 한 번에 접수한다. 코인마다 청산 칸이 따로라(exit_slot) 앞 코인의
        지정가 청산(최대 ~2분)이 끝나기를 기다리지 않는다. 방어는 단건과 같다: POST + confirm=1 + 게이트 + busy_or.
        같은 코인에 롱·숏이 둘 다 있으면 칸이 하나라 두 번째는 already_working 으로 돌려준다(끝난 뒤 다시 누르면 된다).
        주문 코인 표(MANUAL_EXEC_SYMBOLS)에 없는 심볼은 건드리지 않고 unknown_symbol 로 알린다."""
        if request.query.get("confirm") != "1":
            return web.json_response({"ok": False, "error": "confirm=1 required"}, status=400)
        ex = request.query.get("exec", "limit")
        if ex not in ("limit", "market"):
            return web.json_response({"ok": False, "error": "bad_exec", "detail": "exec 는 limit 또는 market"}, status=400)
        if not exec_enabled():
            return web.json_response({"ok": False, "error": "exec_disabled",
                                      "detail": "DASHBOARD_MANUAL_EXEC_ENABLED 가 꺼져 있습니다"}, status=403)
        account = await produce_account()
        if not account.get("ok"):   # «못 읽음»은 «없음»이 아니다
            return web.json_response({"ok": False, "error": "account_unavailable",
                                      "detail": f"계좌 조회 실패: {account.get('error')}"}, status=503)
        targets = list(dict.fromkeys(
            (exec_asset_of(str(p.get("symbol") or "")), str(p.get("side") or "").upper(), str(p.get("symbol") or ""))
            for p in account.get("positions") or []
            if float(p.get("qty") or 0.0) > 0 and str(p.get("side") or "").upper() in ("LONG", "SHORT")))
        if not targets:
            return web.json_response({"ok": False, "error": "no_position", "detail": "열린 포지션이 없습니다"}, status=400)
        results, seen = [], set()
        for asset, side, symbol in targets:
            r = {"asset": asset, "side": side, "symbol": symbol, "ok": False}
            results.append(r)
            if asset is None:
                r.update(error="unknown_symbol", detail=f"{symbol} 은 대시보드 주문 코인이 아닙니다 — 거래소에서 직접 닫으세요")
                continue
            if (asset, side) in seen:   # ETHUSDT·ETHUSDC 가 같은 방향 -- 계획은 한 심볼만 닫는다(other_symbol_open), 칸도 하나
                r.update(error="already_working", detail="같은 코인·방향의 다른 심볼 — 앞 청산이 끝난 뒤 다시 누르세요")
                continue
            seen.add((asset, side))
            st = exit_slot(asset)
            if st is None:
                r.update(error="already_working", detail="이 코인 주문이 진행 중입니다 — 끝난 뒤 다시 누르세요")
                continue
            plan, error = await assemble_exit_plan(side, 1.0, fresh=True, asset=asset, market=ex == "market")
            if error:
                r.update(**error[0])
                continue
            if plan.get("blocked"):
                r.update(error="blocked", detail=plan["blocked"])
                continue
            if exit_slot(asset) is not st:   # 계획 조립(await) 사이에 칸이 바뀌었으면(손절 청산이 잡음) 두 번 내지 않는다
                r.update(error="already_working", detail="이 코인 주문이 막 시작됐습니다 — 끝난 뒤 다시 누르세요")
                continue
            st.clear()
            st.update(phase="submitting", kind="exit", side=side, plan=plan, asset=asset,
                      started_at=datetime.now(timezone.utc).isoformat())
            refresh_tasks["manual_entry" if st is manual_entry_state else f"manual_exit_{asset}"] = asyncio.create_task(
                settle(run_exit(binance_session(), plan, st), st))
            r.update(ok=True, quantity=plan.get("quantity"), type=plan.get("type"), symbol=plan.get("symbol"))
        ok = any(r["ok"] for r in results)
        return web.json_response({"ok": ok, "results": results}, status=200 if ok else 409, headers=NOCACHE)

    async def api_manual_bracket_disarm(request: web.Request) -> web.Response:
        """2026-10-15 사용자 «손절 해제 버튼»(10-14 사고: 바이낸스에서 스탑만 지웠는데 서버 5분봉 종가 감시가 청산했다).
        이 측면의 **우리** 비상 스탑(dbsl algo)을 거래소에서 지우고, 무장에 sl_off 를 남겨 감시가 손절을 쏘지 않게 한다.
        자동 물타기 지정가·TP 는 그대로 두고, 포지션이 닫히면 감시가 평소처럼 정리한다. 다시 무장(새 진입·규칙 물타기)하면 손절이 다시 걸린다.
        🔴거래소 취소가 실패하면 아무것도 안 바꾼다 -- 한 겹만 풀린 상태를 만들지 않는다. 방어는 진입과 같다: POST + confirm=1 + 게이트."""
        if request.query.get("confirm") != "1":
            return web.json_response({"ok": False, "error": "confirm_required", "detail": "confirm=1 이 필요합니다"}, status=400)
        if not exec_enabled():
            return web.json_response({"ok": False, "error": "exec_disabled",
                                      "detail": "DASHBOARD_MANUAL_EXEC_ENABLED 가 꺼져 있습니다"}, status=403)
        side, symbol = request.query.get("side"), request.query.get("symbol")
        if side not in ("LONG", "SHORT") or symbol not in set(MANUAL_EXEC_SYMBOLS.values()):
            return web.json_response({"ok": False, "error": "bad_side", "detail": "side=LONG|SHORT · symbol 이 필요합니다"}, status=400)
        key = bracket_key(symbol, side)
        b = bracket_load().get(key)
        if not b:
            return web.json_response({"ok": False, "error": "not_armed", "detail": "서버가 무장한 손절이 없습니다"}, status=409)
        if b.get("sl_off"):
            return web.json_response({"ok": True, "sl_off": b["sl_off"], "already": True}, headers=NOCACHE)
        r = await clear_bracket(binance_session(), symbol, side, only_sl=True)
        if r.get("errors"):
            return web.json_response({"ok": False, "error": "cancel_failed",
                                      "detail": f"거래소 손절 취소 실패 — 손절은 그대로입니다: {'; '.join(map(str, r['errors']))}"}, status=502)
        st = bracket_load()
        if key not in st or st[key].get("armed_at") != b.get("armed_at"):
            return web.json_response({"ok": False, "error": "rearmed",
                                      "detail": "그 사이 새로 무장됐습니다 — 다시 확인하세요"}, status=409)
        st[key]["sl_off"] = {"at": time.time(), "cancelled": int(r.get("cancelled") or 0)}
        bracket_save(st)
        print(f"bracket {side}: 손절 해제(사용자) 거래소 스탑 {r.get('cancelled')}개 취소 · 5분봉 감시 끔", flush=True)
        return web.json_response({"ok": True, "sl_off": st[key]["sl_off"]}, headers=NOCACHE)

    async def api_manual_bracket_rearm(request: web.Request) -> web.Response:
        """2026-10-15 사용자 «손절 버튼은 다시 누르면 다시 손절을 켜도록». «손절 해제»의 반대: 거래소 비상 스탑(backstop_price)을 다시 걸고
        sl_off 를 지워 5분봉 종가 감시를 다시 켠다. 🔴거래소 스탑을 못 걸면 아무것도 안 바꾼다(감시만 켜진 반쪽 상태를 만들지 않는다).
        마크가가 이미 스탑 너머면 거래소가 거부한다(-2021) -- 그대로 실패로 돌려준다. 방어는 해제와 같다."""
        if request.query.get("confirm") != "1":
            return web.json_response({"ok": False, "error": "confirm_required", "detail": "confirm=1 이 필요합니다"}, status=400)
        if not exec_enabled():
            return web.json_response({"ok": False, "error": "exec_disabled",
                                      "detail": "DASHBOARD_MANUAL_EXEC_ENABLED 가 꺼져 있습니다"}, status=403)
        side, symbol = request.query.get("side"), request.query.get("symbol")
        if side not in ("LONG", "SHORT") or symbol not in set(MANUAL_EXEC_SYMBOLS.values()):
            return web.json_response({"ok": False, "error": "bad_side", "detail": "side=LONG|SHORT · symbol 이 필요합니다"}, status=400)
        key = bracket_key(symbol, side)
        b = bracket_load().get(key)
        if not b:
            return web.json_response({"ok": False, "error": "not_armed", "detail": "서버가 무장한 손절이 없습니다"}, status=409)
        if not b.get("sl_off"):
            return web.json_response({"ok": True, "already": True}, headers=NOCACHE)
        if b.get("lines"):   # 2026-10-28 선 손절 = 거래소 스탑 없음(사용자) -- 감시만 다시 켠다
            st = bracket_load()
            if key not in st or st[key].get("armed_at") != b.get("armed_at"):
                return web.json_response({"ok": False, "error": "rearmed", "detail": "그 사이 새로 무장됐습니다 — 다시 확인하세요"}, status=409)
            st[key].pop("sl_off", None)
            st[key].pop("fired", None)
            st[key]["watch_from"] = time.time()   # 해제 중에 지나간 가격으로 바로 닫지 않게
            bracket_save(st)
            return web.json_response({"ok": True, "lines": True}, headers=NOCACHE)
        price = b.get("backstop_price") or b.get("sl_price")
        if not price:
            return web.json_response({"ok": False, "error": "no_price", "detail": "다시 걸 손절가가 기록에 없습니다"}, status=409)
        r = await place_stop(binance_session(), symbol, side, float(price))
        if not r.get("placed"):
            return web.json_response({"ok": False, "error": "place_failed",
                                      "detail": f"손절 다시 켜기 실패 — 손절은 꺼진 그대로입니다: {r.get('error')}"}, status=502)
        st = bracket_load()
        if key not in st or st[key].get("armed_at") != b.get("armed_at"):
            return web.json_response({"ok": False, "error": "rearmed",
                                      "detail": "그 사이 새로 무장됐습니다 — 다시 확인하세요"}, status=409)
        st[key].pop("sl_off", None)
        st[key]["placed"] = {**(st[key].get("placed") or {}), "backstop": r}
        bracket_save(st)
        print(f"bracket {side}: 손절 다시 켬(사용자) 거래소 스탑 {price} · 5분봉 감시 켬", flush=True)
        return web.json_response({"ok": True, "backstop": r}, headers=NOCACHE)

    async def api_manual_bracket_modify(request: web.Request) -> web.Response:
        """2026-10-29 사용자 «이미 들어간 SLTP 선은 풋프린트에 계속 · 선을 끌면 그대로 주문 수정». 선 무장(lines)만 -- 일반 진입의 −5% 거래소 스탑은 안 건드린다.
        익절 = 우리 post-only 지정가를 지우고 새 가격·포지션 전량으로 다시(place_bracket, 비상 스탑 없음) · 손절 = 서버 감시 레벨만(거래소 주문 없음).
        검사 = 진입 때와 같은 line_bracket(지금가 기준 방향·거리) + 손절은 **진입가**에서 규칙 손절폭(RULE_STOP_PCT)보다 멀게 못 옮긴다(자동 2× 크기의 손실 한도).
        🔴익절 지우기가 실패하면 아무것도 안 바꾼다 · 지운 뒤 새 지정가를 못 걸면 tp_resting False(서버가 1초 닿음으로 닫는다 -- 무방비 아님) ·
        이미 쏜(fired) 무장은 거절 · watch_from = 지금(옮기기 전 지나간 가격으로 바로 닫지 않게). 방어 = 진입과 같다(POST + confirm=1 + 게이트 + 주문 잠금 busy_or)."""
        if request.query.get("confirm") != "1":
            return web.json_response({"ok": False, "error": "confirm_required", "detail": "confirm=1 이 필요합니다"}, status=400)
        if not exec_enabled():
            return web.json_response({"ok": False, "error": "exec_disabled",
                                      "detail": "DASHBOARD_MANUAL_EXEC_ENABLED 가 꺼져 있습니다"}, status=403)
        side, symbol = request.query.get("side"), request.query.get("symbol")
        if side not in ("LONG", "SHORT") or symbol not in set(MANUAL_EXEC_SYMBOLS.values()):
            return web.json_response({"ok": False, "error": "bad_side", "detail": "side=LONG|SHORT · symbol 이 필요합니다"}, status=400)
        lines = query_lines(request)
        if not isinstance(lines, tuple):
            return web.json_response({"ok": False, "error": "bad_lines", "detail": "익절·손절 선(tp·sl)은 둘 다 양수여야 합니다"}, status=400)
        key, asset = bracket_key(symbol, side), exec_asset_of(symbol) or "eth"
        b = bracket_load().get(key)
        if not b or not b.get("lines"):
            return web.json_response({"ok": False, "error": "not_armed_lines", "detail": "선으로 들어간 무장이 없습니다"}, status=409)
        if b.get("fired"):
            return web.json_response({"ok": False, "error": "fired", "detail": "이미 청산이 시작됐습니다"}, status=409)
        account = await swr_cached("binance_account", BINANCE_ACCOUNT_CACHE_SECONDS, produce_account,
                                   max_stale=STALE_GRACE_SECONDS, cache=binance_account_cache)
        pos = next((p for p in account.get("positions") or [] if p.get("symbol") == symbol and p.get("side") == side
                    and abs(float(p.get("qty") or 0.0)) > 0), None)
        if not (account.get("ok") and pos and float(pos.get("entry_price") or 0) > 0):
            return web.json_response({"ok": False, "error": "no_position", "detail": "그 포지션을 계좌에서 못 찾았습니다"}, status=409)
        try:
            book = await fetch_binance_json("https://fapi.binance.com/fapi/v1/ticker/bookTicker", {"symbol": symbol}, error_reason="book_ticker_failed")
            basis, filters = await basis_for(book, asset), await load_filters(binance_session(), symbol)
            pts = sec_prices(asset) or []
            ref = float(pts[-1][1]) if pts else (float(book["bidPrice"]) + float(book["askPrice"])) / 2 / basis   # 지금가(USDT) = 1초 마지막 체결 · 없으면 주문 심볼 중간가 ÷ basis
            nb = line_bracket(position_side=side, ref_level=ref, tp_level=lines[0], sl_level=lines[1], basis=basis, filters=filters)
        except ValueError as exc:
            return web.json_response({"ok": False, "error": "bad_lines", "detail": str(exc)}, status=400)
        except Exception as exc:  # noqa: BLE001 -- 시세·필터를 못 읽으면 아무것도 안 바꾼다
            return web.json_response({"ok": False, "error": "quote_failed", "detail": f"{type(exc).__name__}: {exc}"}, status=502)
        entry_lv = float(pos["entry_price"]) / basis
        far = (1 - lines[1] / entry_lv) if side == "LONG" else (lines[1] / entry_lv - 1)
        if far > RULE_STOP_PCT + 1e-9:
            return web.json_response({"ok": False, "error": "sl_too_far",
                                      "detail": f"손절선이 진입가에서 {100 * far:.2f}% — 규칙 손절폭 {100 * RULE_STOP_PCT:g}% 보다 멀게 못 옮깁니다(손실 한도)"}, status=400)
        tp_moved, tp_res = nb["tp_price"] != b.get("tp_price"), None
        if tp_moved:
            tp_res = await place_bracket(binance_session(), {"tp_price": nb["tp_price"]}, symbol, side)
            if "cleared" not in tp_res or (tp_res.get("cleared") or {}).get("errors"):
                why = "; ".join(map(str, (tp_res.get("cleared") or {}).get("errors") or [])) or tp_res.get("reason") or "?"
                return web.json_response({"ok": False, "error": "tp_replace_failed",
                                          "detail": f"익절 지정가를 못 바꿨습니다 — 원래 주문 그대로일 수 있으니 거래소를 확인하세요: {why}"}, status=502)
        st = bracket_load()
        if key not in st or st[key].get("armed_at") != b.get("armed_at"):
            return web.json_response({"ok": False, "error": "rearmed", "detail": "그 사이 무장이 바뀌었습니다 — 다시 확인하세요"}, status=409)
        cur = st[key]
        cur.update(tp_level=nb["tp_level"], tp_price=nb["tp_price"], sl_level=nb["sl_level"], sl_price=nb["sl_price"], sl_name=nb["sl_name"],
                   watch_from=time.time(), mod={"at": time.time(), "tp": nb["tp_level"], "sl": nb["sl_level"]})
        if tp_moved:
            cur["tp_resting"] = bool((tp_res.get("tp") or {}).get("placed"))
            cur["placed"] = {**(cur.get("placed") or {}), "tp": tp_res.get("tp")}
        bracket_save(st)
        print(f"bracket {side}: 선 수정(사용자) 익절 {b.get('tp_level')}→{nb['tp_level']} 손절 {b.get('sl_level')}→{nb['sl_level']} · "
              f"익절 지정가 {'다시 걸음' if tp_moved else '그대로'} {tp_res and tp_res.get('tp')}", flush=True)
        return web.json_response({"ok": True, "tp_level": nb["tp_level"], "sl_level": nb["sl_level"], "tp_resting": cur.get("tp_resting"),
                                  "tp": tp_res and tp_res.get("tp")}, headers=NOCACHE)

    async def api_manual_entry_status(request: web.Request) -> web.Response:
        """진행 중인 수동 주문 상태. 프런트가 폴링해 «메이커로 채워졌나 / 테이커로 넘어갔나»를
        보여준다. 주문은 최대 하나만 동시에 둔다."""
        return web.json_response({"ok": True, "state": manual_entry_state, "wait": manual_wait_state, "side_exits": side_exit_states,
                                  "exec_enabled": exec_enabled(), "bracket_armed": bracket_load()},
                                 headers=NOCACHE)

    def wait_pending() -> bool:
        return manual_wait_state.get("phase") in ("working", "submitting")

    async def api_manual_wait_action(request: web.Request) -> web.Response:
        """기다리는 진입에 «취소»(잔량 버림 -- 시장가 없음) 또는 «지금 시장가». 진입과 같은 방어: POST + confirm=1 + 게이트.
        집행 루프가 다음 조회(≤3초)에 따른다(run_entry_wait)."""
        action = request.query.get("action")
        if action not in ("cancel", "market") or request.query.get("confirm") != "1":
            return web.json_response({"ok": False, "error": "bad_action",
                                      "detail": "action=cancel|market 과 confirm=1 이 필요합니다"}, status=400)
        if not exec_enabled():
            return web.json_response({"ok": False, "error": "exec_disabled",
                                      "detail": "DASHBOARD_MANUAL_EXEC_ENABLED 가 꺼져 있습니다"}, status=403)
        if not wait_pending():
            return web.json_response({"ok": False, "error": "no_wait", "detail": "기다리는 진입 주문이 없습니다",
                                      "wait": manual_wait_state}, status=409)
        manual_wait_state.update(action=action, action_at=datetime.now(timezone.utc).isoformat(),
                                 action_reason="사용자 취소" if action == "cancel" else None)
        return web.json_response({"ok": True, "wait": manual_wait_state}, headers=NOCACHE)

    async def api_manual_entry_cancel(request: web.Request) -> web.Response:
        """2026-10-11 «따라가기» 진입(peg 지정가 · 120초 뒤 잔량 시장가)의 **대기 취소**. 진입과 같은 방어: POST + confirm=1 + 게이트.
        집행 루프(fill_maker)가 다음 조회(≤3초)에 걸린 GTX 를 취소 확인 뒤 멈추고 잔량은 시장가로 넘기지 않는다 -- 든 몫은 그대로,
        그 몫의 손절·TP 는 진입 종료 경로(entry_then_bracket)가 건다. 확인 못 하면 «거래소에서 직접 확인» 오류로 멈춘다(fill_maker).
        🔴진입에만: 청산 취소는 남은 포지션을 그대로 두는 것이라 안 받는다. 스위칭 반대 진입은 시장가라 기다리는 구간이 없다."""
        if request.query.get("confirm") != "1":
            return web.json_response({"ok": False, "error": "confirm_required", "detail": "confirm=1 이 필요합니다"}, status=400)
        if not exec_enabled():
            return web.json_response({"ok": False, "error": "exec_disabled",
                                      "detail": "DASHBOARD_MANUAL_EXEC_ENABLED 가 꺼져 있습니다"}, status=403)
        st = manual_entry_state
        if (st.get("phase") not in ("working", "submitting") or st.get("kind") == "exit" or st.get("switch")
                or (st.get("plan") or {}).get("type") == "MARKET"):
            return web.json_response({"ok": False, "error": "no_entry", "detail": "기다리는 따라가기 진입이 없습니다",
                                      "state": st}, status=409)
        st.update(action="cancel", action_at=datetime.now(timezone.utc).isoformat(), action_reason="사용자 취소")
        return web.json_response({"ok": True, "state": st}, headers=NOCACHE)

    async def wait_cleanup() -> None:
        """재시작 직후 한 번: 죽은 대기 작업이 남긴 GTX(표식 dbwt)를 지운다. 기록은 상태 칸에 -- 화면이 «재시작으로 취소»를 말한다.
        ponytail: 복구(이어서 기다리기) 대신 정리 -- 배포·재시작 때 대기 주문은 사라진다(이미 든 몫·손절은 그대로)."""
        try:
            r = await cancel_wait_orphans(binance_session(), sorted(set(MANUAL_EXEC_SYMBOLS.values())))
        except Exception as exc:  # noqa: BLE001 -- 기동은 막지 않는다, 대신 남긴다
            r = {"cancelled": [], "errors": [f"{type(exc).__name__}: {exc}"]}
        if r["cancelled"] or r["errors"]:
            manual_wait_state.clear()
            manual_wait_state.update(phase="restart_cancelled", mode="wait", cancelled=r["cancelled"], errors=r["errors"],
                                     done_at=datetime.now(timezone.utc).isoformat())
        print(f"wait cleanup: {r}", flush=True)

    async def start_wait_cleanup(app: web.Application) -> None:
        if exec_enabled():
            refresh_tasks["wait_cleanup"] = asyncio.create_task(wait_cleanup())

    async def busy_or(handler, request: web.Request) -> web.Response:
        if manual_order_lock.locked():
            return web.json_response({"ok": False, "error": "already_working", "state": manual_entry_state}, status=409)
        async with manual_order_lock:
            return await handler(request)

    return SimpleNamespace(
        keep_trip_ledger=keep_trip_ledger,
        start_bracket_watcher=start_bracket_watcher,
        bracket_tick=bracket_tick,   # 시험이 감시 한 틱을 직접 돌린다(test_chart_lines_entry)
        stop_bracket_watcher=stop_bracket_watcher,
        start_wait_cleanup=start_wait_cleanup,
        api_binance_account=api_binance_account,
        api_manual_entry_preview=api_manual_entry_preview,
        busy_or=busy_or,
        api_manual_entry_submit=api_manual_entry_submit,
        api_manual_entry_status=api_manual_entry_status,
        api_manual_wait_action=api_manual_wait_action,
        api_manual_entry_cancel=api_manual_entry_cancel,
        api_manual_bracket_disarm=api_manual_bracket_disarm,
        api_manual_bracket_rearm=api_manual_bracket_rearm,
        api_manual_bracket_modify=api_manual_bracket_modify,
        api_manual_exit_preview=api_manual_exit_preview,
        api_manual_exit_submit=api_manual_exit_submit,
        api_manual_exit_all_submit=api_manual_exit_all_submit,
    )
