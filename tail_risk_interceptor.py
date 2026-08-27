#!/usr/bin/env python3
"""
TailRiskInterceptor — 사후 요격기 (Post-Crash Interceptor)

대규모 강제청산이나 플래시 크래시 같은 '꼬리 위험(Tail Risk)'이 발생했을 때
DSAC 에이전트의 결정을 무시하고 포지션을 강제로 제어하는 모듈입니다.

전략 3종 (3-Stage Rocket):
  1. 동적 Z-Score (탐지기): 고정 금액이 아닌 최근 30분 변동성 대비 비정상 청산 폭발 감지
  2. LAI 청산 흡수 지수 (판별기): 고래의 매집(Squeeze)인지, 찐 붕괴(Cascade)인지 판별
  3. 호크스 감쇠 (타이머): 붕괴 에너지가 소멸하는 바닥/천장을 계산해 역추세 진입

데이터 파이프라인:
  Bootstrap  : DuckDB 최근 30분 로드 (콜드 스타트 방지)
  Live       : websocket(@forceOrder) 실시간 수신 + Numpy 즉각 연산
  Async I/O  : 1분마다 executor에서 DuckDB 저장
"""
from __future__ import annotations

import asyncio
import json
import logging
import math
import os
import time
from collections import deque
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

_ROOT = Path(__file__).resolve().parent

# ── WebSocket URL ────────────────────────────────────────────────────────────
_FORCE_ORDER_WS_URL = "wss://fstream.binance.com/market/ws/{symbol}@forceOrder"

_DB_PATH = str(os.getenv("QUANT_TAIL_DB_PATH", str(_ROOT / "data/live/tail_risk.duckdb")))
_TABLE   = "tail_risk_1m"
# Small event-triggered sibling of dashboard_state.json's tail_risk block (see
# _write_liq_burst_state()) -- own file so a burst-driven write burst never touches the much
# larger, 10s-timer-driven dashboard_state.json trading_bot.py owns.
_LIQ_BURST_STATE_PATH = _ROOT / "data/live/liq_burst_state.json"


class TailRiskInterceptor:
    """
    사후 요격기: 청산 데이터 기반 위기 관리 및 역추세 타점 포착.

    외부 인터페이스:
        start() / stop()
        intercept(action, pos, kelly, current_price, prev_price) -> (action, kelly, reason)
        status_line() -> str
    """

    def __init__(self, symbol: str = "ethusdt"):
        self.symbol = symbol.lower()
        self._table = _TABLE if self.symbol == "ethusdt" else f"{_TABLE}_{self.symbol.replace('usdt', '')}"
        self._liq_burst_state_path = _LIQ_BURST_STATE_PATH if self.symbol == "ethusdt" else _LIQ_BURST_STATE_PATH.with_name(f"liq_burst_state_{self.symbol.replace('usdt', '')}.json")
        self._running = False

        # ── 태스크 핸들 ───────────────────────────────────────────────
        self._ws_task: asyncio.Task | None = None
        self._agg_task: asyncio.Task | None = None

        # ── 실시간 청산 이벤트 버퍼 (최대 10,000건) ───────────────────
        # (ts_ms, side, qty_usd, price)
        self._liq_events: deque[tuple[int, str, float, float]] = deque(maxlen=10_000)
        self._lock = asyncio.Lock()

        # ── 30분 롤링 윈도우 (Z-Score 계산용) ──────────────────────────
        self.window_size = 30
        self._history_long: deque[float] = deque(maxlen=self.window_size)
        self._history_short: deque[float] = deque(maxlen=self.window_size)

        # ── 통계 캐시 (인터셉터가 O(1)로 읽어감) ───────────────────────
        self.mu_long = 0.0
        self.sigma_long = 1.0
        self.mu_short = 0.0
        self.sigma_short = 1.0
        self.is_warmed_up = False

        # ── 호크스(Hawkes) 과정 상태 변수 ──────────────────────────────
        self._hawkes_active = False
        self._crisis_type: str | None = None  # "LONG_CRISIS" or "SHORT_CRISIS"
        self._last_crisis_ts = 0.0
        self._peak_liq_intensity = 0.0
        self._hawkes_decay_level = 0.0
        self._shadow_state: dict[str, float | str] = {
            "shadow_aftershock_prob": 0.0,
            "shadow_decay_half_life": 0.0,
            "shadow_risk_bucket": "normal",
        }

        # ── 퀀트 파라미터 ──────────────────────────────────────────────
        self.z_threshold = float(os.getenv("TR_Z_THRESHOLD", "3.5"))
        self.lai_threshold = float(os.getenv("TR_LAI_THRESHOLD", "300_000_000")) 
        self.hawkes_beta = float(os.getenv("TR_HAWKES_BETA", "0.005")) # 약 4~5분 뒤 에너지가 25% 이하로 감소하는 속도
        self.hawkes_release_ratio = float(os.getenv("TR_HAWKES_RELEASE_RATIO", "0.35"))
        self.hawkes_sniper_ratio = float(os.getenv("TR_HAWKES_SNIPER_RATIO", "0.55"))
        self.liq_cluster_window_sec = int(float(os.getenv("TR_LIQ_CLUSTER_WINDOW_SEC", "1800")))
        self.liq_cluster_bucket_pct = float(os.getenv("TR_LIQ_CLUSTER_BUCKET_PCT", "0.001"))
        self.dsac_intercept_enabled = os.getenv("TR_DSAC_INTERCEPT_ENABLE", "false").strip().lower() in ("1", "true", "yes", "on")
        self._ws_connected = False
        self._ws_connected_since = 0.0
        self._ws_last_msg_ts = 0.0
        # 2026-07-30: 실제 청산 이벤트 없이도 핸드셰이크만 유지되며 ws_connected=True를 77일간 잘못
        # 보고한 사고 이후 도입. 정상 구간(2026-07-19~) 실측 gap 분포: median=2분, p95=16분, p99=29분,
        # 240분 초과 gap은 3675건 중 1건뿐 -- 4시간 무이벤트는 정상적인 "조용한 시장"으로 설명되지
        # 않는 수준이라 스테일 임계값으로 채택.
        self.liq_stale_threshold_sec = float(os.getenv("TR_LIQ_STALE_THRESHOLD_SEC", "14400"))

        self.enabled = os.getenv("TR_ENABLE", "true").strip().lower() in ("1", "true", "yes", "on")

    # ── DuckDB (Bootstrap & Insert) ──────────────────────────────────────────

    def _db_init(self) -> None:
        """초기화 및 최근 30분 데이터 Bootstrap (executor에서 실행)"""
        import duckdb
        os.makedirs(os.path.dirname(_DB_PATH), exist_ok=True)
        con = duckdb.connect(_DB_PATH)
        con.execute(f"""
            CREATE TABLE IF NOT EXISTS {self._table} (
                ts TIMESTAMPTZ,
                long_usd_1m DOUBLE,
                short_usd_1m DOUBLE,
                mu_long DOUBLE,
                sigma_long DOUBLE,
                mu_short DOUBLE,
                sigma_short DOUBLE,
                shadow_aftershock_prob DOUBLE,
                shadow_decay_half_life DOUBLE,
                shadow_risk_bucket VARCHAR
            )
        """)
        existing_cols = {str(r[1]) for r in con.execute(f"PRAGMA table_info('{self._table}')").fetchall()}
        extra_cols = [
            ("ws_connected", "BOOLEAN"),
            ("ws_stale", "BOOLEAN"),
            ("ws_age_sec", "DOUBLE"),
            ("liq_event_count_1m", "INTEGER"),
            ("valid_liq_stream", "BOOLEAN"),
            ("schema_version", "INTEGER"),
        ]
        for col_name, col_type in extra_cols:
            if col_name not in existing_cols:
                con.execute(f"ALTER TABLE {self._table} ADD COLUMN {col_name} {col_type}")
        # 콜드 스타트 방지용 데이터드
        try:
            rows = con.execute(f"""
                SELECT long_usd_1m, short_usd_1m FROM {self._table}
                WHERE ts >= now() - INTERVAL '{self.window_size} minutes'
                ORDER BY ts ASC
            """).fetchall()
            for (l_val, s_val) in rows:
                self._history_long.append(float(l_val or 0.0))
                self._history_short.append(float(s_val or 0.0))
            
            if len(self._history_long) >= 15:
                self.is_warmed_up = True
            
            self._recalculate_stats()
            logger.info("🛡️ TailRiskInterceptor bootstrap: %d rows loaded", len(rows))
        except Exception as e:
            logger.debug("TR bootstrap skip: %s", e)
        con.close()

    def _stream_health(self) -> tuple[bool, float | None, bool]:
        """실제 메시지 흐름 기준 스트림 상태 판정.

        forceOrder는 이벤트가 없으면 조용한 스트림이라 "이벤트 부재 = 장애"로 볼 수 없지만,
        핸드셰이크(ws_connected)만 보는 것도 2026-07-30에 77일간 실데이터 없이 True를 반환한
        사고의 원인이었다. 마지막 실메시지 이후 경과 시간(또는 메시지를 한 번도 못 받았다면
        연결 이후 경과 시간)이 liq_stale_threshold_sec를 넘으면 stale로 판정한다. 임계값은
        정상 구간(2026-07-19~) 실측 gap 분포(p99=29분, 240분 초과 gap 3675건 중 1건)에서
        나온 값이라, 이 정도 무이벤트는 "조용한 시장"으로 설명되지 않는다.

        Returns (ws_stale, ws_age_sec, valid_liq_stream).
        """
        now_ts = time.time()
        if self._ws_last_msg_ts > 0:
            ws_age_sec = now_ts - self._ws_last_msg_ts
        elif self._ws_connected_since > 0:
            ws_age_sec = now_ts - self._ws_connected_since
        else:
            ws_age_sec = None
        ws_stale = (not self._ws_connected) or (ws_age_sec is None) or (ws_age_sec > self.liq_stale_threshold_sec)
        valid_liq_stream = bool(self._ws_connected) and not ws_stale
        return ws_stale, ws_age_sec, valid_liq_stream

    def _db_insert(self, bucket_ts: datetime, long_1m: float, short_1m: float, liq_event_count_1m: int = 0) -> None:
        import duckdb
        try:
            ws_stale, ws_age_sec, valid_liq_stream = self._stream_health()
            con = duckdb.connect(_DB_PATH)
            con.execute(
                f"""
                INSERT INTO {self._table} (
                    ts, long_usd_1m, short_usd_1m, mu_long, sigma_long, mu_short, sigma_short,
                    shadow_aftershock_prob, shadow_decay_half_life, shadow_risk_bucket,
                    ws_connected, ws_stale, ws_age_sec, liq_event_count_1m, valid_liq_stream, schema_version
                ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                """,
                [
                    bucket_ts,
                    long_1m,
                    short_1m,
                    self.mu_long,
                    self.sigma_long,
                    self.mu_short,
                    self.sigma_short,
                    float(self._shadow_state.get("shadow_aftershock_prob", 0.0)),
                    float(self._shadow_state.get("shadow_decay_half_life", 0.0)),
                    str(self._shadow_state.get("shadow_risk_bucket", "normal")),
                    bool(self._ws_connected),
                    bool(ws_stale),
                    float(ws_age_sec) if ws_age_sec is not None else None,
                    int(liq_event_count_1m),
                    bool(valid_liq_stream),
                    3,
                ]
            )
            con.close()
        except Exception as e:
            # 🚨 DB 관련 에러도 숨기지 않고 출력합니다.
            logger.error("🚨 사후 요격기 DB 저장 에러: %s", e, exc_info=True)

    # ── 통계 연산 ─────────────────────────────────────────────────────────────

    def _recalculate_stats(self) -> None:
        """인메모리 버퍼(Numpy)를 통한 Z-Score 통계 갱신"""
        if len(self._history_long) > 0:
            self.mu_long = float(np.mean(self._history_long))
            self.sigma_long = float(np.std(self._history_long) + 1e-6)
            
            self.mu_short = float(np.mean(self._history_short))
            self.sigma_short = float(np.std(self._history_short) + 1e-6)

    def _aggregate_1m(self) -> tuple[float, float]:
        """최근 1분간의 청산 금액 집계"""
        now_ms = int(time.time() * 1000)
        cutoff = now_ms - 60_000
        long_usd = 0.0
        short_usd = 0.0
        
        # 최근 이벤트부터 순회 (최적화)
        for ts, side, usd, _price in reversed(self._liq_events):
            if ts < cutoff:
                break
            if side == "SELL":
                long_usd += usd  # 롱 포지션 청산 = 시장에 SELL
            else:
                short_usd += usd # 숏 포지션 청산 = 시장에 BUY
                
        return long_usd, short_usd

    def _count_liq_events_1m(self) -> int:
        now_ms = int(time.time() * 1000)
        cutoff = now_ms - 60_000
        cnt = 0
        for ts, _side, usd, _price in reversed(self._liq_events):
            if ts < cutoff:
                break
            if usd > 0:
                cnt += 1
        return int(cnt)

    def _compute_liq_cluster(self, current_price: float) -> dict[str, float | int]:
        """최근 청산 분포 기반 자석 방향 추정."""
        if current_price <= 0:
            return {
                "liq_cluster_direction": 0,
                "liq_cluster_strength": 0.0,
                "distance_to_cluster_pct": 1.0,
                "liq_cluster_price": 0.0,
            }
        now_ms = int(time.time() * 1000)
        cutoff = now_ms - int(self.liq_cluster_window_sec * 1000)
        step = max(current_price * self.liq_cluster_bucket_pct, current_price * 0.0005)
        above_short: dict[float, float] = {}
        below_long: dict[float, float] = {}
        total = 0.0
        for ts, side, usd, price in reversed(self._liq_events):
            if ts < cutoff:
                break
            if usd <= 0 or price <= 0:
                continue
            total += usd
            bucket = round(price / step) * step
            if side == "BUY" and price > current_price:
                above_short[bucket] = above_short.get(bucket, 0.0) + usd
            elif side == "SELL" and price < current_price:
                below_long[bucket] = below_long.get(bucket, 0.0) + usd

        best_up_price, best_up_usd = (0.0, 0.0)
        if above_short:
            best_up_price, best_up_usd = max(above_short.items(), key=lambda x: x[1])
        best_dn_price, best_dn_usd = (0.0, 0.0)
        if below_long:
            best_dn_price, best_dn_usd = max(below_long.items(), key=lambda x: x[1])

        if best_up_usd <= 0 and best_dn_usd <= 0:
            return {
                "liq_cluster_direction": 0,
                "liq_cluster_strength": 0.0,
                "distance_to_cluster_pct": 1.0,
                "liq_cluster_price": 0.0,
            }
        if best_up_usd >= best_dn_usd:
            direction = 1
            cluster_price = best_up_price
            cluster_usd = best_up_usd
        else:
            direction = -1
            cluster_price = best_dn_price
            cluster_usd = best_dn_usd
        strength = float(np.clip(cluster_usd / max(total, 1e-8), 0.0, 1.0))
        dist = abs(cluster_price - current_price) / max(current_price, 1e-8)
        return {
            "liq_cluster_direction": int(direction),
            "liq_cluster_strength": float(strength),
            "distance_to_cluster_pct": float(dist),
            "liq_cluster_price": float(cluster_price),
        }

    def _compute_shadow_aftershock(self, long_1m: float, short_1m: float) -> dict[str, float | str]:
        """
        Shadow 전략군 D: 여진 확률 추정 (로그 전용, 매매 미개입).
        """
        z_long = (long_1m - self.mu_long) / max(self.sigma_long, 1.0)
        z_short = (short_1m - self.mu_short) / max(self.sigma_short, 1.0)
        z_peak = max(z_long, z_short, 0.0)
        imbalance = abs(long_1m - short_1m) / (long_1m + short_1m + 1e-8)
        active = float(np.clip(self._hawkes_decay_level, 0.0, 1.0))

        # 보수적 근사치: z + 불균형 + hawkes 활성 상태
        prob = float(np.clip(0.45 * (z_peak / max(self.z_threshold, 1e-8)) + 0.35 * imbalance + 0.20 * active, 0.0, 1.0))
        half_life_min = float(np.log(2.0) / max(self.hawkes_beta * 60.0, 1e-6))
        if prob >= 0.75:
            bucket = "high"
        elif prob >= 0.45:
            bucket = "watch"
        else:
            bucket = "normal"
        return {
            "shadow_aftershock_prob": prob,
            "shadow_decay_half_life": half_life_min,
            "shadow_risk_bucket": bucket,
        }

    def _update_hawkes_state(self, long_1m: float, short_1m: float) -> None:
        """청산 급증 + 지수 감쇠(Hawkes 근사) 상태 갱신."""
        z_long = (long_1m - self.mu_long) / max(self.sigma_long, 1.0)
        z_short = (short_1m - self.mu_short) / max(self.sigma_short, 1.0)
        z_peak = max(z_long, z_short, 0.0)
        now = time.time()
        crisis = "LONG_CRISIS" if z_long >= z_short else "SHORT_CRISIS"

        if z_peak >= self.z_threshold:
            if (not self._hawkes_active) or (self._crisis_type != crisis):
                self._peak_liq_intensity = z_peak
            else:
                self._peak_liq_intensity = max(self._peak_liq_intensity, z_peak)
            self._hawkes_active = True
            self._crisis_type = crisis
            self._last_crisis_ts = now
            self._hawkes_decay_level = 1.0
            return

        if not self._hawkes_active:
            self._hawkes_decay_level = 0.0
            return

        elapsed = max(0.0, now - self._last_crisis_ts)
        decayed = self._peak_liq_intensity * math.exp(-self.hawkes_beta * elapsed)
        self._hawkes_decay_level = float(np.clip(decayed / max(self.z_threshold, 1e-8), 0.0, 1.0))
        if decayed <= self.z_threshold * self.hawkes_release_ratio:
            self._hawkes_active = False
            self._crisis_type = None
            self._peak_liq_intensity = 0.0

    def _write_liq_burst_state(self, long_1m: float, short_1m: float) -> None:
        """Small dedicated burst-alert state file, written from two call sites -- 2026-08-27 user
        request for sub-few-second "sudden liquidation" detection, separate from the once-a-minute
        tail_risk.duckdb persistence and the 10s-cadence dashboard_state.json handoff (both still
        unchanged, still their own writers). (1) _ws_loop() calls this right after each new event is
        appended and _update_hawkes_state() has just run on it, so a burst's ONSET is reflected the
        instant it happens instead of waiting for the next periodic tick. (2) _agg_loop() ALSO calls
        this every 10s regardless of new events -- added 2026-08-27 after discovering that without
        it, hawkes_decay_level/long_usd_1m/short_usd_1m would freeze at whatever they were at the
        last event and could show a "crisis still active" snapshot minutes stale once a cascade had
        actually already decayed away, since decay is a function of elapsed time, not of new events
        arriving. Onset latency stays event-driven (fast); decay/resolution latency is now bounded to
        ~10s instead of unbounded. Cheap by design: a handful of scalar fields, no external calls, so
        even a genuine multi-event-per-second cascade just means a few small writes per second, not a
        bottleneck. dashboard/server.py's load_json_cached() already keys off (mtime, size) rather
        than a timer, so it picks up whichever call wrote most recently on its very next poll."""
        try:
            z_long = (long_1m - self.mu_long) / max(self.sigma_long, 1.0)
            z_short = (short_1m - self.mu_short) / max(self.sigma_short, 1.0)
            ws_stale, ws_age_sec, valid_liq_stream = self._stream_health()
            payload = {
                "updated_at": datetime.utcnow().isoformat() + "Z",
                "hawkes_active": bool(self._hawkes_active),
                "hawkes_decay_level": float(self._hawkes_decay_level),
                "crisis_type": str(self._crisis_type or ""),
                "z_long": float(z_long),
                "z_short": float(z_short),
                "long_usd_1m": float(long_1m),
                "short_usd_1m": float(short_1m),
                "liq_event_count_1m": int(self._count_liq_events_1m()),
                "ws_connected": bool(self._ws_connected),
                "ws_age_sec": (float(ws_age_sec) if ws_age_sec is not None else None),
                "valid_liq_stream": bool(valid_liq_stream),
            }
            tmp_path = self._liq_burst_state_path.with_suffix(".json.tmp")
            tmp_path.write_text(json.dumps(payload))
            tmp_path.replace(self._liq_burst_state_path)  # atomic on POSIX -- never a half-written read
        except Exception:
            logger.debug("liq_burst_state write failed", exc_info=True)

    # ── WebSocket ────────────────────────────────────────────────────────────

    async def _ws_loop(self) -> None:
        """바낸 @forceOrder 스트림 (청산 이벤트 발생 시에만 틱이 옴)"""
        import websockets
        url = _FORCE_ORDER_WS_URL.format(symbol=self.symbol)
        delay = 3.0
        while self._running:
            try:
                async with websockets.connect(url, ping_interval=20, ping_timeout=10) as ws:
                    logger.info("🛡️ TR ForceOrder WS 연결: %s", url)
                    delay = 3.0
                    self._ws_connected = True
                    self._ws_connected_since = time.time()
                    async for raw in ws:
                        if not self._running:
                            break
                        try:
                            msg = json.loads(raw)
                            o = msg.get("o", {})
                            side = str(o.get("S", ""))
                            qty = float(o.get("l", 0.0))
                            price = float(o.get("ap", 0.0))
                            ts_ms = int(msg.get("E", 0))
                            qty_usd = qty * price
                            self._ws_last_msg_ts = time.time()
                            
                            if qty_usd > 0 and side in ("BUY", "SELL"):
                                async with self._lock:
                                    self._liq_events.append((ts_ms, side, qty_usd, price))
                                    long_1m, short_1m = self._aggregate_1m()
                                self._update_hawkes_state(long_1m, short_1m)
                                self._write_liq_burst_state(long_1m, short_1m)
                        except Exception:
                            pass
            except Exception as e:
                self._ws_connected = False
                if self._running:
                    logger.debug("TR WS 재연결 (%.0fs): %s", delay, e)
                    await asyncio.sleep(delay)
                    delay = min(delay * 1.5, 60.0)

    async def _agg_loop(self) -> None:
        last_saved_minute = -1
        while self._running:
            try:
                # 1. 10초 동기화
                now = time.time()
                sleep_sec = 10.0 - (now % 10.0)
                await asyncio.sleep(sleep_sec + 0.05)  # asyncio 시간 오차 방지용 0.05초 버퍼

                if not self._running:
                    break

                async with self._lock:
                    long_1m, short_1m = self._aggregate_1m()
                    liq_event_count_1m = self._count_liq_events_1m()

                self._update_hawkes_state(long_1m, short_1m)
                self._write_liq_burst_state(long_1m, short_1m)
                # 2. 매 10초마다 섀도우 상태 갱신 및 터미널 로그 출력
                self._shadow_state = self._compute_shadow_aftershock(long_1m, short_1m)
                logger.info("%s", self.status_line())

                # 3. 정각(00초) 판별하여 통계 이력(History) 누적 및 DB 저장 (1분에 1번)
                dt_now = datetime.now()
                if dt_now.minute != last_saved_minute and dt_now.second < 15:
                    last_saved_minute = dt_now.minute

                    self._history_long.append(long_1m)
                    self._history_short.append(short_1m)
                    if len(self._history_long) >= 15:
                        self.is_warmed_up = True
                    self._recalculate_stats()

                    # 방금 끝난 1분의 정각 타임스탬프 계산
                    bucket_ts = (dt_now - timedelta(minutes=1)).replace(second=0, microsecond=0)

                    loop = asyncio.get_running_loop()
                    await loop.run_in_executor(None, self._db_insert, bucket_ts, long_1m, short_1m, liq_event_count_1m)

            except Exception as e:
                logger.error("🚨 사후 요격기(_agg_loop) 치명적 에러 발생: %s", e, exc_info=True)

    # ── 🚀 메인 요격 알고리즘 (3-Stage Rocket) ───────────────────────────────

    def intercept(
        self,
        action: int,
        pos: str | None,
        kelly: float,
        current_price: float,
        prev_price: float
    ) -> tuple[int, float, str]:
        """
        DSAC 결정을 가로채어 위험을 관리하거나 스퀴즈 알파를 창출합니다.
        
        Returns:
            (수정된 action, 수정된 kelly, 요격 사유)
        """
        if not self.dsac_intercept_enabled:
            return action, kelly, ""
        if not self.enabled or not self.is_warmed_up:
            return action, kelly, ""

        long_1m, short_1m = self._aggregate_1m()
        z_long = (long_1m - self.mu_long) / max(self.sigma_long, 1.0)
        z_short = (short_1m - self.mu_short) / max(self.sigma_short, 1.0)
        dominant_liq = long_1m if z_long >= z_short else short_1m
        price_change_pct = (current_price - prev_price) / max(abs(prev_price), 1e-8) if prev_price > 0 else 0.0
        lai = dominant_liq / max(abs(price_change_pct), 0.0001)

        # 여진 진행 구간에서는 진입 차단
        if self._hawkes_active:
            return 0, 0.0, "HAWKES_ACTIVE_HOLD"

        # 감쇠 후 역추세 스나이핑
        if self._hawkes_decay_level >= self.hawkes_sniper_ratio and lai >= self.lai_threshold:
            if self._crisis_type == "LONG_CRISIS":
                return 1, min(float(kelly) * 1.3, 1.0), "HAWKES_DECAY_SNIPER_LONG"
            if self._crisis_type == "SHORT_CRISIS":
                return 2, min(float(kelly) * 1.3, 1.0), "HAWKES_DECAY_SNIPER_SHORT"
        return action, kelly, ""

    # ── 시작 / 정지 / 모니터링 ───────────────────────────────────────────────

    def start(self) -> None:
        if not self.enabled:
            logger.info("TailRiskInterceptor disabled (TR_ENABLE=false)")
            return
        loop = asyncio.get_event_loop()
        loop.run_in_executor(None, self._db_init)
        self._running = True
        self._ws_task = asyncio.create_task(self._ws_loop())
        self._agg_task = asyncio.create_task(self._agg_loop())
        logger.info("🛡️ TailRiskInterceptor 시작 (%s)", self.symbol.upper())

    def stop(self) -> None:
        self._running = False
        for t in (self._ws_task, self._agg_task):
            if t and not t.done():
                t.cancel()

    def status_line(self) -> str:
        """트레이딩봇 요약 패널 스타일 로그."""
        if not self.enabled:
            return "🛡️ [사후 요격기] 비활성화됨"
        if not self.is_warmed_up:
            return "🛡️ [사후 요격기] 데이터 수집 및 예열 중 (Warming up)..."

        long_1m, short_1m = self._aggregate_1m()
        z_long = (long_1m - self.mu_long) / max(self.sigma_long, 1.0)
        z_short = (short_1m - self.mu_short) / max(self.sigma_short, 1.0)
        z_status = "평온"
        if z_long >= 2.0:
            z_status = f"롱 청산 급증⚠️ (Z:{z_long:.1f})"
        if z_short >= 2.0:
            z_status = f"숏 청산 급증⚠️ (Z:{z_short:.1f})"

        if self._hawkes_active:
            hawkes_status = f"여진 진행중 ({self._crisis_type})"
        else:
            hawkes_status = "안전 구간"
        ws_stale, ws_age_sec, valid_liq_stream = self._stream_health()
        if not self._ws_connected:
            ws_status = "WS DISCONNECTED"
        elif ws_stale:
            ws_status = (
                f"WS CONNECTED / STALE (no liq {ws_age_sec:.0f}s, threshold={self.liq_stale_threshold_sec:.0f}s)"
                if ws_age_sec is not None else "WS CONNECTED / STALE (no message ever received)"
            )
        else:
            ws_status = f"WS CONNECTED / last_liq={ws_age_sec:.0f}s"

        shadow_prob = float(self._shadow_state.get("shadow_aftershock_prob", 0.0))
        shadow_hl = float(self._shadow_state.get("shadow_decay_half_life", 0.0))
        shadow_bucket = str(self._shadow_state.get("shadow_risk_bucket", "normal"))
        if shadow_bucket == "high":
            shadow_txt = "HOLD 권고"
            dir_txt = "HOLD 유리"
            dir_tag = "[HOLD]"
            reason_txt = "여진확률 높음 + 변동 충격 지속"
        elif shadow_bucket == "watch":
            shadow_txt = "보수적 진입"
            dir_txt = "추세추종 소규모 유리"
            dir_tag = "[LONG/SHORT]"
            reason_txt = "여진 감시 구간(비중 축소)"
        else:
            shadow_txt = "정상 운용"
            dir_txt = "기존 신호 추종 유리"
            dir_tag = "[LONG/SHORT]"
            reason_txt = "여진 위험 낮음"

        return (
            "\n┌─ 🛡️  TAIL RISK PANEL ───────────────────────────────\n"
            f"│ 유리방향  {dir_txt}\n"
            f"│ 근거  {reason_txt} | p={shadow_prob:.2f} bucket={shadow_bucket}\n"
            f"│ 위협  {z_status}\n"
            f"│ 상태  {hawkes_status}\n"
            f"│ 연결  {ws_status}\n"
            f"│ 평균  L={self.mu_long/1e6:.2f}M / S={self.mu_short/1e6:.2f}M (1m baseline)\n"
            f"│ SHDW  D:{shadow_txt} | p={shadow_prob:.2f} ({shadow_bucket}) | half-life={shadow_hl:.1f}m | hawkes={self._hawkes_decay_level:.2f}\n"
            "└────────────────────────────────────────────────────"
        )

    def get_playbook_signal(self, price_change_pct: float = 0.0, current_price: float = 0.0) -> dict[str, float | str]:
        """
        플레이북 라우팅용 경량 신호 스냅샷.
        """
        long_1m, short_1m = self._aggregate_1m()
        z_long = (long_1m - self.mu_long) / max(self.sigma_long, 1.0)
        z_short = (short_1m - self.mu_short) / max(self.sigma_short, 1.0)
        dominant_liq = long_1m if z_long >= z_short else short_1m
        lai = dominant_liq / max(abs(float(price_change_pct)), 0.0001)
        ws_stale, ws_age_sec, valid_liq_stream = self._stream_health()
        out = {
            "z_long": float(z_long),
            "z_short": float(z_short),
            "lai": float(lai),
            "long_usd_1m": float(long_1m),
            "short_usd_1m": float(short_1m),
            "liq_event_count_1m": int(self._count_liq_events_1m()),
            "ws_connected": bool(self._ws_connected),
            "ws_stale": bool(ws_stale),
            "ws_age_sec": float(ws_age_sec) if ws_age_sec is not None else None,
            "valid_liq_stream": bool(valid_liq_stream),
            "hawkes_active": bool(self._hawkes_active),
            "hawkes_decay_level": float(self._hawkes_decay_level),
            "crisis_type": str(self._crisis_type or ""),
        }
        out.update(self._compute_liq_cluster(float(current_price)))
        out.update(self._shadow_state or {})
        return out
