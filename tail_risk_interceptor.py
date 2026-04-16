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
_FORCE_ORDER_WS_URL = "wss://fstream.binance.com/ws/{symbol}@forceOrder"

_DB_PATH = str(_ROOT / "data/live/tail_risk.duckdb")
_TABLE   = "tail_risk_1m"


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
        self.liq_cluster_window_sec = int(float(os.getenv("TR_LIQ_CLUSTER_WINDOW_SEC", "900")))
        self.liq_cluster_bucket_pct = float(os.getenv("TR_LIQ_CLUSTER_BUCKET_PCT", "0.001"))
        self.dsac_intercept_enabled = os.getenv("TR_DSAC_INTERCEPT_ENABLE", "false").strip().lower() in ("1", "true", "yes", "on")
        self._ws_connected = False
        self._ws_last_msg_ts = 0.0
        
        self.enabled = os.getenv("TR_ENABLE", "true").strip().lower() in ("1", "true", "yes", "on")

    # ── DuckDB (Bootstrap & Insert) ──────────────────────────────────────────

    def _db_init(self) -> None:
        """초기화 및 최근 30분 데이터 Bootstrap (executor에서 실행)"""
        import duckdb
        os.makedirs(os.path.dirname(_DB_PATH), exist_ok=True)
        con = duckdb.connect(_DB_PATH)
        con.execute(f"""
            CREATE TABLE IF NOT EXISTS {_TABLE} (
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
        # 콜드 스타트 방지용 데이터드
        try:
            rows = con.execute(f"""
                SELECT long_usd_1m, short_usd_1m FROM {_TABLE}
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

    def _db_insert(self, bucket_ts: datetime, long_1m: float, short_1m: float) -> None:
        import duckdb
        try:
            con = duckdb.connect(_DB_PATH)
            con.execute(
                f"""
                INSERT INTO {_TABLE} (
                    ts, long_usd_1m, short_usd_1m, mu_long, sigma_long, mu_short, sigma_short,
                    shadow_aftershock_prob, shadow_decay_half_life, shadow_risk_bucket
                ) VALUES (?,?,?,?,?,?,?,?,?,?)
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

                self._update_hawkes_state(long_1m, short_1m)
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
                    await loop.run_in_executor(None, self._db_insert, bucket_ts, long_1m, short_1m)

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
        ws_stale = (time.time() - self._ws_last_msg_ts) > 30.0 if self._ws_last_msg_ts > 0 else True
        ws_status = "WS STALE" if ws_stale else "WS LIVE"

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
        out = {
            "z_long": float(z_long),
            "z_short": float(z_short),
            "lai": float(lai),
            "long_usd_1m": float(long_1m),
            "short_usd_1m": float(short_1m),
            "hawkes_active": bool(self._hawkes_active),
            "hawkes_decay_level": float(self._hawkes_decay_level),
            "crisis_type": str(self._crisis_type or ""),
        }
        out.update(self._compute_liq_cluster(float(current_price)))
        out.update(self._shadow_state or {})
        return out
