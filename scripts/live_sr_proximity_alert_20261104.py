#!/usr/bin/env python3
"""지지·저항 «가까워짐» 텔레그램 워커 (2026-11-04 사용자 «닿는 게 아니라 가까워지면 알림»).

대시보드 서버가 10초마다 채우는 /api/sr-watch(화면의 지지1~3·저항1~3 = srLevelsLive 와 같은 목록 + 1초 마지막 체결가)를
읽기만 한다 -- 레벨·가격을 여기서 따로 계산하지 않는다(알림 숫자 = 화면 숫자 · 거래소 REST 호출 0).
뚫림 알림은 서버 sr_break_loop 가 따로 보낸다. 이 워커는 «뚫기 전 다가옴» 만.

판정(코인마다 기준거리 THR_BP): 레벨↔현재가 거리 ≤ 기준 이고 «무장» 상태면 한 번 알리고 무장을 푼다.
무장 = 거리가 기준의 REARM_X 배 이상으로 멀어졌을 때(다시 다가오면 또 알림) · 같은 레벨은 COOLDOWN_S 안에 두 번 안 보낸다.
처음 본 레벨이 이미 기준 안이면 알리지 않는다(재시작·한 시간마다 바뀌는 청산맵 때문에 알림이 쏟아지지 않게) -- «멀리 있다가 다가온» 것만.
⚠️ 알림은 매매 트리거가 아니다 -- 문구는 서술형.

기준 = 그 코인 1시간 고저폭 중앙값(11-04 5분봉 200개 실측 ETH 58 · BTC 46 · SOL 75 · XRP 69 · SNDK 134 · SOXL 272bp)의 약 1/3.
ponytail: 상태는 메모리(재시작 직후 첫 폴링은 기준만 잡는다) · 기준은 상수표 -- 변동성 국면이 바뀌면 THR_BP 를 고친다.

실행: python scripts/live_sr_proximity_alert_20261104.py   (supervisor: scripts/ops/supervisor_sr_proximity.sh · crontab @reboot)
자체 시험: python -m pytest -q test/test_sr_proximity_20261104.py
"""
from __future__ import annotations

import json
import os
import sys
import time
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

URL = os.getenv("SR_WATCH_URL", "http://127.0.0.1:8787/api/sr-watch")
POLL_S = 10
STALE_S = 180          # 1초 마지막 체결이 이보다 오래됐으면 그 코인은 건너뛴다(멈춘 가격으로 «가까움» 을 말하지 않는다)
REARM_X = 2.0
COOLDOWN_S = 30 * 60
FORGET_S = 2 * 3600    # 이 시간 안 보인 레벨 상태는 버린다
THR_BP = {"eth": 20, "btc": 15, "sol": 25, "xrp": 25, "sndk": 45, "soxl": 90,
          "skhynix": 45, "mu": 35}   # 2026-11-13 같은 규칙: vision 1시간봉 10-12~11-11 평일 고저폭 중앙 SKHYNIX 131 · MU 103bp 의 ~1/3 (같은 계산 SNDK 135 = 11-04 실측 134 재현)
DEFAULT_THR_BP = 25


def decide(state: dict, view: dict, now: float) -> list[dict]:
    """한 폴링의 판정(순수 함수 -- state 를 갱신한다). view = /api/sr-watch 의 assets. 반환 = 보낼 알림 [{asset, name, level, px, dist_bp, thr_bp, dp, key}]."""
    out = []
    for asset, v in view.items():
        px, t = float(v.get("px") or 0), float(v.get("t") or 0)
        if px <= 0 or now - t > STALE_S:
            continue
        thr = THR_BP.get(asset, DEFAULT_THR_BP)
        for kind, name, levels in (("sup", "지지", v.get("sup") or []), ("res", "저항", v.get("res") or [])):
            for i, lv in enumerate(levels):
                lv = float(lv)
                dist = abs(lv / px - 1) * 1e4
                k = (asset, kind, round(lv, 8))
                s = state.get(k)
                if s is None:   # 처음 본 레벨: 이미 가까우면 무장 안 함(알림 폭주 방지)
                    s = state[k] = {"armed": dist > thr, "last": 0.0, "seen": now}
                s["seen"] = now
                if dist >= REARM_X * thr:
                    s["armed"] = True
                if dist <= thr and s["armed"] and now - s["last"] >= COOLDOWN_S:
                    s.update(armed=False, last=now)
                    out.append({"asset": asset, "name": f"{name}{i + 1}", "level": lv, "px": px, "dist_bp": dist, "thr_bp": thr,
                                "dp": int(v.get("dp") or 2), "key": k})
    for k in [k for k, s in state.items() if now - s["seen"] > FORGET_S]:
        del state[k]
    return out


def message(asset: str, alerts: list[dict]) -> str:
    dp, px = alerts[0]["dp"], alerts[0]["px"]
    lines = [f"{'🔻' if a['name'].startswith('지지') else '🔺'} <b>{asset.upper()} {a['name']} 가까워짐</b> {a['level']:,.{dp}f} "
             f"({(a['level'] / px - 1) * 100:+.2f}% · 기준 {a['thr_bp'] / 100:.2f}%)" for a in alerts]
    return "\n".join(lines + [f"지금 {px:,.{dp}f}"])


def fetch() -> dict | None:
    try:
        with urllib.request.urlopen(URL, timeout=8) as r:
            return json.loads(r.read()).get("assets") or {}
    except (OSError, ValueError):
        return None   # 대시보드가 꺼져 있으면 다음 주기에 다시 -- 크래시 루프가 되지 않는다


def run() -> None:
    from scripts.ops_watchdog import send_telegram   # .env(TELEGRAM_BOT_TOKEN·CHAT_ID)를 읽는다
    state: dict = {}
    print(f"[sr_proximity] start url={URL} thr_bp={THR_BP}", flush=True)
    while True:
        view = fetch()
        if view:
            now = time.time()
            alerts = decide(state, view, now)
            for asset in dict.fromkeys(a["asset"] for a in alerts):
                mine = [a for a in alerts if a["asset"] == asset]
                sent = send_telegram(message(asset, mine))
                print(f"[sr_proximity] {asset} {[a['name'] for a in mine]} sent={sent}", flush=True)
                if sent is False:   # 보내기 실패 = 다시 무장해 다음 폴링에 재시도(설정이 없어 None 이면 재시도해도 소용없다 -- 그대로 둔다)
                    for a in mine:
                        state[a["key"]].update(armed=True, last=0.0)
        time.sleep(POLL_S)


if __name__ == "__main__":
    try:
        run()
    except KeyboardInterrupt:
        pass
